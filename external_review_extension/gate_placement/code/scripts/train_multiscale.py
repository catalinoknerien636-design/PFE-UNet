from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.pfe_unet import PFEUNet
from models.unet import UNet
from scripts.device_utils import describe_device, get_device
from extension_dataset import ExtensionPCODDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen PFE evidence-extension experiments.")
    parser.add_argument("--model", choices=["unet", "pfe_unet"], required=True)
    parser.add_argument("--input_mode", choices=["rgb_dop", "rgb_s0", "rgb_s1", "rgb_s2", "rgb_gray"], required=True)
    parser.add_argument("--pfe_fusion", choices=["multiply", "conv1x1"], default="multiply")
    parser.add_argument(
        "--active_gate_levels",
        default="L1,L2,L3,L4,L5",
        help="Comma-separated subset of L1..L5. Use an empty string for identity bypass.",
    )
    parser.add_argument("--registration_aug", action="store_true")
    parser.add_argument("--dataset_root", type=Path, default=REPOSITORY_ROOT / "data" / "PCOD_1200" / "full-dataset")
    parser.add_argument("--train_split_file", type=Path, default=REPOSITORY_ROOT / "splits" / "train_split.csv")
    parser.add_argument("--val_split_file", type=Path, default=REPOSITORY_ROOT / "splits" / "val_split.csv")
    parser.add_argument("--extension_root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="directml")
    parser.add_argument("--directml_device", type=int, default=1)
    return parser.parse_args()


def parse_gate_levels(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    levels: list[int] = []
    for token in value.split(","):
        normalized = token.strip().upper()
        if normalized.startswith("L"):
            normalized = normalized[1:]
        level = int(normalized)
        if level not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Gate level must be between L1 and L5: {token!r}")
        levels.append(level)
    if len(set(levels)) != len(levels):
        raise ValueError(f"Duplicate active gate level: {value}")
    return tuple(sorted(levels))


def build_model(
    model_name: str,
    fusion: str,
    base_channels: int,
    active_gate_levels: tuple[int, ...],
) -> nn.Module:
    if model_name == "unet":
        return UNet(in_channels=4, out_channels=1, base_channels=base_channels)
    return PFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=base_channels,
        enhancement="all",
        fusion=fusion,
        active_gate_levels=active_gate_levels,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = torch.clamp(logits, min=0) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    return bce.mean() + dice_loss(logits, targets)


@torch.no_grad()
def batch_counts(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach()).cpu()
    truth = targets.detach().cpu() > 0.5
    preds = probs > 0.5
    return {
        "tp": float(torch.logical_and(preds, truth).sum().item()),
        "fp": float(torch.logical_and(preds, torch.logical_not(truth)).sum().item()),
        "fn": float(torch.logical_and(torch.logical_not(preds), truth).sum().item()),
        "tn": float(torch.logical_and(torch.logical_not(preds), torch.logical_not(truth)).sum().item()),
        "mae": float(torch.mean(torch.abs(probs - targets.detach().cpu())).item()),
    }


def finalize(sums: dict[str, float]) -> dict[str, float]:
    eps = 1e-7
    tp, fp, fn = sums.get("tp", 0.0), sums.get("fp", 0.0), sums.get("fn", 0.0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    mae = sums.get("mae_sum", 0.0) / max(sums.get("batches", 1.0), 1.0)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "mae_batch_mean": mae}


def run_epoch(model: nn.Module, loader: DataLoader, device, optimizer=None) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    sums: dict[str, float] = {}
    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        counts = batch_counts(logits, masks)
        for key in ("tp", "fp", "fn", "tn"):
            sums[key] = sums.get(key, 0.0) + counts[key]
        sums["mae_sum"] = sums.get("mae_sum", 0.0) + counts["mae"]
        sums["batches"] = sums.get("batches", 0.0) + 1.0
    return total_loss / max(len(loader), 1), finalize(sums)


def write_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    active_gate_levels = parse_gate_levels(args.active_gate_levels)
    set_seed(args.seed)
    device = get_device(args.device, args.directml_device)
    checkpoint_dir = args.extension_root / "checkpoints" / args.run_name
    metrics_dir = args.extension_root / "metrics" / args.run_name
    history_path = metrics_dir / "history.csv"
    if history_path.exists() and args.resume_checkpoint is None:
        existing = list(csv.DictReader(history_path.open("r", newline="", encoding="utf-8")))
        if existing:
            raise FileExistsError(
                f"Run already has history and will not be overwritten: {history_path}. Use --resume_checkpoint."
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ExtensionPCODDataset(
        args.dataset_root,
        split="train",
        input_mode=args.input_mode,
        image_size=args.image_size,
        stems_file=args.train_split_file,
        max_samples=args.max_train_samples,
        registration_aug=args.registration_aug,
    )
    val_dataset = ExtensionPCODDataset(
        args.dataset_root,
        split="train",
        input_mode=args.input_mode,
        image_size=args.image_size,
        stems_file=args.val_split_file,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        args.pfe_fusion,
        args.base_channels,
        active_gate_levels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False
    )
    config = vars(args) | {
        "device_description": describe_device(device),
        "in_channels": 4,
        "checkpoint_selection": "highest validation IoU; test not used for selection",
        "active_gate_levels_normalized": [f"L{level}" for level in active_gate_levels],
        "inactive_gate_behavior": "exact identity bypass; gate module is not called",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (metrics_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    history: list[dict] = []
    start_epoch = 1
    best_iou = -1.0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history for resume: {history_path}")
        history = list(csv.DictReader(history_path.open("r", newline="", encoding="utf-8")))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = max(float(row["val_iou"]) for row in history)

    print(f"Run name: {args.run_name}")
    print(f"Device: {describe_device(device)}")
    print(f"Model/input/fusion: {args.model}/{args.input_mode}/{args.pfe_fusion}")
    print(
        "Active gate levels: "
        + (",".join(f"L{level}" for level in active_gate_levels) or "none")
    )
    print(f"Registration augmentation: {args.registration_aug}")
    print(f"Train/validation samples: {len(train_dataset)}/{len(val_dataset)}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_iou": train_metrics["iou"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_mae": train_metrics["mae_batch_mean"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_mae": val_metrics["mae_batch_mean"],
        }
        history.append(row)
        write_history(history, history_path)
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": config,
            "metrics": row,
        }
        torch.save(latest, checkpoint_dir / "latest.pt")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(latest, checkpoint_dir / "best.pt")
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"val_mae={val_metrics['mae_batch_mean']:.4f}"
        )
    print(f"Best val IoU: {best_iou:.6f}")


if __name__ == "__main__":
    main()
