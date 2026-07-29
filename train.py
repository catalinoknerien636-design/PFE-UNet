from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.extension_pcod import ExtensionPCODDataset
from models.pfe_unet import PFEUNet
from models.unet import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets, dim=(1, 2, 3))
    union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"dataset_root", "train_split", "validation_split", "model", "input_mode", "seed"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return config


def parse_active_gate_levels(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = value.replace(";", ",").split(",") if isinstance(value, str) else value
    levels = []
    for item in values:
        token = str(item).strip().upper()
        if not token:
            continue
        if token.startswith("L"):
            token = token[1:]
        levels.append(int(token))
    return tuple(levels)


def build_model(config: dict) -> torch.nn.Module:
    if config["model"] == "unet":
        return UNet(in_channels=4, out_channels=1, base_channels=int(config.get("base_channels", 32)))
    return PFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=int(config.get("base_channels", 32)),
        enhancement="all",
        fusion=config.get("fusion", "multiply"),
        active_gate_levels=parse_active_gate_levels(config.get("active_gate_levels")),
    )


def metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach())
    pred = probs > 0.5
    truth = masks > 0.5
    tp = float(torch.logical_and(pred, truth).sum())
    fp = float(torch.logical_and(pred, ~truth).sum())
    fn = float(torch.logical_and(~pred, truth).sum())
    eps = 1e-7
    return {"iou": tp / (tp + fp + fn + eps), "mae_image_sum": float(torch.abs(probs - masks).mean(dim=(1,2,3)).sum()), "images": masks.shape[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = Path(cfg["dataset_root"])
    train_set = ExtensionPCODDataset(root, "train", cfg["input_mode"], int(cfg["image_size"]), Path(cfg["train_split"]), args.max_train_samples)
    val_set = ExtensionPCODDataset(root, "train", cfg["input_mode"], int(cfg["image_size"]), Path(cfg["validation_split"]), args.max_validation_samples)
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]), foreach=False)
    out = Path(cfg["checkpoint_dir"])
    metrics_out = Path(cfg["evaluation_output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    metrics_out.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), mask)
            loss.backward()
            optimizer.step()
        model.eval()
        total_tp = total_fp = total_fn = 0.0
        image_mae_sum = images = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image, mask = batch["image"].to(device), batch["mask"].to(device)
                logits = model(image)
                probs = torch.sigmoid(logits)
                pred, truth = probs > 0.5, mask > 0.5
                total_tp += float(torch.logical_and(pred, truth).sum())
                total_fp += float(torch.logical_and(pred, ~truth).sum())
                total_fn += float(torch.logical_and(~pred, truth).sum())
                image_mae_sum += float(torch.abs(probs-mask).mean(dim=(1,2,3)).sum())
                images += mask.shape[0]
        val_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
        val_mae = image_mae_sum / max(images, 1)
        row = {"epoch": epoch, "val_iou": val_iou, "val_image_mae": val_mae}
        history.append(row)
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg, "metrics": row}
        torch.save(state, out / "latest.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(state, out / "best.pt")
        with (metrics_out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
    (metrics_out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
