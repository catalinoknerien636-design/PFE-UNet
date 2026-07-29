from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (CODE_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.device_utils import describe_device, get_device
from extension_dataset import ExtensionPCODDataset
from train_multiscale import build_model, parse_gate_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one frozen extension checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model", choices=["unet", "pfe_unet"], required=True)
    parser.add_argument("--input_mode", choices=["rgb_dop", "rgb_s0", "rgb_s1", "rgb_s2", "rgb_gray"], required=True)
    parser.add_argument("--pfe_fusion", choices=["multiply", "conv1x1"], default="multiply")
    parser.add_argument("--active_gate_levels", default="L1,L2,L3,L4,L5")
    parser.add_argument("--dataset_root", type=Path, default=REPOSITORY_ROOT / "data" / "PCOD_1200" / "full-dataset")
    parser.add_argument("--extension_root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="directml")
    parser.add_argument("--directml_device", type=int, default=1)
    return parser.parse_args()


def per_image_metrics(prob: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    pred = prob > 0.5
    truth = mask > 0.5
    tp = float(torch.logical_and(pred, truth).sum().item())
    fp = float(torch.logical_and(pred, torch.logical_not(truth)).sum().item())
    fn = float(torch.logical_and(torch.logical_not(pred), truth).sum().item())
    tn = float(torch.logical_and(torch.logical_not(pred), torch.logical_not(truth)).sum().item())
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    mae = float(torch.mean(torch.abs(prob - mask)).item())
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mae_image": mae,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate(rows: list[dict], batch_maes: list[float]) -> dict[str, float]:
    eps = 1e-7
    tp = sum(float(row["tp"]) for row in rows)
    fp = sum(float(row["fp"]) for row in rows)
    fn = sum(float(row["fn"]) for row in rows)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    mae_batch = float(np.mean(batch_maes))
    mae_image = float(np.mean([float(row["mae_image"]) for row in rows]))
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mae": mae_batch,
        "mae_batch_mean": mae_batch,
        "mae_image_mean": mae_image,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(prob: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.rint(prob.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    args = parse_args()
    active_gate_levels = parse_gate_levels(args.active_gate_levels)
    device = get_device(args.device, args.directml_device)
    dataset = ExtensionPCODDataset(
        args.dataset_root,
        split="test",
        input_mode=args.input_mode,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(
        args.model,
        args.pfe_fusion,
        args.base_channels,
        active_gate_levels,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict] = []
    batch_maes: list[float] = []
    pred_dir = args.extension_root / "predictions" / args.run_name
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval:{args.run_name}"):
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks_device = batch["mask"].to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(images)).detach().cpu()
            masks = masks_device.detach().cpu()
            batch_maes.append(float(torch.mean(torch.abs(probs - masks)).item()))
            for index, stem in enumerate(batch["stem"]):
                metrics = per_image_metrics(probs[index], masks[index])
                rows.append({"stem": stem, **metrics})
                save_prediction(probs[index], pred_dir / f"{stem}.png")

    final = aggregate(rows, batch_maes)
    metrics_dir = args.extension_root / "metrics" / args.run_name
    write_csv(metrics_dir / "test_metrics.csv", [final])
    write_csv(metrics_dir / "test_per_sample.csv", rows)
    (metrics_dir / "evaluation_device.txt").write_text(
        describe_device(device) + "\n", encoding="utf-8"
    )
    print(f"Evaluated {len(rows)} official test images.")
    for key, value in final.items():
        print(f"{key}: {value:.8f}")
    print(f"Predictions: {pred_dir}")


if __name__ == "__main__":
    main()
