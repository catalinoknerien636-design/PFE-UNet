from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.extension_pcod import ExtensionPCODDataset
from train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.dataset_root or Path(cfg["dataset_root"])
    dataset = ExtensionPCODDataset(root, "test", cfg["input_mode"], int(cfg["image_size"]), max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    tp = fp = fn = 0.0
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"]))
            pred, truth = probs > 0.5, batch["mask"] > 0.5
            tp += float(torch.logical_and(pred, truth).sum())
            fp += float(torch.logical_and(pred, ~truth).sum())
            fn += float(torch.logical_and(~pred, truth).sum())
            for stem, prob, mask in zip(batch["stem"], probs, batch["mask"]):
                rows.append({"stem": stem, "image_mae": float(torch.abs(prob-mask).mean())})
    result = {"images": len(rows), "iou": tp/(tp+fp+fn+1e-7), "image_averaged_mae": sum(float(r["image_mae"]) for r in rows)/max(len(rows),1)}
    out = Path(cfg["evaluation_output_dir"]); out.mkdir(parents=True, exist_ok=True)
    with (out/"test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(result)); writer.writeheader(); writer.writerow(result)
    print(result)


if __name__ == "__main__":
    main()
