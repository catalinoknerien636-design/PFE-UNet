from __future__ import annotations

import argparse
import csv
import json
import platform
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from PIL import Image

from datasets.extension_pcod import ExtensionPCODDataset
from models.pfe_unet import PFEUNet
from train import segmentation_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, default=Path("splits/train_split.csv"))
    parser.add_argument("--validation-split", type=Path, default=Path("splits/val_split.csv"))
    parser.add_argument("--formal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    log.append(f"Python: {platform.python_version()}")
    log.append(f"OS: {platform.platform()}")
    log.append(f"PyTorch: {torch.__version__}")
    log.append(f"NumPy: {np.__version__}; pandas: {pd.__version__}; SciPy: {scipy.__version__}; Pillow: {Image.__version__}; matplotlib: {matplotlib.__version__}")
    train_rows = list(csv.DictReader(args.train_split.open("r", encoding="utf-8-sig", newline="")))
    val_rows = list(csv.DictReader(args.validation_split.open("r", encoding="utf-8-sig", newline="")))
    if len(train_rows) != 3104 or len(val_rows) != 776:
        raise AssertionError(f"Unexpected split sizes: {len(train_rows)}/{len(val_rows)}")
    if set(row["stem"] for row in train_rows) & set(row["stem"] for row in val_rows):
        raise AssertionError("Train/validation split overlap detected")
    log.append("Split manifest check: PASS (3104 train, 776 validation, zero stem overlap)")
    dataset = ExtensionPCODDataset(args.dataset_root, "train", "rgb_dop", image_size=64, stems_file=args.train_split, max_samples=2)
    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)
    model = PFEUNet(in_channels=4, out_channels=1, base_channels=8, enhancement="all", fusion="multiply")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = segmentation_loss(logits, mask)
    loss.backward(); optimizer.step()
    log.append(f"Small batch forward/backward/train step: PASS, loss={float(loss):.6f}")
    tiny_path = args.output_dir / "tiny_checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, tiny_path)
    model.load_state_dict(torch.load(tiny_path, map_location="cpu", weights_only=False)["model_state"])
    log.append("Tiny checkpoint save/load: PASS")
    formal = PFEUNet(in_channels=4, out_channels=1, base_channels=32, enhancement="all", fusion="multiply")
    payload = torch.load(args.formal_checkpoint, map_location="cpu", weights_only=False)
    formal.load_state_dict(payload["model_state"], strict=True)
    formal.eval()
    test_set = ExtensionPCODDataset(args.dataset_root, "test", "rgb_dop", image_size=352, max_samples=1)
    with torch.no_grad():
        formal_prob = torch.sigmoid(formal(test_set[0]["image"].unsqueeze(0)))
    eval_mae = float(torch.abs(formal_prob - test_set[0]["mask"].unsqueeze(0)).mean())
    log.append(f"Formal checkpoint strict load and one-image evaluation: PASS, MAE={eval_mae:.6f}")
    table_script = Path(__file__).resolve().parents[1] / "reproduce_tables.py"
    import subprocess, sys
    subprocess.run([sys.executable, str(table_script)], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    table_path = Path(__file__).resolve().parents[1] / "results" / "Table_1_reproduced.csv"
    rows = list(csv.DictReader(table_path.open("r", encoding="utf-8-sig", newline="")))
    if len(rows) != 5 or not any(row["Method"] == "PFE-DoLP multiply" and row["IoU"].startswith("0.4718") for row in rows):
        raise AssertionError("Table 1 reproduction mismatch")
    log.append("Table 1 reproduction: PASS")
    fig, ax = plt.subplots(figsize=(4.5, 3.0)); ax.imshow(formal_prob[0,0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off"); fig.tight_layout(pad=0)
    figure_path = args.output_dir / "smoke_prediction.png"; fig.savefig(figure_path, dpi=200); plt.close(fig)
    log.append(f"Deterministic smoke figure: PASS ({figure_path})")
    (args.output_dir / "smoke_test.json").write_text(json.dumps({"status":"PASS", "formal_eval_image_mae":eval_mae, "steps":log}, indent=2), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
