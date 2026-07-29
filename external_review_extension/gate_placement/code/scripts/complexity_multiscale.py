from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (CODE_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.device_utils import describe_device, get_device
from scripts.model_complexity import (
    count_parameters,
    estimate_conv_macs,
    estimate_parameter_size_mb,
    measure_inference_ms,
)
from train_multiscale import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure frozen extension model complexity.")
    parser.add_argument("--extension_root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="directml")
    parser.add_argument("--directml_device", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device, args.directml_device)
    specs = [
        ("M0", "No active gate", ()),
        ("M1", "L1 only", (1,)),
        ("M3", "L5 only", (5,)),
        ("M4", "Shallow", (1, 2, 3)),
        ("M5", "Deep", (3, 4, 5)),
        ("M6", "All five levels", (1, 2, 3, 4, 5)),
    ]
    rows: list[dict] = []
    for variant, description, levels in specs:
        model = build_model("pfe_unet", "multiply", 32, levels).to(device)
        dummy = torch.randn(1, 4, 352, 352, device=device, dtype=torch.float32)
        total, trainable = count_parameters(model)
        macs = estimate_conv_macs(model, dummy)
        avg_ms = measure_inference_ms(model, dummy, args.warmup, args.repeat)
        rows.append(
            {
                "variant": variant,
                "method": description,
                "model": "pfe_unet",
                "fusion": "multiply",
                "active_gate_levels": ",".join(f"L{level}" for level in levels),
                "input_channels": 4,
                "image_size": 352,
                "device": describe_device(device),
                "total_params": total,
                "trainable_params": trainable,
                "parameter_size_mb": estimate_parameter_size_mb(model),
                "conv_gmacs": macs / 1e9,
                "conv_gflops": 2.0 * macs / 1e9,
                "avg_inference_ms": avg_ms,
                "fps": 1000.0 / avg_ms,
            }
        )
        del model
    out = args.extension_root / "model_complexity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved complexity table: {out}")


if __name__ == "__main__":
    main()
