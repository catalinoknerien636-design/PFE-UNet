from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.device_utils import describe_device, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model parameter count and inference speed.")
    parser.add_argument(
        "--model",
        choices=["unet", "unetpp", "attention_unet", "deeplabv3plus", "pfe_unet"],
        required=True,
    )
    parser.add_argument("--input_channels", type=int, required=True)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--pfe_enhancement", choices=["all", "shallow", "deep", "none"], default="all")
    parser.add_argument("--pfe_fusion", choices=["none", "add", "multiply", "residual", "conv1x1"], default="residual")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    parser.add_argument("--directml_device", type=int, default=1)
    parser.add_argument("--out_csv", type=Path, default=ROOT / "outputs" / "metrics" / "model_complexity.csv")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def estimate_parameter_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


@torch.no_grad()
def estimate_conv_macs(model: torch.nn.Module, dummy: torch.Tensor) -> int:
    total_macs = 0
    handles = []

    def count_conv(module: torch.nn.Conv2d, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total_macs
        batch_size, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        kernel_macs = module.in_channels // module.groups * kernel_height * kernel_width
        total_macs += batch_size * out_channels * out_height * out_width * kernel_macs

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
    try:
        _ = model(dummy).cpu()
    finally:
        for handle in handles:
            handle.remove()
    return total_macs


@torch.no_grad()
def measure_inference_ms(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    warmup: int,
    repeat: int,
) -> float:
    model.eval()
    for _ in range(warmup):
        _ = model(dummy).cpu()
    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy).cpu()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(repeat, 1)


def append_csv(path: Path, row: dict[str, int | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    # Keep the reusable complexity helpers importable in extension packages
    # that provide their own model builder.
    from scripts.train_unet import build_model

    args = parse_args()
    device = get_device(args.device, args.directml_device)
    model = build_model(
        args.model,
        args.input_channels,
        args.base_channels,
        pfe_enhancement=args.pfe_enhancement,
        pfe_fusion=args.pfe_fusion,
    ).to(device)
    dummy = torch.randn(
        args.batch_size,
        args.input_channels,
        args.image_size,
        args.image_size,
        device=device,
        dtype=torch.float32,
    )

    total_params, trainable_params = count_parameters(model)
    size_mb = estimate_parameter_size_mb(model)
    conv_macs = estimate_conv_macs(model, dummy)
    avg_ms = measure_inference_ms(model, dummy, args.warmup, args.repeat)
    fps = 1000.0 * args.batch_size / avg_ms

    row = {
        "model": args.model,
        "pfe_enhancement": args.pfe_enhancement if args.model == "pfe_unet" else "none",
        "pfe_fusion": args.pfe_fusion if args.model == "pfe_unet" else "none",
        "input_channels": args.input_channels,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": describe_device(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_size_mb": round(size_mb, 4),
        "conv_gmacs": round(conv_macs / 1e9, 4),
        "conv_gflops": round(2 * conv_macs / 1e9, 4),
        "avg_inference_ms": round(avg_ms, 4),
        "fps": round(fps, 4),
    }
    append_csv(args.out_csv, row)

    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
