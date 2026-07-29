from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("PFE_SOURCE_PROJECT_ROOT", REPOSITORY_ROOT)).resolve()
EVIDENCE_ROOT = Path(os.environ.get("PFE_EVIDENCE_ROOT", REPOSITORY_ROOT)).resolve()
MULTISCALE_CODE = (
    EXTENSION_ROOT / "gate_placement" / "code"
)
for path in (PROJECT_ROOT, MULTISCALE_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.pfe_unet_reference import PFEUNet
from models.unet import UNet
from scripts.device_utils import describe_device, get_device
from scripts.extension_dataset import ExtensionPCODDataset


METHODS = {
    "direct": {"label": "Direct RGB+DoLP", "model": "unet", "fusion": "none"},
    "pfe_multiply": {"label": "PFE multiply", "model": "pfe_unet", "fusion": "multiply"},
    "pfe_1x1": {"label": "PFE 1x1", "model": "pfe_unet", "fusion": "conv1x1"},
}
SEEDS = (42, 2026, 3407, 7819, 12011)


@dataclass(frozen=True)
class ArtifactSource:
    run_name: str
    metrics_root: Path
    checkpoint_root: Path

    @property
    def history_path(self) -> Path:
        return self.metrics_root / self.run_name / "history.csv"

    @property
    def config_path(self) -> Path:
        return self.metrics_root / self.run_name / "config.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_root / self.run_name / "best.pt"


def source_for(method: str, seed: int) -> ArtifactSource:
    if method == "direct":
        if seed in {42, 2026, 3407}:
            suffix = "" if seed == 42 else f"_seed{seed}"
            run = f"grouped_rgb_dop_unet_directml_50ep{suffix}"
            return ArtifactSource(
                run,
                PROJECT_ROOT / "outputs" / "metrics",
                PROJECT_ROOT / "checkpoints",
            )
        return ArtifactSource(
            f"ext_rgb_dolp_unet_seed{seed}",
            EVIDENCE_ROOT / "metrics",
            EVIDENCE_ROOT / "checkpoints",
        )
    if method == "pfe_multiply":
        if seed in {42, 2026, 3407}:
            suffix = "" if seed == 42 else f"_seed{seed}"
            run = f"grouped_pfe_unet_multiply_directml_50ep{suffix}"
            return ArtifactSource(
                run,
                PROJECT_ROOT / "outputs" / "metrics",
                PROJECT_ROOT / "checkpoints",
            )
        return ArtifactSource(
            f"ext_pfe_dolp_multiply_seed{seed}",
            EVIDENCE_ROOT / "metrics",
            EVIDENCE_ROOT / "checkpoints",
        )
    if method == "pfe_1x1":
        if seed in {42, 2026, 3407}:
            suffix = "" if seed == 42 else f"_seed{seed}"
            run = f"grouped_pfe_unet_conv1x1_directml_50ep{suffix}"
            return ArtifactSource(
                run,
                PROJECT_ROOT / "outputs" / "metrics",
                PROJECT_ROOT / "checkpoints",
            )
        return ArtifactSource(
            f"ext_pfe_dolp_conv1x1_seed{seed}",
            EVIDENCE_ROOT / "metrics",
            EVIDENCE_ROOT / "checkpoints",
        )
    raise ValueError(f"Unsupported method: {method}")


def validate_formal_sources() -> list[dict[str, str]]:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            source = source_for(method, seed)
            missing = [
                str(path)
                for path in (
                    source.history_path,
                    source.config_path,
                    source.checkpoint_path,
                )
                if not path.exists()
            ]
            rows.append(
                {
                    "method": method,
                    "seed": str(seed),
                    "run_name": source.run_name,
                    "history_path": str(source.history_path),
                    "config_path": str(source.config_path),
                    "checkpoint_path": str(source.checkpoint_path),
                    "status": "VERIFIED" if not missing else "MISSING",
                    "missing": ";".join(missing),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_model(method: str) -> torch.nn.Module:
    if method == "direct":
        return UNet(in_channels=4, out_channels=1, base_channels=32)
    return PFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=32,
        enhancement="all",
        fusion=METHODS[method]["fusion"],
    )


def load_model(method: str, seed: int, device) -> tuple[torch.nn.Module, ArtifactSource]:
    source = source_for(method, seed)
    model = build_model(method).to(device)
    checkpoint = torch.load(source.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, source


def shift_reflect(aux: torch.Tensor, pixels: int, direction: str) -> torch.Tensor:
    if pixels == 0:
        return aux
    if direction == "left":
        padded = F.pad(aux, (0, pixels, 0, 0), mode="reflect")
        return padded[..., pixels : pixels + aux.shape[-1]]
    if direction == "right":
        padded = F.pad(aux, (pixels, 0, 0, 0), mode="reflect")
        return padded[..., : aux.shape[-1]]
    if direction == "up":
        padded = F.pad(aux, (0, 0, 0, pixels), mode="reflect")
        return padded[..., pixels : pixels + aux.shape[-2], :]
    if direction == "down":
        padded = F.pad(aux, (0, 0, pixels, 0), mode="reflect")
        return padded[..., : aux.shape[-2], :]
    raise ValueError(f"Unsupported shift direction: {direction}")


def rotate_reflect(aux: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    if abs(angle_degrees) < 1e-12:
        return aux
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    theta = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
        dtype=aux.dtype,
        device=aux.device,
    ).unsqueeze(0)
    theta = theta.repeat(aux.shape[0], 1, 1)
    grid = F.affine_grid(theta, aux.shape, align_corners=False)
    return F.grid_sample(
        aux,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def gaussian_blur_reflect(aux: torch.Tensor, sigma: float) -> tuple[torch.Tensor, int]:
    if sigma <= 0:
        return aux, 1
    radius = int(math.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    coordinates = torch.arange(
        -radius, radius + 1, dtype=aux.dtype, device=aux.device
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1)
    result = F.pad(aux, (radius, radius, 0, 0), mode="reflect")
    result = F.conv2d(result, horizontal)
    result = F.pad(result, (0, 0, radius, radius), mode="reflect")
    result = F.conv2d(result, vertical)
    return result, kernel_size


def perturbation_seed(kind: str, level: str, realization: int) -> int:
    digest = hashlib.sha256(f"{kind}|{level}|{realization}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def apply_perturbation(
    images: torch.Tensor,
    kind: str,
    level: float | int,
    direction: str,
    realization: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, str | int | float]]:
    rgb = images[:, :3]
    aux = images[:, 3:4]
    metadata: dict[str, str | int | float] = {
        "interpolation": "bilinear where applicable",
        "padding": "reflection where applicable",
    }
    if kind == "clean":
        transformed = aux
    elif kind == "translation":
        transformed = shift_reflect(aux, int(level), direction)
    elif kind == "rotation":
        transformed = rotate_reflect(aux, float(level))
    elif kind == "blur":
        transformed, kernel_size = gaussian_blur_reflect(aux, float(level))
        metadata["kernel_size"] = kernel_size
    elif kind == "noise":
        if generator is None:
            raise ValueError("Noise perturbation requires a fixed CPU generator.")
        noise = torch.randn(
            aux.shape,
            generator=generator,
            dtype=aux.dtype,
            device="cpu",
        ).to(aux.device)
        transformed = torch.clamp(aux + noise * float(level), 0.0, 1.0)
        metadata["noise_seed"] = perturbation_seed(kind, str(level), realization)
    elif kind == "quantization":
        bins = 2 ** int(level) - 1
        transformed = torch.round(aux * bins) / bins
        metadata["quantization_formula"] = "round(x*(2^b-1))/(2^b-1)"
    else:
        raise ValueError(f"Unsupported perturbation: {kind}")
    return torch.cat([rgb, transformed], dim=1), metadata


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


def aggregate(rows: list[dict]) -> dict[str, float]:
    eps = 1e-7
    tp = sum(float(row["tp"]) for row in rows)
    fp = sum(float(row["fp"]) for row in rows)
    fn = sum(float(row["fn"]) for row in rows)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mae_image_mean": float(np.mean([float(row["mae_image"]) for row in rows])),
    }


def save_prediction(prob: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.rint(prob.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def condition_id(
    kind: str,
    level: float | int,
    direction: str,
    realization: int,
) -> str:
    level_text = str(level).replace("-", "m").replace(".", "p")
    pieces = [kind, level_text]
    if direction:
        pieces.append(direction)
    if kind == "noise":
        pieces.append(f"r{realization}")
    return "_".join(pieces)


def condition_complete(condition_dir: Path) -> bool:
    return (
        (condition_dir / "summary.csv").exists()
        and (condition_dir / "config.json").exists()
        and len(list((condition_dir / "predictions").glob("*.png"))) == 230
        and len(read_csv(condition_dir / "per_image.csv")) == 230
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def best_validation(history_path: Path) -> tuple[int, float]:
    rows = read_csv(history_path)
    if len(rows) != 50:
        raise ValueError(f"Expected 50 validation epochs: {history_path}")
    best = max(rows, key=lambda row: float(row["val_iou"]))
    return int(best["epoch"]), float(best["val_iou"])


def evaluate_condition(
    model: torch.nn.Module,
    source: ArtifactSource,
    method: str,
    seed: int,
    device,
    loader: DataLoader,
    output_root: Path,
    kind: str,
    level: float | int,
    direction: str = "",
    realization: int = 0,
) -> dict:
    identifier = condition_id(kind, level, direction, realization)
    condition_dir = output_root / "runs" / f"{method}_seed{seed}" / identifier
    if condition_complete(condition_dir):
        return read_csv(condition_dir / "summary.csv")[0]

    generator = None
    noise_seed = ""
    if kind == "noise":
        noise_seed = perturbation_seed(kind, str(level), realization)
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)

    rows: list[dict] = []
    metadata: dict[str, str | int | float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{method}:{seed}:{identifier}", leave=False):
            images_cpu = batch["image"].to(dtype=torch.float32)
            perturbed_cpu, metadata = apply_perturbation(
                images_cpu,
                kind=kind,
                level=level,
                direction=direction,
                realization=realization,
                generator=generator,
            )
            images = perturbed_cpu.to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)
            probabilities = torch.sigmoid(model(images)).detach().cpu()
            masks_cpu = masks.detach().cpu()
            for index, stem in enumerate(batch["stem"]):
                metrics = per_image_metrics(probabilities[index], masks_cpu[index])
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "perturbation_type": kind,
                        "perturbation_level": level,
                        "direction": direction,
                        "noise_realization": realization if kind == "noise" else "",
                        "noise_seed": noise_seed,
                        "stem": stem,
                        **metrics,
                    }
                )
                save_prediction(
                    probabilities[index],
                    condition_dir / "predictions" / f"{stem}.png",
                )

    summary = {
        "method": method,
        "method_label": METHODS[method]["label"],
        "seed": seed,
        "perturbation_type": kind,
        "perturbation_level": level,
        "direction": direction,
        "noise_realization": realization if kind == "noise" else "",
        "noise_seed": noise_seed,
        **aggregate(rows),
        "checkpoint_path": str(source.checkpoint_path),
        "history_path": str(source.history_path),
        "source_config_path": str(source.config_path),
        "prediction_path": str(condition_dir / "predictions"),
        "per_image_metrics_path": str(condition_dir / "per_image.csv"),
        "verified": "yes",
    }
    best_epoch, validation_iou = best_validation(source.history_path)
    summary["best_epoch"] = best_epoch
    summary["validation_iou"] = validation_iou
    write_csv(condition_dir / "per_image.csv", rows)
    write_csv(condition_dir / "summary.csv", [summary])
    config = {
        "method": method,
        "seed": seed,
        "source_run": source.run_name,
        "checkpoint": str(source.checkpoint_path),
        "checkpoint_selection": "highest validation IoU; test not used for selection",
        "dataset_root": str(
            PROJECT_ROOT / "datasets" / "PCOD_1200" / "full-dataset"
        ),
        "test_images": 230,
        "image_size": 352,
        "batch_size": 4,
        "perturbation_type": kind,
        "perturbation_level": level,
        "direction": direction,
        "noise_realization": realization if kind == "noise" else None,
        "noise_seed": noise_seed or None,
        "transform_metadata": metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": describe_device(device),
    }
    (condition_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    if not condition_complete(condition_dir):
        raise RuntimeError(f"Incomplete condition after evaluation: {condition_dir}")
    return summary


def make_loader() -> DataLoader:
    dataset = ExtensionPCODDataset(
        PROJECT_ROOT / "datasets" / "PCOD_1200" / "full-dataset",
        split="test",
        input_mode="rgb_dop",
        image_size=352,
    )
    if len(dataset) != 230:
        raise ValueError(f"Expected 230 official test images, found {len(dataset)}")
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)


def resolve_device(name: str, directml_device: int):
    return get_device(name, directml_device)
