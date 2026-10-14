from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PCODSample:
    stem: str
    rgb_path: Path
    mask_path: Path
    dop_path: Path | None = None
    s0_path: Path | None = None
    s1_path: Path | None = None
    s2_path: Path | None = None


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def stem_map(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in list_images(directory):
        if path.stem in mapping:
            raise ValueError(f"Duplicate stem in {directory}: {path.stem}")
        mapping[path.stem] = path
    return mapping


def split_dirs(dataset_root: Path, split: str) -> dict[str, Path]:
    split_root = dataset_root / split
    stokes_root = split_root / f"{split}-Stokes"
    return {
        "rgb": split_root / f"{split}-rgb",
        "mask": split_root / f"{split}-gt",
        "dop": split_root / f"{split}-dop",
        "s0": stokes_root / "s0",
        "s1": stokes_root / "s1",
        "s2": stokes_root / "s2",
    }


def load_selected_stems(stems_file: Path) -> list[str]:
    with Path(stems_file).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "stem" not in (reader.fieldnames or []):
            raise ValueError(f"Split file must contain a stem column: {stems_file}")
        return [row["stem"] for row in reader]


def build_samples(
    dataset_root: Path,
    split: str,
    max_samples: int | None = None,
    stems_file: Path | None = None,
) -> list[PCODSample]:
    dirs = split_dirs(dataset_root, split)
    rgb_map = stem_map(dirs["rgb"])
    mask_map = stem_map(dirs["mask"])
    dop_map = stem_map(dirs["dop"])
    s0_map = stem_map(dirs["s0"])
    s1_map = stem_map(dirs["s1"])
    s2_map = stem_map(dirs["s2"])

    stems = load_selected_stems(stems_file) if stems_file is not None else sorted(rgb_map)
    unknown = sorted(set(stems) - set(rgb_map))
    if unknown:
        raise ValueError(f"{split}: split file contains unknown rgb stems, first examples: {unknown[:5]}")
    for name, mapping in {
        "mask": mask_map,
        "dop": dop_map,
        "s0": s0_map,
        "s1": s1_map,
        "s2": s2_map,
    }.items():
        missing = sorted(set(stems) - set(mapping))
        if missing:
            raise ValueError(f"{split}: missing {name} files, first examples: {missing[:5]}")

    if max_samples is not None:
        stems = stems[:max_samples]

    return [
        PCODSample(
            stem=stem,
            rgb_path=rgb_map[stem],
            mask_path=mask_map[stem],
            dop_path=dop_map[stem],
            s0_path=s0_map[stem],
            s1_path=s1_map[stem],
            s2_path=s2_map[stem],
        )
        for stem in stems
    ]


class PCODDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        input_mode: str = "rgb",
        image_size: int = 352,
        max_samples: int | None = None,
        stems_file: Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.input_mode = input_mode
        self.image_size = image_size
        self.samples = build_samples(
            self.dataset_root,
            split,
            max_samples=max_samples,
            stems_file=stems_file,
        )
        self.channels = parse_input_mode(input_mode)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def in_channels(self) -> int:
        total = 0
        for channel in self.channels:
            total += 3 if channel == "rgb" else 1
        return total

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_parts = [load_channel(sample, channel, self.image_size) for channel in self.channels]
        image = np.concatenate(image_parts, axis=0).astype(np.float32)
        mask = load_mask(sample.mask_path, self.image_size)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "stem": sample.stem,
        }


def parse_input_mode(input_mode: str) -> list[str]:
    presets = {
        "rgb": ["rgb"],
        "dop": ["dop"],
        "rgb_dop": ["rgb", "dop"],
        "rgb_aop": ["rgb", "aop_cos", "aop_sin"],
        "rgb_dop_aop": ["rgb", "dop", "aop_cos", "aop_sin"],
        "rgb_stokes": ["rgb", "s0", "s1", "s2"],
        "rgb_polar_all": ["rgb", "dop", "s0", "s1", "s2"],
    }
    if input_mode in presets:
        return presets[input_mode]
    channels = [part.strip() for part in input_mode.split(",") if part.strip()]
    allowed = {"rgb", "dop", "s0", "s1", "s2", "aop_cos", "aop_sin"}
    invalid = sorted(set(channels) - allowed)
    if not channels or invalid:
        raise ValueError(f"Invalid input_mode={input_mode}. Invalid channels: {invalid}")
    return channels


def load_channel(sample: PCODSample, channel: str, image_size: int) -> np.ndarray:
    if channel == "rgb":
        image = Image.open(sample.rgb_path).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return array

    if channel in {"aop_cos", "aop_sin"}:
        if sample.s1_path is None or sample.s2_path is None:
            raise FileNotFoundError(f"Missing Stokes S1/S2 paths for {sample.stem}")
        s1 = load_stokes_signed(sample.s1_path, image_size)
        s2 = load_stokes_signed(sample.s2_path, image_size)
        magnitude = np.sqrt(s1 * s1 + s2 * s2)
        eps = 1e-6
        if channel == "aop_cos":
            return s1 / (magnitude + eps)
        return s2 / (magnitude + eps)

    path = getattr(sample, f"{channel}_path")
    if path is None:
        raise FileNotFoundError(f"Missing {channel} path for {sample.stem}")
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    return array


def load_stokes_signed(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    return (array - 0.5) * 2.0


def load_mask(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.float32) > 127).astype(np.float32)
    return array[None, ...]
