"""Manifest-only RGBP-Glass input adapter; no training or model construction."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXCLUDED_IDS = frozenset({"20210808013065"})
MODALITY_SUFFIXES = {"image": "_rgb.tiff", "dolp": "_dolp.tiff", "mask": "_mask.png"}


def resolve_file(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("A nonempty dataset-relative path is required")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"File outside dataset root: {relative}")
    return path


def read_manifest(root: Path, manifest: Path, partition: str) -> list[dict]:
    if partition not in {"train", "validation", "test", "train_pool"}:
        raise ValueError(f"Unknown partition: {partition}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "split", *MODALITY_SUFFIXES}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest lacks fields: {required}")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)) or set(ids) & EXCLUDED_IDS:
        raise ValueError("Empty, duplicate, or excluded sample IDs")
    source_split = "test" if partition == "test" else "train"
    for row in rows:
        if row["split"] != source_split:
            raise ValueError("Test samples cannot enter a training/validation partition")
        for modality, suffix in MODALITY_SUFFIXES.items():
            expected = f"{source_split}/{modality}/{row['sample_id']}{suffix}"
            if row[modality] != expected:
                raise ValueError(f"Unexpected paired filename: {row[modality]}")
            resolve_file(root, row[modality])
    return rows


def decode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    # imdecode handles Unicode Windows paths without relying on cv2 filename codecs.
    array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if array is None:
        raise ValueError(f"Could not decode: {path}")
    return array


def load_arrays(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = decode(resolve_file(root, row["image"]))
    dolp = decode(resolve_file(root, row["dolp"]))
    mask = decode(resolve_file(root, row["mask"]))
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Native protocol requires three-channel uint16 RGB")
    if dolp.dtype != np.uint8 or dolp.shape != rgb.shape:
        raise ValueError("DoLP must be matched three-channel uint8")
    if mask.dtype != np.uint8 or mask.shape != rgb.shape[:2]:
        raise ValueError("Mask must be matched single-channel uint8")
    values = np.unique(mask)
    if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
        raise ValueError(f"Non-binary mask labels: {values.tolist()}")
    # RGB: OpenCV BGR -> RGB. Polar TIFFs follow official PIL-channel reversal,
    # which is already the channel order returned by OpenCV IMREAD_UNCHANGED.
    rgb = rgb[..., ::-1].astype(np.float32) / np.float32(65535)
    dolp = dolp.astype(np.float32) / np.float32(255)
    mask = (mask > 0).astype(np.float32)
    return rgb, dolp, mask


def transform_arrays(rgb, dolp, mask, image_size, flip=False):
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if height < 1 or width < 1:
        raise ValueError("Positive image dimensions required")
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    dolp = cv2.resize(dolp, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if flip:
        rgb, dolp, mask = rgb[:, ::-1], dolp[:, ::-1], mask[:, ::-1]
    return (np.ascontiguousarray(rgb.transpose(2, 0, 1)),
            np.ascontiguousarray(dolp.transpose(2, 0, 1)),
            np.ascontiguousarray(mask[None]))


class RGBPGlassDataset(Dataset):
    """Returns the existing framework's image/mask/stem dictionary on CPU.

    rgb_dolp has six channels, ordered RGB followed by official-order DoLP.
    Normalization is a fixed storage-range mapping, not radiometric calibration.
    AoLP is intentionally excluded; the official PGSNet reference is separate.
    """

    def __init__(self, dataset_root, manifest, partition, input_mode="rgb_dolp",
                 image_size=352, horizontal_flip_probability=0.0):
        self.root = Path(dataset_root).resolve(strict=True)
        self.manifest = Path(manifest)
        self.partition = partition
        if input_mode not in {"rgb", "dolp", "rgb_dolp"}:
            raise ValueError("Supported inputs: rgb, dolp, rgb_dolp")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("Invalid flip probability")
        if partition != "train" and horizontal_flip_probability:
            raise ValueError("Augmentation is permitted only for a training split")
        self.rows = read_manifest(self.root, self.manifest, partition)
        self.input_mode = input_mode
        self.image_size = image_size
        self.flip_probability = horizontal_flip_probability

    def __len__(self):
        return len(self.rows)

    @property
    def in_channels(self):
        return 6 if self.input_mode == "rgb_dolp" else 3

    def __getitem__(self, index):
        row = self.rows[index]
        flip = self.flip_probability > 0 and torch.rand(()).item() < self.flip_probability
        rgb, dolp, mask = transform_arrays(*load_arrays(self.root, row), self.image_size, flip)
        image = {"rgb": rgb, "dolp": dolp}.get(self.input_mode)
        if image is None:
            image = np.concatenate([rgb, dolp], axis=0)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask),
                "stem": row["sample_id"], "group_id": row.get("group_id", "")}
