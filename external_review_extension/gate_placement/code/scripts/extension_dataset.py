from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.pcod_dataset import build_samples, load_mask


MODE_TO_AUX = {
    "rgb_dop": "dop",
    "rgb_s0": "s0",
    "rgb_s1": "s1",
    "rgb_s2": "s2",
    "rgb_gray": "gray",
}


class ExtensionPCODDataset(Dataset):
    """PCOD loader with frozen auxiliary-channel controls.

    RGB and auxiliary channels are resized independently with bilinear
    interpolation and normalized to [0, 1]. Masks use nearest-neighbor resize.
    Registration transforms affect only the auxiliary channel.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        input_mode: str,
        image_size: int = 352,
        stems_file: Path | None = None,
        max_samples: int | None = None,
        registration_aug: bool = False,
        registration_aug_probability: float = 0.5,
        fixed_shift: tuple[int, int] | None = None,
        fixed_rotation: float | None = None,
    ) -> None:
        if input_mode not in MODE_TO_AUX:
            raise ValueError(f"Unsupported extension input mode: {input_mode}")
        if registration_aug and split != "train":
            raise ValueError("Registration augmentation is allowed only for the training subset.")
        if registration_aug and input_mode != "rgb_dop":
            raise ValueError("The preregistered registration augmentation applies only to DoLP.")
        self.samples = build_samples(
            Path(dataset_root),
            split,
            max_samples=max_samples,
            stems_file=stems_file,
        )
        self.input_mode = input_mode
        self.aux_name = MODE_TO_AUX[input_mode]
        self.image_size = int(image_size)
        self.registration_aug = bool(registration_aug)
        self.registration_aug_probability = float(registration_aug_probability)
        self.fixed_shift = fixed_shift
        self.fixed_rotation = fixed_rotation

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def in_channels(self) -> int:
        return 4

    def _load_rgb(self, path: Path) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return image, array

    def _load_aux(self, sample, rgb_array: np.ndarray) -> Image.Image:
        if self.aux_name == "gray":
            gray = (
                0.299 * rgb_array[..., 0]
                + 0.587 * rgb_array[..., 1]
                + 0.114 * rgb_array[..., 2]
            )
            return Image.fromarray(np.rint(gray * 255.0).clip(0, 255).astype(np.uint8), mode="L")

        path = getattr(sample, f"{self.aux_name}_path")
        if path is None:
            raise FileNotFoundError(f"Missing {self.aux_name} for {sample.stem}")
        image = Image.open(path).convert("L")
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    @staticmethod
    def _transform_aux(image: Image.Image, dx: int, dy: int, angle: float) -> Image.Image:
        if dx == 0 and dy == 0 and abs(angle) < 1e-12:
            return image
        return image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            center=None,
            translate=(dx, dy),
            fillcolor=0,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        _, rgb_array = self._load_rgb(sample.rgb_path)
        aux_image = self._load_aux(sample, rgb_array)

        dx = dy = 0
        angle = 0.0
        if self.fixed_shift is not None:
            dx, dy = self.fixed_shift
        if self.fixed_rotation is not None:
            angle = float(self.fixed_rotation)
        if self.registration_aug and random.random() < self.registration_aug_probability:
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            angle = random.uniform(-1.0, 1.0)
        aux_image = self._transform_aux(aux_image, dx=dx, dy=dy, angle=angle)

        rgb = rgb_array.transpose(2, 0, 1)
        aux = np.asarray(aux_image, dtype=np.float32)[None, ...] / 255.0
        image = np.concatenate([rgb, aux], axis=0).astype(np.float32)
        mask = load_mask(sample.mask_path, self.image_size)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "stem": sample.stem,
        }
