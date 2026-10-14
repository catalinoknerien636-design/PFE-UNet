"""PCOD baselines with explicit fusion and historical loss semantics."""
from pathlib import Path
import sys

import torch
from torch import nn

REFERENCE = Path(__file__).resolve().parent.parent / "13_registration_evaluation/reference"
sys.path.insert(0, str(REFERENCE / "04_multiscale_ablation/code"))
from models.pfe_unet_reference import PFEUNet
from models.unet import UNet

for name in ("models.pfe_unet_reference", "models.unet"):
    if not Path(sys.modules[name].__file__).resolve().is_relative_to(REFERENCE):
        raise RuntimeError(f"Unexpected model source: {name}")

STANDARD_METHOD = "deeplabv3plus_resnet50"
METHODS = ("direct_reference", "pfe_reference", "learned_fusion_reference", "global_film_style", "wide_direct", STANDARD_METHOD)
FORMAL_WIDTHS = {method: (41 if method == "wide_direct" else 32) for method in METHODS}
FORMAL_WIDTHS[STANDARD_METHOD] = None


def standard_module():
    directory = Path(__file__).resolve().parent.parent / "16_standard_backbone"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import standard_model
    if Path(standard_model.__file__).resolve() != (directory / "standard_model.py").resolve():
        raise RuntimeError("Unexpected standard-backbone adapter source")
    return standard_model


class GlobalAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.affine = nn.Conv2d(channels, 2 * channels, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, appearance, polarization):
        delta, offset = self.affine(self.pool(polarization)).chunk(2, dim=1)
        return appearance * (1 + delta) + offset


class GlobalFiLMUNet(PFEUNet):
    """Global channel-affine control; not a reproduction of the original FiLM task."""
    def __init__(self, base_channels=32):
        super().__init__(in_channels=4, out_channels=1, base_channels=base_channels,
                         enhancement="all", fusion="none")
        for level in range(1, 6):
            setattr(self, f"gate{level}", nn.Identity())
            setattr(self, f"affine{level}", GlobalAffine(base_channels * 2 ** (level - 1)))

    def enhance(self, level, gate, rgb_feature, polar_feature):
        return getattr(self, f"affine{level}")(rgb_feature, polar_feature)


def build_model(method, *, base_channels=32, seed=42):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == STANDARD_METHOD:
        if base_channels is not None:
            raise ValueError("A standard ResNet backbone has no U-Net base width")
        return standard_module().build_model(seed)
    if type(base_channels) is not int or base_channels < 1:
        raise ValueError("Invalid base width")
    # Historical PCOD shuffling uses the global RNG after model initialization.
    torch.manual_seed(seed)
    if method in ("direct_reference", "wide_direct"):
        return UNet(in_channels=4, out_channels=1, base_channels=base_channels)
    if method == "global_film_style":
        return GlobalFiLMUNet(base_channels)
    return PFEUNet(in_channels=4, out_channels=1, base_channels=base_channels,
                   enhancement="all", fusion="multiply" if method == "pfe_reference" else "conv1x1")


def segmentation_loss(logits, masks):
    if logits.shape != masks.shape:
        raise ValueError("Logit/mask shape mismatch")
    bce = torch.clamp(logits, min=0) - logits * masks + torch.log1p(torch.exp(-torch.abs(logits)))
    probability = logits.sigmoid()
    intersection = (probability * masks).sum((1, 2, 3))
    total = probability.sum((1, 2, 3)) + masks.sum((1, 2, 3))
    return bce.mean() + (1 - ((2 * intersection + 1e-6) / (total + 1e-6)).mean())
