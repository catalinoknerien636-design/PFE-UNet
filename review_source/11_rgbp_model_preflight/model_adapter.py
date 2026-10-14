"""RGBP-Glass model construction and bounded diagnostics, not a training CLI."""
from __future__ import annotations

import torch
from torch import nn

from reference_models.pfe_unet import PFEUNet
from reference_models.unet import UNet

CORE_METHODS = ("rgb_only", "direct", "pfe_multiply", "learned_fusion")
METHODS = CORE_METHODS + ("global_film_style",)


class GlobalAffine(nn.Module):
    """Global FiLM-style conditioning, not a reproduction of original FiLM."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.affine = nn.Conv2d(channels, 2 * channels, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, rgb, polar):
        delta, offset = self.affine(self.pool(polar)).chunk(2, dim=1)
        return rgb * (1 + delta) + offset


class GlobalFiLMUNet(PFEUNet):
    def __init__(self, base_channels=32):
        super().__init__(in_channels=6, base_channels=base_channels,
                         enhancement="all", fusion="none")
        for level in range(1, 6):
            # Remove the inherited unused gates so parameter counts are honest.
            setattr(self, f"gate{level}", nn.Identity())
            setattr(self, f"affine{level}", GlobalAffine(base_channels * 2 ** (level - 1)))

    def enhance(self, level, gate, rgb_feature, polar_feature):
        return getattr(self, f"affine{level}")(rgb_feature, polar_feature)


class RGBPModel(nn.Module):
    """Reject incorrectly shaped inputs before dispatching to a reference model."""

    def __init__(self, backbone, input_channels):
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(f"Expected NCHW with {self.input_channels} channels")
        if image.dtype != torch.float32 or min(image.shape[-2:]) < 32:
            raise ValueError("Current FP32 protocol requires spatial dimensions >=32")
        return self.backbone(image)


def build_model(method, *, base_channels=32, seed=42):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if not isinstance(base_channels, int) or base_channels < 1:
        raise ValueError("base_channels must be a positive integer")
    # Isolate model initialization from data-order and augmentation RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if method in {"rgb_only", "direct"}:
            channels = 3 if method == "rgb_only" else 6
            backbone = UNet(in_channels=channels, base_channels=base_channels)
        elif method == "global_film_style":
            channels, backbone = 6, GlobalFiLMUNet(base_channels)
        else:
            channels = 6
            backbone = PFEUNet(in_channels=6, base_channels=base_channels,
                               enhancement="all",
                               fusion="multiply" if method == "pfe_multiply" else "conv1x1")
    return RGBPModel(backbone, channels)


def segmentation_loss(logits, mask):
    if logits.shape != mask.shape:
        raise ValueError("Logits and binary mask must have the same shape")
    probability = torch.sigmoid(logits)
    intersection = (probability * mask).sum(dim=(1, 2, 3))
    total = probability.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (total + 1e-6)).mean()
    return nn.functional.binary_cross_entropy_with_logits(logits, mask) + dice


def gradient_audit(model):
    counts = {"registered_parameters": 0, "parameters_with_grad_tensor": 0,
              "parameters_in_tensors_with_nonzero_grad": 0}
    disconnected = []
    branch_gradient_l1 = {}
    for name, parameter in model.named_parameters():
        counts["registered_parameters"] += parameter.numel()
        if parameter.grad is None:
            disconnected.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Nonfinite gradient: {name}")
        counts["parameters_with_grad_tensor"] += parameter.numel()
        magnitude = float(parameter.grad.abs().sum())
        if magnitude > 0:
            counts["parameters_in_tensors_with_nonzero_grad"] += parameter.numel()
        branch = name.split(".")[1]
        branch_gradient_l1[branch] = branch_gradient_l1.get(branch, 0) + magnitude
    return {**counts, "disconnected": disconnected, "branch_gradient_l1": branch_gradient_l1}
