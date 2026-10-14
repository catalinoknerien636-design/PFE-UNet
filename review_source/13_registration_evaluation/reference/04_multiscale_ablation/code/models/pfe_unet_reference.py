from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.unet import DoubleConv, Down, Up


class PolarizationGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, polar_feature: torch.Tensor) -> torch.Tensor:
        gate = self.net(polar_feature)
        return gate


class Conv1x1FeatureFusion(nn.Module):
    """Dual-encoder learnable feature fusion with a 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feature: torch.Tensor, polar_feature: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rgb_feature, polar_feature], dim=1))


class PFEUNet(nn.Module):
    """U-Net with polarization-prior-guided feature enhancement."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        enhancement: str = "all",
        fusion: str = "residual",
    ) -> None:
        super().__init__()
        if in_channels < 4:
            raise ValueError("PFEUNet expects RGB plus at least one polarization prior channel.")
        if enhancement not in {"all", "shallow", "deep", "none"}:
            raise ValueError(f"Unsupported enhancement mode: {enhancement}")
        if fusion not in {"none", "add", "multiply", "residual", "conv1x1"}:
            raise ValueError(f"Unsupported fusion mode: {fusion}")
        self.enhancement = enhancement
        self.fusion = fusion
        self.polar_channels = in_channels - 3

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.rgb_inc = DoubleConv(3, c1)
        self.rgb_down1 = Down(c1, c2)
        self.rgb_down2 = Down(c2, c3)
        self.rgb_down3 = Down(c3, c4)
        self.rgb_down4 = Down(c4, c5)

        self.polar_inc = DoubleConv(self.polar_channels, c1)
        self.polar_down1 = Down(c1, c2)
        self.polar_down2 = Down(c2, c3)
        self.polar_down3 = Down(c3, c4)
        self.polar_down4 = Down(c4, c5)

        if self.fusion == "conv1x1":
            self.gate1 = nn.Identity()
            self.gate2 = nn.Identity()
            self.gate3 = nn.Identity()
            self.gate4 = nn.Identity()
            self.gate5 = nn.Identity()
            self.fuse1 = Conv1x1FeatureFusion(c1)
            self.fuse2 = Conv1x1FeatureFusion(c2)
            self.fuse3 = Conv1x1FeatureFusion(c3)
            self.fuse4 = Conv1x1FeatureFusion(c4)
            self.fuse5 = Conv1x1FeatureFusion(c5)
        else:
            self.gate1 = PolarizationGate(c1)
            self.gate2 = PolarizationGate(c2)
            self.gate3 = PolarizationGate(c3)
            self.gate4 = PolarizationGate(c4)
            self.gate5 = PolarizationGate(c5)

        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def should_enhance(self, level: int) -> bool:
        if self.enhancement == "all":
            return True
        if self.enhancement == "shallow":
            return level in {1, 2}
        if self.enhancement == "deep":
            return level in {4, 5}
        return False

    def enhance(
        self,
        level: int,
        gate: PolarizationGate,
        rgb_feature: torch.Tensor,
        polar_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.should_enhance(level):
            if self.fusion == "none":
                return rgb_feature
            if self.fusion == "add":
                return rgb_feature + polar_feature
            if self.fusion == "conv1x1":
                fusion_layer = getattr(self, f"fuse{level}")
                return fusion_layer(rgb_feature, polar_feature)
            weights = gate(polar_feature)
            if self.fusion == "multiply":
                return rgb_feature * weights
            return rgb_feature * (1.0 + weights)
        return rgb_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3, :, :]
        polar = x[:, 3:, :, :]

        r1 = self.rgb_inc(rgb)
        p1 = self.polar_inc(polar)
        x1 = self.enhance(1, self.gate1, r1, p1)

        r2 = self.rgb_down1(x1)
        p2 = self.polar_down1(p1)
        x2 = self.enhance(2, self.gate2, r2, p2)

        r3 = self.rgb_down2(x2)
        p3 = self.polar_down2(p2)
        x3 = self.enhance(3, self.gate3, r3, p3)

        r4 = self.rgb_down3(x3)
        p4 = self.polar_down3(p3)
        x4 = self.enhance(4, self.gate4, r4, p4)

        r5 = self.rgb_down4(x4)
        p5 = self.polar_down4(p4)
        x5 = self.enhance(5, self.gate5, r5, p5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
