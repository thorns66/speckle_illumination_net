from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


def blur_lateral_3d(value: Tensor) -> Tensor:
    """Apply a fixed binomial low-pass without changing shape or depth."""
    if value.ndim != 5:
        raise ValueError("blur_lateral_3d expects [B,C,Z,H,W]")
    channels = value.shape[1]
    kernel_2d = value.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    kernel = kernel_2d.view(1, 1, 1, 3, 3).expand(channels, 1, 1, 3, 3)
    padded = F.pad(value, (1, 1, 1, 1, 0, 0), mode="replicate")
    return F.conv3d(padded, kernel, groups=channels)


def blur_lateral_2d(value: Tensor) -> Tensor:
    """Apply the same fixed binomial low-pass to frame features."""
    if value.ndim != 4:
        raise ValueError("blur_lateral_2d expects [B,C,H,W]")
    channels = value.shape[1]
    kernel_2d = value.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    kernel = kernel_2d.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def normalize_feature_input(value: Tensor, dims: tuple[int, ...], eps: float = 1e-6) -> Tensor:
    scale = value.detach().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(eps)
    return value / scale
