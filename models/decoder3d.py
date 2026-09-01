from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvBlock3D


class ResidualDecoder3D(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c0, c1, c2 = channels
        self.deep = ConvBlock3D(c2, c2)
        self.decode1 = ConvBlock3D(c2 + c1, c1)
        self.decode0 = ConvBlock3D(c1 + c0, c0)
        self.head = nn.Conv3d(c0, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        level0, level1, level2 = features
        value = self.deep(level2)
        value = F.interpolate(value, size=level1.shape[-3:], mode="trilinear", align_corners=False)
        value = self.decode1(torch.cat([value, level1], dim=1))
        value = F.interpolate(value, size=level0.shape[-3:], mode="trilinear", align_corners=False)
        value = self.decode0(torch.cat([value, level0], dim=1))
        return self.head(value)
