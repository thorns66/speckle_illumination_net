from __future__ import annotations

from torch import Tensor, nn

from .blocks import ConvBlock3D


class Encoder3D(nn.Module):
    """Three lateral scales with an unchanged depth axis."""

    def __init__(self, channels: list[int] | tuple[int, int, int]) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Encoder3D requires exactly three channel scales")
        c0, c1, c2 = (int(value) for value in channels)
        self.channels = (c0, c1, c2)
        self.level0 = ConvBlock3D(1, c0)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=(1, 2, 2), padding=1)
        self.level1 = ConvBlock3D(c1, c1)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1)
        self.level2 = ConvBlock3D(c2, c2)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level0 = self.level0(value)
        level1 = self.level1(self.down1(level0))
        level2 = self.level2(self.down2(level1))
        return level0, level1, level2
