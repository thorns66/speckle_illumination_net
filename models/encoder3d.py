from __future__ import annotations

from torch import Tensor, nn

from .blocks import ConvBlock3D, blur_lateral_3d


class Encoder3D(nn.Module):
    """Three lateral scales with an unchanged depth axis."""

    def __init__(self, channels: list[int] | tuple[int, int, int], *, anti_alias: bool = False) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Encoder3D requires exactly three channel scales")
        self.anti_alias = bool(anti_alias)
        c0, c1, c2 = (int(value) for value in channels)
        self.channels = (c0, c1, c2)
        self.level0 = ConvBlock3D(1, c0)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=(1, 2, 2), padding=1)
        self.level1 = ConvBlock3D(c1, c1)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1)
        self.level2 = ConvBlock3D(c2, c2)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level0 = self.level0(value)
        down1_input = blur_lateral_3d(level0) if self.anti_alias else level0
        level1 = self.level1(self.down1(down1_input))
        down2_input = blur_lateral_3d(level1) if self.anti_alias else level1
        level2 = self.level2(self.down2(down2_input))
        return level0, level1, level2
