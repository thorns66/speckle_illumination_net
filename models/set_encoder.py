from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .blocks import ConvBlock2D, ConvBlock3D


class SharedFrameEncoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c0, c1, c2 = channels
        self.level0 = ConvBlock2D(1, c0)
        self.down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.level1 = ConvBlock2D(c1, c1)
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.level2 = ConvBlock2D(c2, c2)

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level0 = self.level0(frames)
        level1 = self.level1(self.down1(level0))
        level2 = self.level2(self.down2(level1))
        return level0, level1, level2


class SetEncoder(nn.Module):
    """Shared frame CNN followed by permutation-invariant mean/std pooling."""

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int],
        *,
        frame_chunk_size: int = 8,
        z_scale_um: float = 100.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SetEncoder requires exactly three channel scales")
        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        self.channels = tuple(int(value) for value in channels)
        self.frame_chunk_size = int(frame_chunk_size)
        self.z_scale_um = float(z_scale_um)
        self.use_checkpoint = bool(use_checkpoint)
        self.shared = SharedFrameEncoder(self.channels)
        self.aggregate = nn.ModuleList(
            [nn.Conv2d(2 * channel, channel, 1) for channel in self.channels]
        )
        self.lift = nn.ModuleList(
            [ConvBlock3D(channel + 1, channel) for channel in self.channels]
        )

    def aggregate_frames(self, residual_frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if residual_frames.ndim != 5 or residual_frames.shape[2] != 1:
            raise ValueError(
                f"residual_frames must be [B,N,1,H,W], got {tuple(residual_frames.shape)}"
            )
        batch, frame_count, _, height, width = residual_frames.shape
        if frame_count < 1:
            raise ValueError("At least one residual frame is required")
        sums: list[Tensor | None] = [None, None, None]
        square_sums: list[Tensor | None] = [None, None, None]
        for start in range(0, frame_count, self.frame_chunk_size):
            chunk = residual_frames[:, start : start + self.frame_chunk_size]
            chunk_count = chunk.shape[1]
            flat = chunk.reshape(batch * chunk_count, 1, height, width)
            if self.use_checkpoint and self.training:
                encoded = checkpoint(self.shared, flat, use_reentrant=False)
            else:
                encoded = self.shared(flat)
            for scale, feature in enumerate(encoded):
                feature = feature.reshape(batch, chunk_count, *feature.shape[1:])
                chunk_sum = feature.sum(dim=1)
                chunk_square_sum = feature.square().sum(dim=1)
                sums[scale] = chunk_sum if sums[scale] is None else sums[scale] + chunk_sum
                square_sums[scale] = (
                    chunk_square_sum
                    if square_sums[scale] is None
                    else square_sums[scale] + chunk_square_sum
                )
        aggregated: list[Tensor] = []
        for scale in range(3):
            assert sums[scale] is not None and square_sums[scale] is not None
            mean = sums[scale] / frame_count
            variance = (square_sums[scale] / frame_count - mean.square()).clamp_min(0.0)
            std = torch.sqrt(variance + 1e-8)
            aggregated.append(self.aggregate[scale](torch.cat([mean, std], dim=1)))
        return tuple(aggregated)  # type: ignore[return-value]

    def forward(self, residual_frames: Tensor, z_values_um: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features_2d = self.aggregate_frames(residual_frames)
        if z_values_um.ndim == 1:
            z_values_um = z_values_um.unsqueeze(0)
        if z_values_um.ndim != 2:
            raise ValueError("z_values_um must be [Z] or [B,Z]")
        batch = residual_frames.shape[0]
        if z_values_um.shape[0] == 1 and batch > 1:
            z_values_um = z_values_um.expand(batch, -1)
        if z_values_um.shape[0] != batch:
            raise ValueError("z_values_um batch does not match residual frames")
        z_count = z_values_um.shape[1]
        outputs: list[Tensor] = []
        for scale, feature in enumerate(features_2d):
            _, _, height, width = feature.shape
            broadcast = feature.unsqueeze(2).expand(-1, -1, z_count, -1, -1)
            z_channel = (z_values_um / self.z_scale_um)[:, None, :, None, None]
            z_channel = z_channel.expand(-1, 1, -1, height, width).to(feature)
            outputs.append(self.lift[scale](torch.cat([broadcast, z_channel], dim=1)))
        return tuple(outputs)  # type: ignore[return-value]
