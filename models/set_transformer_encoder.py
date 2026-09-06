from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .blocks import ConvBlock3D
from .set_encoder import SharedFrameEncoder


class MultiheadAttentionBlock(nn.Module):
    """Pre-normalized attention block used by ISAB and PMA."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.query_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.key_norm(key_value),
            self.key_norm(key_value),
            need_weights=False,
        )
        value = query + attended
        return value + self.feed_forward(self.output_norm(value))


class InducedSetAttentionBlock(nn.Module):
    """Permutation-equivariant Set Transformer block with learned inducing tokens."""

    def __init__(self, channels: int, heads: int, inducing_points: int) -> None:
        super().__init__()
        if inducing_points < 1:
            raise ValueError("inducing_points must be positive")
        self.inducing = nn.Parameter(torch.empty(1, inducing_points, channels))
        nn.init.normal_(self.inducing, std=0.02)
        self.induce = MultiheadAttentionBlock(channels, heads)
        self.broadcast = MultiheadAttentionBlock(channels, heads)

    def forward(self, tokens: Tensor) -> Tensor:
        inducing = self.inducing.expand(tokens.shape[0], -1, -1)
        compressed = self.induce(inducing, tokens)
        return self.broadcast(tokens, compressed)


class PoolingByMultiheadAttention(nn.Module):
    """One-seed permutation-invariant pooling used as global set context."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, channels))
        nn.init.normal_(self.seed, std=0.02)
        self.pool = MultiheadAttentionBlock(channels, heads)

    def forward(self, tokens: Tensor) -> Tensor:
        seed = self.seed.expand(tokens.shape[0], -1, -1)
        return self.pool(seed, tokens)


class SetTransformerScale(nn.Module):
    """Turn per-frame descriptors into invariant frame weights."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        inducing_points: int,
        layers: int,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.encoder = nn.ModuleList(
            [
                InducedSetAttentionBlock(channels, heads, inducing_points)
                for _ in range(layers)
            ]
        )
        self.pool = PoolingByMultiheadAttention(channels, heads)
        self.score = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )
        # Start extremely close to uniform mean/std pooling while preserving a
        # gradient path into the attention blocks on the first optimization step.
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, descriptors: Tensor) -> Tensor:
        tokens = descriptors
        for block in self.encoder:
            tokens = block(tokens)
        context = self.pool(tokens).expand(-1, tokens.shape[1], -1)
        logits = self.score(torch.cat([tokens, context], dim=-1)).squeeze(-1)
        return torch.softmax(logits, dim=1)


class SetTransformerEncoder(nn.Module):
    """Memory-aware Set Transformer over unordered residual frames.

    Attention acts on one globally pooled token per frame and scale. The learned
    invariant frame weights then pool the full spatial feature maps, retaining
    the weighted mean and standard deviation used by the original SetEncoder.
    This avoids infeasible attention over every pixel x every frame.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int],
        *,
        frame_chunk_size: int = 8,
        z_scale_um: float = 100.0,
        use_checkpoint: bool = True,
        anti_alias: bool = False,
        heads: int = 4,
        inducing_points: int = 16,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SetTransformerEncoder requires exactly three channel scales")
        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        self.channels = tuple(int(value) for value in channels)
        if any(channel % heads != 0 for channel in self.channels):
            raise ValueError("Every set channel count must be divisible by attention heads")
        self.frame_chunk_size = int(frame_chunk_size)
        self.z_scale_um = float(z_scale_um)
        self.use_checkpoint = bool(use_checkpoint)
        self.shared = SharedFrameEncoder(self.channels, anti_alias=anti_alias)
        self.transformers = nn.ModuleList(
            [
                SetTransformerScale(
                    channel,
                    heads=heads,
                    inducing_points=inducing_points,
                    layers=layers,
                )
                for channel in self.channels
            ]
        )
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

        per_scale_chunks: list[list[Tensor]] = [[], [], []]
        for start in range(0, frame_count, self.frame_chunk_size):
            chunk = residual_frames[:, start : start + self.frame_chunk_size]
            chunk_count = chunk.shape[1]
            flat = chunk.reshape(batch * chunk_count, 1, height, width)
            if self.use_checkpoint and self.training:
                encoded = checkpoint(self.shared, flat, use_reentrant=False)
            else:
                encoded = self.shared(flat)
            for scale, feature in enumerate(encoded):
                per_scale_chunks[scale].append(
                    feature.reshape(batch, chunk_count, *feature.shape[1:])
                )

        aggregated: list[Tensor] = []
        for scale, chunks in enumerate(per_scale_chunks):
            features = torch.cat(chunks, dim=1)
            descriptors = features.mean(dim=(-2, -1))
            weights = self.transformers[scale](descriptors)
            spatial_weights = weights[:, :, None, None, None]
            mean = (spatial_weights * features).sum(dim=1)
            second_moment = (spatial_weights * features.square()).sum(dim=1)
            variance = (second_moment - mean.square()).clamp_min(0.0)
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
