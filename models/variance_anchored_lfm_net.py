from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from physics.lfm_operator import LFMOperator

from .blocks import normalize_feature_input
from .decoder3d import ResidualDecoder3D
from .encoder3d import Encoder3D
from .gated_fusion import GatedSetFusion
from .set_encoder import SetEncoder


@dataclass
class ModelOutput:
    reconstruction: Tensor
    residual: Tensor
    gates: tuple[Tensor, Tensor, Tensor]
    alphas: Tensor
    beta: Tensor
    variance_features: tuple[Tensor, Tensor, Tensor]
    mean_features: tuple[Tensor, Tensor, Tensor]
    set_features: tuple[Tensor, Tensor, Tensor] | None


def anchor_preserving_positive_map(
    anchor: Tensor,
    residual: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Apply signed residual corrections while preserving a nonnegative anchor."""
    if anchor.shape != residual.shape:
        raise ValueError("anchor and residual must have the same shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    anchor = anchor.clamp_min(0.0)
    scale_floor = max(float(eps), torch.finfo(anchor.dtype).tiny)
    scale = anchor.clamp_min(scale_floor)
    negative_residual = residual.clamp_max(0.0)
    negative_correction = anchor * torch.exp(negative_residual / scale)
    return torch.where(residual >= 0, anchor + residual, negative_correction)


class VarianceAnchoredLFMNet(nn.Module):
    def __init__(
        self,
        *,
        var_channels: list[int] | tuple[int, int, int] = (16, 32, 64),
        mean_channels: list[int] | tuple[int, int, int] = (16, 32, 64),
        set_channels: list[int] | tuple[int, int, int] = (8, 16, 32),
        decoder_channels: list[int] | tuple[int, int, int] = (16, 32, 64),
        set_frame_chunk_size: int = 8,
        alpha_init: float = 0.05,
        beta_range: float = 0.2,
        positivity_eps: float = 1e-8,
        z_scale_um: float = 100.0,
        set_use_checkpoint: bool = True,
        use_variance_branch: bool = True,
        use_mean_branch: bool = True,
        use_set_branch: bool = True,
        use_gate: bool = True,
        use_network_refinement: bool = True,
    ) -> None:
        super().__init__()
        var_channels = tuple(int(v) for v in var_channels)
        mean_channels = tuple(int(v) for v in mean_channels)
        set_channels = tuple(int(v) for v in set_channels)
        decoder_channels = tuple(int(v) for v in decoder_channels)
        if not all(len(values) == 3 for values in (var_channels, mean_channels, set_channels, decoder_channels)):
            raise ValueError("All channel configurations must contain three scales")
        self.var_encoder = Encoder3D(var_channels)
        self.mean_encoder = Encoder3D(mean_channels)
        self.set_encoder = SetEncoder(
            set_channels,
            frame_chunk_size=set_frame_chunk_size,
            z_scale_um=z_scale_um,
            use_checkpoint=set_use_checkpoint,
        )
        self.fusion = GatedSetFusion(
            var_channels,
            mean_channels,
            set_channels,
            decoder_channels,
            alpha_init=alpha_init,
        )
        self.decoder = ResidualDecoder3D(decoder_channels)
        self.var_channels = var_channels
        self.mean_channels = mean_channels
        self.beta_range = float(beta_range)
        if positivity_eps <= 0:
            raise ValueError("positivity_eps must be positive")
        self.positivity_eps = float(positivity_eps)
        self.use_variance_branch = bool(use_variance_branch)
        self.use_mean_branch = bool(use_mean_branch)
        self.use_set_branch = bool(use_set_branch)
        self.use_gate = bool(use_gate)
        self.use_network_refinement = bool(use_network_refinement)
        self.register_buffer("beta0", torch.tensor(1.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self) -> Tensor:
        return self.beta0 * (1.0 + self.beta_range * torch.tanh(self.raw_beta))

    @torch.no_grad()
    def initialize_beta(self, f_var: Tensor, measured_mean: Tensor, operator: LFMOperator, eps: float = 1e-8) -> float:
        prediction = operator(f_var)
        numerator = (prediction * measured_mean).sum()
        denominator = prediction.square().sum() + eps
        beta0 = (numerator / denominator).clamp_min(eps)
        self.beta0.copy_(beta0.to(self.beta0))
        self.raw_beta.zero_()
        return float(self.beta0.item())

    @staticmethod
    def _pad_lateral(value: Tensor, multiple: int = 4) -> tuple[Tensor, tuple[int, int]]:
        height, width = value.shape[-2:]
        pad_h = (-height) % multiple
        pad_w = (-width) % multiple
        if pad_h or pad_w:
            value = F.pad(value, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
        return value, (height, width)

    @staticmethod
    def _zero_pyramid(reference: Tensor, channels: tuple[int, int, int]) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, depth, height, width = reference.shape
        return (
            reference.new_zeros((batch, channels[0], depth, height, width)),
            reference.new_zeros((batch, channels[1], depth, (height + 1) // 2, (width + 1) // 2)),
            reference.new_zeros((batch, channels[2], depth, (height + 3) // 4, (width + 3) // 4)),
        )

    def forward(
        self,
        f_var: Tensor,
        g_mean: Tensor,
        residual_frames: Tensor,
        z_values_um: Tensor,
    ) -> ModelOutput:
        if f_var.shape != g_mean.shape or f_var.ndim != 5 or f_var.shape[1] != 1:
            raise ValueError("f_var and g_mean must share shape [B,1,Z,H,W]")
        if residual_frames.shape[0] != f_var.shape[0]:
            raise ValueError("Residual-frame batch does not match volume batch")
        padded_var, original_shape = self._pad_lateral(f_var)
        padded_mean, _ = self._pad_lateral(g_mean)
        pad_h = padded_var.shape[-2] - residual_frames.shape[-2]
        pad_w = padded_var.shape[-1] - residual_frames.shape[-1]
        if pad_h < 0 or pad_w < 0:
            residual_frames = F.interpolate(
                residual_frames.flatten(0, 1),
                size=padded_var.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, residual_frames.shape[:2])
        elif pad_h or pad_w:
            residual_frames = F.pad(
                residual_frames, (0, pad_w, 0, pad_h, 0, 0), mode="replicate"
            )

        var_input = normalize_feature_input(padded_var, dims=(-3, -2, -1))
        mean_input = normalize_feature_input(padded_mean, dims=(-3, -2, -1))
        variance_features = (
            self.var_encoder(var_input)
            if self.use_variance_branch
            else self._zero_pyramid(padded_var, self.var_channels)
        )
        mean_features = (
            self.mean_encoder(mean_input)
            if self.use_mean_branch
            else self._zero_pyramid(padded_mean, self.mean_channels)
        )
        set_features = (
            self.set_encoder(residual_frames, z_values_um) if self.use_set_branch else None
        )
        fused, gates, alphas = self.fusion(
            variance_features,
            mean_features,
            set_features,
            use_set=self.use_set_branch,
            use_gate=self.use_gate,
        )
        residual = self.decoder(fused)
        if not self.use_network_refinement:
            residual = torch.zeros_like(residual)
        height, width = original_shape
        residual = residual[..., :height, :width]
        gates = tuple(gate[..., :height, :width] for gate in gates)  # type: ignore[assignment]
        reconstruction = anchor_preserving_positive_map(
            self.beta * f_var,
            residual,
            eps=self.positivity_eps,
        )
        return ModelOutput(
            reconstruction=reconstruction,
            residual=residual,
            gates=gates,
            alphas=alphas,
            beta=self.beta,
            variance_features=variance_features,
            mean_features=mean_features,
            set_features=set_features,
        )
