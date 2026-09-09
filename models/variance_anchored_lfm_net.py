from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from physics.lfm_operator import LFMOperator

from .blocks import group_count, normalize_feature_input
from .decoder3d import ResidualDecoder3D
from .output_parameterizations import axial_lateral_decoupled_map
from .encoder3d import Encoder3D
from .gated_fusion import GatedSetFusion
from .set_encoder import SetEncoder
from .set_transformer_encoder import SetTransformerEncoder


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
    detail_residual: Tensor | None
    axial_mass_fraction: Tensor | None
    lateral_log_modulation: Tensor | None
    detail_strength: float


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


def bounded_log_multiplicative_map(
    anchor: Tensor,
    residual: Tensor,
    *,
    bound: float = 4.0,
    eps: float = 1e-8,
) -> Tensor:
    """Apply a symmetric bounded log correction relative to a positive anchor."""
    if anchor.shape != residual.shape:
        raise ValueError("anchor and residual must have the same shape")
    if bound <= 0.0 or eps <= 0.0:
        raise ValueError("bound and eps must be positive")
    safe_anchor = anchor.clamp_min(eps)
    log_modulation = float(bound) * torch.tanh(residual / float(bound))
    return safe_anchor * torch.exp(log_modulation)


class VarianceDetailHead(nn.Module):
    """Native-XY-resolution detail path without cross-depth mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv3d(channels, 1, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.body(value))


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
        set_encoder_type: str = "mean_std",
        set_transformer_heads: int = 4,
        set_transformer_inducing_points: int = 16,
        set_transformer_layers: int = 2,
        use_role_separated_detail: bool = False,
        detail_lowpass_passes: int = 2,
        detail_lowpass_auxiliary_features: bool = True,
        detail_application: str = "legacy_additive",
        detail_log_modulation_bound: float = 0.15,
        network_context_pad_xy: int = 0,
        network_context_pad_mode: str = "reflect",
        anti_alias_downsampling: bool = False,
        coarse_application: str = "legacy_anchor_positive",
        coarse_log_residual_bound: float = 4.0,
        lateral_log_residual_bound: float = 0.5,
        axial_logit_scale: float = 1.0,
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
        if int(detail_lowpass_passes) < 1:
            raise ValueError("detail_lowpass_passes must be positive")
        if detail_application not in {"legacy_additive", "mass_preserving_multiplicative"}:
            raise ValueError(
                "detail_application must be 'legacy_additive' or "
                "'mass_preserving_multiplicative'"
            )
        if float(detail_log_modulation_bound) <= 0.0:
            raise ValueError("detail_log_modulation_bound must be positive")
        if int(network_context_pad_xy) < 0 or int(network_context_pad_xy) % 4 != 0:
            raise ValueError("network_context_pad_xy must be a nonnegative multiple of four")
        if network_context_pad_mode not in {"reflect", "replicate"}:
            raise ValueError("network_context_pad_mode must be 'reflect' or 'replicate'")
        if coarse_application not in {
            "legacy_anchor_positive",
            "bounded_log_multiplicative",
            "axial_lateral_decoupled",
        }:
            raise ValueError(
                "coarse_application must be 'legacy_anchor_positive', "
                "'bounded_log_multiplicative', or 'axial_lateral_decoupled'"
            )
        if float(coarse_log_residual_bound) <= 0.0:
            raise ValueError("coarse_log_residual_bound must be positive")
        if float(lateral_log_residual_bound) < 0.0:
            raise ValueError("lateral_log_residual_bound must be nonnegative")
        if float(axial_logit_scale) <= 0.0:
            raise ValueError("axial_logit_scale must be positive")
        self.anti_alias_downsampling = bool(anti_alias_downsampling)
        self.var_encoder = Encoder3D(var_channels, anti_alias=self.anti_alias_downsampling)
        self.mean_encoder = Encoder3D(mean_channels, anti_alias=self.anti_alias_downsampling)
        self.set_encoder_type = str(set_encoder_type)
        set_common = {
            "frame_chunk_size": set_frame_chunk_size,
            "z_scale_um": z_scale_um,
            "use_checkpoint": set_use_checkpoint,
            "anti_alias": self.anti_alias_downsampling,
        }
        if self.set_encoder_type == "mean_std":
            self.set_encoder = SetEncoder(set_channels, **set_common)
        elif self.set_encoder_type == "set_transformer":
            baseline_rng_consumer = SetEncoder(set_channels, **set_common)
            del baseline_rng_consumer
            with torch.random.fork_rng(devices=[]):
                self.set_encoder = SetTransformerEncoder(
                    set_channels,
                    heads=set_transformer_heads,
                    inducing_points=set_transformer_inducing_points,
                    layers=set_transformer_layers,
                    **set_common,
                )
        else:
            raise ValueError(
                "set_encoder_type must be 'mean_std' or 'set_transformer', "
                f"got {self.set_encoder_type!r}"
            )
        self.fusion = GatedSetFusion(
            var_channels,
            mean_channels,
            set_channels,
            decoder_channels,
            alpha_init=alpha_init,
        )
        self.decoder = ResidualDecoder3D(decoder_channels)
        self.use_role_separated_detail = bool(use_role_separated_detail)
        self.detail_lowpass_passes = int(detail_lowpass_passes)
        self.detail_lowpass_auxiliary_features = bool(detail_lowpass_auxiliary_features)
        self.detail_application = str(detail_application)
        self.detail_log_modulation_bound = float(detail_log_modulation_bound)
        self.network_context_pad_xy = int(network_context_pad_xy)
        self.network_context_pad_mode = str(network_context_pad_mode)
        self.coarse_application = str(coarse_application)
        self.coarse_log_residual_bound = float(coarse_log_residual_bound)
        self.lateral_log_residual_bound = float(lateral_log_residual_bound)
        self.axial_logit_scale = float(axial_logit_scale)
        if self.use_role_separated_detail:
            # The candidate branch must not shift initialization of common modules.
            with torch.random.fork_rng(devices=[]):
                self.detail_head: VarianceDetailHead | None = VarianceDetailHead(var_channels[0])
        else:
            self.detail_head = None
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

    def beta_from_base(self, beta0: Tensor) -> Tensor:
        """Apply the shared bounded correction to per-item analytic gains."""
        if not beta0.is_floating_point():
            raise TypeError("beta0 must be floating point")
        if not torch.isfinite(beta0).all() or bool((beta0 <= 0).any()):
            raise ValueError("Every beta0 value must be finite and positive")
        return beta0 * (1.0 + self.beta_range * torch.tanh(self.raw_beta))

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

    @staticmethod
    def _lowpass_lateral(value: Tensor, passes: int) -> Tensor:
        result = value
        for _ in range(int(passes)):
            padded = F.pad(result, (1, 1, 1, 1, 0, 0), mode="replicate")
            result = F.avg_pool3d(padded, kernel_size=(1, 3, 3), stride=1)
        return result

    def _add_network_context(self, value: Tensor) -> Tensor:
        pad = self.network_context_pad_xy
        if pad == 0:
            return value
        if self.network_context_pad_mode == "reflect" and (
            pad >= value.shape[-2] or pad >= value.shape[-1]
        ):
            raise ValueError(
                "reflect network context must be smaller than both lateral dimensions"
            )
        return F.pad(
            value,
            (pad, pad, pad, pad, 0, 0),
            mode=self.network_context_pad_mode,
        )

    def _crop_network_context(
        self,
        value: Tensor,
        original_shape: tuple[int, int],
        *,
        scale: int = 1,
    ) -> Tensor:
        height, width = original_shape
        offset = self.network_context_pad_xy // scale
        target_height = (height + scale - 1) // scale
        target_width = (width + scale - 1) // scale
        return value[
            ...,
            offset : offset + target_height,
            offset : offset + target_width,
        ]

    def forward(
        self,
        f_var: Tensor,
        g_mean: Tensor,
        residual_frames: Tensor,
        z_values_um: Tensor,
        *,
        var_feature_volume: Tensor | None = None,
        detail_strength: float = 1.0,
        beta0: Tensor | None = None,
    ) -> ModelOutput:
        if f_var.shape != g_mean.shape or f_var.ndim != 5 or f_var.shape[1] != 1:
            raise ValueError("f_var and g_mean must share shape [B,1,Z,H,W]")
        if residual_frames.shape[0] != f_var.shape[0]:
            raise ValueError("Residual-frame batch does not match volume batch")
        if var_feature_volume is None:
            var_feature_volume = f_var
        if var_feature_volume.shape != f_var.shape:
            raise ValueError("var_feature_volume must have the same shape as f_var")
        if not 0.0 <= float(detail_strength) <= 1.0:
            raise ValueError("detail_strength must lie in [0,1]")
        original_shape = f_var.shape[-2:]
        padded_var, _ = self._pad_lateral(
            self._add_network_context(var_feature_volume)
        )
        padded_mean, _ = self._pad_lateral(self._add_network_context(g_mean))
        residual_frames = self._add_network_context(residual_frames)
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
        if self.use_role_separated_detail and self.detail_lowpass_auxiliary_features:
            mean_features = tuple(
                self._lowpass_lateral(feature, self.detail_lowpass_passes)
                for feature in mean_features
            )  # type: ignore[assignment]
            if set_features is not None:
                set_features = tuple(
                    self._lowpass_lateral(feature, self.detail_lowpass_passes)
                    for feature in set_features
                )  # type: ignore[assignment]
        fused, gates, alphas = self.fusion(
            variance_features,
            mean_features,
            set_features,
            use_set=self.use_set_branch,
            use_gate=self.use_gate,
        )
        residual = self.decoder(fused)
        detail_residual: Tensor | None = None
        if self.use_role_separated_detail:
            assert self.detail_head is not None
            raw_detail = self.detail_head(variance_features[0])
            detail_residual = raw_detail - self._lowpass_lateral(
                raw_detail, self.detail_lowpass_passes
            )
            if self.detail_application == "legacy_additive":
                residual = residual + float(detail_strength) * detail_residual
        if not self.use_network_refinement:
            residual = torch.zeros_like(residual)
            if detail_residual is not None:
                detail_residual = torch.zeros_like(detail_residual)
        residual = self._crop_network_context(residual, original_shape)
        if detail_residual is not None:
            detail_residual = self._crop_network_context(detail_residual, original_shape)
        gates = tuple(
            self._crop_network_context(gate, original_shape, scale=2**index)
            for index, gate in enumerate(gates)
        )  # type: ignore[assignment]
        if beta0 is None:
            beta_value = self.beta
            anchor = beta_value * f_var
        else:
            beta0 = beta0.to(device=f_var.device, dtype=f_var.dtype).reshape(-1)
            if beta0.numel() != f_var.shape[0]:
                raise ValueError("beta0 must contain exactly one value per batch item")
            beta_value = self.beta_from_base(beta0)
            anchor = beta_value[:, None, None, None, None] * f_var
        axial_mass_fraction: Tensor | None = None
        lateral_log_modulation: Tensor | None = None
        if self.coarse_application == "legacy_anchor_positive":
            base_reconstruction = anchor_preserving_positive_map(
                anchor,
                residual,
                eps=self.positivity_eps,
            )
        elif self.coarse_application == "bounded_log_multiplicative":
            base_reconstruction = bounded_log_multiplicative_map(
                anchor,
                residual,
                bound=self.coarse_log_residual_bound,
                eps=self.positivity_eps,
            )
        else:
            (
                base_reconstruction,
                axial_mass_fraction,
                lateral_log_modulation,
            ) = axial_lateral_decoupled_map(
                anchor,
                residual,
                lateral_log_bound=self.lateral_log_residual_bound,
                axial_logit_scale=self.axial_logit_scale,
                eps=self.positivity_eps,
            )
        reconstruction = base_reconstruction
        if (
            detail_residual is not None
            and self.detail_application == "mass_preserving_multiplicative"
        ):
            log_modulation = (
                float(detail_strength)
                * self.detail_log_modulation_bound
                * torch.tanh(detail_residual)
            )
            modulated = base_reconstruction * torch.exp(log_modulation)
            base_mass = base_reconstruction.sum(dim=(-2, -1), keepdim=True)
            modulated_mass = modulated.sum(dim=(-2, -1), keepdim=True)
            reconstruction = modulated * (
                base_mass / modulated_mass.clamp_min(self.positivity_eps)
            )
            detail_residual = log_modulation
        return ModelOutput(
            reconstruction=reconstruction,
            residual=residual,
            gates=gates,
            alphas=alphas,
            beta=beta_value,
            variance_features=variance_features,
            mean_features=mean_features,
            set_features=set_features,
            detail_residual=detail_residual,
            axial_mass_fraction=axial_mass_fraction,
            lateral_log_modulation=lateral_log_modulation,
            detail_strength=float(detail_strength),
        )
