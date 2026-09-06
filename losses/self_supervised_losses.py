from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from physics.lfm_operator import LFMOperator


class VariancePhysicsModel(nn.Module):
    def forward(self, reconstruction: Tensor, measured_mean: Tensor) -> Tensor:
        raise NotImplementedError


class TaylorH2VarianceModel(VariancePhysicsModel):
    """V1 approximation: V = H**2(g**2) + alpha*I_bar + sigma_read**2."""

    def __init__(
        self,
        operator: LFMOperator,
        *,
        alpha_noise: float = 0.0,
        sigma_read: float = 0.0,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.alpha_noise = float(alpha_noise)
        self.sigma_read = float(sigma_read)

    def forward(self, reconstruction: Tensor, measured_mean: Tensor) -> Tensor:
        signal_variance = self.operator.forward_squared(reconstruction.square()).clamp_min(0.0)
        noise = self.alpha_noise * measured_mean + self.sigma_read**2
        return signal_variance + noise


def _gaussian_blur_2d(value: Tensor) -> Tensor:
    if value.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(value.shape)}")
    kernel_1d = value.new_tensor([1.0, 2.0, 1.0]) / 4.0
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, 3, 3)
    kernel = kernel.expand(value.shape[1], 1, 3, 3)
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=value.shape[1])


def undecimated_laplacian_bands(value: Tensor, levels: int = 2) -> tuple[Tensor, ...]:
    """Return aligned high-to-low band-pass coefficients without decimation."""
    if levels < 1:
        raise ValueError("levels must be positive")
    current = value
    bands: list[Tensor] = []
    for _ in range(int(levels)):
        blurred = _gaussian_blur_2d(current)
        bands.append(current - blurred)
        current = blurred
    return tuple(bands)


@torch.no_grad()
def compute_multiband_reliability(
    split_variance_a: Tensor,
    split_variance_b: Tensor,
    *,
    levels: int = 2,
    log_eps: float = 1e-6,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Estimate reproducible frequency bands from disjoint frame subsets."""
    if split_variance_a.shape != split_variance_b.shape:
        raise ValueError("Split variance maps must share shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    log_a = torch.log(split_variance_a.clamp_min(0.0) + float(log_eps))
    log_b = torch.log(split_variance_b.clamp_min(0.0) + float(log_eps))
    bands_a = undecimated_laplacian_bands(log_a, levels)
    bands_b = undecimated_laplacian_bands(log_b, levels)
    correlations: list[Tensor] = []
    for band_a, band_b in zip(bands_a, bands_b):
        centered_a = band_a - band_a.mean(dim=(-2, -1), keepdim=True)
        centered_b = band_b - band_b.mean(dim=(-2, -1), keepdim=True)
        numerator = (centered_a * centered_b).sum(dim=(-2, -1))
        denominator = torch.sqrt(
            centered_a.square().sum(dim=(-2, -1))
            * centered_b.square().sum(dim=(-2, -1))
        ).clamp_min(torch.finfo(band_a.dtype).eps)
        correlations.append((numerator / denominator).mean().clamp(-1.0, 1.0))
    correlation_tensor = torch.stack(correlations)
    weights = torch.where(
        correlation_tensor >= float(threshold),
        correlation_tensor.clamp(0.0, 1.0),
        torch.zeros_like(correlation_tensor),
    )
    return weights, correlation_tensor


def multiband_log_variance_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    levels: int = 2,
    log_eps: float = 1e-6,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must share shape")
    if weights.numel() != int(levels):
        raise ValueError("One reliability weight is required per band")
    log_prediction = torch.log(prediction.clamp_min(0.0) + float(log_eps))
    log_target = torch.log(target.clamp_min(0.0) + float(log_eps))
    prediction_bands = undecimated_laplacian_bands(log_prediction, levels)
    target_bands = undecimated_laplacian_bands(log_target, levels)
    losses = torch.stack(
        [
            F.smooth_l1_loss(prediction_band, target_band)
            for prediction_band, target_band in zip(prediction_bands, target_bands)
        ]
    )
    typed_weights = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * typed_weights).sum() / typed_weights.sum().clamp_min(
        torch.finfo(losses.dtype).eps
    )


@dataclass
class LossBreakdown:
    total: Tensor
    raw_mean: Tensor
    normalized_mean: Tensor
    weighted_mean: Tensor
    raw_var: Tensor
    normalized_var: Tensor
    weighted_var: Tensor
    var_band: Tensor
    weighted_var_band: Tensor
    tv: Tensor
    weighted_tv: Tensor
    predicted_mean: Tensor
    predicted_variance: Tensor

    def scalar_metrics(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach().item()),
            "raw_mean_loss": float(self.raw_mean.detach().item()),
            "normalized_mean_loss": float(self.normalized_mean.detach().item()),
            "weighted_mean_loss": float(self.weighted_mean.detach().item()),
            "raw_var_loss": float(self.raw_var.detach().item()),
            "normalized_var_loss": float(self.normalized_var.detach().item()),
            "weighted_var_loss": float(self.weighted_var.detach().item()),
            "var_band_loss": float(self.var_band.detach().item()),
            "weighted_var_band_loss": float(self.weighted_var_band.detach().item()),
            "tv_loss": float(self.tv.detach().item()),
            "weighted_tv_loss": float(self.weighted_tv.detach().item()),
        }


def _pointwise_loss(prediction: Tensor, target: Tensor, loss_type: str) -> Tensor:
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported loss type: {loss_type!r}")


def total_variation_3d(volume: Tensor, z_weight: float = 1.0) -> Tensor:
    dx = (volume[..., 1:] - volume[..., :-1]).abs().mean()
    dy = (volume[..., 1:, :] - volume[..., :-1, :]).abs().mean()
    if volume.shape[-3] > 1:
        dz = (volume[:, :, 1:] - volume[:, :, :-1]).abs().mean()
    else:
        dz = volume.new_zeros(())
    return dx + dy + float(z_weight) * dz


def _run_physics_model(
    function: nn.Module,
    *inputs: Tensor,
    use_checkpoint: bool,
) -> Tensor:
    if use_checkpoint and torch.is_grad_enabled() and any(value.requires_grad for value in inputs):
        return checkpoint(
            function,
            *inputs,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return function(*inputs)


def compute_self_supervised_loss(
    reconstruction: Tensor,
    measured_mean: Tensor,
    measured_variance: Tensor,
    operator: LFMOperator,
    variance_model: VariancePhysicsModel,
    *,
    lambda_mean: float = 1.0,
    lambda_var: float = 1.0,
    lambda_var_band: float = 0.0,
    lambda_tv: float = 1e-5,
    lambda_tv_z: float = 0.5,
    var_log_eps: float = 1e-6,
    var_band_levels: int = 2,
    var_band_weights: Tensor | None = None,
    mean_loss_type: str = "smooth_l1",
    physics_use_checkpoint: bool = False,
) -> LossBreakdown:
    predicted_mean = _run_physics_model(
        operator,
        reconstruction,
        use_checkpoint=physics_use_checkpoint,
    )
    predicted_variance = _run_physics_model(
        variance_model,
        reconstruction,
        measured_mean,
        use_checkpoint=physics_use_checkpoint,
    )

    raw_mean = _pointwise_loss(predicted_mean, measured_mean, mean_loss_type)
    mean_scale = measured_mean.detach().abs().mean().clamp_min(1e-8)
    normalized_mean = _pointwise_loss(
        predicted_mean / mean_scale, measured_mean / mean_scale, mean_loss_type
    )

    raw_var = F.smooth_l1_loss(predicted_variance, measured_variance)
    log_prediction = torch.log(predicted_variance.clamp_min(0.0) + var_log_eps)
    log_measurement = torch.log(measured_variance.clamp_min(0.0) + var_log_eps)
    normalized_var = F.smooth_l1_loss(log_prediction, log_measurement)
    if var_band_weights is None:
        var_band = reconstruction.new_zeros(())
    else:
        var_band = multiband_log_variance_loss(
            predicted_variance,
            measured_variance,
            var_band_weights,
            levels=var_band_levels,
            log_eps=var_log_eps,
        )

    tv = total_variation_3d(reconstruction, z_weight=lambda_tv_z)
    weighted_mean = float(lambda_mean) * normalized_mean
    weighted_var = float(lambda_var) * normalized_var
    weighted_var_band = float(lambda_var_band) * var_band
    weighted_tv = float(lambda_tv) * tv
    total = weighted_mean + weighted_var + weighted_var_band + weighted_tv
    return LossBreakdown(
        total=total,
        raw_mean=raw_mean,
        normalized_mean=normalized_mean,
        weighted_mean=weighted_mean,
        raw_var=raw_var,
        normalized_var=normalized_var,
        weighted_var=weighted_var,
        var_band=var_band,
        weighted_var_band=weighted_var_band,
        tv=tv,
        weighted_tv=weighted_tv,
        predicted_mean=predicted_mean,
        predicted_variance=predicted_variance,
    )
