from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


@dataclass
class LossBreakdown:
    total: Tensor
    raw_mean: Tensor
    normalized_mean: Tensor
    weighted_mean: Tensor
    raw_var: Tensor
    normalized_var: Tensor
    weighted_var: Tensor
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


def compute_self_supervised_loss(
    reconstruction: Tensor,
    measured_mean: Tensor,
    measured_variance: Tensor,
    operator: LFMOperator,
    variance_model: VariancePhysicsModel,
    *,
    lambda_mean: float = 1.0,
    lambda_var: float = 1.0,
    lambda_tv: float = 1e-5,
    lambda_tv_z: float = 0.5,
    var_log_eps: float = 1e-6,
    mean_loss_type: str = "smooth_l1",
) -> LossBreakdown:
    predicted_mean = operator(reconstruction)
    predicted_variance = variance_model(reconstruction, measured_mean)

    raw_mean = _pointwise_loss(predicted_mean, measured_mean, mean_loss_type)
    mean_scale = measured_mean.detach().abs().mean().clamp_min(1e-8)
    normalized_mean = _pointwise_loss(
        predicted_mean / mean_scale, measured_mean / mean_scale, mean_loss_type
    )

    raw_var = F.smooth_l1_loss(predicted_variance, measured_variance)
    log_prediction = torch.log(predicted_variance.clamp_min(0.0) + var_log_eps)
    log_measurement = torch.log(measured_variance.clamp_min(0.0) + var_log_eps)
    normalized_var = F.smooth_l1_loss(log_prediction, log_measurement)

    tv = total_variation_3d(reconstruction, z_weight=lambda_tv_z)
    weighted_mean = float(lambda_mean) * normalized_mean
    weighted_var = float(lambda_var) * normalized_var
    weighted_tv = float(lambda_tv) * tv
    total = weighted_mean + weighted_var + weighted_tv
    return LossBreakdown(
        total=total,
        raw_mean=raw_mean,
        normalized_mean=normalized_mean,
        weighted_mean=weighted_mean,
        raw_var=raw_var,
        normalized_var=normalized_var,
        weighted_var=weighted_var,
        tv=tv,
        weighted_tv=weighted_tv,
        predicted_mean=predicted_mean,
        predicted_variance=predicted_variance,
    )
