from __future__ import annotations

import torch
from torch import Tensor


def axial_lateral_decoupled_map(
    anchor: Tensor,
    residual: Tensor,
    *,
    lateral_log_bound: float = 0.5,
    axial_logit_scale: float = 1.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Separate axial mass from normalized within-layer structure.

    The network emits one residual field. Its lateral mean produces unconstrained
    axial logits, while its exactly zero-mean component can only modify the
    normalized structure inside each depth layer. No independent per-depth
    trainable scale is introduced.
    """
    if anchor.shape != residual.shape:
        raise ValueError("anchor and residual must have the same shape")
    if anchor.ndim != 5:
        raise ValueError("anchor and residual must have shape [B,C,Z,H,W]")
    if lateral_log_bound < 0.0:
        raise ValueError("lateral_log_bound must be nonnegative")
    if axial_logit_scale <= 0.0 or eps <= 0.0:
        raise ValueError("axial_logit_scale and eps must be positive")

    nonnegative_anchor = anchor.clamp_min(0.0)
    total_mass = nonnegative_anchor.sum(dim=(-3, -2, -1), keepdim=True)
    residual_scale = nonnegative_anchor.detach().mean(
        dim=(-3, -2, -1), keepdim=True
    ).clamp_min(float(eps))
    dimensionless_residual = residual / residual_scale

    lateral_mean = dimensionless_residual.mean(dim=(-2, -1), keepdim=True)
    axial_logits = float(axial_logit_scale) * lateral_mean
    axial_logits = axial_logits - axial_logits.mean(dim=-3, keepdim=True)
    axial_mass_fraction = torch.softmax(axial_logits, dim=-3)

    safe_anchor = nonnegative_anchor.clamp_min(float(eps))
    anchor_layer_shape = safe_anchor / safe_anchor.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(float(eps))
    centered_lateral_residual = dimensionless_residual - lateral_mean
    if lateral_log_bound == 0.0:
        lateral_log_modulation = torch.zeros_like(centered_lateral_residual)
    else:
        bound = float(lateral_log_bound)
        lateral_log_modulation = bound * torch.tanh(
            centered_lateral_residual / bound
        )
    modulated_shape = anchor_layer_shape * torch.exp(lateral_log_modulation)
    normalized_layer_shape = modulated_shape / modulated_shape.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(float(eps))
    reconstruction = total_mass * axial_mass_fraction * normalized_layer_shape
    return reconstruction, axial_mass_fraction, lateral_log_modulation
