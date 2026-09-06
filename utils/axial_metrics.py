from __future__ import annotations

from typing import Any

import numpy as np
import torch


def axial_lateral_metrics(output: Any, z_values_um: np.ndarray) -> dict[str, float]:
    """Return detached diagnostics for the axial/lateral parameterization."""
    if output.axial_mass_fraction is None:
        return {}

    axial = output.axial_mass_fraction.detach()[0, 0, :, 0, 0]
    reconstruction = output.reconstruction.detach()[0, 0]
    layer_mass = reconstruction.sum(dim=(-2, -1))
    actual = layer_mass / layer_mass.sum().clamp_min(torch.finfo(layer_mass.dtype).tiny)
    entropy = -(axial * axial.clamp_min(torch.finfo(axial.dtype).tiny).log()).sum()
    centroid = (actual * torch.as_tensor(z_values_um, device=actual.device)).sum()
    peak_index = int(actual.argmax().item())

    metrics = {
        "axial_entropy": float(entropy.item()),
        "axial_mass_consistency_max_abs": float((axial - actual).abs().max().item()),
        "depth_peak_um": float(z_values_um[peak_index]),
        "depth_centroid_um": float(centroid.item()),
        "mass_40_60_fraction": float(
            actual[
                torch.as_tensor(
                    (z_values_um >= 40.0) & (z_values_um <= 60.0),
                    device=actual.device,
                )
            ].sum().item()
        ),
        "reconstruction_max_voxel_fraction": float(
            reconstruction.max().div(reconstruction.sum().clamp_min(torch.finfo(reconstruction.dtype).tiny)).item()
        ),
    }
    if output.lateral_log_modulation is not None:
        lateral = output.lateral_log_modulation.detach()
        metrics["lateral_log_modulation_rms"] = float(lateral.square().mean().sqrt().item())
        metrics["lateral_log_modulation_max_abs"] = float(lateral.abs().max().item())

    for index, z_um in enumerate(z_values_um):
        label = f"{float(z_um):g}".replace(".", "p")
        metrics[f"axial_mass_fraction_z{label}_um"] = float(axial[index].item())
        metrics[f"reconstruction_mass_fraction_z{label}_um"] = float(actual[index].item())
    return metrics
