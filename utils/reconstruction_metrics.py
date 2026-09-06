from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _ssim_2d(reference: np.ndarray, estimate: np.ndarray) -> float:
    x = torch.from_numpy(reference.astype(np.float32))[None, None]
    y = torch.from_numpy(estimate.astype(np.float32))[None, None]
    data_range = float(max(reference.max(), estimate.max()) - min(reference.min(), estimate.min()))
    if data_range <= 0:
        return 1.0
    kernel = torch.ones((1, 1, 11, 11), dtype=torch.float32) / 121.0
    mu_x = F.conv2d(x, kernel, padding=5)
    mu_y = F.conv2d(y, kernel, padding=5)
    sigma_x = F.conv2d(x.square(), kernel, padding=5) - mu_x.square()
    sigma_y = F.conv2d(y.square(), kernel, padding=5) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, padding=5) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return float(score.mean().item())


def reconstruction_metrics(
    reconstruction_zyx: np.ndarray,
    ground_truth_zyx: np.ndarray,
    z_um: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(reconstruction_zyx, dtype=np.float64)
    truth = np.asarray(ground_truth_zyx, dtype=np.float64)
    z = np.asarray(z_um, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3 or z.shape != (prediction.shape[0],):
        raise ValueError("Metric inputs have incompatible shapes")
    prediction = np.maximum(prediction, 0)
    truth = np.maximum(truth, 0)
    raw_nrmse = np.linalg.norm((prediction - truth).ravel()) / max(
        np.linalg.norm(truth.ravel()), 1e-12
    )
    scale = np.vdot(prediction.ravel(), truth.ravel()).real / max(
        np.vdot(prediction.ravel(), prediction.ravel()).real, 1e-12
    )
    aligned_nrmse = np.linalg.norm((scale * prediction - truth).ravel()) / max(
        np.linalg.norm(truth.ravel()), 1e-12
    )
    pred_mass = prediction.sum(axis=(1, 2))
    truth_mass = truth.sum(axis=(1, 2))
    pred_fraction = pred_mass / max(pred_mass.sum(), 1e-12)
    truth_fraction = truth_mass / max(truth_mass.sum(), 1e-12)
    dz = float(np.median(np.diff(z))) if len(z) > 1 else 0.0
    w1 = float(np.abs(np.cumsum(pred_fraction) - np.cumsum(truth_fraction)).sum() * dz)
    occupied = truth_fraction > 1e-8
    expanded = occupied.copy()
    expanded[:-1] |= occupied[1:]
    expanded[1:] |= occupied[:-1]
    outside = float(pred_fraction[~expanded].sum())
    xy_prediction = prediction.max(axis=0)
    xy_truth = truth.max(axis=0)
    return {
        "gt_raw_nrmse": float(raw_nrmse),
        "gt_scale_aligned_nrmse": float(aligned_nrmse),
        "gt_axial_w1_um": w1,
        "gt_axial_mass_l1": float(np.abs(pred_fraction - truth_fraction).sum()),
        "gt_support_outside_pm10_mass": outside,
        "gt_xy_mip_ssim": _ssim_2d(xy_truth, xy_prediction),
        "predicted_depth_peak_um": float(z[int(np.argmax(pred_fraction))]),
        "predicted_depth_centroid_um": float((z * pred_fraction).sum()),
        "truth_depth_centroid_um": float((z * truth_fraction).sum()),
    }
