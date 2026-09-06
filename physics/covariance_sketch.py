from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import Tensor

from physics.lfm_operator import LFMOperator


@dataclass(frozen=True)
class CsPreprocessReport:
    source_shape: tuple[int, int]
    source_peak_yx: tuple[int, int]
    support_shape: tuple[int, int]
    center_before_normalization: float
    symmetry_relative_error_before: float
    symmetry_relative_error_after: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_preprocess_cs(
    path: str | Path,
    *,
    support_threshold_fraction: float = 1.0 / 255.0,
) -> tuple[Tensor, CsPreprocessReport]:
    """Load a lateral covariance shape and extract an odd, symmetric support.

    The returned kernel is normalized by its zero-lag value, not by its sum.
    Positive-semidefinite projection is deliberately deferred until the target
    image shape is known, because it is performed on the zero-padded circulant
    embedding used by :func:`stationary_covariance_action`.
    """

    source = np.asarray(tifffile.imread(Path(path).expanduser().resolve()))
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D Cs TIFF, got shape {source.shape}")
    source = source.astype(np.float64, copy=False)
    if not np.isfinite(source).all() or source.max(initial=0.0) <= 0.0:
        raise ValueError("Cs must be finite and contain a positive center peak")
    if not 0.0 <= support_threshold_fraction < 1.0:
        raise ValueError("support_threshold_fraction must lie in [0,1)")

    peak_y, peak_x = np.unravel_index(int(np.argmax(source)), source.shape)
    threshold = float(source[peak_y, peak_x]) * float(support_threshold_fraction)
    support_y, support_x = np.nonzero(source >= threshold)
    if support_y.size == 0:
        raise ValueError("No Cs samples passed the support threshold")
    radius_y = int(np.max(np.abs(support_y - peak_y)))
    radius_x = int(np.max(np.abs(support_x - peak_x)))
    radius_y = min(radius_y, peak_y, source.shape[0] - 1 - peak_y)
    radius_x = min(radius_x, peak_x, source.shape[1] - 1 - peak_x)
    cropped = source[
        peak_y - radius_y : peak_y + radius_y + 1,
        peak_x - radius_x : peak_x + radius_x + 1,
    ].copy()
    center = float(cropped[radius_y, radius_x])
    if center <= 0.0:
        raise ValueError("Cs zero-lag sample must be positive")
    inversion = np.flip(cropped, axis=(0, 1))
    norm = max(float(np.linalg.norm(cropped)), np.finfo(np.float64).eps)
    symmetry_before = float(np.linalg.norm(cropped - inversion) / norm)
    cropped = 0.5 * (cropped + inversion)
    cropped /= float(cropped[radius_y, radius_x])
    symmetry_after = float(
        np.linalg.norm(cropped - np.flip(cropped, axis=(0, 1)))
        / max(float(np.linalg.norm(cropped)), np.finfo(np.float64).eps)
    )
    report = CsPreprocessReport(
        source_shape=tuple(int(value) for value in source.shape),
        source_peak_yx=(int(peak_y), int(peak_x)),
        support_shape=tuple(int(value) for value in cropped.shape),
        center_before_normalization=center,
        symmetry_relative_error_before=symmetry_before,
        symmetry_relative_error_after=symmetry_after,
    )
    return torch.from_numpy(np.ascontiguousarray(cropped)).to(torch.float64), report


def _covariance_embedding(
    kernel: Tensor,
    image_shape: tuple[int, int],
) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Build and PSD-project a non-wrapping circulant embedding."""

    if kernel.ndim != 2 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError("Cs kernel must be an odd 2D array")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    kh, kw = int(kernel.shape[0]), int(kernel.shape[1])
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    fft_height, fft_width = height + kh - 1, width + kw - 1
    center_y, center_x = kh // 2, kw // 2
    grid = kernel.new_zeros((fft_height, fft_width))
    for row in range(kh):
        target_row = (row - center_y) % fft_height
        for col in range(kw):
            target_col = (col - center_x) % fft_width
            grid[target_row, target_col] = kernel[row, col]
    spectrum_raw = torch.fft.rfft2(grid).real
    negative = spectrum_raw < 0
    negative_mass = (-spectrum_raw.clamp_max(0)).sum()
    absolute_mass = spectrum_raw.abs().sum().clamp_min(torch.finfo(kernel.dtype).eps)
    spectrum = spectrum_raw.clamp_min(0.0)
    projected_grid = torch.fft.irfft2(spectrum, s=(fft_height, fft_width))
    report: dict[str, float | int] = {
        "fft_height": fft_height,
        "fft_width": fft_width,
        "negative_frequency_fraction": float(negative.double().mean().item()),
        "negative_spectral_mass_fraction": float((negative_mass / absolute_mass).item()),
        "minimum_raw_eigenvalue": float(spectrum_raw.min().item()),
        "minimum_projected_eigenvalue": float(spectrum.min().item()),
    }
    return spectrum, projected_grid, report


def covariance_psd_report(
    kernel: Tensor,
    image_shape: tuple[int, int],
) -> dict[str, float | int]:
    typed = torch.as_tensor(kernel, dtype=torch.float64, device="cpu")
    _, projected, report = _covariance_embedding(typed, image_shape)
    kh, kw = typed.shape
    cy, cx = kh // 2, kw // 2
    reconstructed = typed.new_empty(typed.shape)
    for row in range(kh):
        for col in range(kw):
            reconstructed[row, col] = projected[
                (row - cy) % projected.shape[0],
                (col - cx) % projected.shape[1],
            ]
    report["support_projection_relative_error"] = float(
        (
            torch.linalg.vector_norm(reconstructed - typed)
            / torch.linalg.vector_norm(typed).clamp_min(torch.finfo(typed.dtype).eps)
        ).item()
    )
    report["projected_center"] = float(projected[0, 0].item())
    return report


def stationary_covariance_action(volume: Tensor, kernel: Tensor) -> Tensor:
    """Apply the same lateral stationary Cs independently at every depth.

    ``volume`` is [B,C,Z,H,W]. Zero padding and a PSD-projected circulant
    embedding avoid the periodic wrap-around of an unpadded FFT convolution.
    """

    if volume.ndim != 5:
        raise ValueError(f"Expected volume [B,C,Z,H,W], got {tuple(volume.shape)}")
    typed_kernel = torch.as_tensor(kernel, dtype=volume.dtype, device=volume.device)
    spectrum, _, _ = _covariance_embedding(typed_kernel, volume.shape[-2:])
    height, width = volume.shape[-2:]
    fft_height = int(spectrum.shape[0])
    fft_width = 2 * (int(spectrum.shape[1]) - 1)
    # rfft has one fewer represented column when the original width is odd.
    fft_width = width + int(typed_kernel.shape[1]) - 1
    flattened = volume.reshape(-1, 1, height, width)
    padded = F.pad(flattened, (0, fft_width - width, 0, fft_height - height))
    transformed = torch.fft.rfft2(padded, s=(fft_height, fft_width))
    result = torch.fft.irfft2(
        transformed * spectrum,
        s=(fft_height, fft_width),
    )[..., :height, :width]
    return result.reshape_as(volume)


def empirical_covariance_action(frames: Tensor, probes: Tensor) -> Tensor:
    """Compute Yc(Yc^T q)/(N-1) without materializing sensor covariance."""

    if frames.ndim != 3:
        raise ValueError(f"Expected frames [N,H,W], got {tuple(frames.shape)}")
    if probes.ndim != 3 or probes.shape[-2:] != frames.shape[-2:]:
        raise ValueError("Probes must have shape [K,H,W] matching the frames")
    if frames.shape[0] < 2:
        raise ValueError("At least two frames are required")
    centered = frames - frames.mean(dim=0, keepdim=True)
    weights = torch.einsum("nhw,khw->nk", centered, probes)
    return torch.einsum("nhw,nk->khw", centered, weights) / (frames.shape[0] - 1)


def theoretical_covariance_action(
    reconstruction: Tensor,
    probes: Tensor,
    operator: LFMOperator,
    cs_kernel: Tensor,
    *,
    covariance_scale: float | Tensor = 1.0,
    noise_variance: Tensor | None = None,
) -> Tensor:
    """Apply (kappa H Dg Cs Dg H^T + Cn) to sensor probes."""

    if reconstruction.ndim != 5 or reconstruction.shape[0] != 1:
        raise ValueError("Validation currently expects one reconstruction [1,1,Z,H,W]")
    if probes.ndim == 3:
        probes = probes[:, None]
    if probes.ndim != 4 or probes.shape[1] != 1:
        raise ValueError("Probes must have shape [K,H,W] or [K,1,H,W]")
    backprojected = operator.adjoint(probes)
    reconstruction_batch = reconstruction.expand(probes.shape[0], -1, -1, -1, -1)
    right_weighted = reconstruction_batch * backprojected
    correlated = stationary_covariance_action(right_weighted, cs_kernel)
    signal = operator(reconstruction_batch * correlated)
    scale = torch.as_tensor(covariance_scale, dtype=signal.dtype, device=signal.device)
    result = scale * signal
    if noise_variance is not None:
        typed_noise = torch.as_tensor(
            noise_variance, dtype=result.dtype, device=result.device
        )
        result = result + typed_noise * probes
    return result[:, 0]
