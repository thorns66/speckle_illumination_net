from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CutoffEstimate:
    radius_px: float
    resolution_um: float
    status: str


@dataclass(frozen=True)
class FTCCurve:
    radii_px: np.ndarray
    resolution_um: np.ndarray
    ftc: np.ndarray
    valid_fraction: np.ndarray
    matlab_cutoff: CutoffEstimate
    robust_cutoff: CutoffEstimate
    robust_ftc: np.ndarray


def peak_normalize(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Image contains NaN or Inf")
    value = np.clip(value, 0.0, None)
    maximum = float(value.max(initial=0.0))
    if maximum <= 0.0:
        raise ValueError("Image must contain positive signal")
    return value / maximum


def prepare_resolution_image(image: np.ndarray, upsample_factor: float = 2.0) -> np.ndarray:
    """Bicubic upsampling followed by the peak normalization used by MATLAB."""
    if upsample_factor <= 0:
        raise ValueError("upsample_factor must be positive")
    normalized = peak_normalize(image)
    if upsample_factor != 1.0:
        normalized = ndimage.zoom(
            normalized,
            zoom=float(upsample_factor),
            order=3,
            mode="nearest",
            prefilter=True,
        )
    return peak_normalize(normalized)


def matlab_round_positive(value: np.ndarray) -> np.ndarray:
    """Match MATLAB round for the positive coordinates used by the circle scan."""
    return np.floor(np.asarray(value) + 0.5).astype(np.int64)


def sample_quarter_circle(
    image: np.ndarray,
    radius_px: float,
    *,
    center_xy_1based: tuple[float, float],
    angular_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample the MATLAB first-quadrant arc with nearest-neighbour indexing."""
    value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError("image must be 2D")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if angular_samples < 4:
        raise ValueError("angular_samples must be at least four")
    center_x, center_y = (float(v) for v in center_xy_1based)
    theta = np.linspace(0.0, np.pi / 2.0, angular_samples, endpoint=True)
    x_1based = matlab_round_positive(radius_px * np.cos(theta) + center_x)
    y_1based = matlab_round_positive(radius_px * np.sin(theta) + center_y)
    valid = (
        (x_1based >= 1)
        & (x_1based <= value.shape[1])
        & (y_1based >= 1)
        & (y_1based <= value.shape[0])
    )
    samples = value[y_1based[valid] - 1, x_1based[valid] - 1]
    return samples, x_1based[valid], y_1based[valid], float(valid.mean())


def fourier_contrast(samples: np.ndarray, harmonic: int = 10) -> float:
    """Return 2*|FFT[harmonic]|/|FFT[DC]| (MATLAB index harmonic+1)."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) <= harmonic:
        raise ValueError("Not enough one-dimensional samples for the requested harmonic")
    spectrum = np.abs(np.fft.fft(values))
    dc = float(spectrum[0])
    if dc <= np.finfo(np.float64).eps:
        return float("nan")
    return float(2.0 * spectrum[harmonic] / dc)


def _resolution_um(radius_px: np.ndarray, pixel_size_um: float, line_pairs: int) -> np.ndarray:
    if pixel_size_um <= 0 or line_pairs <= 0:
        raise ValueError("pixel_size_um and line_pairs must be positive")
    return np.asarray(radius_px, dtype=np.float64) * pixel_size_um * 2.0 * np.pi / line_pairs


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if window == 1:
        return np.asarray(values, dtype=np.float64).copy()
    return ndimage.median_filter(np.asarray(values, dtype=np.float64), size=window, mode="nearest")


def matlab_cutoff_estimate(
    radii_px: np.ndarray,
    resolution_um: np.ndarray,
    ftc: np.ndarray,
    *,
    threshold: float,
) -> CutoffEstimate:
    """Reproduce the supplied MATLAB reverse scan, with explicit censoring states."""
    finite = np.isfinite(ftc)
    if not finite.any():
        raise ValueError("FTC curve has no finite samples")
    valid_indices = np.flatnonzero(finite)
    valid_ftc = ftc[valid_indices]
    if np.all(valid_ftc >= threshold):
        i = int(valid_indices[0])
        return CutoffEstimate(float(radii_px[i]), float(resolution_um[i]), "better_than_range")
    if np.all(valid_ftc < threshold):
        i = int(valid_indices[-1])
        return CutoffEstimate(float(radii_px[i]), float(resolution_um[i]), "worse_than_range")
    for i in valid_indices[::-1]:
        if ftc[i] < threshold:
            j = min(int(i) + 1, len(radii_px) - 1)
            return CutoffEstimate(float(radii_px[j]), float(resolution_um[j]), "crossing")
    raise AssertionError("Unreachable cutoff state")


def robust_cutoff_estimate(
    radii_px: np.ndarray,
    resolution_um: np.ndarray,
    ftc: np.ndarray,
    *,
    threshold: float,
    smoothing_window: int = 9,
    consecutive_below: int = 5,
) -> tuple[CutoffEstimate, np.ndarray]:
    """Reject isolated dips before applying the MATLAB inside/outside cutoff rule."""
    if consecutive_below < 1:
        raise ValueError("consecutive_below must be positive")
    smoothed = _median_smooth(ftc, smoothing_window)
    finite = np.isfinite(smoothed)
    if not finite.any():
        raise ValueError("Smoothed FTC curve has no finite samples")
    below = finite & (smoothed < threshold)
    above = finite & ~below
    if not below.any():
        i = int(np.flatnonzero(finite)[0])
        return (
            CutoffEstimate(float(radii_px[i]), float(resolution_um[i]), "better_than_range"),
            smoothed,
        )
    if not above.any():
        i = int(np.flatnonzero(finite)[-1])
        return (
            CutoffEstimate(float(radii_px[i]), float(resolution_um[i]), "worse_than_range"),
            smoothed,
        )
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_below in enumerate(below):
        if is_below and start is None:
            start = i
        if start is not None and (not is_below or i == len(below) - 1):
            end = i if is_below and i == len(below) - 1 else i - 1
            if end - start + 1 >= consecutive_below:
                runs.append((start, end))
            start = None
    if not runs:
        fallback = matlab_cutoff_estimate(
            radii_px, resolution_um, smoothed, threshold=threshold
        )
        return CutoffEstimate(fallback.radius_px, fallback.resolution_um, "noisy_fallback"), smoothed
    _, outermost_below = max(runs, key=lambda pair: pair[1])
    cutoff_index = min(outermost_below + 1, len(radii_px) - 1)
    status = "crossing" if cutoff_index > outermost_below else "worse_than_range"
    return (
        CutoffEstimate(
            float(radii_px[cutoff_index]),
            float(resolution_um[cutoff_index]),
            status,
        ),
        smoothed,
    )


def compute_ftc_curve(
    image: np.ndarray,
    *,
    center_xy_1based: tuple[float, float] = (22.0, 22.0),
    pixel_size_um: float = 5.2 / 8.93,
    line_pairs: int = 40,
    harmonic: int = 10,
    angular_samples: int = 1000,
    max_radius_px: int = 500,
    threshold: float = 0.1,
    smoothing_window: int = 9,
    consecutive_below: int = 5,
) -> FTCCurve:
    value = peak_normalize(image)
    center_x, center_y = center_xy_1based
    complete_radius = int(
        np.floor(min(value.shape[1] - center_x, value.shape[0] - center_y))
    )
    usable_radius = min(int(max_radius_px), complete_radius)
    if usable_radius < 1:
        raise ValueError("Center leaves no complete first-quadrant circle in the image")
    radii = np.arange(1, usable_radius + 1, dtype=np.float64)
    ftc = np.empty_like(radii)
    valid_fraction = np.empty_like(radii)
    for index, radius in enumerate(radii):
        samples, _, _, valid = sample_quarter_circle(
            value,
            radius,
            center_xy_1based=center_xy_1based,
            angular_samples=angular_samples,
        )
        ftc[index] = fourier_contrast(samples, harmonic=harmonic)
        valid_fraction[index] = valid
    resolution = _resolution_um(radii, pixel_size_um, line_pairs)
    matlab = matlab_cutoff_estimate(radii, resolution, ftc, threshold=threshold)
    robust, robust_ftc = robust_cutoff_estimate(
        radii,
        resolution,
        ftc,
        threshold=threshold,
        smoothing_window=smoothing_window,
        consecutive_below=consecutive_below,
    )
    return FTCCurve(
        radii_px=radii,
        resolution_um=resolution,
        ftc=ftc,
        valid_fraction=valid_fraction,
        matlab_cutoff=matlab,
        robust_cutoff=robust,
        robust_ftc=robust_ftc,
    )
