from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.resolution import peak_normalize, sample_quarter_circle


@dataclass(frozen=True)
class CenterCalibration:
    center_xy_1based: tuple[float, float]
    score: float
    median_ftc: float
    phase_coherence: float
    harmonic_purity: float
    calibration_radii_px: np.ndarray
    search_bounds_xy_1based: tuple[float, float, float, float]


def harmonic_profile_metrics(
    image: np.ndarray,
    *,
    center_xy_1based: tuple[float, float],
    radii_px: np.ndarray,
    harmonic: int = 10,
    angular_samples: int = 1000,
    purity_half_width: int = 2,
) -> tuple[float, float, float]:
    """Return target strength, phase coherence, and local harmonic purity."""
    value = peak_normalize(image)
    radii = np.asarray(radii_px, dtype=np.float64)
    if radii.ndim != 1 or len(radii) < 2 or np.any(radii <= 0):
        raise ValueError("radii_px must contain at least two positive radii")
    if harmonic < 1 or purity_half_width < 1:
        raise ValueError("harmonic and purity_half_width must be positive")

    amplitudes: list[float] = []
    coefficients: list[complex] = []
    purities: list[float] = []
    epsilon = np.finfo(np.float64).eps
    for radius in radii:
        samples, _, _, valid_fraction = sample_quarter_circle(
            value,
            float(radius),
            center_xy_1based=center_xy_1based,
            angular_samples=angular_samples,
        )
        if valid_fraction < 1.0 or len(samples) <= harmonic + purity_half_width:
            raise ValueError("Calibration requires complete arcs with enough samples")
        spectrum = np.fft.fft(np.asarray(samples, dtype=np.float64))
        dc = float(abs(spectrum[0]))
        if dc <= epsilon:
            continue
        coefficient = 2.0 * spectrum[harmonic] / dc
        lo = max(1, harmonic - purity_half_width)
        hi = harmonic + purity_half_width + 1
        local_energy = float(np.abs(spectrum[lo:hi]).sum())
        amplitudes.append(float(abs(coefficient)))
        coefficients.append(complex(coefficient))
        purities.append(float(abs(spectrum[harmonic]) / (local_energy + epsilon)))

    if len(coefficients) < 2:
        raise ValueError("No usable center-calibration profiles")
    coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    weights = np.maximum(np.abs(coefficient_array), epsilon)
    unit_phase = coefficient_array / weights
    phase_coherence = float(abs(np.sum(weights * unit_phase) / np.sum(weights)))
    return (
        float(np.median(amplitudes)),
        phase_coherence,
        float(np.median(purities)),
    )


def calibrate_star_center(
    reference_image: np.ndarray,
    *,
    initial_center_xy_1based: tuple[float, float] = (22.0, 22.0),
    search_radius_px: int = 16,
    harmonic: int = 10,
    angular_samples: int = 1000,
    calibration_radii_px: np.ndarray | None = None,
    max_radius_px: int = 500,
) -> CenterCalibration:
    """Fit the star center on resolved annuli of one reference image."""
    value = peak_normalize(reference_image)
    if search_radius_px < 0:
        raise ValueError("search_radius_px must be non-negative")
    initial_x, initial_y = (float(v) for v in initial_center_xy_1based)
    x_min = max(1, int(np.floor(initial_x - search_radius_px)))
    x_max = min(value.shape[1], int(np.ceil(initial_x + search_radius_px)))
    y_min = max(1, int(np.floor(initial_y - search_radius_px)))
    y_max = min(value.shape[0], int(np.ceil(initial_y + search_radius_px)))

    if calibration_radii_px is None:
        complete_at_initial = int(
            np.floor(min(value.shape[1] - initial_x, value.shape[0] - initial_y))
        )
        usable = min(int(max_radius_px), complete_at_initial)
        calibration_radii_px = np.rint(
            np.linspace(0.28 * usable, 0.68 * usable, 6)
        ).astype(np.float64)
    radii = np.unique(np.asarray(calibration_radii_px, dtype=np.float64))

    best: tuple[float, float, float, float, float, float, float] | None = None
    for center_y in range(y_min, y_max + 1):
        for center_x in range(x_min, x_max + 1):
            if radii.max(initial=0.0) > min(
                value.shape[1] - center_x, value.shape[0] - center_y
            ):
                continue
            median_ftc, coherence, purity = harmonic_profile_metrics(
                value,
                center_xy_1based=(float(center_x), float(center_y)),
                radii_px=radii,
                harmonic=harmonic,
                angular_samples=angular_samples,
            )
            score = median_ftc * coherence * purity
            distance = (center_x - initial_x) ** 2 + (center_y - initial_y) ** 2
            candidate = (
                score,
                -distance,
                float(center_x),
                float(center_y),
                median_ftc,
                coherence,
                purity,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError("No center candidate supports all calibration radii")
    score, _, center_x, center_y, median_ftc, coherence, purity = best
    return CenterCalibration(
        center_xy_1based=(center_x, center_y),
        score=float(score),
        median_ftc=float(median_ftc),
        phase_coherence=float(coherence),
        harmonic_purity=float(purity),
        calibration_radii_px=radii,
        search_bounds_xy_1based=(float(x_min), float(x_max), float(y_min), float(y_max)),
    )
