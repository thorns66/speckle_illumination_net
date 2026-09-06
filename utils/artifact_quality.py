from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from utils.resolution import peak_normalize, sample_quarter_circle


@dataclass(frozen=True)
class VisualArtifactMetrics:
    radial_to_tangential_gradient_energy: float
    off_harmonic_angular_energy_fraction: float
    target_harmonic_phase_coherence: float
    normalized_laplacian_energy: float
    inner_radius_px: float
    outer_radius_px: float
    evaluated_radius_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AnnularSimilarity:
    intensity_correlation: float
    affine_normalized_rmse: float
    gradient_cosine_similarity: float
    evaluated_pixel_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _annulus_mask(
    shape: tuple[int, int],
    center_xy_1based: tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if inner_radius_px <= 0 or outer_radius_px <= inner_radius_px:
        raise ValueError("Require 0 < inner_radius_px < outer_radius_px")
    center_x = float(center_xy_1based[0]) - 1.0
    center_y = float(center_xy_1based[1]) - 1.0
    yy, xx = np.indices(shape, dtype=np.float64)
    dx = xx - center_x
    dy = yy - center_y
    radius = np.hypot(dx, dy)
    mask = (
        (dx >= 0.0)
        & (dy >= 0.0)
        & (radius >= float(inner_radius_px))
        & (radius <= float(outer_radius_px))
    )
    if int(mask.sum()) < 32:
        raise ValueError("The requested first-quadrant annulus contains too few pixels")
    return mask, dx, dy, radius


def compute_visual_artifact_metrics(
    image: np.ndarray,
    *,
    center_xy_1based: tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
    harmonic: int = 10,
    angular_samples: int = 1000,
    radius_step_px: int = 2,
    harmonic_half_width: int = 1,
) -> VisualArtifactMetrics:
    """Measure radial barbs and angular energy that cannot belong to an ideal star.

    All measurements use the externally supplied center. The function never
    recenters the reconstruction, because recentering can hide geometric drift.
    """
    value = peak_normalize(image)
    if harmonic < 1 or angular_samples < 4:
        raise ValueError("harmonic and angular_samples must be positive")
    if radius_step_px < 1 or harmonic_half_width < 0:
        raise ValueError("radius_step_px must be positive and harmonic_half_width nonnegative")
    mask, dx, dy, radius = _annulus_mask(
        value.shape,
        center_xy_1based,
        inner_radius_px,
        outer_radius_px,
    )

    gradient_y, gradient_x = np.gradient(value)
    safe_radius = np.maximum(radius, np.finfo(np.float64).eps)
    radial_gradient = gradient_x * dx / safe_radius + gradient_y * dy / safe_radius
    tangential_gradient = -gradient_x * dy / safe_radius + gradient_y * dx / safe_radius
    radial_energy = float(np.mean(np.square(radial_gradient[mask])))
    tangential_energy = float(np.mean(np.square(tangential_gradient[mask])))
    epsilon = np.finfo(np.float64).eps
    radial_ratio = radial_energy / max(tangential_energy, epsilon)

    laplacian = ndimage.laplace(value, mode="nearest")
    total_gradient_energy = float(
        np.mean(np.square(gradient_x[mask]) + np.square(gradient_y[mask]))
    )
    normalized_laplacian = float(np.mean(np.square(laplacian[mask]))) / max(
        total_gradient_energy, epsilon
    )

    radii = np.arange(
        int(np.ceil(inner_radius_px)),
        int(np.floor(outer_radius_px)) + 1,
        int(radius_step_px),
        dtype=np.float64,
    )
    off_harmonic_fractions: list[float] = []
    target_coefficients: list[complex] = []
    for radius_px in radii:
        samples, _, _, valid_fraction = sample_quarter_circle(
            value,
            float(radius_px),
            center_xy_1based=center_xy_1based,
            angular_samples=angular_samples,
        )
        if valid_fraction < 1.0 or len(samples) <= harmonic:
            continue
        centered = np.asarray(samples, dtype=np.float64) - float(np.mean(samples))
        spectrum = np.fft.rfft(centered)
        power = np.square(np.abs(spectrum))
        if len(power) <= 1 or float(power[1:].sum()) <= epsilon:
            continue
        allowed = np.zeros_like(power, dtype=bool)
        target = harmonic
        while target < len(power):
            lower = max(1, target - harmonic_half_width)
            upper = min(len(power), target + harmonic_half_width + 1)
            allowed[lower:upper] = True
            target += 2 * harmonic
        off_harmonic_fractions.append(
            float(power[1:][~allowed[1:]].sum() / power[1:].sum())
        )
        target_coefficients.append(complex(spectrum[harmonic]))

    if len(target_coefficients) < 2:
        raise ValueError("No complete angular profiles were available for artifact metrics")
    coefficients = np.asarray(target_coefficients, dtype=np.complex128)
    weights = np.maximum(np.abs(coefficients), epsilon)
    phase_coherence = float(abs(np.sum(coefficients) / np.sum(weights)))
    return VisualArtifactMetrics(
        radial_to_tangential_gradient_energy=float(radial_ratio),
        off_harmonic_angular_energy_fraction=float(
            np.median(np.asarray(off_harmonic_fractions, dtype=np.float64))
        ),
        target_harmonic_phase_coherence=phase_coherence,
        normalized_laplacian_energy=normalized_laplacian,
        inner_radius_px=float(inner_radius_px),
        outer_radius_px=float(outer_radius_px),
        evaluated_radius_count=len(target_coefficients),
    )


def compare_annular_structure(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    center_xy_1based: tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
) -> AnnularSimilarity:
    reference_value = peak_normalize(reference)
    reconstruction_value = peak_normalize(reconstruction)
    if reference_value.shape != reconstruction_value.shape:
        raise ValueError("reference and reconstruction must have the same shape")
    mask, _, _, _ = _annulus_mask(
        reference_value.shape,
        center_xy_1based,
        inner_radius_px,
        outer_radius_px,
    )
    reference_vector = reference_value[mask]
    reconstruction_vector = reconstruction_value[mask]
    design = np.stack(
        [reconstruction_vector, np.ones_like(reconstruction_vector)], axis=1
    )
    scale, offset = np.linalg.lstsq(design, reference_vector, rcond=None)[0]
    fitted = scale * reconstruction_vector + offset
    reference_std = max(float(np.std(reference_vector)), np.finfo(np.float64).eps)
    normalized_rmse = float(np.sqrt(np.mean(np.square(fitted - reference_vector)))) / reference_std
    correlation = float(np.corrcoef(reference_vector, reconstruction_vector)[0, 1])

    reference_gradient_y, reference_gradient_x = np.gradient(reference_value)
    reconstruction_gradient_y, reconstruction_gradient_x = np.gradient(reconstruction_value)
    reference_gradients = np.concatenate(
        [reference_gradient_x[mask], reference_gradient_y[mask]]
    )
    reconstruction_gradients = np.concatenate(
        [reconstruction_gradient_x[mask], reconstruction_gradient_y[mask]]
    )
    denominator = float(
        np.linalg.norm(reference_gradients) * np.linalg.norm(reconstruction_gradients)
    )
    gradient_cosine = float(
        np.dot(reference_gradients, reconstruction_gradients)
        / max(denominator, np.finfo(np.float64).eps)
    )
    return AnnularSimilarity(
        intensity_correlation=correlation,
        affine_normalized_rmse=normalized_rmse,
        gradient_cosine_similarity=gradient_cosine,
        evaluated_pixel_count=int(mask.sum()),
    )
