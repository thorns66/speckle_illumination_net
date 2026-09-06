from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

import tools.validate_covariance_probe_reliability as stage_a


def _make_extended_probe_families(
    count: int,
    shape: tuple[int, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    height, width = shape
    raw = np.random.default_rng(seed).choice(
        (-1.0, 1.0), size=(count, height, width)
    ).astype(np.float32)
    families: dict[str, np.ndarray] = {}
    for sigma in (8.0, 16.0):
        blurred = np.stack(
            [gaussian_filter(probe, sigma=sigma, mode="reflect") for probe in raw]
        )
        blurred -= blurred.mean(axis=(1, 2), keepdims=True)
        families[f"lowpass_sigma{sigma:g}"] = stage_a._unit_norm(blurred)

    y = np.arange(height, dtype=np.float64) + 0.5
    x = np.arange(width, dtype=np.float64) + 0.5
    frequency_pairs = sorted(
        ((fy, fx) for fy in range(8) for fx in range(8)),
        key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, pair[0], pair[1]),
    )[:count]
    dct = np.stack(
        [
            np.cos(np.pi * fy * y[:, None] / height)
            * np.cos(np.pi * fx * x[None, :] / width)
            for fy, fx in frequency_pairs
        ]
    )
    families["fixed_low_frequency_dct"] = stage_a._unit_norm(dct)
    return families


if __name__ == "__main__":
    stage_a._make_probe_families = _make_extended_probe_families
    stage_a.main()
