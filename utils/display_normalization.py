"""Display-only normalization; never apply this to quantitative evaluation."""

import numpy as np


def normalize_display_volume(volume: np.ndarray) -> tuple[np.ndarray, float]:
    """Map an entire nonnegative ZYX volume to [0, 1] with one scalar divisor."""
    value = np.asarray(volume, dtype=np.float64)
    if value.ndim != 3 or not value.size:
        raise ValueError("Display input must be a nonempty ZYX volume")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("Display input must be finite and nonnegative")
    maximum = float(value.max())
    return (value / maximum if maximum > 0 else value.copy()), maximum
