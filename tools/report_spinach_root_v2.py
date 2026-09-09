"""Round-off-safe wrapper for the spinach-root comparison report."""
from __future__ import annotations

import numpy as np

from tools import report_spinach_root as base


def shape_rmse(pred, target, log=False):
    p = pred.astype(np.float64) / max(float(pred.mean()), 1e-30)
    t = target.astype(np.float64) / max(float(target.mean()), 1e-30)
    if log:
        # FFT projection leaves harmless negative round-off values below 3e-9.
        # Physical intensity and variance are non-negative.
        p = np.log(np.maximum(p, 0) + 1e-6)
        t = np.log(np.maximum(t, 0) + 1e-6)
    return float(np.sqrt(np.mean((p - t) ** 2)))


base.shape_rmse = shape_rmse
report = base.report

