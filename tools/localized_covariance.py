"""Data-independent soft windows for covariance-vector diagnostic objectives.

Measures P_w C_y P_w r on each window, not P_w C_y r or only its diagonal.
No object support, target depth, estimated covariance, or truth chooses windows.
"""
from __future__ import annotations

import math
import numpy as np

from tools.scan_covariance_sketch_depth import _lowpass_probes


def localized_probe_bank(count, shape, *, seed, window_sigma, probe_sigma=0.):
    if count < 1 or window_sigma < 0 or probe_sigma < 0:
        raise ValueError("Nonnegative widths and a positive probe count required")
    probes = _lowpass_probes(count, shape, sigma=probe_sigma, seed=seed)
    if window_sigma == 0:
        return probes, np.ones_like(probes)
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError("Windowed bank requires a square number of probes for uniform grid coverage")
    height, width = shape
    ys = (np.arange(side)+.5)*height/side-.5
    xs = (np.arange(side)+.5)*width/side-.5
    yy, xx = np.mgrid[:height, :width]
    windows = np.stack([np.exp(-((yy-y)**2+(xx-x)**2)/(2*window_sigma**2))
                        for y in ys for x in xs]).astype(np.float32)
    windows /= windows.max(axis=(1,2), keepdims=True)
    # Keep the same zero-DC convention as the global baseline, but enforce it
    # after localization. Both input and output use the fixed same window.
    offset = (probes*windows).sum(axis=(1,2), keepdims=True)/windows.sum(axis=(1,2), keepdims=True)
    probes = (probes-offset)*windows
    norms = np.linalg.norm(probes.reshape(count, -1), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Degenerate localized probes")
    probes /= norms[:, None, None]
    return probes, windows
