import numpy as np
import torch


def toy_psf(z: int = 2, period: int = 3, kh: int = 5, kw: int = 7, dtype=torch.float64):
    generator = torch.Generator().manual_seed(20260831)
    h = torch.rand((z, period, period, kh, kw), generator=generator, dtype=dtype)
    h[h < 0.35] = 0
    return h


def numpy_matlab_forward(volume: np.ndarray, h: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    z_count, period, _, _, _ = h.shape
    output = np.zeros(volume.shape[-2:], dtype=np.float64)
    for z in range(z_count):
        for aa in range(period):
            for bb in range(period):
                masked = np.zeros_like(output)
                masked[aa::period, bb::period] = volume[z, aa::period, bb::period]
                output += convolve2d(masked, h[z, aa, bb], mode="same", boundary="fill")
    return output
