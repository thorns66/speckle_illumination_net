"""Population intensity covariance of the *linear* finite random-phase source.

Not valid after realization-wise max normalization. For the real circular-pupil
transfer B and independent unit-modulus phases,
  mean(I) = (B**2) @ 1
  Cov(I) = (B @ B.T)**2 - (B**2) @ (B**2).T.
The fourth-phase-cumulant subtraction is essential; Siegert alone is not exact.
No illumination realizations, spatial stationarity, or Gaussian-field limit are
used. The dense representation is a diagnostic, not the production default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physics.speckle_oracle import SpeckleGeneratorConfig, circular_pupil


@torch.no_grad()
def finite_phase_transfer(config, *, device="cpu", dtype=torch.float64, row_chunk=256):
    side = int(config.sampling)
    if side <= 0 or side % 2 or row_chunk < 1:
        raise ValueError("Even positive side and positive chunk required for the existing padded generator")
    pupil = circular_pupil(config, device="cpu", dtype=torch.float64)
    if not torch.equal(pupil, circular_pupil(config, device="cpu", dtype=torch.float32).double()):
        raise ValueError("Float32 generator and Float64 reference disagree on pupil support")
    impulse = torch.fft.ifft2(torch.fft.ifftshift(pupil))
    if float(impulse.imag.abs().max()) > max(1e-14, float(impulse.real.abs().max())*1e-12):
        raise ValueError("This real-transfer formula does not support a complex/asymmetric pupil")
    kernel = impulse.real.to(device=device, dtype=dtype).contiguous()
    index = torch.arange(side*side, device=device)
    ys, xs = index//side, index%side
    transfer = torch.empty((side*side, side*side), device=device, dtype=dtype)
    for start in range(0, side*side, row_chunk):
        dy = (ys[start:start+row_chunk,None]-ys[None]) % (2*side)
        dx = (xs[start:start+row_chunk,None]-xs[None]) % (2*side)
        transfer[start:start+row_chunk] = kernel[dy,dx]
    return transfer


@torch.no_grad()
def dense_phase_covariance(transfer, *, row_chunk=512, consume_transfer=False, source_chunk=0):
    """Exact moment identity up to arithmetic rounding; no PSD projection.

    consume_transfer replaces B by B**2 in place, allowing the real-size build
    to fit on one A40. Callers must not reuse B as a field transfer afterwards.
    """
    if transfer.ndim != 2 or row_chunk < 1 or transfer.is_complex():
        raise ValueError("Real field transfer matrix and positive chunk required")
    if source_chunk < 0:
        raise ValueError("Source chunk cannot be negative")
    if source_chunk:
        # Long FP32 dot products lose small impulse-tail contributions. Bound
        # their length and accumulate partial products in Float64, without
        # storing another full matrix or using slow full-size FP64 GEMM.
        covariance = torch.empty((len(transfer), len(transfer)), device=transfer.device, dtype=transfer.dtype)
        def gram_rows(matrix, start):
            rows = matrix[start:start+row_chunk]
            accumulator = torch.zeros((len(rows),len(matrix)),device=matrix.device,dtype=torch.float64)
            for source in range(0,matrix.shape[1],source_chunk):
                partial = rows[:,source:source+source_chunk]@matrix[:,source:source+source_chunk].T
                accumulator.add_(partial)
            return accumulator
        for start in range(0,len(transfer),row_chunk):
            covariance[start:start+row_chunk].copy_(gram_rows(transfer,start).square_())
    else:
        covariance = transfer @ transfer.T
        covariance.square_()
    power = transfer.square_() if consume_transfer else transfer.square()
    mean = power.sum(1)
    for start in range(0, len(transfer), row_chunk):
        fourth = gram_rows(power,start) if source_chunk else power[start:start+row_chunk]@power.T
        covariance[start:start+row_chunk].sub_(fourth)
    # Do not add a white-illumination nugget: even a small one would introduce
    # information that the physical band-limited illumination does not have.
    return covariance, mean


class DenseFinitePhaseCs:
    def __init__(self, covariance, shape, *, system_mean=1.):
        count = int(np.prod(shape))
        if covariance.shape != (count,count) or len(shape) != 2 or system_mean <= 0:
            raise ValueError("Covariance/shape mismatch or invalid fixed system mean")
        self.covariance = covariance.detach()
        self.shape = tuple(shape)
        self.system_mean = float(system_mean)

    @classmethod
    def load(cls, directory, *, device, system_mean):
        directory = Path(directory)
        report = json.loads((directory/"report.json").read_text())
        if not report["complete"]:
            raise ValueError("Incomplete covariance artifact")
        # Copy-on-write mapping does not modify the stored file and avoids
        # exposing a non-writable numpy array to PyTorch.
        array = np.load(directory/"covariance_raw.npy", mmap_mode="c", allow_pickle=False)
        matrix = torch.from_numpy(array).to(device=device, copy=True)
        return cls(matrix, report["object_shape"], system_mean=system_mean)

    def action(self, value):
        if tuple(value.shape[-2:]) != self.shape:
            raise ValueError("Input does not match the calibrated object grid")
        flat = value.reshape(-1, self.covariance.shape[0])
        return ((flat@self.covariance.T)/(self.system_mean**2)).reshape_as(value)
