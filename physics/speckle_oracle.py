from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import j1
from torch import Tensor


@dataclass(frozen=True)
class SpeckleGeneratorConfig:
    sampling: int = 260
    numerical_aperture: float = 0.05
    wavelength_m: float = 488e-9
    pixel_size_m: float = 4.5e-6 / 4.0
    seed: int = 20260904

    @property
    def padded_sampling(self) -> int:
        return 2 * int(self.sampling)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def circular_pupil(
    config: SpeckleGeneratorConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    size = config.padded_sampling
    frequency = (
        torch.arange(size, device=device, dtype=dtype) - size / 2
    ) / (size * config.pixel_size_m)
    fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
    return (torch.sqrt(fx.square() + fy.square()) <= (
        config.numerical_aperture / config.wavelength_m
    )).to(dtype)


def generate_speckle_ensemble(
    count: int,
    config: SpeckleGeneratorConfig,
    *,
    device: torch.device | str,
    batch_size: int = 32,
    output_device: torch.device | str | None = None,
) -> Tensor:
    """Reproduce active lines 52--82 of generate_speckle_NA05.m.

    The RNG is an independent reproducible draw from the same distribution;
    the MATLAB script did not record a seed, so it cannot recreate the exact
    100 historical realizations.
    """

    if count < 2 or batch_size < 1:
        raise ValueError("count must be >=2 and batch_size must be positive")
    device = torch.device(device)
    output_device = device if output_device is None else torch.device(output_device)
    generator = torch.Generator(device=device).manual_seed(int(config.seed))
    pupil = circular_pupil(config, device=device)
    sampling = int(config.sampling)
    pad = sampling // 2
    batches: list[Tensor] = []
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        phase = torch.rand(
            (current, sampling, sampling),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ) * (2.0 * torch.pi)
        slm = torch.polar(torch.ones_like(phase), phase)
        padded = F.pad(slm, (pad, pad, pad, pad))
        phase_fft = torch.fft.fftshift(
            torch.fft.fft2(padded), dim=(-2, -1)
        )
        field = torch.fft.ifft2(
            torch.fft.ifftshift(phase_fft * pupil, dim=(-2, -1))
        )
        intensity = field.abs().square()[..., pad : pad + sampling, pad : pad + sampling]
        intensity = intensity / intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(intensity.dtype).eps
        )
        batches.append(intensity.to(output_device))
    return torch.cat(batches, dim=0)


def analytic_intensity_covariance(
    config: SpeckleGeneratorConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Ideal stationary jinc-squared intensity covariance for a circular pupil."""

    radius = int(config.sampling) - 1
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    distance = np.hypot(x, y) * float(config.pixel_size_m)
    argument = (
        2.0
        * np.pi
        * float(config.numerical_aperture)
        * distance
        / float(config.wavelength_m)
    )
    kernel = np.ones_like(argument)
    nonzero = argument != 0
    kernel[nonzero] = (2.0 * j1(argument[nonzero]) / argument[nonzero]) ** 2
    return torch.as_tensor(kernel, device=device, dtype=dtype)


def stationary_covariance_from_ensemble(patterns: Tensor) -> Tensor:
    """Translation-average the exact cropped, max-normalized generator ensemble."""

    if patterns.ndim != 3 or patterns.shape[0] < 2:
        raise ValueError("patterns must have shape [L,H,W], L>=2")
    count, height, width = patterns.shape
    centered = patterns - patterns.mean(dim=0, keepdim=True)
    fft_shape = (2 * height - 1, 2 * width - 1)
    spectrum = torch.fft.rfft2(centered, s=fft_shape)
    autocorrelation = torch.fft.irfft2(
        spectrum.abs().square().mean(dim=0), s=fft_shape
    )
    autocorrelation = torch.fft.fftshift(autocorrelation, dim=(-2, -1))
    lag_y = torch.arange(-(height - 1), height, device=patterns.device)
    lag_x = torch.arange(-(width - 1), width, device=patterns.device)
    overlap = (height - lag_y.abs())[:, None] * (width - lag_x.abs())[None, :]
    kernel = autocorrelation / overlap.to(autocorrelation.dtype)
    kernel = 0.5 * (kernel + torch.flip(kernel, dims=(-2, -1)))
    return kernel / kernel[height - 1, width - 1].clamp_min(
        torch.finfo(kernel.dtype).eps
    )


def stationary_covariance_action(inputs: Tensor, kernel: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Apply a PSD-projected stationary covariance without circular image wrap."""

    if inputs.ndim < 2:
        raise ValueError("inputs must end in [H,W]")
    height, width = inputs.shape[-2:]
    typed_kernel = torch.as_tensor(kernel, dtype=inputs.dtype, device=inputs.device)
    kh, kw = typed_kernel.shape
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("kernel dimensions must be odd")
    fft_shape = (height + kh - 1, width + kw - 1)
    embedded = F.pad(
        typed_kernel,
        (0, fft_shape[1] - kw, 0, fft_shape[0] - kh),
    )
    embedded = torch.roll(embedded, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    raw_spectrum = torch.fft.rfft2(embedded).real
    negative_mass = (-raw_spectrum.clamp_max(0)).sum()
    total_mass = raw_spectrum.abs().sum().clamp_min(torch.finfo(inputs.dtype).eps)
    spectrum = raw_spectrum.clamp_min(0.0)
    flattened = inputs.reshape(-1, 1, height, width)
    padded = F.pad(
        flattened,
        (0, fft_shape[1] - width, 0, fft_shape[0] - height),
    )
    result = torch.fft.irfft2(
        torch.fft.rfft2(padded) * spectrum,
        s=fft_shape,
    )[..., :height, :width]
    report = {
        "negative_spectral_mass_fraction": float((negative_mass / total_mass).item()),
        "minimum_raw_eigenvalue": float(raw_spectrum.min().item()),
    }
    return result.reshape_as(inputs), report


def ensemble_covariance_action(inputs: Tensor, patterns: Tensor) -> Tensor:
    """Apply the exact nonstationary ensemble covariance S_c^T S_c/(L-1)."""

    if inputs.shape[-2:] != patterns.shape[-2:] or patterns.ndim != 3:
        raise ValueError("patterns [L,H,W] must match the input lateral shape")
    height, width = inputs.shape[-2:]
    flat_inputs = inputs.reshape(-1, height * width)
    centered = (patterns - patterns.mean(dim=0, keepdim=True)).reshape(
        patterns.shape[0], -1
    )
    weights = centered @ flat_inputs.T
    result = (weights.T @ centered) / float(patterns.shape[0] - 1)
    return result.reshape_as(inputs)
