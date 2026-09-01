from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

OperatorMode = Literal["reference", "optimized"]


def _matlab_conv2_same(inputs: Tensor, kernels: Tensor, groups: int = 1) -> Tensor:
    """MATLAB conv2(..., 'same') using PyTorch cross-correlation.

    inputs is [B,C,H,W]. kernels is [C_out,C/groups,Kh,Kw]. Building the
    full convolution and centrally cropping also fixes MATLAB's even-kernel
    alignment instead of relying on framework-specific ``padding='same'``.
    """

    kh, kw = kernels.shape[-2:]
    full = F.conv2d(
        inputs,
        torch.flip(kernels, dims=(-2, -1)),
        padding=(kh - 1, kw - 1),
        groups=groups,
    )
    row0, col0 = kh // 2, kw // 2
    return full[..., row0 : row0 + inputs.shape[-2], col0 : col0 + inputs.shape[-1]]


def _fft_conv2_same(inputs: Tensor, kernels: Tensor) -> Tensor:
    """Batched equivalent of the original MATLAB ``conv2FFT`` function."""

    if inputs.ndim != 4 or kernels.ndim != 3:
        raise ValueError("Expected inputs [B,C,H,W] and kernels [C,Kh,Kw]")
    if inputs.shape[1] != kernels.shape[0]:
        raise ValueError("Input channels and kernel count must match")
    height, width = inputs.shape[-2:]
    kh, kw = kernels.shape[-2:]
    def matlab_fft_extent(image_size: int, kernel_size: int) -> int:
        needed = image_size + kernel_size // 2
        power_of_two = 2 ** math.ceil(math.log2(needed))
        block_128 = 128 * math.ceil(needed / 128)
        return min(power_of_two, block_128)

    fft_shape = (
        matlab_fft_extent(height, kh),
        matlab_fft_extent(width, kw),
    )
    input_fft = torch.fft.rfft2(inputs, s=fft_shape)
    padded_kernels = F.pad(
        kernels,
        (0, fft_shape[1] - kw, 0, fft_shape[0] - kh),
    )
    centered_kernels = torch.roll(
        padded_kernels, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1)
    )
    kernel_fft = torch.fft.rfft2(centered_kernels)
    result = torch.fft.irfft2(input_fft * kernel_fft.unsqueeze(0), s=fft_shape)
    return result[..., :height, :width]


class LFMOperator(nn.Module):
    """Periodic shift-variant LFM forward model.

    The PSF tensor is never interpreted as a shift-invariant 3D kernel. Its
    canonical shape is [Z, phase_row, phase_col, kernel_row, kernel_col].
    MATLAB phase ``aa`` maps exactly to Python ``aa - 1 :: Nnum``.
    """

    def __init__(
        self,
        H: Tensor,
        Ht: Tensor | None = None,
        *,
        mode: OperatorMode = "optimized",
        phase_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        h = torch.as_tensor(H)
        if h.ndim != 5:
            raise ValueError(f"H must have shape [Z,P,P,Kh,Kw], got {tuple(h.shape)}")
        if h.shape[1] != h.shape[2]:
            raise ValueError(f"H phase dimensions must be square, got {tuple(h.shape)}")
        if not h.is_floating_point():
            h = h.float()
        self.register_buffer("H", h.contiguous(), persistent=False)
        if Ht is not None:
            ht = torch.as_tensor(Ht, dtype=h.dtype, device=h.device)
            if ht.shape != h.shape:
                raise ValueError(f"Ht shape {tuple(ht.shape)} does not match H {tuple(h.shape)}")
            self.register_buffer("Ht", ht.contiguous(), persistent=False)
        else:
            self.register_buffer("Ht", None, persistent=False)
        if mode not in ("reference", "optimized"):
            raise ValueError(f"Unsupported operator mode: {mode}")
        if phase_chunk_size < 1:
            raise ValueError("phase_chunk_size must be positive")
        self.mode = mode
        self.phase_chunk_size = int(phase_chunk_size)

    @property
    def num_depths(self) -> int:
        return int(self.H.shape[0])

    @property
    def phase_period(self) -> int:
        return int(self.H.shape[1])

    @property
    def kernel_shape(self) -> tuple[int, int]:
        return int(self.H.shape[-2]), int(self.H.shape[-1])

    def _validate_volume(self, volume: Tensor) -> Tensor:
        if volume.ndim == 4:
            volume = volume.unsqueeze(1)
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(f"Volume must be [B,1,Z,H,W] or [B,Z,H,W], got {tuple(volume.shape)}")
        if volume.shape[2] != self.num_depths:
            raise ValueError(f"Volume has Z={volume.shape[2]}, PSF has Z={self.num_depths}")
        if not volume.is_floating_point():
            raise TypeError("Volume must be floating point")
        return volume

    def _validate_sensor(self, sensor: Tensor) -> Tensor:
        if sensor.ndim == 3:
            sensor = sensor.unsqueeze(1)
        if sensor.ndim != 4 or sensor.shape[1] != 1:
            raise ValueError(f"Sensor image must be [B,1,H,W] or [B,H,W], got {tuple(sensor.shape)}")
        return sensor

    def _phase_mask(self, flat_phases: Tensor, height: int, width: int, dtype: torch.dtype) -> Tensor:
        period = self.phase_period
        aa = torch.div(flat_phases, period, rounding_mode="floor")
        bb = torch.remainder(flat_phases, period)
        rows = torch.arange(height, device=flat_phases.device)
        cols = torch.arange(width, device=flat_phases.device)
        row_mask = torch.remainder(rows[None, :], period) == aa[:, None]
        col_mask = torch.remainder(cols[None, :], period) == bb[:, None]
        return (row_mask[:, :, None] & col_mask[:, None, :]).to(dtype=dtype)

    def forward(self, volume: Tensor) -> Tensor:
        return self._project(volume, squared=False)

    def forward_squared(self, volume_squared: Tensor) -> Tensor:
        """Apply H**2 to an already-squared object volume."""

        return self._project(volume_squared, squared=True)

    def _project(self, volume: Tensor, *, squared: bool) -> Tensor:
        volume = self._validate_volume(volume)
        if volume.device != self.H.device:
            raise ValueError(f"Volume is on {volume.device}, PSF is on {self.H.device}")
        if volume.dtype != self.H.dtype:
            raise ValueError(f"Volume dtype {volume.dtype} must match PSF dtype {self.H.dtype}")
        if self.mode == "reference":
            return self._project_reference(volume, squared=squared)
        return self._project_optimized(volume, squared=squared)

    def _project_reference(self, volume: Tensor, *, squared: bool) -> Tensor:
        batch, _, _, height, width = volume.shape
        projection = volume.new_zeros((batch, 1, height, width))
        period = self.phase_period
        for z in range(self.num_depths):
            for aa in range(period):
                for bb in range(period):
                    phase_volume = volume.new_zeros((batch, 1, height, width))
                    source = volume[:, :, z, aa::period, bb::period]
                    mask = phase_volume.clone()
                    mask[:, :, aa::period, bb::period] = source
                    kernel = self.H[z, aa, bb]
                    if squared:
                        kernel = kernel.square()
                    projection = projection + _matlab_conv2_same(mask, kernel[None, None])
        return projection

    def _project_optimized(self, volume: Tensor, *, squared: bool) -> Tensor:
        batch, _, _, height, width = volume.shape
        projection = volume.new_zeros((batch, 1, height, width))
        phase_count = self.phase_period**2
        all_phases = torch.arange(phase_count, device=volume.device)
        kernels = self.H.reshape(self.num_depths, phase_count, *self.kernel_shape)
        for start in range(0, phase_count, self.phase_chunk_size):
            phase_ids = all_phases[start : start + self.phase_chunk_size]
            masks = self._phase_mask(phase_ids, height, width, volume.dtype)
            for z in range(self.num_depths):
                phase_kernels = kernels[z, phase_ids]
                if squared:
                    phase_kernels = phase_kernels.square()
                image = volume[:, :, z]
                phase_images = image * masks.unsqueeze(0)
                convolved = _fft_conv2_same(phase_images, phase_kernels)
                projection = projection + convolved.sum(dim=1, keepdim=True)
        return projection

    def adjoint(self, sensor: Tensor, *, mode: OperatorMode | None = None) -> Tensor:
        """Exact adjoint of ``forward`` derived from H, not a learned backward."""

        sensor = self._validate_sensor(sensor)
        if sensor.device != self.H.device or sensor.dtype != self.H.dtype:
            raise ValueError("Sensor device and dtype must match the PSF")
        selected_mode = mode or self.mode
        if selected_mode == "reference":
            return self._adjoint_reference(sensor)
        if selected_mode != "optimized":
            raise ValueError(f"Unsupported operator mode: {selected_mode}")
        return self._adjoint_optimized(sensor)

    def _adjoint_reference(self, sensor: Tensor) -> Tensor:
        batch, _, height, width = sensor.shape
        result = sensor.new_zeros((batch, 1, self.num_depths, height, width))
        period = self.phase_period
        for z in range(self.num_depths):
            for aa in range(period):
                for bb in range(period):
                    rotated = torch.flip(self.H[z, aa, bb], dims=(-2, -1))
                    back = _matlab_conv2_same(sensor, rotated[None, None])
                    result[:, :, z, aa::period, bb::period] = back[:, :, aa::period, bb::period]
        return result

    def _adjoint_optimized(self, sensor: Tensor) -> Tensor:
        batch, _, height, width = sensor.shape
        result = sensor.new_zeros((batch, 1, self.num_depths, height, width))
        phase_count = self.phase_period**2
        all_phases = torch.arange(phase_count, device=sensor.device)
        kernels = torch.flip(
            self.H.reshape(self.num_depths, phase_count, *self.kernel_shape), dims=(-2, -1)
        )
        for start in range(0, phase_count, self.phase_chunk_size):
            phase_ids = all_phases[start : start + self.phase_chunk_size]
            masks = self._phase_mask(phase_ids, height, width, sensor.dtype)
            for z in range(self.num_depths):
                repeated_sensor = sensor.expand(-1, len(phase_ids), -1, -1)
                convolved = _fft_conv2_same(repeated_sensor, kernels[z, phase_ids])
                result[:, :, z] = result[:, :, z] + (
                    convolved * masks.unsqueeze(0)
                ).sum(dim=1, keepdim=True)
        return result

    def backward_with_matlab_ht(self, sensor: Tensor, *, mode: OperatorMode | None = None) -> Tensor:
        """Reproduce backwardProjectGPU(Ht, sensor) for exported comparisons."""

        if self.Ht is None:
            raise RuntimeError("No Ht was loaded")
        sensor = self._validate_sensor(sensor)
        selected_mode = mode or self.mode
        if selected_mode == "optimized":
            return self._backward_matlab_ht_optimized(sensor)
        if selected_mode != "reference":
            raise ValueError(f"Unsupported operator mode: {selected_mode}")
        batch, _, height, width = sensor.shape
        result = sensor.new_zeros((batch, 1, self.num_depths, height, width))
        period = self.phase_period
        for z in range(self.num_depths):
            depth_back = sensor.new_zeros((batch, 1, height, width))
            for aa in range(period):
                for bb in range(period):
                    masked = sensor.new_zeros(sensor.shape)
                    masked[:, :, aa::period, bb::period] = sensor[:, :, aa::period, bb::period]
                    depth_back = depth_back + _matlab_conv2_same(
                        masked, self.Ht[z, aa, bb][None, None]
                    )
            result[:, :, z] = depth_back
        return result

    def _backward_matlab_ht_optimized(self, sensor: Tensor) -> Tensor:
        assert self.Ht is not None
        batch, _, height, width = sensor.shape
        result = sensor.new_zeros((batch, 1, self.num_depths, height, width))
        phase_count = self.phase_period**2
        all_phases = torch.arange(phase_count, device=sensor.device)
        kernels = self.Ht.reshape(self.num_depths, phase_count, *self.kernel_shape)
        for start in range(0, phase_count, self.phase_chunk_size):
            phase_ids = all_phases[start : start + self.phase_chunk_size]
            masks = self._phase_mask(phase_ids, height, width, sensor.dtype)
            phase_sensors = sensor * masks.unsqueeze(0)
            for z in range(self.num_depths):
                convolved = _fft_conv2_same(phase_sensors, kernels[z, phase_ids])
                result[:, :, z] = result[:, :, z] + convolved.sum(dim=1, keepdim=True)
        return result
