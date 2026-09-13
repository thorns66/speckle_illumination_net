"""Exact mixed-resolution LFM operators for the V5 simulation/real experiment.

The validated 260x260 simulation path keeps the historical sparse matrix.  The
1029x1421 real path evaluates the same periodic PSF with FFT convolution.  Its
custom autograd function stores only the input shape because both H and H**2 are
linear maps; backward is the exact transpose projection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from physics.lfm_operator import _fft_conv2_same


class _SparseProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, owner: "SparseOnlyLFM", squared: bool) -> Tensor:
        ctx.owner = owner
        ctx.squared = bool(squared)
        ctx.shape = tuple(value.shape)
        matrix = owner._a2 if squared else owner._a
        return torch.sparse.mm(matrix, value.flatten(1).T.contiguous()).T.reshape(
            value.shape[0], 1, owner.height, owner.width
        )

    @staticmethod
    def backward(ctx: Any, gradient: Tensor):
        matrix = ctx.owner._at2 if ctx.squared else ctx.owner._at
        value = torch.sparse.mm(matrix, gradient.flatten(1).T.contiguous()).T.reshape(
            ctx.shape
        )
        return value, None, None


class SparseOnlyLFM(nn.Module):
    """Historical exact sparse operator without retaining the 3.7-GiB PSF tensor."""

    def __init__(self, cache: str | Path, device: torch.device) -> None:
        super().__init__()
        import json

        cache = Path(cache)
        spec = json.loads((cache / "complete.json").read_text(encoding="utf-8"))
        self.height = int(spec["height"])
        self.width = int(spec["width"])
        self.num_depths = int(spec["H_shape"][0])
        self.phase_period = int(spec["H_shape"][1])
        self.phase_chunk_size = 0
        self.mode = "exact_sparse_without_psf_copy"
        self.register_buffer("H", torch.empty((), dtype=torch.float32, device=device), persistent=False)
        rows = self.height * self.width
        columns = rows * self.num_depths
        for name, prefix, shape in (
            ("_a", "forward", (rows, columns)),
            ("_at", "transpose", (columns, rows)),
        ):
            arrays = {
                field: torch.as_tensor(
                    np.load(cache / f"{prefix}_{field}.npy", mmap_mode="c"), device=device
                )
                for field in ("indptr", "indices", "data")
            }
            matrix = torch.sparse_csr_tensor(
                arrays["indptr"], arrays["indices"], arrays["data"], size=shape, device=device
            )
            squared = torch.sparse_csr_tensor(
                arrays["indptr"], arrays["indices"], arrays["data"].square(),
                size=shape, device=device,
            )
            self.register_buffer(name, matrix, persistent=False)
            self.register_buffer(name + "2", squared, persistent=False)

    def _validate_volume(self, value: Tensor) -> Tensor:
        if value.ndim == 4:
            value = value.unsqueeze(1)
        expected = (self.num_depths, self.height, self.width)
        if value.ndim != 5 or value.shape[1] != 1 or tuple(value.shape[-3:]) != expected:
            raise ValueError(f"Sparse simulation volume must end in {expected}, got {tuple(value.shape)}")
        return value

    def forward(self, value: Tensor) -> Tensor:
        return _SparseProject.apply(self._validate_volume(value), self, False)

    def forward_squared(self, value: Tensor) -> Tensor:
        return _SparseProject.apply(self._validate_volume(value), self, True)

    def adjoint(self, sensor: Tensor, *, squared: bool = False) -> Tensor:
        matrix = self._at2 if squared else self._at
        return torch.sparse.mm(matrix, sensor.flatten(1).T.contiguous()).T.reshape(
            sensor.shape[0], 1, self.num_depths, self.height, self.width
        )


def _tensor_identity(value: Tensor) -> tuple[Any, ...] | None:
    # Tensors created inside torch.inference_mode() intentionally have no version
    # counter.  The detached-forward cache is only an optimization for the later
    # gradient-enabled H(q), so inference tensors can safely bypass it.
    try:
        version = value._version
    except RuntimeError as exception:
        if "do not track version counter" not in str(exception):
            raise
        return None
    return (
        value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape),
        tuple(value.stride()), value.device, value.dtype, version,
    )


class _FullProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, owner: "MixedResolutionLFM", squared: bool) -> Tensor:
        ctx.owner = owner
        ctx.squared = bool(squared)
        ctx.shape = tuple(value.shape)
        return owner._full_project_plain(value, squared=bool(squared))

    @staticmethod
    def backward(ctx: Any, gradient: Tensor):
        return ctx.owner._full_adjoint_plain(gradient, squared=ctx.squared), None, None


class _ReuseLinearForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, value: Tensor, cached: Tensor, owner: "MixedResolutionLFM", squared: bool
    ) -> Tensor:
        ctx.owner = owner
        ctx.squared = bool(squared)
        ctx.input_shape = tuple(value.shape)
        return cached

    @staticmethod
    def backward(ctx: Any, gradient: Tensor):
        return ctx.owner.adjoint(gradient, squared=ctx.squared), None, None, None


class MixedResolutionLFM(nn.Module):
    """Dispatch exact physics by spatial shape and reuse the detached E3 H(q)."""

    def __init__(
        self,
        sparse_cache: str | Path,
        device: torch.device,
        *,
        full_h: Tensor | None,
        phase_chunk_size: int = 32,
        real_shape: tuple[int, int] = (1029, 1421),
        load_sparse_simulation: bool = True,
    ) -> None:
        super().__init__()
        if phase_chunk_size < 1:
            raise ValueError("phase_chunk_size must be positive")
        self.simulation = SparseOnlyLFM(sparse_cache, device) if load_sparse_simulation else None
        self.simulation_shape = (260, 260)
        if full_h is None:
            if self.simulation is None:
                raise ValueError("At least the sparse simulation operator or full PSF must be loaded")
            self.register_buffer("full_h", None, persistent=False)
            self.register_buffer("H", self.simulation.H, persistent=False)
            self.num_depths = self.simulation.num_depths
            self.phase_period = self.simulation.phase_period
        else:
            if full_h.ndim != 5 or full_h.shape[1] != full_h.shape[2]:
                raise ValueError(f"Unexpected selected PSF shape: {tuple(full_h.shape)}")
            self.register_buffer("full_h", full_h.contiguous(), persistent=False)
            self.register_buffer("H", self.full_h, persistent=False)
            self.num_depths = int(full_h.shape[0])
            self.phase_period = int(full_h.shape[1])
        self.phase_chunk_size = int(phase_chunk_size)
        self.real_shape = tuple(int(v) for v in real_shape)
        self.mode = "mixed_exact_sparse260_recomputed_fft_fullfield"
        self._linear_cache: tuple[tuple[Any, ...], Tensor] | None = None

    @property
    def kernel_shape(self) -> tuple[int, int]:
        if self.full_h is None:
            raise RuntimeError("This rank has no full-field PSF")
        return int(self.full_h.shape[-2]), int(self.full_h.shape[-1])

    def _is_simulation(self, value: Tensor) -> bool:
        return tuple(value.shape[-2:]) == self.simulation_shape

    def _validate_full_volume(self, value: Tensor) -> Tensor:
        if value.ndim == 4:
            value = value.unsqueeze(1)
        if self.full_h is None:
            raise RuntimeError("A real full-field sample reached a rank without the full PSF")
        if value.ndim != 5 or value.shape[1] != 1:
            raise ValueError(f"Expected [B,1,Z,H,W], got {tuple(value.shape)}")
        if int(value.shape[2]) != self.num_depths or tuple(value.shape[-2:]) not in {
            self.real_shape, self.simulation_shape
        }:
            raise ValueError(
                f"FFT volume must use Z={self.num_depths} and shape {self.real_shape} or "
                f"{self.simulation_shape}, got {tuple(value.shape)}"
            )
        if value.dtype != self.full_h.dtype or value.device != self.full_h.device:
            raise ValueError("Real volume dtype/device must match the PSF")
        return value

    def _phase_mask(self, ids: Tensor, height: int, width: int, dtype: torch.dtype) -> Tensor:
        aa = torch.div(ids, self.phase_period, rounding_mode="floor")
        bb = torch.remainder(ids, self.phase_period)
        rows = torch.arange(height, device=ids.device)
        cols = torch.arange(width, device=ids.device)
        row_mask = torch.remainder(rows[None], self.phase_period) == aa[:, None]
        col_mask = torch.remainder(cols[None], self.phase_period) == bb[:, None]
        return (row_mask[:, :, None] & col_mask[:, None, :]).to(dtype=dtype)

    @torch.no_grad()
    def _full_project_plain(self, value: Tensor, *, squared: bool) -> Tensor:
        value = self._validate_full_volume(value)
        assert self.full_h is not None
        batch, _, _, height, width = value.shape
        result = value.new_zeros((batch, 1, height, width))
        phase_count = self.phase_period**2
        ids = torch.arange(phase_count, device=value.device)
        kernels = self.full_h.reshape(self.num_depths, phase_count, *self.kernel_shape)
        for start in range(0, phase_count, self.phase_chunk_size):
            phase_ids = ids[start : start + self.phase_chunk_size]
            masks = self._phase_mask(phase_ids, height, width, value.dtype)
            for z in range(self.num_depths):
                selected = kernels[z, phase_ids]
                if squared:
                    selected = selected.square()
                phase_images = value[:, :, z] * masks.unsqueeze(0)
                result.add_(_fft_conv2_same(phase_images, selected).sum(1, keepdim=True))
        return result

    @torch.no_grad()
    def _full_adjoint_plain(self, sensor: Tensor, *, squared: bool) -> Tensor:
        if self.full_h is None:
            raise RuntimeError("This rank has no full-field PSF")
        if sensor.ndim == 3:
            sensor = sensor.unsqueeze(1)
        if sensor.ndim != 4 or sensor.shape[1] != 1 or tuple(sensor.shape[-2:]) not in {
            self.real_shape, self.simulation_shape
        }:
            raise ValueError(f"Unexpected FFT sensor shape: {tuple(sensor.shape)}")
        batch, _, height, width = sensor.shape
        result = sensor.new_zeros((batch, 1, self.num_depths, height, width))
        phase_count = self.phase_period**2
        ids = torch.arange(phase_count, device=sensor.device)
        kernels = torch.flip(
            self.full_h.reshape(self.num_depths, phase_count, *self.kernel_shape), dims=(-2, -1)
        )
        for start in range(0, phase_count, self.phase_chunk_size):
            phase_ids = ids[start : start + self.phase_chunk_size]
            masks = self._phase_mask(phase_ids, height, width, sensor.dtype)
            for z in range(self.num_depths):
                selected = kernels[z, phase_ids]
                if squared:
                    selected = selected.square()
                repeated = sensor.expand(-1, len(phase_ids), -1, -1)
                convolved = _fft_conv2_same(repeated, selected)
                result[:, :, z].add_((convolved * masks.unsqueeze(0)).sum(1, keepdim=True))
        return result

    def _project(self, value: Tensor, *, squared: bool) -> Tensor:
        if self._is_simulation(value) and self.simulation is not None:
            function = self.simulation.forward_squared if squared else self.simulation.forward
            return function(value)
        value = self._validate_full_volume(value)
        identity = _tensor_identity(value)
        if (
            not squared and torch.is_grad_enabled() and value.requires_grad
            and identity is not None and self._linear_cache is not None
            and self._linear_cache[0] == identity
        ):
            return _ReuseLinearForward.apply(value, self._linear_cache[1], self, False)
        if torch.is_grad_enabled() and value.requires_grad:
            return _FullProject.apply(value, self, squared)
        result = self._full_project_plain(value, squared=squared)
        if not squared and identity is not None:
            self._linear_cache = (identity, result.detach())
        return result

    def forward(self, value: Tensor) -> Tensor:
        return self._project(value, squared=False)

    def forward_squared(self, value: Tensor) -> Tensor:
        return self._project(value, squared=True)

    def adjoint(self, sensor: Tensor, *, squared: bool = False) -> Tensor:
        if tuple(sensor.shape[-2:]) == self.simulation_shape and self.simulation is not None:
            return self.simulation.adjoint(sensor, squared=squared)
        return self._full_adjoint_plain(sensor, squared=squared)
