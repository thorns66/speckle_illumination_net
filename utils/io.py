from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile


def _convert_intensity(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "preserve":
        return values.astype(np.float32, copy=False)
    if mode != "matlab_im2double":
        raise ValueError(f"Unsupported TIFF intensity mode: {mode!r}")
    if np.issubdtype(values.dtype, np.integer):
        maximum = np.iinfo(values.dtype).max
        return values.astype(np.float32) / float(maximum)
    return values.astype(np.float32, copy=False)


def load_tiff_stack(path: str | Path, *, intensity_mode: str = "matlab_im2double") -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tif:
        pages = [page.asarray() for page in tif.pages]
    if not pages:
        raise ValueError(f"TIFF contains no pages: {path}")
    if any(page.ndim != 2 for page in pages):
        raise ValueError(f"Expected grayscale 2D TIFF pages in {path}")
    stack = np.stack(pages, axis=0)
    return _convert_intensity(stack, intensity_mode)


def select_frame_indices(
    total_frames: int,
    num_frames: int,
    configured_indices: Sequence[int] | None,
    *,
    seed: int,
) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("Variance estimation requires at least two speckle frames")
    if configured_indices is not None:
        indices = np.asarray(configured_indices, dtype=np.int64)
        if len(indices) != num_frames:
            raise ValueError(
                f"frame_indices contains {len(indices)} entries but num_speckle_frames={num_frames}"
            )
    else:
        if num_frames > total_frames:
            raise ValueError(f"Requested {num_frames} frames but TIFF contains {total_frames}")
        indices = np.sort(np.random.default_rng(seed).choice(total_frames, num_frames, replace=False))
    if len(np.unique(indices)) != len(indices):
        raise ValueError("frame_indices must be unique")
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= total_frames:
        raise IndexError(f"frame_indices must use zero-based values in [0,{total_frames - 1}]")
    return indices


def load_volume_tiff(
    path: str | Path,
    expected_depths: int,
    *,
    intensity_mode: str = "matlab_im2double",
) -> np.ndarray:
    volume = load_tiff_stack(path, intensity_mode=intensity_mode)
    if volume.shape[0] != expected_depths:
        raise ValueError(f"Volume {path} has Z={volume.shape[0]}, expected {expected_depths}")
    return volume


def save_volume_tiff(path: str | Path, volume_zxy: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(volume_zxy, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("Volume must be [Z,H,W]")
    tifffile.imwrite(path, volume, photometric="minisblack", metadata={"axes": "ZYX"})
