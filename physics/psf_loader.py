from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PSFMetadata:
    path: Path
    mat_format: str
    variable_shapes_matlab: dict[str, tuple[int, ...]]
    z_all_um: np.ndarray
    selected_indices: np.ndarray
    selected_z_um: np.ndarray
    phase_period: int
    caindex: np.ndarray | None


@dataclass(frozen=True)
class PSFData:
    """Selected PSFs in canonical [Z, phase_row, phase_col, row, col] order."""

    H: np.ndarray | None
    Ht: np.ndarray | None
    metadata: PSFMetadata


def _is_hdf5(path: Path) -> bool:
    try:
        import h5py

        return h5py.is_hdf5(path)
    except ImportError:
        return False


def _as_um(values: np.ndarray, unit: str = "auto") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if unit == "um":
        return values
    if unit == "m":
        return values * 1e6
    if unit != "auto":
        raise ValueError(f"Unsupported depth unit: {unit!r}")
    finite = np.abs(values[np.isfinite(values)])
    if finite.size and finite.max(initial=0.0) < 1.0:
        return values * 1e6
    return values


def _select_depths(z_all_um: np.ndarray, requested_um: Sequence[float], atol: float) -> np.ndarray:
    requested = np.asarray(requested_um, dtype=np.float64).reshape(-1)
    if requested.size == 0:
        raise ValueError("z_values_um must not be empty")
    indices: list[int] = []
    for z in requested:
        matches = np.flatnonzero(np.isclose(z_all_um, z, rtol=0.0, atol=atol))
        if matches.size != 1:
            raise ValueError(
                f"Requested depth {z:g} um has {matches.size} exact matches in "
                f"{z_all_um.tolist()} (atol={atol:g} um)"
            )
        indices.append(int(matches[0]))
    if len(set(indices)) != len(indices):
        raise ValueError("Requested physical depths map to duplicate PSF indices")
    return np.asarray(indices, dtype=np.int64)


def _matlab_shape_from_hdf5(shape: Iterable[int]) -> tuple[int, ...]:
    return tuple(reversed(tuple(int(v) for v in shape)))


def _validate_kernel_shape(shape: tuple[int, ...], variable: str) -> None:
    if len(shape) != 5:
        raise ValueError(f"{variable} must be a 5D LF-PSF library, got MATLAB shape {shape}")
    if shape[2] != shape[3]:
        raise ValueError(f"{variable} phase axes must be square, got MATLAB shape {shape}")


def _read_hdf5_vector(handle, name: str) -> np.ndarray | None:
    if name not in handle:
        return None
    return np.asarray(handle[name]).reshape(-1)


def _read_hdf5_caindex(handle) -> np.ndarray | None:
    if "CAindex" not in handle:
        return None
    # MATLAB [Z,2] is stored by v7.3 HDF5 as [2,Z].
    value = np.asarray(handle["CAindex"])
    return value.T.copy() if value.ndim == 2 else value


def _inspect_hdf5(
    path: Path,
    h_name: str,
    ht_name: str,
    fallback_z_um: Sequence[float] | None,
    depth_unit: str,
) -> tuple[dict[str, tuple[int, ...]], np.ndarray, int, np.ndarray | None]:
    import h5py

    shapes: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as handle:
        for name in (h_name, ht_name):
            if name in handle:
                shapes[name] = _matlab_shape_from_hdf5(handle[name].shape)
                _validate_kernel_shape(shapes[name], name)
        if h_name not in shapes and ht_name not in shapes:
            raise KeyError(f"Neither {h_name!r} nor {ht_name!r} exists in {path}")
        z_raw = _read_hdf5_vector(handle, "x3objspace")
        z_all_um = _as_um(z_raw, depth_unit) if z_raw is not None else None
        if z_all_um is None:
            if fallback_z_um is None:
                raise ValueError("PSF has no x3objspace; psf_z_all_um must be configured")
            z_all_um = np.asarray(fallback_z_um, dtype=np.float64)
        reference_shape = shapes.get(h_name, shapes.get(ht_name))
        assert reference_shape is not None
        if len(z_all_um) != reference_shape[4]:
            raise ValueError(
                f"Depth vector has {len(z_all_um)} values but PSF has {reference_shape[4]} layers"
            )
        nnum_raw = _read_hdf5_vector(handle, "Nnum")
        phase_period = int(round(float(nnum_raw[0]))) if nnum_raw is not None else reference_shape[2]
        caindex = _read_hdf5_caindex(handle)
        zspacing_raw = _read_hdf5_vector(handle, "zspacing")
        if zspacing_raw is not None and len(z_all_um) > 1:
            declared = float(_as_um(zspacing_raw, depth_unit)[0])
            actual = float(np.median(np.diff(z_all_um)))
            if not np.isclose(declared, actual, rtol=1e-5, atol=1e-5):
                LOGGER.warning(
                    "Ignoring inconsistent zspacing %.6g um; x3objspace spacing is %.6g um",
                    declared,
                    actual,
                )
    return shapes, z_all_um, phase_period, caindex


def _inspect_classic(
    path: Path,
    h_name: str,
    ht_name: str,
    fallback_z_um: Sequence[float] | None,
    depth_unit: str,
) -> tuple[dict[str, tuple[int, ...]], np.ndarray, int, np.ndarray | None]:
    from scipy.io import loadmat, whosmat

    shapes = {name: tuple(shape) for name, shape, _ in whosmat(path) if name in (h_name, ht_name)}
    for name, shape in shapes.items():
        _validate_kernel_shape(shape, name)
    if not shapes:
        raise KeyError(f"Neither {h_name!r} nor {ht_name!r} exists in {path}")
    aux_names = ["x3objspace", "Nnum", "CAindex", "zspacing"]
    aux = loadmat(path, variable_names=aux_names, squeeze_me=False)
    if "x3objspace" in aux:
        z_all_um = _as_um(aux["x3objspace"], depth_unit)
    elif fallback_z_um is not None:
        z_all_um = np.asarray(fallback_z_um, dtype=np.float64)
    else:
        raise ValueError("PSF has no x3objspace; psf_z_all_um must be configured")
    reference_shape = shapes.get(h_name, shapes.get(ht_name))
    assert reference_shape is not None
    if len(z_all_um) != reference_shape[4]:
        raise ValueError(f"Depth vector has {len(z_all_um)} values but PSF has {reference_shape[4]} layers")
    phase_period = int(np.asarray(aux.get("Nnum", [[reference_shape[2]]])).reshape(-1)[0])
    caindex = np.asarray(aux["CAindex"]) if "CAindex" in aux else None
    if "zspacing" in aux and len(z_all_um) > 1:
        declared = float(_as_um(aux["zspacing"], depth_unit)[0])
        actual = float(np.median(np.diff(z_all_um)))
        if not np.isclose(declared, actual, rtol=1e-5, atol=1e-5):
            LOGGER.warning(
                "Ignoring inconsistent zspacing %.6g um; x3objspace spacing is %.6g um",
                declared,
                actual,
            )
    return shapes, z_all_um, phase_period, caindex


def inspect_psf(
    path: str | Path,
    z_values_um: Sequence[float],
    *,
    h_variable_name: str = "H",
    ht_variable_name: str = "Ht",
    psf_z_all_um: Sequence[float] | None = None,
    depth_unit: str = "auto",
    depth_atol_um: float = 1e-4,
) -> PSFMetadata:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _is_hdf5(path):
        mat_format = "v7.3-hdf5"
        shapes, z_all_um, phase_period, caindex = _inspect_hdf5(
            path, h_variable_name, ht_variable_name, psf_z_all_um, depth_unit
        )
    else:
        mat_format = "classic"
        shapes, z_all_um, phase_period, caindex = _inspect_classic(
            path, h_variable_name, ht_variable_name, psf_z_all_um, depth_unit
        )
    indices = _select_depths(z_all_um, z_values_um, depth_atol_um)
    selected_z_um = z_all_um[indices]
    if phase_period != shapes.get(h_variable_name, shapes.get(ht_variable_name))[2]:
        raise ValueError(f"Nnum={phase_period} disagrees with PSF phase dimension")
    metadata = PSFMetadata(
        path=path,
        mat_format=mat_format,
        variable_shapes_matlab=shapes,
        z_all_um=z_all_um,
        selected_indices=indices,
        selected_z_um=selected_z_um,
        phase_period=phase_period,
        caindex=caindex[indices].copy() if caindex is not None else None,
    )
    LOGGER.info(
        "PSF depth crop: indices=%s, z_um=%s, phase=%d",
        indices.tolist(),
        selected_z_um.tolist(),
        phase_period,
    )
    return metadata


def _hdf5_selected_kernel(path: Path, variable: str, indices: np.ndarray) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        if variable not in handle:
            raise KeyError(f"Variable {variable!r} does not exist in {path}")
        order = np.argsort(indices)
        sorted_raw = np.asarray(handle[variable][indices[order].tolist(), ...], dtype=np.float32)
        raw = sorted_raw[np.argsort(order)]
    # v7.3 raw axes are [Z, phase_col, phase_row, kernel_col, kernel_row].
    return np.ascontiguousarray(raw.transpose(0, 2, 1, 4, 3))


def _classic_selected_kernel(path: Path, variable: str, indices: np.ndarray) -> np.ndarray:
    from scipy.io import loadmat

    values = loadmat(path, variable_names=[variable])[variable]
    # Classic MATLAB axes are [kernel_row, kernel_col, phase_row, phase_col, Z].
    selected = values[..., indices]
    return np.ascontiguousarray(selected.transpose(4, 2, 3, 0, 1), dtype=np.float32)


def load_psf(
    path: str | Path,
    z_values_um: Sequence[float],
    *,
    h_variable_name: str = "H",
    ht_variable_name: str = "Ht",
    psf_z_all_um: Sequence[float] | None = None,
    depth_unit: str = "auto",
    depth_atol_um: float = 1e-4,
    load_h: bool = True,
    load_ht: bool = False,
    ht_path: str | Path | None = None,
) -> PSFData:
    if not load_h and not load_ht:
        raise ValueError("At least one of load_h/load_ht must be true")
    metadata = inspect_psf(
        path,
        z_values_um,
        h_variable_name=h_variable_name,
        ht_variable_name=ht_variable_name,
        psf_z_all_um=psf_z_all_um,
        depth_unit=depth_unit,
        depth_atol_um=depth_atol_um,
    )
    reader = _hdf5_selected_kernel if metadata.mat_format == "v7.3-hdf5" else _classic_selected_kernel
    h = reader(metadata.path, h_variable_name, metadata.selected_indices) if load_h else None
    ht = None
    if load_ht:
        ht_metadata = metadata
        if ht_path is not None and Path(ht_path).expanduser().resolve() != metadata.path:
            ht_metadata = inspect_psf(
                ht_path,
                z_values_um,
                h_variable_name=h_variable_name,
                ht_variable_name=ht_variable_name,
                psf_z_all_um=psf_z_all_um,
                depth_unit=depth_unit,
                depth_atol_um=depth_atol_um,
            )
            if ht_metadata.phase_period != metadata.phase_period:
                raise ValueError("H and Ht files have different phase periods")
            if not np.allclose(ht_metadata.selected_z_um, metadata.selected_z_um, rtol=0.0, atol=depth_atol_um):
                raise ValueError("H and Ht files selected different physical depths")
        ht_reader = (
            _hdf5_selected_kernel
            if ht_metadata.mat_format == "v7.3-hdf5"
            else _classic_selected_kernel
        )
        ht = ht_reader(ht_metadata.path, ht_variable_name, ht_metadata.selected_indices)
    expected_prefix = (len(metadata.selected_indices), metadata.phase_period, metadata.phase_period)
    for name, value in ((h_variable_name, h), (ht_variable_name, ht)):
        if value is not None and value.shape[:3] != expected_prefix:
            raise ValueError(f"Loaded {name} has unexpected canonical shape {value.shape}")
    return PSFData(H=h, Ht=ht, metadata=metadata)
