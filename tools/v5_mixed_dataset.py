"""Mixed P01-P11 and real-field dataset/scheduler for the V5 comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset
from training.global_batch_schedule import FixedGlobalBatchScheduler


EXPECTED_Z = np.arange(10, 101, 10, dtype=np.float32)


def mixed_fingerprint(simulation_fingerprint: str, real_root: str | Path) -> str:
    final = Path(real_root).resolve() / "final_manifest.json"
    if not final.is_file():
        raise FileNotFoundError(f"Real training data are not finalized: {final}")
    digest = hashlib.sha256()
    digest.update(b"v5-sim-real-no-p12-6-1-1\0")
    digest.update(simulation_fingerprint.encode("ascii"))
    digest.update(final.read_bytes())
    return digest.hexdigest()


def _matlab_volume(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        value = np.asarray(handle[name], dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"Expected a MATLAB YXZ volume in {path}:{name}, got {value.shape}")
    return np.ascontiguousarray(value.transpose(0, 2, 1))


class RealFieldDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.items = [
            (field, subset, self.root / field / f"subset_{subset:02d}")
            for field in ("45", "55") for subset in range(1, 11)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        field, subset_index, folder = self.items[index]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["field_id"] != field or int(manifest["subset_index"]) != subset_index:
            raise ValueError(f"Real subset ownership changed: {folder}")
        selected = np.asarray(manifest["input_indices"], dtype=np.int64)
        holdout = np.asarray(manifest["holdout_indices"], dtype=np.int64)
        if selected.shape != (10,) or holdout.shape != (90,):
            raise ValueError(f"Real subset is not 10/90: {folder}")
        if set(selected.tolist()) & set(holdout.tolist()) or set(selected.tolist()) | set(holdout.tolist()) != set(range(1, 101)):
            raise ValueError(f"Real subset has frame leakage: {folder}")
        frames = np.stack([tifffile.imread(path) for path in manifest["selected_files"]]).astype(np.float32, copy=False)
        input_mean = tifffile.imread(manifest["mean_tiff"]).astype(np.float32, copy=False)
        recomputed = frames.mean(0, dtype=np.float64).astype(np.float32)
        if not np.array_equal(recomputed, input_mean):
            raise ValueError(f"Saved real input mean changed: {folder}")
        residual = frames - input_mean[None]
        mean_volume = _matlab_volume(Path(manifest["output_mat"]["mean"]), "reconstruction_raw")
        taylor_volume = _matlab_volume(Path(manifest["output_mat"]["taylor"]), "reconstruction_sqrt")
        measured_mean = tifffile.imread(manifest["holdout_mean_tiff"]).astype(np.float32, copy=False)
        measured_variance = tifffile.imread(manifest["holdout_variance_tiff"]).astype(np.float32, copy=False)
        arrays = {
            "f_var": taylor_volume,
            "g_mean": mean_volume,
            "input_mean": input_mean,
            "residual_frames": residual,
            "measured_mean": measured_mean,
            "measured_variance": measured_variance,
        }
        for name, value in arrays.items():
            if not np.isfinite(value).all() or (name != "residual_frames" and np.any(value < 0)):
                raise ValueError(f"Invalid real {name}: {folder}")
        z = np.asarray(manifest["z_um"], dtype=np.float32)
        if not np.array_equal(z, EXPECTED_Z):
            raise ValueError(f"Real depth grid changed: {folder}")
        return {
            "sample_id": f"real_{field}",
            "field_id": field,
            "domain": "real_train",
            "has_ground_truth": False,
            "subset_index": subset_index,
            "split": "train",
            "input_indices": selected,
            "holdout_indices": holdout,
            "z_values_um": torch.from_numpy(z).float(),
            "f_var": torch.from_numpy(taylor_volume[None]).float(),
            "f_var_feature": torch.from_numpy(taylor_volume[None]).float(),
            "g_mean": torch.from_numpy(mean_volume[None]).float(),
            "input_mean": torch.from_numpy(input_mean[None]).float(),
            "residual_frames": torch.from_numpy(residual[:, None]).float(),
            "measured_mean": torch.from_numpy(measured_mean[None]).float(),
            "measured_variance": torch.from_numpy(measured_variance[None]).float(),
        }


class MixedExperimentDataset(Dataset[dict[str, Any]]):
    """Trainer-compatible route: mixed train, unchanged simulation validation/test."""

    def __init__(
        self,
        simulation_root: str | Path,
        split: str,
        *,
        real_root: str | Path,
        cache_dir: str | Path | None = None,
        include_ground_truth: bool | None = None,
        var_feature_representation: str = "sqrt",
    ) -> None:
        if var_feature_representation != "sqrt":
            raise ValueError("V5 keeps Taylor sqrt as the VAR encoder input")
        self.split = split
        self.simulation = MatlabMultiVolumeDataset(
            simulation_root, split, cache_dir=cache_dir,
            include_ground_truth=include_ground_truth,
            var_feature_representation=var_feature_representation,
        )
        self.real = RealFieldDataset(real_root) if split == "train" else None
        if split == "train" and len(self.simulation) != 110:
            raise ValueError(f"P01-P11 simulation view must contain 110 items, got {len(self.simulation)}")
        if split in ("validation", "test") and len(self.simulation) != 30:
            raise ValueError(f"Simulation {split} must contain 30 items")
        self.dataset_fingerprint = mixed_fingerprint(self.simulation.dataset_fingerprint, real_root)

    def __len__(self) -> int:
        return len(self.simulation) + (len(self.real) if self.real is not None else 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < len(self.simulation):
            item = dict(self.simulation[index])
            item.update(domain="simulation", has_ground_truth=self.split != "train")
            return item
        if self.real is None:
            raise IndexError(index)
        return self.real[index - len(self.simulation)]

    def precompute(self) -> None:
        self.simulation.precompute()


class MixedSixOneOneScheduler:
    """Exact 6 simulation + one field45 + one field55 batch, with resumable queues."""

    def __init__(self, dataset_size: int, global_batch_size: int, seed: int) -> None:
        if dataset_size != 130 or global_batch_size != 8:
            raise ValueError("V5 mixed schedule requires 130 candidates and global batch eight")
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.simulation = FixedGlobalBatchScheduler(110, 6, seed)
        self.field45 = FixedGlobalBatchScheduler(10, 1, seed + 45)
        self.field55 = FixedGlobalBatchScheduler(10, 1, seed + 55)

    @property
    def epoch(self) -> int:
        return self.simulation.epoch

    def next_batch(self) -> list[int]:
        # Positions six and seven reach DDP ranks zero and one when world_size=6.
        return self.simulation.next_batch() + [110 + self.field45.next_batch()[0], 120 + self.field55.next_batch()[0]]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "mixed-6-1-1-v1",
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "simulation": self.simulation.state_dict(),
            "field45": self.field45.state_dict(),
            "field55": self.field55.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "mixed-6-1-1-v1":
            raise ValueError("Incompatible mixed scheduler checkpoint")
        if int(state["dataset_size"]) != self.dataset_size or int(state["global_batch_size"]) != self.global_batch_size:
            raise ValueError("Mixed scheduler dimensions changed")
        self.simulation.load_state_dict(state["simulation"])
        self.field45.load_state_dict(state["field45"])
        self.field55.load_state_dict(state["field55"])

