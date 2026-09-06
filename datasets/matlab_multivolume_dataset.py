from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_Z_UM = np.arange(10, 101, 10, dtype=np.float32)
EXPECTED_SPLITS = {
    "train": ("P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10"),
    "validation": ("P09", "V01", "V02"),
    "test": ("P07", "T01", "T02"),
}


@dataclass(frozen=True)
class DatasetItemKey:
    sample_id: str
    subset_index: int
    split: str
    sample_dir: Path


def _matlab_yx_to_yx(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset[()])
    if value.ndim != 2:
        raise ValueError(f"Expected a MATLAB YX matrix, got shape {value.shape}")
    return np.ascontiguousarray(value.T)


def _matlab_yxz_to_zyx(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset[()])
    if value.ndim != 3:
        raise ValueError(f"Expected a MATLAB YXZ volume, got shape {value.shape}")
    return np.ascontiguousarray(value.transpose(0, 2, 1))


def _vector(dataset: h5py.Dataset, dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(dataset[()]).reshape(-1).astype(dtype, copy=False)


def _require_finite_nonnegative(name: str, value: np.ndarray) -> None:
    if value.size == 0 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite, nonnegative, and nonempty")


def _source_signature(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def load_dataset_index(root: str | Path) -> tuple[dict[str, list[DatasetItemKey]], str]:
    root = Path(root).expanduser().resolve()
    split_path = root / "dataset_splits.json"
    final_path = root / "FINAL_DATASET_MANIFEST.json"
    if not split_path.is_file() or not final_path.is_file():
        raise FileNotFoundError("Dataset root lacks dataset_splits.json or FINAL_DATASET_MANIFEST.json")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if not splits.get("dataset_complete") or not final.get("complete"):
        raise ValueError("Dataset manifests do not declare a complete dataset")
    final_map = {item["sample_id"]: item for item in final["samples"]}
    split_map = {item["sample_id"]: item for item in splits["samples"]}
    if set(final_map) != set(split_map):
        raise ValueError("Final and split manifests contain different objects")
    actual: dict[str, list[str]] = {name: [] for name in EXPECTED_SPLITS}
    indexed: dict[str, list[DatasetItemKey]] = {name: [] for name in EXPECTED_SPLITS}
    signature_paths = [split_path, final_path]
    for sample_id, item in split_map.items():
        split = str(item["split"])
        if split not in indexed:
            raise ValueError(f"Unsupported split {split!r} for {sample_id}")
        sample_dir = root / sample_id
        if Path(final_map[sample_id]["sample_dir"]).resolve() != sample_dir:
            raise ValueError(f"{sample_id} points outside the unified dataset root")
        validation = sample_dir / "validation_manifest.json"
        prepared = sample_dir / "prepared.mat"
        if not validation.is_file() or not prepared.is_file():
            raise FileNotFoundError(f"{sample_id} is missing prepared or validation metadata")
        validation_record = json.loads(validation.read_text(encoding="utf-8"))
        if not validation_record.get("complete"):
            raise ValueError(f"{sample_id} has not passed MATLAB validation")
        actual[split].append(sample_id)
        signature_paths.extend((validation, prepared))
        for subset_index in range(1, 11):
            subset_path = sample_dir / "subsets" / f"subset_{subset_index:02d}.mat"
            if not subset_path.is_file():
                raise FileNotFoundError(subset_path)
            signature_paths.append(subset_path)
            indexed[split].append(DatasetItemKey(sample_id, subset_index, split, sample_dir))
        signature_paths.extend(sorted((sample_dir / "sensor_frames").glob("frame_*.mat")))
    for split, expected in EXPECTED_SPLITS.items():
        if tuple(actual[split]) != expected:
            raise ValueError(
                f"Authoritative {split} split is {actual[split]}, expected {list(expected)}"
            )
    return indexed, _source_signature(signature_paths)


def _read_input(key: DatasetItemKey) -> dict[str, Any]:
    subset_path = key.sample_dir / "subsets" / f"subset_{key.subset_index:02d}.mat"
    with h5py.File(subset_path, "r") as handle:
        sample_id = "".join(chr(int(v)) for v in handle["sample_id"][()].reshape(-1))
        subset_index = int(np.asarray(handle["subset_index"][()]).item())
        iterations = int(np.asarray(handle["iterations"][()]).item())
        if sample_id != key.sample_id or subset_index != key.subset_index or iterations != 3:
            raise ValueError(f"Invalid owner, subset index, or RL iterations in {subset_path}")
        input_indices = _vector(handle["input_indices"], np.int64)
        holdout_indices = _vector(handle["holdout_indices"], np.int64)
        z_um = _vector(handle["z_um"], np.float32)
        g_mean = _matlab_yxz_to_zyx(handle["physics_mean_raw"]).astype(np.float32)
        f_var = _matlab_yxz_to_zyx(handle["physics_taylor_sqrt_float"]).astype(np.float32)
        input_mean = _matlab_yx_to_yx(handle["input_physics_mean_float"]).astype(np.float32)
    if input_indices.shape != (10,) or holdout_indices.shape != (90,):
        raise ValueError(f"{subset_path} is not a 10/90 split")
    if set(input_indices.tolist()) & set(holdout_indices.tolist()):
        raise ValueError(f"Input/target leakage in {subset_path}")
    if set(input_indices.tolist()) | set(holdout_indices.tolist()) != set(range(1, 101)):
        raise ValueError(f"Incomplete frame partition in {subset_path}")
    if not np.array_equal(z_um, EXPECTED_Z_UM):
        raise ValueError(f"Unexpected depth grid in {subset_path}")
    frames = []
    for frame_index in input_indices:
        frame_path = key.sample_dir / "sensor_frames" / f"frame_{frame_index:03d}.mat"
        with h5py.File(frame_path, "r") as handle:
            frame = _matlab_yx_to_yx(handle["sensor_pre_detector"]).astype(np.float32)
        frames.append(frame)
    input_frames = np.stack(frames, axis=0)
    recomputed_mean = input_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    if not np.array_equal(recomputed_mean, input_mean):
        raise ValueError(f"Saved input mean does not exactly match its ten frames: {subset_path}")
    residual_frames = input_frames - input_mean[None]
    for name, value in {
        "f_var": f_var,
        "g_mean": g_mean,
        "input_mean": input_mean,
        "input_frames": input_frames,
    }.items():
        _require_finite_nonnegative(name, value)
    return {
        "sample_id": key.sample_id,
        "subset_index": key.subset_index,
        "split": key.split,
        "input_indices": input_indices,
        "z_values_um": z_um,
        "f_var": f_var[None],
        "g_mean": g_mean[None],
        "input_mean": input_mean[None],
        "residual_frames": residual_frames[:, None],
    }


def _read_targets(
    key: DatasetItemKey, *, include_ground_truth: bool
) -> dict[str, np.ndarray]:
    subset_path = key.sample_dir / "subsets" / f"subset_{key.subset_index:02d}.mat"
    with h5py.File(subset_path, "r") as handle:
        measured_mean = _matlab_yx_to_yx(handle["holdout_physics_mean_float"]).astype(np.float32)
        measured_variance = _matlab_yx_to_yx(
            handle["holdout_physics_variance_nminus1_float"]
        ).astype(np.float32)
        holdout_indices = _vector(handle["holdout_indices"], np.int64)
    for name, value in {
        "measured_mean": measured_mean,
        "measured_variance": measured_variance,
    }.items():
        _require_finite_nonnegative(name, value)
    result = {
        "holdout_indices": holdout_indices,
        "measured_mean": measured_mean[None],
        "measured_variance": measured_variance[None],
    }
    if include_ground_truth:
        with h5py.File(key.sample_dir / "prepared.mat", "r") as handle:
            ground_truth = _matlab_yxz_to_zyx(handle["ground_truth"]).astype(np.float32)
        _require_finite_nonnegative("ground_truth", ground_truth)
        result["ground_truth"] = ground_truth[None]
    return result


def load_inference_input(sample_dir: str | Path, subset_index: int) -> dict[str, Any]:
    sample_dir = Path(sample_dir).expanduser().resolve()
    key = DatasetItemKey(sample_dir.name, int(subset_index), "inference", sample_dir)
    return _read_input(key)


class MatlabMultiVolumeDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        cache_dir: str | Path | None = None,
        include_ground_truth: bool | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        indexed, self.dataset_fingerprint = load_dataset_index(self.root)
        if split not in indexed:
            raise ValueError(f"Unknown split {split!r}")
        self.split = split
        self.keys = indexed[split]
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        self.include_ground_truth = split != "train" if include_ground_truth is None else bool(
            include_ground_truth
        )
        if split == "train" and self.include_ground_truth:
            raise ValueError("Ground truth is forbidden in the training data path")

    def __len__(self) -> int:
        return len(self.keys)

    def _cache_path(self, key: DatasetItemKey) -> Path:
        assert self.cache_dir is not None
        gt_mode = "eval_gt" if self.include_ground_truth else "train_nogt"
        return self.cache_dir / self.dataset_fingerprint[:16] / key.split / gt_mode / (
            f"{key.sample_id}_subset_{key.subset_index:02d}.npz"
        )

    def _load_numpy(self, index: int) -> dict[str, Any]:
        key = self.keys[index]
        if self.cache_dir is None:
            return {
                **_read_input(key),
                **_read_targets(key, include_ground_truth=self.include_ground_truth),
            }
        cache_path = self._cache_path(key)
        if not cache_path.is_file():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            item = {
                **_read_input(key),
                **_read_targets(key, include_ground_truth=self.include_ground_truth),
            }
            temporary = cache_path.with_suffix(f".tmp-{os.getpid()}.npz")
            np.savez(temporary, **item, dataset_fingerprint=self.dataset_fingerprint)
            os.replace(temporary, cache_path)
        with np.load(cache_path, allow_pickle=False) as stored:
            if str(stored["dataset_fingerprint"].item()) != self.dataset_fingerprint:
                raise ValueError(f"Stale cache entry: {cache_path}")
            item = {name: stored[name] for name in stored.files if name != "dataset_fingerprint"}
        for name in ("sample_id", "split"):
            if isinstance(item[name], np.ndarray):
                item[name] = str(item[name].item())
        item["subset_index"] = int(np.asarray(item["subset_index"]).item())
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._load_numpy(index)
        tensor_names = {
            "f_var", "g_mean", "input_mean", "residual_frames",
            "measured_mean", "measured_variance", "ground_truth", "z_values_um",
        }
        return {
            name: torch.from_numpy(np.asarray(value)).float() if name in tensor_names else value
            for name, value in item.items()
        }

    def precompute(self) -> None:
        for index in range(len(self)):
            self._load_numpy(index)
