"""Prepare deterministic 10/90 subsets and RL3 inputs for real-field training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

import h5py
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/spinach_real_preserved_20260910_run01"
SOURCE_OUTPUT = ROOT / "outputs/spinach_real_reprocess_20260910_run01"
DEFAULT_DATA = ROOT / "data/spinach_real_mixed_training_20260910_run01"
DEFAULT_OUTPUT = ROOT / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run01"
MATLAB = "/workspace/xyx/MATLAB/R2023b/bin/matlab"
PSF = ROOT / "psf/NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat"
SEED = 20260901
FIXED_SUBSET_01 = (9, 10, 27, 44, 54, 74, 77, 83, 85, 92)
FIELDS = ("45", "55")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_indices(seed: int = SEED) -> list[list[int]]:
    fixed = set(FIXED_SUBSET_01)
    remaining = np.asarray(sorted(set(range(1, 101)) - fixed), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(remaining)
    subsets = [list(FIXED_SUBSET_01)] + [
        shuffled[start : start + 10].tolist() for start in range(0, 90, 10)
    ]
    flat = [index for subset in subsets for index in subset]
    if len(subsets) != 10 or len(flat) != 100 or set(flat) != set(range(1, 101)):
        raise AssertionError("The ten real subsets must partition frames 1..100 exactly once")
    return subsets


def statistics(frames: np.ndarray, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    selected = frames[np.asarray(indices, dtype=np.int64) - 1]
    return (
        selected.mean(0, dtype=np.float64).astype(np.float32),
        selected.var(0, ddof=1, dtype=np.float64).astype(np.float32),
    )


def _read_source_hashes(field: str) -> dict[str, str]:
    source_manifest = json.loads((SOURCE_OUTPUT / field / "manifest.json").read_text(encoding="utf-8"))
    return source_manifest["rectified_hashes"]


def prepare(data_root: Path, output_root: Path) -> None:
    marker = data_root / "preparation_complete.json"
    if marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if not record.get("complete") or record.get("seed") != SEED:
            raise RuntimeError(f"Incompatible existing preparation: {marker}")
        return
    if data_root.exists() or output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing V5 data or output directory")
    data_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    subsets = subset_indices()
    common = {
        "schema_version": 1,
        "simulation_train_objects": [f"P{i:02d}" for i in range(1, 12)],
        "simulation_excluded_objects": ["P12"],
        "real_train_fields": list(FIELDS),
        "real_validation_fields": [],
        "real_test_fields": [],
        "real_fields_have_ground_truth": False,
        "field_ids_are_not_depth": True,
        "subset_seed": SEED,
        "subset_indices": subsets,
        "subset_policy": "subset_01 preserved; remaining 90 frames deterministically partitioned into nine disjoint groups",
        "mix_per_global_batch": {"simulation": 6, "field_45": 1, "field_55": 1},
        "source_dataset": str(SOURCE),
        "psf_path": str(PSF),
        "psf_sha256": sha256(PSF),
        "z_um": list(range(10, 101, 10)),
        "iterations": 3,
        "input_policy": "preserved float32 rectified intensity; no second /255; no per-frame maximum normalization",
    }
    write_json(data_root / "dataset_manifest.json", common)
    write_json(output_root / "data_contract.json", common)
    for field in FIELDS:
        frame_paths = [SOURCE / field / "rectified_float" / f"frame_{index:03d}.tif" for index in range(1, 101)]
        if not all(path.is_file() for path in frame_paths):
            raise FileNotFoundError(f"Incomplete rectified frame inventory for field {field}")
        frames = np.stack([tifffile.imread(path) for path in frame_paths])
        if frames.shape != (100, 1029, 1421) or frames.dtype != np.float32:
            raise ValueError(f"Unexpected field {field} stack: {frames.shape} {frames.dtype}")
        if not np.isfinite(frames).all() or frames.min() < 0:
            raise ValueError(f"Invalid field {field} intensities")
        source_hashes = _read_source_hashes(field)
        for subset_index, selected in enumerate(subsets, 1):
            holdout = sorted(set(range(1, 101)) - set(selected))
            input_mean, input_variance = statistics(frames, selected)
            holdout_mean, holdout_variance = statistics(frames, holdout)
            folder = data_root / field / f"subset_{subset_index:02d}"
            folder.mkdir(parents=True)
            arrays = {
                "mean": input_mean,
                "variance": input_variance,
                "holdout_mean": holdout_mean,
                "holdout_variance": holdout_variance,
            }
            for name, value in arrays.items():
                tifffile.imwrite(folder / f"{name}.tif", value, photometric="minisblack")
            manifest = {
                **common,
                "field_id": field,
                "split": "train",
                "subset_index": subset_index,
                "image_shape_yx": [1029, 1421],
                "input_indices": selected,
                "holdout_indices": holdout,
                "selected_files": [str(frame_paths[index - 1]) for index in selected],
                "holdout_files": [str(frame_paths[index - 1]) for index in holdout],
                "mean_tiff": str(folder / "mean.tif"),
                "variance_tiff": str(folder / "variance.tif"),
                "holdout_mean_tiff": str(folder / "holdout_mean.tif"),
                "holdout_variance_tiff": str(folder / "holdout_variance.tif"),
                "output_mat": {
                    "mean": str(folder / "mean_rl3.mat"),
                    "taylor": str(folder / "taylor_rl3.mat"),
                },
                "rectified_hashes": source_hashes,
                "gpu_uuid": {},
            }
            write_json(folder / "manifest.json", manifest)
        del frames
    for field in FIELDS:
        source = SOURCE / field / "subset_01"
        destination = data_root / field / "subset_01"
        for name in ("mean_rl3.mat", "taylor_rl3.mat"):
            shutil.copy2(source / name, destination / name)
        with h5py.File(destination / "mean_rl3.mat", "r") as handle:
            observed = np.asarray(handle["input_indices"]).reshape(-1).astype(int).tolist()
        if observed != list(FIXED_SUBSET_01):
            raise ValueError(f"Reused field {field} subset_01 has changed frame indices")
    write_json(marker, {**common, "complete": True, "prepared_unix": time.time(), "reused_rl_volumes": 4})


def inventory() -> list[dict[str, Any]]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = []
    for line in rows.splitlines():
        index, uuid, name, free, utilization = [part.strip() for part in line.split(",")]
        result.append({"index": int(index), "uuid": uuid, "name": name, "free": int(free), "utilization": int(utilization)})
    return result


def _matlab_job(manifest_path: Path, mode: str, gpu: dict[str, Any], output_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gpu_uuid"][mode] = gpu["uuid"]
    job_manifest = output_root / "jobs" / (
        f"rl_{manifest['field_id']}_subset_{manifest['subset_index']:02d}_{mode}.json"
    )
    write_json(job_manifest, manifest)
    log = output_root / "logs" / f"rl_{manifest['field_id']}_subset_{manifest['subset_index']:02d}_{mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "matlab_code" / name for name in ("real_data", "pilot_dataset", "Util", "Solver")]
    quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    target = str(job_manifest).replace("'", "''")
    expression = (
        f"addpath({quoted});spinach_root_rl_worker_v2('{target}','{mode}');"
        f"p='{manifest['output_mat'][mode]}';s=load(p);s.input_policy='{manifest['input_policy']}';"
        "save(p,'-struct','s','-v7.3');"
    )
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu["uuid"],
        MATLAB_PREFDIR=str(output_root / "matlab_preferences" / f"{manifest['field_id']}_{manifest['subset_index']:02d}_{mode}"),
    )
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [MATLAB, "-singleCompThread", "-softwareopengl", "-batch", expression],
            cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"MATLAB RL failed; see {log}")
    output = Path(manifest["output_mat"][mode])
    if not output.is_file():
        raise FileNotFoundError(output)


def run_rl(data_root: Path, output_root: Path) -> None:
    pending = [
        (data_root / field / f"subset_{subset:02d}" / "manifest.json", mode)
        for field in FIELDS for subset in range(2, 11) for mode in ("mean", "taylor")
        if not (data_root / field / f"subset_{subset:02d}" / f"{mode}_rl3.mat").is_file()
    ]
    while pending:
        cards = [
            gpu for gpu in inventory()
            if gpu["index"] in range(6) and "A40" in gpu["name"].upper() and gpu["free"] >= 30000
        ]
        cards.sort(key=lambda gpu: (gpu["utilization"] > 5, -gpu["free"]))
        if not cards:
            raise RuntimeError("No A40 currently has 30 GiB free for full-field RL3")
        batch = pending[: min(2, len(cards))]
        pending = pending[len(batch):]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [pool.submit(_matlab_job, task[0], task[1], gpu, output_root) for task, gpu in zip(batch, cards)]
            for future in futures:
                future.result()
        write_json(output_root / "rl_progress.json", {"remaining_jobs": len(pending), "updated_unix": time.time()})
    expected = [data_root / field / f"subset_{subset:02d}" / f"{mode}_rl3.mat"
                for field in FIELDS for subset in range(1, 11) for mode in ("mean", "taylor")]
    if not all(path.is_file() for path in expected):
        raise RuntimeError("RL inventory is incomplete")
    validation_rows = []
    for field in FIELDS:
        for subset in range(1, 11):
            folder = data_root / field / f"subset_{subset:02d}"
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            for mode in ("mean", "taylor"):
                path = folder / f"{mode}_rl3.mat"
                with h5py.File(path, "r") as handle:
                    indices = np.asarray(handle["input_indices"]).reshape(-1).astype(int).tolist()
                    raw = np.asarray(handle["reconstruction_raw"], dtype=np.float32)
                    if indices != manifest["input_indices"] or int(np.asarray(handle["iterations"]).item()) != 3:
                        raise ValueError(f"RL ownership or iteration mismatch: {path}")
                    if raw.shape != (10, 1421, 1029) or not np.isfinite(raw).all() or np.any(raw < 0):
                        raise ValueError(f"Invalid full-field RL volume: {path}")
                    root_error = None
                    if mode == "taylor":
                        square_root = np.asarray(handle["reconstruction_sqrt"], dtype=np.float32)
                        root_error = float(np.max(np.abs(square_root - np.sqrt(raw))))
                        if root_error > 2e-7:
                            raise ValueError(f"Taylor sqrt mismatch: {path}")
                    validation_rows.append({
                        "field_id": field, "subset_index": subset, "mode": mode,
                        "shape_matlab_zyx": list(raw.shape), "minimum": float(raw.min()),
                        "maximum": float(raw.max()), "taylor_sqrt_max_abs_error": root_error,
                    })
    write_json(output_root / "rl_validation.json", {
        "complete": True, "rows": validation_rows, "validated_volumes": len(validation_rows),
        "all_iterations": 3, "all_nonnegative_finite": True,
    })
    artifact_paths = []
    for field in FIELDS:
        for subset in range(1, 11):
            folder = data_root / field / f"subset_{subset:02d}"
            artifact_paths.extend(
                folder / name for name in (
                    "manifest.json", "mean.tif", "variance.tif", "holdout_mean.tif",
                    "holdout_variance.tif", "mean_rl3.mat", "taylor_rl3.mat",
                )
            )
    artifacts = {
        str(path.relative_to(data_root)): {"sha256": sha256(path), "size": path.stat().st_size}
        for path in artifact_paths
    }
    preparation = json.loads((data_root / "preparation_complete.json").read_text(encoding="utf-8"))
    write_json(data_root / "final_manifest.json", {
        **preparation,
        "complete": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "source_rectified_hashes": {
            field: _read_source_hashes(field) for field in FIELDS
        },
        "finalized_unix": time.time(),
    })
    write_json(output_root / "rl_complete.json", {
        "complete": True, "volumes": len(expected), "reused": 4, "new": 36,
        "finished_unix": time.time(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "rl", "all"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root, output_root = args.data.resolve(), args.output.resolve()
    if args.stage in ("prepare", "all"):
        prepare(data_root, output_root)
    if args.stage in ("rl", "all"):
        run_rl(data_root, output_root)


if __name__ == "__main__":
    main()
