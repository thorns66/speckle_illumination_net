#!/usr/bin/env python3
"""Run the validation iteration/operator audit with one MATLAB worker per GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[int, dict[str, int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: dict[int, dict[str, int | str]] = {}
    for line in output.strip().splitlines():
        index, uuid, used, utilization, total = [part.strip() for part in line.split(",")]
        rows[int(index)] = {
            "uuid": uuid,
            "used_mib": int(used),
            "utilization_percent": int(utilization),
            "total_mib": int(total),
        }
    return rows


def matlab_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def expected_outputs(output: Path) -> list[Path]:
    paths = []
    for sample in ("P09", "V01", "V02"):
        for mode in ("mean", "taylor"):
            for subset in range(1, 11):
                paths.append(
                    output
                    / "historical_isra"
                    / sample
                    / f"subset_{subset:02d}"
                    / mode
                    / "complete.json"
                )
            paths.append(
                output / "historical_isra" / sample / "full_100" / mode / "complete.json"
            )
            for tag in ("subset_01", "full_100"):
                paths.append(output / "standard_rl" / sample / tag / mode / "complete.json")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--matlab", default="/workspace/xyx/MATLAB/R2023b/bin/matlab"
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    data_root = repo / "data/matlab_cells_pilot_v2_r04"
    script = Path(__file__).resolve().with_suffix(".m")
    assert data_root.is_dir() and script.is_file()
    assert output != data_root and data_root not in output.parents

    gpu_ids = [int(item) for item in args.gpus.split(",")]
    assert gpu_ids and len(set(gpu_ids)) == len(gpu_ids)
    initial_gpus = gpu_snapshot()
    for gpu in gpu_ids:
        assert gpu in initial_gpus, f"GPU {gpu} not available"

    split_manifest = json.loads((data_root / "dataset_splits.json").read_text())
    validation_samples = [
        row["sample_id"]
        for row in split_manifest["samples"]
        if row["split"] == "validation"
    ]
    assert validation_samples == ["P09", "V01", "V02"], validation_samples
    tasks = [
        (sample, mode)
        for sample in validation_samples
        for mode in ("mean", "taylor")
    ]

    source_paths = [
        script,
        Path(__file__).resolve(),
        data_root / "dataset_splits.json",
        repo / "data/.cache/psf/selected_H_d57d0714b548a9d8c888.json",
        repo / "matlab_code/Solver/deconvRL.m",
        repo / "matlab_code/Util/forwardProjectGPU.m",
        repo / "matlab_code/Util/backwardProjectGPU.m",
        repo / "matlab_code/pilot_dataset/pilot_load_psf.m",
        repo / "matlab_code/pilot_dataset/pilot_load_psf_gpu.m",
        repo / "matlab_code/pilot_dataset/pilot_reconstruct_volume.m",
    ]
    for sample in validation_samples:
        sample_root = data_root / sample
        source_paths.extend(
            [
                sample_root / "prepared.mat",
                sample_root / "geometry.json",
                sample_root / "ground_truth_float.tif",
                sample_root / "illumination_3d.mat",
            ]
        )
        source_paths.extend(sorted((sample_root / "sensor_frames").glob("frame_*.mat")))
        source_paths.extend(sorted((sample_root / "subsets").glob("subset_*.mat")))
    assert all(path.is_file() for path in source_paths)
    source_hashes = {str(path): sha256(path) for path in source_paths}

    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    prefs = output / "matlab_prefs"
    prefs.mkdir(exist_ok=True)
    manifest_path = output / "run_manifest.json"
    assert not manifest_path.exists(), f"Refusing to replace {manifest_path}"

    manifest = {
        "status": "running",
        "purpose": "validation iteration scan plus historical-vs-standard RL operator audit",
        "samples": validation_samples,
        "modes": ["mean", "taylor"],
        "historical_isra": {
            "inputs": "ten fixed 10-frame subsets plus full 100 frames",
            "saved_iterations": [5, 10, 20, 50],
            "regression_iteration": 3,
            "trajectory_count": 66,
        },
        "standard_rl_algorithm_audit": {
            "inputs": ["fixed subset_01", "full 100 frames"],
            "saved_iterations": [5, 10, 20, 50],
            "trajectory_count": 12,
            "caveat": "Taylor variance is not asserted to have a Poisson likelihood",
        },
        "gpu_policy": "prefer truly idle GPUs; share only because --allow-busy-gpus was explicit and free-memory threshold passed",
        "shared_gpus_allowed": args.allow_busy_gpus,
        "minimum_free_memory_gib": args.min_free_gib,
        "initial_gpu_snapshot": initial_gpus,
        "source_sha256": source_hashes,
        "started_unix": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def worker(gpu: int, sample: str, mode: str) -> dict[str, object]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(initial_gpus[gpu]["uuid"])
        pref = prefs / f"{sample}_{mode}"
        pref.mkdir(exist_ok=True)
        env["MATLAB_PREFDIR"] = str(pref)
        log_path = logs / f"{sample}_{mode}.log"
        expression = (
            f"addpath({matlab_quote(script.parent)}); "
            f"issues=checkcode({matlab_quote(script)},'-id'); "
            "assert(~any(strcmp({issues.id},'PARSE')),'MATLAB parse error'); "
            f"run_validation_iteration_audit({matlab_quote(repo)},"
            f"{matlab_quote(output)},{matlab_quote(sample)},{matlab_quote(mode)});"
        )
        print(f"LAUNCH sample={sample} mode={mode} physical_gpu={gpu}", flush=True)
        started = time.monotonic()
        with log_path.open("x") as stream:
            completed = subprocess.run(
                [args.matlab, "-batch", expression],
                cwd=repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        result = {
            "sample": sample,
            "mode": mode,
            "physical_gpu": gpu,
            "returncode": completed.returncode,
            "seconds": time.monotonic() - started,
            "log": str(log_path),
        }
        print(json.dumps(result), flush=True)
        return result

    pending = list(tasks)
    active: dict[concurrent.futures.Future, int] = {}
    results: list[dict[str, object]] = []
    max_workers = min(args.max_workers, len(gpu_ids), len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            current = gpu_snapshot()
            reserved = set(active.values())
            candidates = []
            for gpu in gpu_ids:
                if gpu in reserved:
                    continue
                row = current[gpu]
                free_mib = int(row["total_mib"]) - int(row["used_mib"])
                idle = int(row["used_mib"]) < 1024 and int(row["utilization_percent"]) <= 10
                eligible = free_mib >= args.min_free_gib * 1024 and (
                    idle or args.allow_busy_gpus
                )
                if eligible:
                    candidates.append(
                        (
                            not idle,
                            int(row["used_mib"]),
                            int(row["utilization_percent"]),
                            gpu,
                        )
                    )
            for busy, used, utilization, gpu in sorted(candidates):
                if not pending or len(active) >= max_workers:
                    break
                sample, mode = pending.pop(0)
                state = "shared" if busy else "idle"
                print(
                    f"SCHEDULE gpu={gpu} state={state} used={used}MiB util={utilization}% task={sample}/{mode}",
                    flush=True,
                )
                active[executor.submit(worker, gpu, sample, mode)] = gpu
            if active:
                done, _ = concurrent.futures.wait(
                    active,
                    timeout=10,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    results.append(future.result())
                    del active[future]
            elif pending:
                print("WAIT no eligible GPU; retrying in 10 seconds", flush=True)
                time.sleep(10)

    changed_sources = [
        path for path, digest in source_hashes.items() if sha256(Path(path)) != digest
    ]
    outputs = expected_outputs(output)
    missing_outputs = [str(path) for path in outputs if not path.is_file()]
    operator_outputs = [
        output / "operator_audit/operator_audit.mat",
        output / "operator_audit/operator_audit.json",
    ]
    missing_outputs.extend(str(path) for path in operator_outputs if not path.is_file())
    success = (
        all(result["returncode"] == 0 for result in results)
        and not missing_outputs
        and not changed_sources
    )
    manifest.update(
        status="complete" if success else "failed",
        results=results,
        expected_complete_json_count=len(outputs),
        missing_outputs=missing_outputs,
        source_files_unchanged=not changed_sources,
        changed_source_files=changed_sources,
        finished_unix=time.time(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not success:
        raise SystemExit("Validation audit failed; inspect run_manifest.json and logs")
    print(f"COMPLETE trajectories={len(outputs)} sources_unchanged=true", flush=True)


if __name__ == "__main__":
    main()
