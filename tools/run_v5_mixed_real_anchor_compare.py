"""Prepare and run the dated V5 no-P12 simulation/real anchor comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import three_way_experiment as old
from tools import v5_mixed_real_anchor_experiment as exp
from tools.v5_mixed_dataset import MixedExperimentDataset, MixedSixOneOneScheduler


OUTPUT = exp.OUTPUT
ARMS = exp.ARMS
GPU_POOL = tuple(range(6))
MIN_SIM_FREE_MIB = 12 * 1024
MIN_REAL_FREE_MIB = 30 * 1024


def write_json(path: Path, value: Any) -> None:
    old.write_json(path, value)


def config(arm: str) -> dict[str, Any]:
    source = yaml.safe_load(
        (ROOT / "outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100.yaml").read_text(encoding="utf-8")
    )
    value = copy.deepcopy(source)
    anchor = exp.ANCHORS[arm]
    value["experiment"].update(name=f"v5_mixed_real_{arm}", seed=20260901, output_dir=str(OUTPUT / arm))
    value["data"].update(
        root=str(exp.SIMULATION_DATA), real_root=str(exp.REAL_DATA),
        cache_dir=str(OUTPUT / "data_cache"), precompute_cache=False,
        var_feature_representation="sqrt", subset_policy="simulation frozen; real deterministic partition",
    )
    value["model"].update(reconstruction_anchor=anchor, activation_checkpoint_segments=True)
    value["optimization"].update(
        max_steps=600, global_batch_size=8, micro_batch_per_gpu=1,
        lr_network=1e-3, lr_beta=1e-4, validate_every=20, checkpoint_every=20,
    )
    value["runtime"].update(
        amp=False, tensorboard=True, operator_phase_chunk_size=32,
        max_peak_memory_gib=42, required_free_reserve_gib=4,
    )
    value["noise"].update(alpha_noise=0.0, sigma_read=0.0)
    value["loss"].update(lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5)
    value["three_way"].update(
        kind="e3", lr_gain=1e-4,
        selection_rule="simulation validation object-macro normalized mean + normalized variance + weighted TV; no GT/test/real selection",
    )
    value["v3_compare"] = {
        "kind": arm,
        "shared_initial_state": str(exp.SHARED_INITIAL_STATE),
        "common_seed": 20260901,
        "from_scratch": True,
        "shape_gradient_budget": 1.0,
        "shape_gradient_eps": 1e-12,
        "shape_coefficient_cap": 1.0,
        "ramp_steps": 50,
        "lr_drop_after_steps": 200,
        "lr_network_after_drop": 1e-4,
        "lr_scalar_after_drop": 1e-5,
    }
    value["v5_mixed"] = {
        "kind": arm,
        "shared_initial_state": str(exp.SHARED_INITIAL_STATE),
        "simulation_manifest_root": str(exp.SIMULATION_DATA),
        "real_final_manifest": str(exp.REAL_DATA / "final_manifest.json"),
        "p12_in_training": False,
        "simulation_train_objects": [f"P{i:02d}" for i in range(1, 12)],
        "real_train_fields": ["45", "55"],
        "simulation_per_batch": 6,
        "field45_per_batch": 1,
        "field55_per_batch": 1,
        "shape_gradient_budget": 1.0,
        "ramp_steps": 50,
        "lr_drop_after_steps": 200,
        "spatial_policy": "real full field 1029x1421; no resize; no tiles",
        "beta_anchor": anchor,
        "beta_cache_key": "sample,subset,anchor,domain,shape",
    }
    return value


def source_paths() -> list[Path]:
    return [
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "models/configurable_anchor_lfm_net.py",
        ROOT / "tools/mixed_resolution_lfm.py",
        ROOT / "tools/v5_mixed_real_data.py",
        ROOT / "tools/v5_mixed_dataset.py",
        ROOT / "tools/v5_mixed_real_anchor_experiment.py",
        Path(__file__).resolve(),
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
    ]


def validate_real_artifacts(final_path: Path) -> dict[str, Any]:
    record = json.loads(final_path.read_text(encoding="utf-8"))
    if not record.get("complete") or int(record.get("artifact_count", -1)) != 140:
        raise ValueError("Real final manifest is incomplete")
    for relative, expected in record["artifacts"].items():
        path = exp.REAL_DATA / relative
        if path.stat().st_size != int(expected["size"]) or old.sha256(path) != expected["sha256"]:
            raise ValueError(f"Real training artifact changed: {path}")
    for field, paths in record["source_rectified_hashes"].items():
        if len(paths) != 100:
            raise ValueError(f"Field {field} does not retain 100 source frame hashes")
        for path, digest in paths.items():
            if old.sha256(path) != digest:
                raise ValueError(f"Rectified source frame changed: {path}")
    return record


def prepare(*, resume: bool) -> dict[str, Any]:
    data_final = exp.REAL_DATA / "final_manifest.json"
    if not data_final.is_file():
        raise RuntimeError("The 40 real RL3 volumes are not finalized")
    real_record = validate_real_artifacts(data_final)
    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"V5 preflight already exists; use --resume: {marker}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for path, digest in {**record["source_hashes"], **record["config_hashes"]}.items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen V5 input changed: {path}")
        return record
    OUTPUT.mkdir(parents=True, exist_ok=True)
    probe = MixedExperimentDataset(
        exp.SIMULATION_DATA, "train", real_root=exp.REAL_DATA,
        cache_dir=OUTPUT / "data_cache", var_feature_representation="sqrt",
    )
    validation = MixedExperimentDataset(
        exp.SIMULATION_DATA, "validation", real_root=exp.REAL_DATA,
        cache_dir=OUTPUT / "data_cache", var_feature_representation="sqrt",
    )
    test = MixedExperimentDataset(
        exp.SIMULATION_DATA, "test", real_root=exp.REAL_DATA,
        cache_dir=OUTPUT / "data_cache", var_feature_representation="sqrt",
    )
    if (len(probe), len(validation), len(test)) != (130, 30, 30):
        raise ValueError("Unexpected V5 split counts")
    if len({probe.dataset_fingerprint, validation.dataset_fingerprint, test.dataset_fingerprint}) != 1:
        raise ValueError("V5 split fingerprints disagree")
    config_hashes: dict[str, str] = {}
    state_hashes: dict[str, str] = {}
    schedule_hashes: dict[str, str] = {}
    for arm in ARMS:
        current = config(arm)
        exp.validate_config(current)
        path = OUTPUT / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
        config_hashes[str(path.resolve())] = old.sha256(path)
        model = exp.build_model(current, initial=True)
        common = {name: tensor for name, tensor in model.state_dict().items() if name != "mean_gain_gamma"}
        state_hashes[arm] = old.state_hash(common)
        scheduler = MixedSixOneOneScheduler(130, 8, 20260901)
        schedule = [scheduler.next_batch() for _ in range(600)]
        if any(len(batch) != 8 or sum(index < 110 for index in batch) != 6 or sum(110 <= index < 120 for index in batch) != 1 or sum(index >= 120 for index in batch) != 1 for batch in schedule):
            raise AssertionError("Mixed schedule violated 6+1+1")
        schedule_hashes[arm] = hashlib.sha256(json.dumps(schedule).encode("utf-8")).hexdigest()
    initial = torch.load(exp.SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
    if set(state_hashes.values()) != {initial["common_state_sha256"]} or len(set(schedule_hashes.values())) != 1:
        raise ValueError("Arms do not share initialization and schedule")
    sources = source_paths()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    record = {
        "complete": True,
        "plan": "P12 excluded; P01-P11 plus full-field real 45/55; Taylor versus Mean anchor; 600 steps",
        "output": str(OUTPUT),
        "prepared_unix": time.time(),
        "dataset_fingerprint": probe.dataset_fingerprint,
        "split_counts": {"train_candidates": 130, "validation": 30, "test": 30},
        "mixed_batch": {"simulation": 6, "field45": 1, "field55": 1},
        "config_hashes": config_hashes,
        "source_hashes": source_hashes,
        "real_final_manifest": str(data_final),
        "real_final_manifest_sha256": old.sha256(data_final),
        "real_artifact_count": real_record["artifact_count"],
        "psf_sha256": old.sha256(Path(config(ARMS[0])["psf"]["H_path"])),
        "shared_initial_file_sha256": old.sha256(exp.SHARED_INITIAL_STATE),
        "shared_initial_state_sha256": initial["common_state_sha256"],
        "arm_initial_state_sha256": state_hashes,
        "sample_schedule_sha256": next(iter(schedule_hashes.values())),
        "max_steps_per_arm": 600,
        "checkpoint_steps": [200, 400, 600],
        "lr_schedule": {
            "1-200": {"network": 1e-3, "beta_gain": 1e-4},
            "201-600": {"network": 1e-4, "beta_gain": 1e-5},
        },
        "real_spatial_policy": "full field 1029x1421; no resize; no tiling",
        "real_results_role": "training-field diagnostic, never independent test",
    }
    write_json(marker, record)
    return record


def inventory() -> dict[int, dict[str, Any]]:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    )
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    busy: dict[str, list[int]] = {}
    for line in process_rows.splitlines():
        if line.strip():
            uuid, pid = [field.strip() for field in line.split(",")]
            busy.setdefault(uuid, []).append(int(pid))
    result = {}
    for line in gpu_rows.splitlines():
        index, uuid, name, total, used, free, utilization = [field.strip() for field in line.split(",")]
        result[int(index)] = {
            "uuid": uuid, "name": name, "total": int(total), "used": int(used), "free": int(free),
            "utilization": int(utilization), "pids": busy.get(uuid, []), "busy": uuid in busy,
        }
    return result


def ordered_cards(*, resume_checkpoint: Path | None = None) -> list[int]:
    current = inventory()
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if int(payload["world_size"]) != 6:
            raise ValueError("V5 resume checkpoint was not made with six GPUs")
    available = [index for index in GPU_POOL if index in current and "A40" in current[index]["name"].upper() and current[index]["free"] >= MIN_SIM_FREE_MIB]
    available.sort(key=lambda index: (current[index]["busy"], current[index]["utilization"], -current[index]["free"]))
    real = [index for index in available if current[index]["free"] >= MIN_REAL_FREE_MIB]
    if len(available) != 6 or len(real) < 2:
        raise RuntimeError(f"Need all six A40s and two cards with 30 GiB free; inventory={current}")
    first = real[:2]
    return first + [index for index in available if index not in first]


def launch_train(arm: str, *, resume: bool) -> subprocess.Popen:
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    cards = ordered_cards(resume_checkpoint=checkpoint if checkpoint.exists() else None)
    current = inventory()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(current[index]["uuid"] for index in cards),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", SPECKLE_PHYSICAL_GPUS=",".join(map(str, cards)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        V5_MIXED_OUTPUT=str(OUTPUT), V5_REAL_DATA=str(exp.REAL_DATA),
    )
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=6",
        str(Path(__file__).resolve()), "--worker", "train", "--experiment", arm,
    ]
    if resume and checkpoint.exists():
        command.append("--resume")
    log = OUTPUT / "logs" / f"train_{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    process._v5_log_handle = handle
    process._v5_job = OUTPUT / "jobs" / f"train_{arm}.json"
    write_json(process._v5_job, {
        "status": "running", "pid": process.pid, "arm": arm, "gpus_in_rank_order": cards,
        "real_sample_ranks": {"rank0": cards[0], "rank1": cards[1]},
        "inventory_at_launch": {str(index): current[index] for index in cards},
        "started_unix": time.time(), "log": str(log), "command": command,
    })
    return process


def wait_job(process: subprocess.Popen) -> None:
    code = process.wait()
    process._v5_log_handle.close()
    record = json.loads(process._v5_job.read_text(encoding="utf-8"))
    record.update(status="complete" if code == 0 else "failed", exit_code=code, finished_unix=time.time())
    write_json(process._v5_job, record)
    if code:
        raise RuntimeError(f"Training failed; inspect {record['log']}")


def worker(arm: str, resume: bool) -> None:
    exp.run_train(OUTPUT / f"{arm}.yaml", resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "all"), default="all")
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("train",))
    args = parser.parse_args()
    if args.worker:
        worker(args.experiment, args.resume)
        return
    prepare(resume=args.resume)
    if args.stage == "preflight":
        return
    selected = ARMS if args.experiment == "all" else (args.experiment,)
    for arm in selected:
        complete = OUTPUT / arm / "training_complete.json"
        if args.resume and complete.exists():
            continue
        process = launch_train(arm, resume=args.resume)
        wait_job(process)


if __name__ == "__main__":
    main()
