"""Run the frozen V3 baseline/E3/bounded-mean comparison end to end."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import three_way_experiment as old
from tools import v3_compare_experiment as exp


OUTPUT = exp.OUTPUT
DATA = exp.DATA
ARMS = list(exp.ARMS)
TEMPLATE = ROOT / "outputs/parallel_validation_scale_cov_mean_20260907/e3.yaml"
GPU_POLICY = {
    "training": "arms run sequentially; each arm uses every idle A40 in the requested pool, capped at 8",
    "global_batch_size": 8,
    "micro_batch_per_gpu": 1,
    "resume": "exact same world size and optimizer/scheduler state",
    "inference": "one checkpoint job per idle A40",
}


def _write_json(path: Path, value) -> None:
    old.write_json(path, value)


def _verify_dataset_hashes() -> dict:
    final = json.loads((DATA / "FINAL_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "final_hash_audit.json").read_text(encoding="utf-8"))
    reader = json.loads((DATA / "python_reader_verification.json").read_text(encoding="utf-8"))
    if not final.get("complete") or not audit.get("complete") or audit.get("errors"):
        raise ValueError("The V3 dataset has not passed its final hash audit")
    if final.get("training_started") is not False:
        raise ValueError("The V3 manifest does not identify a pre-training immutable dataset")
    expected_counts = {"train": 110, "validation": 30, "test": 30}
    if not reader.get("complete") or not reader.get("all_170_items_read") or reader.get("counts") != expected_counts:
        raise ValueError(f"The V3 Python reader audit is incomplete: {reader}")
    checked = 0
    total_bytes = 0
    sample_counts = {}
    for sample in final["samples"]:
        sample_id = sample["sample_id"]
        sample_dir = DATA / sample_id
        manifest = json.loads((sample_dir / "validation_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("sample_id") != sample_id:
            raise ValueError(f"Invalid validation manifest for {sample_id}")
        artifacts = manifest["artifacts"]
        if len(artifacts) != int(sample["artifact_count"]) or len(artifacts) != int(manifest["artifact_count"]):
            raise ValueError(f"Artifact count mismatch for {sample_id}")
        for artifact in artifacts:
            path = sample_dir / artifact["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"Artifact byte count changed: {path}")
            if old.sha256(path) != artifact["sha256"]:
                raise ValueError(f"Artifact hash changed: {path}")
            checked += 1
            total_bytes += path.stat().st_size
        sample_counts[sample_id] = len(artifacts)
    if checked != int(audit["immutable_artifacts"]) or checked != 6926:
        raise ValueError(f"Expected 6926 immutable artifacts, verified {checked}")
    return {
        "objects": len(final["samples"]),
        "artifacts_verified": checked,
        "bytes_verified": total_bytes,
        "sample_artifact_counts": sample_counts,
        "declared_fingerprint": reader["fingerprint"],
        "final_hash_audit_sha256": old.sha256(DATA / "final_hash_audit.json"),
        "final_manifest_sha256": old.sha256(DATA / "FINAL_DATASET_MANIFEST.json"),
        "split_manifest_sha256": old.sha256(DATA / "dataset_splits.json"),
    }


def _base_config() -> dict:
    import yaml

    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


def _new_config(template: dict, arm: str) -> dict:
    config = copy.deepcopy(template)
    config["experiment"].update(
        name=f"v3_compare_{arm}", seed=20260901, output_dir=str(OUTPUT / arm)
    )
    config["data"].update(
        root=str(DATA), cache_dir=str(OUTPUT / "data_cache"), precompute_cache=False
    )
    config["optimization"].update(
        max_steps=400,
        global_batch_size=8,
        micro_batch_per_gpu=1,
        lr_network=1e-3,
        lr_beta=1e-4,
        validate_every=20,
        checkpoint_every=20,
    )
    config["runtime"].update(amp=False, tensorboard=True)
    config["loss"].update(lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5)
    config["three_way"].update(
        kind="baseline" if arm == "baseline" else "e3",
        lr_gain=1e-4,
        selection_rule=(
            "baseline absolute log-variance + TV object macro"
            if arm == "baseline"
            else "common E3 normalized mean + normalized variance + TV object macro; bounded shape term excluded"
        ),
    )
    config["v3_compare"] = {
        "kind": arm,
        "shared_initial_state": str(OUTPUT / "shared_initial_state.pt"),
        "common_seed": 20260901,
        "from_scratch": True,
        "shape_gradient_budget": 0.05 if arm == "e3_mean005" else 0.0,
        "shape_gradient_eps": 1e-12,
        "shape_coefficient_cap": 1.0,
        "ramp_steps": 50,
        "gradient_limit_location": "per-sample normalized reconstruction q",
        "shape_mean_definition": "SmoothL1(H(q)/mean(H(q)), mu90/mean(mu90))",
        "lr_drop_after_steps": 200,
        "lr_network_after_drop": 1e-4,
        "lr_scalar_after_drop": 1e-5,
        "numerical_precision": "full FP32; AMP and TF32 disabled",
        "best_selection_uses_gt": False,
        "illumination_NA": 0.05,
        "detection_NA": 0.15,
        "real_system_NA_status": "deferred by user; no optical setting changed in this comparison",
    }
    return config


def _training_sources() -> list[Path]:
    return [
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "tools/v3_compare_checks.py",
        Path(__file__).resolve(),
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "utils/dataset_splits.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
        ROOT / "physics/psf_loader.py",
        ROOT / "tools/three_way_experiment.py",
        ROOT / "tools/three_way_physics.py",
        ROOT / "train_volume.py",
    ]


def prepare(*, resume: bool, gpu_request: str) -> dict:
    import numpy as np
    import torch
    import yaml
    from datasets.matlab_multivolume_dataset import load_dataset_index
    from train_volume import _model_from_config

    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"{OUTPUT} already exists; use --resume")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for path, digest in record["training_source_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen training source changed: {path}")
        for path, digest in record["config_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen experiment config changed: {path}")
        indexed, fingerprint = load_dataset_index(DATA)
        if fingerprint != record["dataset_fingerprint"]:
            raise ValueError("V3 dataset fingerprint changed")
        if old.sha256(OUTPUT / "shared_initial_state.pt") != record["shared_initial_file_sha256"]:
            raise ValueError("Shared initialization file changed")
        return record

    dataset_hashes = _verify_dataset_hashes()
    indexed, fingerprint = load_dataset_index(DATA)
    counts = {split: len(items) for split, items in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(f"Wrong indexed split counts: {counts}")
    if fingerprint != dataset_hashes["declared_fingerprint"]:
        raise ValueError("Current dataset fingerprint differs from the completed reader audit")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    template = _base_config()
    # Create the common network once.  Extra E3 gamma parameters are deliberately
    # absent here and are initialized to scalar zero per E3 arm.
    random.seed(20260901)
    np.random.seed(20260901)
    torch.manual_seed(20260901)
    common_model = _model_from_config(template)
    common_state = {name: value.detach().cpu().clone() for name, value in common_model.state_dict().items()}
    common_hash = old.state_hash(common_state)
    torch.save(
        {"format": "v3_compare_shared_network_initial_v1", "seed": 20260901,
         "model_state": common_state, "common_state_sha256": common_hash},
        OUTPUT / "shared_initial_state.pt",
    )
    del common_model, common_state

    configs = {}
    common_hashes = {}
    full_hashes = {}
    for arm in ARMS:
        config = _new_config(template, arm)
        config_path = OUTPUT / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        model = exp.build_model(config, initial=True)
        common = {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
        common_hashes[arm] = old.state_hash(common)
        full_hashes[arm] = old.state_hash(model.state_dict())
        if hasattr(model, "mean_gain_gamma") and float(model.mean_gain_gamma) != 0:
            raise ValueError(f"{arm} gamma did not initialize to zero")
        configs[str(config_path.resolve())] = old.sha256(config_path)
        del model
    if set(common_hashes.values()) != {common_hash}:
        raise ValueError(f"The three common initial states differ: {common_hashes}")

    sources = _training_sources()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    psf_path = Path(template["psf"]["H_path"])
    record = {
        "complete": True,
        "plan": "V3 baseline vs complete E3 vs E3 plus per-sample q mean-gradient cap 5%",
        "prepared_unix": time.time(),
        "dataset_root": str(DATA),
        "dataset_fingerprint": fingerprint,
        "dataset_hash_verification": dataset_hashes,
        "split_counts": counts,
        "split_objects": {
            "train": [f"P{i:02d}" for i in range(1, 12)],
            "validation": ["V01", "V02", "V03"],
            "test": ["T02", "T03", "T04"],
        },
        "training_source_hashes": source_hashes,
        "config_hashes": configs,
        "shared_initial_common_state_sha256": common_hash,
        "shared_initial_file_sha256": old.sha256(OUTPUT / "shared_initial_state.pt"),
        "arm_common_initial_hashes": common_hashes,
        "arm_full_initial_hashes": full_hashes,
        "psf_path": str(psf_path),
        "psf_sha256": old.sha256(psf_path),
        "gpu_policy": GPU_POLICY,
        "gpu_request_at_preflight": gpu_request,
        "max_steps": 400,
        "global_batch_size": 8,
        "nominal_train_subsets_seen": 3200,
        "nominal_dataset_passes": 3200 / 110,
        "validation_interval": 20,
        "lr_schedule": {"steps_1_200": {"network": 1e-3, "beta_gamma": 1e-4},
                        "steps_201_400": {"network": 1e-4, "beta_gamma": 1e-5}},
        "numerical_precision": "full FP32; AMP=false; cuDNN TF32=false; matmul TF32=false",
        "no_training_gt": True,
        "taylor_saved_sqrt_used_once": True,
    }
    _write_json(marker, record)
    print(json.dumps({"preflight_prepared": True, "output": str(OUTPUT),
                      "artifacts_verified": dataset_hashes["artifacts_verified"],
                      "common_initial_state": common_hash}, ensure_ascii=False), flush=True)
    return record


def inventory() -> dict[int, dict]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    result = {}
    for line in rows.splitlines():
        index, uuid, name, memory, utilization = [field.strip() for field in line.split(",")]
        result[int(index)] = {"uuid": uuid, "name": name, "memory": int(memory),
                              "utilization": int(utilization), "busy": uuid in busy}
    return result


def parse_gpus(value: str):
    if value.strip().lower() == "auto":
        return None
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field.isdigit() for field in fields):
        raise ValueError("--gpus must be auto or a comma-separated list of GPU indices")
    indices = [int(field) for field in fields]
    if len(indices) != len(set(indices)):
        raise ValueError("--gpus must not contain duplicates")
    return indices


def _pool(inv: dict[int, dict], request: str) -> list[int]:
    selected = parse_gpus(request)
    if selected is None:
        return sorted(index for index, info in inv.items() if "A40" in info["name"].upper())
    for index in selected:
        if index not in inv:
            raise ValueError(f"GPU {index} does not exist")
        if "A40" not in inv[index]["name"].upper():
            raise ValueError(f"GPU {index} is not an A40")
    return selected


def free_gpus(request: str = "auto", *, required: int | None = None) -> list[int]:
    inv = inventory()
    pool = _pool(inv, request)
    for _ in range(10):
        if not any(not inv[index]["busy"] and inv[index]["memory"] <= 1024 and inv[index]["utilization"] > 10 for index in pool):
            break
        time.sleep(1)
        inv = inventory()
        pool = _pool(inv, request)
    available = [index for index in pool if not inv[index]["busy"] and inv[index]["memory"] <= 1024 and inv[index]["utilization"] <= 10]
    needed = 1 if required is None else required
    if len(available) < needed:
        raise RuntimeError(f"Need {needed} idle A40 GPUs but found {available}; occupied cards will not be shared")
    return available if required is None else available[:required]


def training_gpus(arm: str, request: str, *, resume: bool) -> list[int]:
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        world_size = int(payload["world_size"])
        if int(payload["config"]["optimization"]["global_batch_size"]) != 8:
            raise ValueError("Resume checkpoint global batch size changed")
        del payload
        return free_gpus(request, required=world_size)
    return free_gpus(request)[:8]


def launch(worker: str, gpus: list[int], *, arm: str | None = None, resume: bool = False):
    if not gpus:
        raise ValueError("At least one idle GPU is required")
    inv = inventory()
    for index in gpus:
        info = inv[index]
        if info["busy"] or info["memory"] > 1024 or info["utilization"] > 10:
            raise RuntimeError(f"GPU {index} became occupied: {info}")
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(inv[index]["uuid"] for index in gpus),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SPECKLE_PHYSICAL_GPUS=",".join(map(str, gpus)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
    )
    arguments = [str(Path(__file__).resolve()), "--worker", worker]
    if arm:
        arguments += ["--experiment", arm]
    if resume:
        arguments.append("--resume")
    command = [sys.executable, *arguments]
    if worker == "train" and len(gpus) > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={len(gpus)}", *arguments]
    log = OUTPUT / "logs" / f"{worker}_{arm or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    record_path = OUTPUT / "jobs" / f"{worker}_{arm or 'all'}.json"
    _write_json(record_path, {"pid": process.pid, "worker": worker, "experiment": arm,
                              "gpus": gpus, "command": command, "started_unix": time.time(),
                              "status": "running", "log": str(log)})
    process.record_path = record_path
    print(json.dumps({"started": worker, "experiment": arm, "pid": process.pid,
                      "gpus": gpus, "log": str(log)}), flush=True)
    return process


def wait(process) -> None:
    code = process.wait()
    record = json.loads(process.record_path.read_text(encoding="utf-8"))
    record.update(exit_code=code, status="complete" if code == 0 else "failed", finished_unix=time.time())
    _write_json(process.record_path, record)
    if code:
        raise RuntimeError(f"Worker failed; inspect {record['log']}")


def _run_report() -> None:
    from tools.v3_compare_report import build_report

    build_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "infer", "report"), default="all")
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("train", "gpu-check", "infer"))
    args = parser.parse_args()
    try:
        parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    selected = ARMS if args.experiment == "all" else [args.experiment]
    if args.dry_run:
        print(json.dumps({"dry_run_no_changes": True, "output": str(OUTPUT), "dataset": str(DATA),
                          "experiments": selected, "steps": 400, "global_batch": 8,
                          "lr": {"1-200": [1e-3, 1e-4], "201-400": [1e-4, 1e-5]},
                          "gpu_policy": GPU_POLICY, "checkpoints": [200, "best", 400],
                          "primary_predictions": 564}, indent=2, ensure_ascii=False))
        return
    if args.worker:
        if args.worker == "train":
            exp.run_train(OUTPUT / f"{args.experiment}.yaml", resume=args.resume)
        elif args.worker == "gpu-check":
            from tools.v3_compare_checks import gpu_checks

            gpu_checks()
        else:
            from tools.v3_compare_evaluation import evaluate_arm

            evaluate_arm(args.experiment, resume=args.resume)
        return

    if args.stage in ("all", "preflight", "train"):
        prepare(resume=args.resume, gpu_request=args.gpus)
    if args.stage in ("all", "preflight", "train") and not (OUTPUT / "gpu_checks.json").exists():
        wait(launch("gpu-check", free_gpus(args.gpus, required=1)))
    if args.stage == "preflight":
        return
    if args.stage in ("all", "train"):
        gpu_check = json.loads((OUTPUT / "gpu_checks.json").read_text(encoding="utf-8"))
        if not gpu_check.get("passed"):
            raise RuntimeError("GPU/data numerical checks did not pass")
        for arm in selected:
            if args.resume and (OUTPUT / arm / "training_complete.json").exists():
                continue
            cards = training_gpus(arm, args.gpus, resume=args.resume)
            print(json.dumps({"training_resource_selection": arm, "gpus": cards,
                              "world_size": len(cards), "global_batch_size": 8}), flush=True)
            checkpoint_exists = (OUTPUT / arm / "checkpoint_last.pt").exists()
            wait(launch("train", cards, arm=arm, resume=args.resume and checkpoint_exists))
    if args.stage in ("all", "infer"):
        pending = [arm for arm in selected if not (args.resume and (OUTPUT / "evaluation" / arm / "complete.json").exists())]
        while pending:
            cards = free_gpus(args.gpus)
            batch = pending[: len(cards)]
            pending = pending[len(batch):]
            jobs = [launch("infer", [card], arm=arm, resume=args.resume) for card, arm in zip(cards, batch)]
            for job in jobs:
                wait(job)
    if args.stage in ("all", "report"):
        _run_report()


if __name__ == "__main__":
    main()

