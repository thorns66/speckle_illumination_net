"""Run the four E3 mean/depth-protection continuations end to end."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import mean_depth_experiment as exp
from tools import run_v3_compare as v3run
from tools import three_way_experiment as old

OUTPUT = exp.OUTPUT
ARMS = list(exp.ARMS)
GPU_POLICY = {
    "training": "arms sequential; every idle A40 in requested pool, capped at global batch 8",
    "evaluation": "one arm per idle A40 in parallel batches",
    "global_batch_size": 8,
    "occupied_gpu_policy": "never share a GPU with an existing compute process",
    "resume": "same world size, optimizer, scheduler, RNG, and sampler state",
}


def write_json(path: Path, value) -> None:
    old.write_json(path, value)


def training_sources() -> list[Path]:
    return [
        ROOT / "tools/mean_depth_experiment.py",
        ROOT / "tools/mean_depth_checks.py",
        Path(__file__).resolve(),
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "utils/dataset_splits.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "models/output_parameterizations.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
        ROOT / "physics/psf_loader.py",
        ROOT / "tools/three_way_experiment.py",
        ROOT / "tools/three_way_physics.py",
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "train_volume.py",
    ]


def new_config(source: dict, arm: str, source_hash: str) -> dict:
    config = copy.deepcopy(source)
    config["experiment"].update(
        name=f"mean_depth_{arm}", seed=20260901, output_dir=str(OUTPUT / arm)
    )
    config["data"].update(
        root=str(exp.DATA), cache_dir=str(exp.SOURCE_OUTPUT / "data_cache"), precompute_cache=False
    )
    config["optimization"].update(
        max_steps=200,
        global_batch_size=8,
        micro_batch_per_gpu=1,
        lr_network=1e-4,
        lr_beta=1e-5,
        validate_every=20,
        checkpoint_every=20,
    )
    config["three_way"].update(
        kind="e3",
        lr_gain=1e-5,
        selection_rule="common original E3 validation mean + normalized variance + TV object macro; no GT",
    )
    config["runtime"].update(amp=False, tensorboard=True)
    config["loss"].update(lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5)
    use_mean = arm in ("r1_mean005", "r3_mean005_depth")
    use_depth = arm in ("r2_depth", "r3_mean005_depth")
    config["mean_depth"] = {
        "kind": arm,
        "source_checkpoint": str(exp.SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_hash,
        "source_total_steps": 400,
        "additional_steps": 200,
        "reset_optimizer": True,
        "preserve_source_gamma": True,
        "use_mean_shape": use_mean,
        "mean_gradient_budget": 0.05 if use_mean else 0.0,
        "mean_ramp_steps": 50,
        "gradient_eps": 1e-12,
        "use_depth_protection": use_depth,
        "depth_window_pixels": [9, 17],
        "depth_window_stride": 4,
        "z_spacing_um": 10.0,
        "depth_tolerance_um": 1.0,
        "reference_floor_fraction": 0.01,
        "lambda_depth": 1.0,
        "depth_gradient_budget": 0.25 if use_depth else 0.0,
        "depth_ramp_steps": 50,
        "teacher": "frozen V3 E3 step-400 final; same ten-frame input; no GT/holdout target",
        "depth_metric": "two-scale local native-Z W1 trust region with squared hinge beyond 1 um",
        "mean_budget_denominator": "original normalized-variance q-gradient only",
        "best_selection_uses_mean_shape_or_depth": False,
        "numerical_precision": "full FP32; AMP and TF32 disabled",
    }
    return config


def prepare(*, resume: bool, gpu_request: str) -> dict:
    import torch
    import yaml
    from datasets.matlab_multivolume_dataset import load_dataset_index

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
                raise ValueError(f"Frozen config changed: {path}")
        if old.sha256(exp.SOURCE_CHECKPOINT) != record["source_checkpoint_sha256"]:
            raise ValueError("Source E3 checkpoint changed")
        _index, fingerprint = load_dataset_index(exp.DATA)
        if fingerprint != record["dataset_fingerprint"]:
            raise ValueError("Dataset fingerprint changed")
        return record

    prior_acceptance = json.loads((exp.SOURCE_OUTPUT / "final_acceptance.json").read_text())
    if not prior_acceptance.get("passed") or prior_acceptance.get("primary_predictions") != 564:
        raise ValueError("The source V3 comparison has not passed final acceptance")
    dataset_hashes = v3run._verify_dataset_hashes()
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(values) for split, values in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(f"Unexpected V3 split counts: {counts}")
    if fingerprint != dataset_hashes["declared_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from immutable hash audit")

    source_hash = old.sha256(exp.SOURCE_CHECKPOINT)
    checkpoint = torch.load(exp.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["completed_steps"]) != 400 or checkpoint["config"]["v3_compare"]["kind"] != "e3":
        raise ValueError("Source is not the actual V3 E3 final step-400 checkpoint")
    source_config = checkpoint["config"]
    source_gamma = float(checkpoint["model_state"]["mean_gain_gamma"])

    OUTPUT.mkdir(parents=True, exist_ok=False)
    configs = {}
    initial_hashes = {}
    network_hashes = {}
    for arm in ARMS:
        config = new_config(source_config, arm, source_hash)
        path = OUTPUT / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        model = exp.build_model(config, initial=True)
        initial_hashes[arm] = old.state_hash(model.state_dict())
        network_hashes[arm] = old.state_hash({k: v for k, v in model.state_dict().items() if k != "mean_gain_gamma"})
        if float(model.mean_gain_gamma) != source_gamma:
            raise ValueError(f"{arm} did not preserve the source gamma")
        configs[str(path.resolve())] = old.sha256(path)
        del model
    if len(set(initial_hashes.values())) != 1 or len(set(network_hashes.values())) != 1:
        raise ValueError("The four continuation arms do not share identical weights")

    sources = training_sources()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    record = {
        "complete": True,
        "prepared_unix": time.time(),
        "plan": "2x2 continuation: optional bounded mean shape x stable bounded local depth protection",
        "dataset_root": str(exp.DATA),
        "dataset_fingerprint": fingerprint,
        "dataset_hash_verification": dataset_hashes,
        "split_counts": counts,
        "split_objects": {"train": [f"P{i:02d}" for i in range(1, 12)], "validation": ["V01", "V02", "V03"], "test": ["T02", "T03", "T04"]},
        "source_checkpoint": str(exp.SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_hash,
        "source_completed_steps": 400,
        "source_gamma": source_gamma,
        "all_arms_preserve_source_gamma": True,
        "initial_state_hashes": initial_hashes,
        "network_initial_hashes": network_hashes,
        "training_source_hashes": source_hashes,
        "config_hashes": configs,
        "prior_acceptance_sha256": old.sha256(exp.SOURCE_OUTPUT / "final_acceptance.json"),
        "psf_path": str(source_config["psf"]["H_path"]),
        "psf_sha256": old.sha256(source_config["psf"]["H_path"]),
        "gpu_policy": GPU_POLICY,
        "gpu_request_at_preflight": gpu_request,
        "additional_steps": 200,
        "final_total_steps": 600,
        "global_batch_size": 8,
        "validation_interval": 20,
        "learning_rates": {"network": 1e-4, "beta": 1e-5, "gamma": 1e-5},
        "no_training_gt": True,
        "teacher_uses_input_only": True,
        "numerical_precision": "full FP32; AMP=false; cuDNN TF32=false; matmul TF32=false",
    }
    write_json(marker, record)
    print(json.dumps({"preflight_prepared": True, "output": str(OUTPUT), "source_step": 400, "arms_equal": True, "dataset_artifacts": dataset_hashes["artifacts_verified"]}, ensure_ascii=False), flush=True)
    return record


def inventory() -> dict[int, dict]:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    result = {}
    for line in rows.splitlines():
        index, uuid, name, memory, utilization = [part.strip() for part in line.split(",")]
        result[int(index)] = {"uuid": uuid, "name": name, "memory": int(memory), "utilization": int(utilization), "busy": uuid in busy}
    return result


def parse_gpus(value: str):
    if value.strip().lower() == "auto":
        return None
    fields = [part.strip() for part in value.split(",")]
    if not fields or any(not part.isdigit() for part in fields):
        raise ValueError("--gpus must be auto or comma-separated GPU indices")
    values = [int(part) for part in fields]
    if len(values) != len(set(values)):
        raise ValueError("--gpus contains a duplicate")
    return values


def pool(inv: dict[int, dict], request: str) -> list[int]:
    selected = parse_gpus(request)
    if selected is None:
        return sorted(i for i, info in inv.items() if "A40" in info["name"].upper())
    for i in selected:
        if i not in inv or "A40" not in inv[i]["name"].upper():
            raise ValueError(f"GPU {i} is unavailable or not an A40")
    return selected


def free_gpus(request: str = "auto", required: int | None = None) -> list[int]:
    inv = inventory()
    choices = pool(inv, request)
    available = [i for i in choices if not inv[i]["busy"] and inv[i]["memory"] <= 1024 and inv[i]["utilization"] <= 10]
    needed = 1 if required is None else required
    if len(available) < needed:
        raise RuntimeError(f"Need {needed} idle A40 GPUs, found {available}; occupied GPUs will not be shared")
    return available if required is None else available[:required]


def training_gpus(arm: str, request: str, resume: bool) -> list[int]:
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        count = int(state["world_size"])
        if int(state["config"]["optimization"]["global_batch_size"]) != 8:
            raise ValueError("Resume checkpoint has the wrong global batch size")
        del state
        return free_gpus(request, count)
    return free_gpus(request)[:8]


def launch(worker: str, gpus: list[int], arm: str | None = None, resume: bool = False):
    if not gpus:
        raise ValueError("A worker requires at least one GPU")
    inv = inventory()
    for gpu in gpus:
        if gpu not in inv or inv[gpu]["busy"] or inv[gpu]["memory"] > 1024:
            raise RuntimeError(f"GPU {gpu} became occupied: {inv.get(gpu)}")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(inv[i]["uuid"] for i in gpus),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SPECKLE_PHYSICAL_GPUS=",".join(map(str, gpus)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
    )
    args = [str(Path(__file__).resolve()), "--worker", worker]
    if arm:
        args += ["--experiment", arm]
    if resume:
        args.append("--resume")
    command = [sys.executable, *args]
    if worker == "train" and len(gpus) > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}", *args]
    log = OUTPUT / "logs" / f"{worker}_{arm or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    marker = OUTPUT / "jobs" / f"{worker}_{arm or 'all'}.json"
    write_json(marker, {"pid": process.pid, "worker": worker, "experiment": arm, "gpus": gpus, "command": command, "log": str(log), "status": "running", "started_unix": time.time()})
    process.marker = marker
    print(json.dumps({"started": worker, "experiment": arm, "pid": process.pid, "gpus": gpus, "log": str(log)}), flush=True)
    return process


def wait(process) -> None:
    code = process.wait()
    record = json.loads(process.marker.read_text())
    record.update(status="complete" if code == 0 else "failed", exit_code=code, finished_unix=time.time())
    write_json(process.marker, record)
    if code:
        raise RuntimeError(f"Worker failed; inspect {record['log']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "evaluate", "report"), default="all")
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("gpu-check", "train", "evaluate"))
    args = parser.parse_args()
    selected = ARMS if args.experiment == "all" else [args.experiment]
    parse_gpus(args.gpus)
    if args.dry_run:
        print(json.dumps({"output": str(OUTPUT), "arms": selected, "source": str(exp.SOURCE_CHECKPOINT), "source_total_steps": 400, "additional_steps": 200, "final_total_steps": 600, "global_batch": 8, "mean_budget": 0.05, "depth_windows": [9, 17], "depth_tolerance_um": 1.0, "lambda_depth": 1.0, "depth_gradient_budget": 0.25, "depth_ramp_steps": 50, "gpu_policy": GPU_POLICY}, indent=2))
        return
    if args.worker:
        if args.worker == "gpu-check":
            from tools.mean_depth_checks import gpu_checks
            gpu_checks()
        elif args.worker == "train":
            exp.run_train(OUTPUT / f"{args.experiment}.yaml", resume=args.resume)
        else:
            from tools.mean_depth_evaluation import evaluate_arm
            evaluate_arm(args.experiment, resume=args.resume)
        return
    if args.stage in ("all", "preflight", "train"):
        prepare(resume=args.resume, gpu_request=args.gpus)
    if args.stage in ("all", "preflight", "train") and not (OUTPUT / "gpu_checks.json").exists():
        wait(launch("gpu-check", free_gpus(args.gpus, 1)))
    if args.stage == "preflight":
        return
    if args.stage in ("all", "train"):
        if not json.loads((OUTPUT / "gpu_checks.json").read_text()).get("passed"):
            raise RuntimeError("GPU preflight did not pass")
        for arm in selected:
            if args.resume and (OUTPUT / arm / "training_complete.json").exists():
                continue
            gpus = training_gpus(arm, args.gpus, args.resume)
            print(json.dumps({"training_resource_selection": arm, "gpus": gpus, "world_size": len(gpus), "global_batch": 8}), flush=True)
            wait(launch("train", gpus, arm, resume=args.resume and (OUTPUT / arm / "checkpoint_last.pt").exists()))
    if args.stage in ("all", "evaluate"):
        pending = [arm for arm in selected if not (args.resume and (OUTPUT / "evaluation" / arm / "complete.json").exists())]
        while pending:
            available = free_gpus(args.gpus)
            batch, pending = pending[:len(available)], pending[len(available):]
            jobs = [launch("evaluate", [gpu], arm, resume=args.resume) for arm, gpu in zip(batch, available)]
            for job in jobs:
                wait(job)
    if args.stage in ("all", "report"):
        from tools.mean_depth_report import build_report
        build_report()


if __name__ == "__main__":
    main()
