"""Run E3+100% step-600 continuation and from-scratch no-Set ablation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import numpy as np
import torch
import yaml

import training.multivolume_trainer as trainer
from datasets.matlab_multivolume_dataset import load_dataset_index
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean100_ablation_experiment as exp
from tools import three_way_experiment as old
from tools import run_v3_compare as original_runner
from tools import v3_compare_checks as original_checks
from tools import v3_compare_experiment as base_exp


OUTPUT = exp.OUTPUT
ARMS = exp.ARMS
BASELINE_RECORD = ROOT / "CURRENT_BASELINE.json"
SOURCE_CHECKPOINT = exp.BASELINE_ARM / "checkpoint_last.pt"
SOURCE_BEST = exp.BASELINE_ARM / "checkpoint_best.pt"
ORIGINAL_GPUS = (1, 2, 3, 4, 5)


def write_json(path: Path, value) -> None:
    old.write_json(path, value)


def config_for(arm: str) -> dict:
    source = yaml.safe_load((exp.BASELINE_OUTPUT / "e3_mean100.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    config["experiment"]["output_dir"] = str(OUTPUT / arm)
    config["optimization"]["max_steps"] = 600
    if arm == "noset600":
        config["experiment"]["name"] = "v3_e3_mean100_noset600"
        config["ablation"]["use_set_branch"] = False
    return config


def source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        ROOT / "tools/mean100_ablation_experiment.py",
        ROOT / "tools/mean100_ablation_evaluation.py",
        ROOT / "tools/mean100_ablation_report.py",
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "tools/v3_compare_evaluation.py",
        ROOT / "tools/v3_compare_local_audit.py",
        ROOT / "tools/v3_compare_report.py",
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "models/gated_fusion.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
        ROOT / "physics/psf_loader.py",
        ROOT / "tools/three_way_experiment.py",
        ROOT / "tools/priority_validation_analysis.py",
        ROOT / "train_volume.py",
    ]


def _stage_continuation() -> dict:
    destination = OUTPUT / "extend600"
    destination.mkdir(parents=True, exist_ok=True)
    copies = {
        SOURCE_CHECKPOINT: destination / "checkpoint_last.pt",
        SOURCE_BEST: destination / "checkpoint_best.pt",
    }
    for source, target in copies.items():
        shutil.copy2(source, target)
    shutil.copy2(SOURCE_CHECKPOINT, destination / "checkpoint_step_000400.pt")
    for name in ("training_metrics.csv", "validation_metrics.csv"):
        shutil.copy2(exp.BASELINE_ARM / name, destination / name)
    for pattern in (
        "gradient_diagnostics_rank*.jsonl",
        "learning_rate_rank*.jsonl",
        "validation_step_*.json",
    ):
        for source in exp.BASELINE_ARM.glob(pattern):
            shutil.copy2(source, destination / source.name)
    return {
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_sha256": old.sha256(SOURCE_CHECKPOINT),
        "staged_sha256": old.sha256(destination / "checkpoint_last.pt"),
        "best_source_sha256": old.sha256(SOURCE_BEST),
        "world_size": 5,
        "physical_gpus": list(ORIGINAL_GPUS),
    }


def prepare(*, resume: bool, gpu_request: str) -> dict:
    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"{OUTPUT} exists; use --resume")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for group in ("training_source_hashes", "config_hashes", "reference_hashes"):
            for name, digest in record[group].items():
                if old.sha256(name) != digest:
                    raise ValueError(f"Frozen {group} entry changed: {name}")
        _indexed, fingerprint = load_dataset_index(exp.DATA)
        if fingerprint != record["dataset_fingerprint"]:
            raise ValueError("Dataset fingerprint changed")
        return record

    baseline = json.loads(BASELINE_RECORD.read_text(encoding="utf-8"))
    if baseline["baseline_id"] != "e3_mean100_v3_400_20260908_run01":
        raise ValueError("The registered baseline changed")
    if old.sha256(SOURCE_CHECKPOINT) != baseline["checkpoints"]["final"]["sha256"]:
        raise ValueError("Registered final checkpoint hash mismatch")
    dataset = original_runner._verify_dataset_hashes()
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(rows) for split, rows in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(f"Unexpected split counts: {counts}")
    if fingerprint != dataset["declared_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from its immutable audit")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    configs = {}
    for arm in ARMS:
        config = config_for(arm)
        exp.validate_config(config)
        path = OUTPUT / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[str(path)] = old.sha256(path)
    staged = _stage_continuation()

    source_state = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    extend_config = yaml.safe_load((OUTPUT / "extend600.yaml").read_text(encoding="utf-8"))
    if trainer._resume_contract(source_state["config"]) != trainer._resume_contract(extend_config):
        raise ValueError("Step-400 source is not an exact-resume match for extend600")
    if int(source_state["completed_steps"]) != 400 or int(source_state["world_size"]) != 5:
        raise ValueError("Unexpected continuation checkpoint state")
    lrs = [float(group["lr"]) for group in source_state["optimizer_state"]["param_groups"]]
    if lrs != [1e-4, 1e-5, 1e-5]:
        raise ValueError(f"Step-400 optimizer does not contain the expected low LRs: {lrs}")

    initial = torch.load(exp.SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
    no_set_config = yaml.safe_load((OUTPUT / "noset600.yaml").read_text(encoding="utf-8"))
    no_set_model = exp.build_model(no_set_config, initial=True)
    common = {name: value for name, value in no_set_model.state_dict().items() if name != "mean_gain_gamma"}
    if old.state_hash(common) != initial["common_state_sha256"]:
        raise ValueError("No-Set common initialization differs")
    contract = trainer._configure_trainable_parameters(no_set_model)
    required_prefixes = ("set_encoder.", "fusion.set_projection.", "fusion.gate.")
    if not all(
        (not parameter.requires_grad)
        for name, parameter in no_set_model.named_parameters()
        if name.startswith(required_prefixes) or name == "fusion.raw_alpha"
    ):
        raise ValueError("Set-specific parameters were not frozen")
    del no_set_model, source_state, initial

    sources = source_paths()
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError([str(path) for path in sources if not path.is_file()])
    source_hashes = {str(path): old.sha256(path) for path in sources}
    references = [
        BASELINE_RECORD,
        SOURCE_CHECKPOINT,
        SOURCE_BEST,
        exp.BASELINE_OUTPUT / "e3_mean100.yaml",
        exp.BASELINE_OUTPUT / "final_acceptance.json",
        exp.BASELINE_OUTPUT / "evaluation/e3_mean100/complete.json",
        exp.SHARED_INITIAL_STATE,
    ]
    record = {
        "complete": True,
        "prepared_unix": time.time(),
        "output": str(OUTPUT),
        "dataset_root": str(exp.DATA),
        "dataset_fingerprint": fingerprint,
        "dataset_hash_verification": dataset,
        "split_counts": counts,
        "config_hashes": configs,
        "training_source_hashes": source_hashes,
        "reference_hashes": {str(path): old.sha256(path) for path in references},
        "baseline_record": str(BASELINE_RECORD),
        "continuation": staged,
        "no_set_parameter_contract": contract,
        "max_steps": 600,
        "global_batch_size": 8,
        "validation_interval": 20,
        "checkpoint_roles": {
            "extend600": ["baseline_step400", "best", "step600"],
            "noset600": ["best", "step400", "step600"],
        },
        "gpu_request": gpu_request,
        "precision": "full FP32; AMP and TF32 disabled",
        "no_training_gt": True,
    }
    write_json(marker, record)
    snapshot = OUTPUT / "source_snapshot"
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps({"preflight_prepared": True, "output": str(OUTPUT)}, ensure_ascii=False), flush=True)
    return record


def _gpu_item(config: dict, device: torch.device):
    previous = original_checks.exp
    original_checks.exp = exp
    try:
        return original_checks._gpu_item(config, device)
    finally:
        original_checks.exp = previous


def gpu_checks() -> dict:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    exp.configure_precision()
    config = yaml.safe_load((OUTPUT / "noset600.yaml").read_text(encoding="utf-8"))
    model = exp.build_model(config, initial=True).to(device).train()
    parameter_contract = trainer._configure_trainable_parameters(model)
    optimizer = exp.optimizer(model, config)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    frozen = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
    if optimized & frozen:
        raise ValueError("Frozen Set parameters entered the optimizer")
    operator = exp.load_operator(config, device)
    item = _gpu_item(config, device)
    beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
    model.eval()
    with torch.inference_mode():
        first, _ = exp.forward(model, item, operator, beta0, config=config)
        changed_item = dict(item)
        changed_item["residual_frames"] = torch.randn_like(item["residual_frames"]) * 1000
        changed, _ = exp.forward(model, changed_item, operator, beta0, config=config)
    residual_error = float(
        (first.reconstruction - changed.reconstruction).norm()
        / first.reconstruction.norm().clamp_min(1e-30)
    )
    if residual_error != 0.0:
        raise ValueError(f"No-Set output still depends on residual frames: {residual_error}")

    model.train()
    base_exp._STATE.update(completed_steps=48, micro_call=0, phase="gpu_check", evaluation=False)
    check_config = copy.deepcopy(config)
    check_config["experiment"]["output_dir"] = "/tmp/mean100_ablation_gpu_check/noset600"
    output, _ = exp.forward(model, item, operator, beta0, config=check_config)
    loss = exp.loss(
        output,
        item,
        operator,
        TaylorH2VarianceModel(operator, **config["noise"]),
        check_config,
    )
    metrics = loss.scalar_metrics()
    if not math.isfinite(float(loss.total)) or float(metrics["mean_shape_gradient_ratio"]) > 0.98001:
        raise ValueError("Step-49 mean-gradient bound failed")
    loss.total.backward()
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError("Frozen Set parameters received gradients")

    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    source_model = exp.build_model(source["config"], initial=False).to(device).eval()
    source_model.load_state_dict(source["model_state"], strict=True)
    source_operator = exp.load_operator(source["config"], device)
    source_item = _gpu_item(source["config"], device)
    source_beta0 = trainer._analytic_beta0(source_operator, source_item["f_var"], source_item["input_mean"])
    with torch.inference_mode():
        reproduced, _ = exp.forward(
            source_model, source_item, source_operator, source_beta0, config=source["config"]
        )
    saved = np.load(
        exp.BASELINE_OUTPUT / "evaluation/e3_mean100/final/V01_subset_01/reconstruction.npy",
        allow_pickle=False,
    )
    prediction = reproduced.reconstruction[0, 0].float().cpu().numpy()
    reproduction_error = float(np.linalg.norm(prediction - saved) / max(np.linalg.norm(saved), 1e-30))
    if reproduction_error > 1e-4:
        raise ValueError(f"Step-400 baseline reproduction failed: {reproduction_error}")
    record = {
        "passed": True,
        "no_set_residual_frame_relative_l2": residual_error,
        "step49_actual_gradient_ratio": float(metrics["mean_shape_gradient_ratio"]),
        "step400_reproduction_relative_l2": reproduction_error,
        "no_set_parameter_contract": parameter_contract,
        "finished_unix": time.time(),
    }
    write_json(OUTPUT / "gpu_checks.json", record)
    return record


def inventory():
    return original_runner.inventory()


def parse_gpus(value: str):
    return original_runner.parse_gpus(value)


def pool(inv, request):
    return original_runner._pool(inv, request)


def free_gpus(request="auto", *, required=None) -> list[int]:
    inv = inventory()
    available = [
        index for index in pool(inv, request)
        if not inv[index]["busy"] and inv[index]["memory"] <= 1024 and inv[index]["utilization"] <= 10
    ]
    needed = 1 if required is None else required
    if len(available) < needed:
        raise RuntimeError(f"Need {needed} idle A40 GPUs but found {available}")
    return available if required is None else available[:required]


def training_gpus(arm: str, request: str, *, resume: bool) -> list[int]:
    if arm == "extend600":
        available = set(free_gpus(request))
        if not set(ORIGINAL_GPUS).issubset(available):
            raise RuntimeError(f"Exact continuation requires idle original GPUs {ORIGINAL_GPUS}")
        return list(ORIGINAL_GPUS)
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return free_gpus(request, required=int(payload["world_size"]))
    return free_gpus(request)[:8]


def launch(worker: str, gpus: list[int], *, arm=None, resume=False):
    inv = inventory()
    for index in gpus:
        if inv[index]["busy"] or inv[index]["memory"] > 1024 or inv[index]["utilization"] > 10:
            raise RuntimeError(f"GPU {index} became occupied: {inv[index]}")
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
    args = [str(Path(__file__).resolve()), "--worker", worker]
    if arm:
        args += ["--experiment", arm]
    if resume:
        args.append("--resume")
    command = [sys.executable, *args]
    if worker == "train" and len(gpus) > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(gpus)}", *args,
        ]
    log = OUTPUT / "logs" / f"{worker}_{arm or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    record = OUTPUT / "jobs" / f"{worker}_{arm or 'all'}.json"
    write_json(record, {
        "pid": process.pid, "worker": worker, "experiment": arm, "gpus": gpus,
        "command": command, "started_unix": time.time(), "status": "running", "log": str(log),
    })
    process.record_path = record
    print(json.dumps({"started": worker, "experiment": arm, "pid": process.pid, "gpus": gpus}), flush=True)
    return process


def wait(process) -> None:
    code = process.wait()
    record = json.loads(process.record_path.read_text(encoding="utf-8"))
    record.update(exit_code=code, status="complete" if code == 0 else "failed", finished_unix=time.time())
    write_json(process.record_path, record)
    if code:
        raise RuntimeError(f"Worker failed; inspect {record['log']}")


def worker(name: str, arm: str | None, resume: bool) -> None:
    if name == "gpu-check":
        print(json.dumps(gpu_checks(), ensure_ascii=False), flush=True)
    elif name == "train":
        exp.run_train(OUTPUT / f"{arm}.yaml", resume=resume)
    elif name == "infer":
        from tools.mean100_ablation_evaluation import evaluate_arm
        print(json.dumps(evaluate_arm(str(arm), resume=resume), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "infer", "report"), default="all")
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("gpu-check", "train", "infer"))
    args = parser.parse_args()
    parse_gpus(args.gpus)
    selected = ARMS if args.experiment == "all" else (args.experiment,)
    if args.dry_run:
        print(json.dumps({
            "dry_run_no_changes": True, "output": str(OUTPUT), "experiments": selected,
            "extend": "exact step 400 to 600", "noset": "from scratch to 600",
            "global_batch": 8, "roles": {"extend600": ["best", 600], "noset600": ["best", 400, 600]},
        }, ensure_ascii=False, indent=2))
        return
    if args.worker:
        worker(args.worker, args.experiment, args.resume)
        return
    if args.stage in ("all", "preflight", "train"):
        prepare(resume=args.resume, gpu_request=args.gpus)
    if args.stage in ("all", "preflight", "train") and not (OUTPUT / "gpu_checks.json").exists():
        wait(launch("gpu-check", free_gpus(args.gpus, required=1)))
    if args.stage == "preflight":
        return
    if args.stage in ("all", "train"):
        if not json.loads((OUTPUT / "gpu_checks.json").read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("GPU preflight did not pass")
        for arm in selected:
            if args.resume and (OUTPUT / arm / "training_complete.json").exists():
                continue
            resume_arm = arm == "extend600" or (args.resume and (OUTPUT / arm / "checkpoint_last.pt").exists())
            cards = training_gpus(arm, args.gpus, resume=resume_arm)
            print(json.dumps({"training": arm, "gpus": cards, "resume": resume_arm}), flush=True)
            wait(launch("train", cards, arm=arm, resume=resume_arm))
    if args.stage in ("all", "infer"):
        pending = [
            arm for arm in selected
            if not (args.resume and (OUTPUT / "evaluation" / arm / "complete.json").exists())
        ]
        while pending:
            cards = free_gpus(args.gpus)
            batch, pending = pending[:len(cards)], pending[len(cards):]
            jobs = [launch("infer", [card], arm=arm, resume=args.resume) for card, arm in zip(cards, batch)]
            for job in jobs:
                wait(job)
    if args.stage in ("all", "report"):
        from tools.mean100_ablation_report import build_report
        print(json.dumps(build_report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
