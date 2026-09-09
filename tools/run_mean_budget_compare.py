"""Run the isolated E3 50%/100% mean-structure comparison end to end."""
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

import torch
import yaml

from datasets.matlab_multivolume_dataset import load_dataset_index
from tools import mean_budget_experiment as exp
from tools import three_way_experiment as old
from tools import run_v3_compare as original_runner


OUTPUT = exp.OUTPUT
ARMS = exp.ARMS
GPU_POLICY = {
    "training": "arms sequential; each arm uses all idle requested A40 cards, capped at 8",
    "global_batch_size": 8,
    "micro_batch_per_gpu": 1,
    "resume": "same world size, optimizer, scheduler and completed update count",
    "inference": "one checkpoint-pair job per idle A40",
}


def _write_json(path: Path, value) -> None:
    old.write_json(path, value)


def _config(arm: str) -> dict:
    source = yaml.safe_load((exp.REFERENCE_OUTPUT / "e3_mean005.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    config["experiment"].update(
        name=f"v3_mean_budget_{arm}", seed=20260901, output_dir=str(OUTPUT / arm)
    )
    config["data"].update(root=str(exp.DATA), cache_dir=str(OUTPUT / "data_cache"), precompute_cache=False)
    config["optimization"].update(
        max_steps=400, global_batch_size=8, micro_batch_per_gpu=1,
        lr_network=1e-3, lr_beta=1e-4, validate_every=20, checkpoint_every=20,
    )
    config["runtime"].update(amp=False, tensorboard=True)
    config["loss"].update(lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5)
    config["three_way"].update(
        kind="e3", lr_gain=1e-4,
        selection_rule="common E3 normalized mean + normalized variance + TV object macro; bounded shape term excluded",
    )
    config["v3_compare"].update(
        kind=arm,
        shared_initial_state=str(exp.SHARED_INITIAL_STATE),
        common_seed=20260901,
        from_scratch=True,
        shape_gradient_budget=exp.BUDGETS[arm],
        shape_gradient_eps=1e-12,
        shape_coefficient_cap=1.0,
        ramp_steps=50,
        gradient_limit_location="per-sample normalized reconstruction q",
        shape_mean_definition="SmoothL1(H(q)/mean(H(q)), mu90/mean(mu90))",
        lr_drop_after_steps=200,
        lr_network_after_drop=1e-4,
        lr_scalar_after_drop=1e-5,
        numerical_precision="full FP32; AMP and TF32 disabled",
        best_selection_uses_gt=False,
    )
    return config


def _source_paths() -> list[Path]:
    return [
        ROOT / "tools/mean_budget_experiment.py",
        ROOT / "tools/mean_budget_checks.py",
        ROOT / "tools/mean_budget_report.py",
        Path(__file__).resolve(),
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "tools/v3_compare_evaluation.py",
        ROOT / "tools/v3_compare_local_audit.py",
        ROOT / "tools/v3_compare_report.py",
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "utils/dataset_splits.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
        ROOT / "physics/psf_loader.py",
        ROOT / "tools/three_way_experiment.py",
        ROOT / "tools/priority_validation_analysis.py",
        ROOT / "train_volume.py",
    ]


def prepare(*, resume: bool, gpu_request: str) -> dict:
    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"{OUTPUT} already exists; use --resume")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for name, digest in record["implementation_source_hashes"].items():
            if old.sha256(name) != digest: raise ValueError(f"Frozen source changed: {name}")
        for name, digest in record["config_hashes"].items():
            if old.sha256(name) != digest: raise ValueError(f"Frozen config changed: {name}")
        for name, digest in record["reference_hashes"].items():
            if old.sha256(name) != digest: raise ValueError(f"Reference result changed: {name}")
        _, fingerprint = load_dataset_index(exp.DATA)
        if fingerprint != record["dataset_fingerprint"]: raise ValueError("Dataset fingerprint changed")
        return record

    dataset = original_runner._verify_dataset_hashes()
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(items) for split, items in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30} or fingerprint != dataset["declared_fingerprint"]:
        raise ValueError(f"Unexpected V3 split or fingerprint: {counts}")
    source_acceptance_path = exp.REFERENCE_OUTPUT / "final_acceptance.json"
    source_acceptance = json.loads(source_acceptance_path.read_text())
    if not source_acceptance.get("passed"):
        raise ValueError("Original baseline/E3/5% comparison is not accepted")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    configs = {}
    initial_hashes = {}
    for arm in ARMS:
        config = _config(arm)
        exp.validate_config(config)
        path = OUTPUT / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[str(path.resolve())] = old.sha256(path)
        model = exp.build_model(config, initial=True)
        common = {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
        initial_hashes[arm] = old.state_hash(common)
        if float(model.mean_gain_gamma) != 0.0: raise ValueError(f"{arm} gamma is not zero")
        del model
    expected_initial = source_acceptance["training"]["e3"]["checkpoints"]
    reference_preflight = json.loads((exp.REFERENCE_OUTPUT / "preflight.json").read_text())
    if set(initial_hashes.values()) != {reference_preflight["shared_initial_common_state_sha256"]}:
        raise ValueError("New arms do not reproduce the original shared initialization")

    sources = _source_paths()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    reference_files = [
        source_acceptance_path,
        exp.REFERENCE_OUTPUT / "preflight.json",
        exp.SHARED_INITIAL_STATE,
        *(exp.REFERENCE_OUTPUT / arm / filename for arm in ("baseline", "e3", "e3_mean005")
          for filename in ("checkpoint_best.pt", "checkpoint_last.pt")),
        *(exp.REFERENCE_OUTPUT / "evaluation" / arm / "complete.json" for arm in ("baseline", "e3", "e3_mean005")),
        *(exp.REFERENCE_OUTPUT / "analysis" / filename for filename in
          ("summary.csv", "per_object.csv", "local_summary.csv", "t02_tubes.csv",
           "t03_lines.csv", "t04_axial.csv", "v03_beads.csv", "fixed_profiles.csv")),
    ]
    reference_hashes = {str(path.resolve()): old.sha256(path) for path in reference_files}
    psf = Path(_config(ARMS[0])["psf"]["H_path"])
    record = {
        "complete": True,
        "plan": "from-scratch E3 plus 50% and 100% per-sample q mean-structure gradient caps",
        "prepared_unix": time.time(), "dataset_root": str(exp.DATA),
        "dataset_fingerprint": fingerprint, "dataset_hash_verification": dataset,
        "split_counts": counts, "config_hashes": configs,
        "implementation_source_hashes": source_hashes,
        "training_source_hashes": source_hashes,
        "reference_hashes": reference_hashes,
        "shared_initial_state": str(exp.SHARED_INITIAL_STATE),
        "shared_initial_file_sha256": old.sha256(exp.SHARED_INITIAL_STATE),
        "shared_initial_common_state_sha256": reference_preflight["shared_initial_common_state_sha256"],
        "arm_common_initial_hashes": initial_hashes,
        "psf_path": str(psf), "psf_sha256": old.sha256(psf),
        "budgets": exp.BUDGETS, "coefficient_cap": 1.0, "ramp_steps": 50,
        "max_steps": 400, "global_batch_size": 8, "validation_interval": 20,
        "lr_schedule": {"steps_1_200": {"network": 1e-3, "beta_gamma": 1e-4},
                        "steps_201_400": {"network": 1e-4, "beta_gamma": 1e-5}},
        "numerical_precision": "full FP32; AMP and TF32 disabled",
        "gpu_policy": GPU_POLICY, "gpu_request_at_preflight": gpu_request,
        "reference_checkpoint_index": expected_initial,
        "no_training_gt": True, "taylor_saved_sqrt_used_once": True,
    }
    _write_json(marker, record)
    print(json.dumps({"preflight_prepared": True, "output": str(OUTPUT),
                      "artifacts_verified": dataset["artifacts_verified"],
                      "common_initial_state": record["shared_initial_common_state_sha256"]}, ensure_ascii=False), flush=True)
    return record


def inventory():
    return original_runner.inventory()


def parse_gpus(value: str):
    return original_runner.parse_gpus(value)


def _pool(inv, request):
    return original_runner._pool(inv, request)


def free_gpus(request="auto", *, required=None):
    inv = inventory(); pool = _pool(inv, request)
    available = [i for i in pool if not inv[i]["busy"] and inv[i]["memory"] <= 1024 and inv[i]["utilization"] <= 10]
    needed = 1 if required is None else required
    if len(available) < needed:
        raise RuntimeError(f"Need {needed} idle A40 GPUs but found {available}")
    return available if required is None else available[:required]


def training_gpus(arm: str, request: str, *, resume: bool) -> list[int]:
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume: raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        world_size = int(state["world_size"]); del state
        return free_gpus(request, required=world_size)
    return free_gpus(request)[:8]


def launch(worker: str, gpus: list[int], *, arm=None, resume=False):
    if not gpus: raise ValueError("At least one idle GPU is required")
    inv = inventory()
    for index in gpus:
        if inv[index]["busy"] or inv[index]["memory"] > 1024 or inv[index]["utilization"] > 10:
            raise RuntimeError(f"GPU {index} became occupied: {inv[index]}")
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(inv[index]["uuid"] for index in gpus),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", SPECKLE_PHYSICAL_GPUS=",".join(map(str, gpus)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
    )
    args = [str(Path(__file__).resolve()), "--worker", worker]
    if arm: args += ["--experiment", arm]
    if resume: args.append("--resume")
    command = [sys.executable, *args]
    if worker == "train" and len(gpus) > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={len(gpus)}", *args]
    log = OUTPUT / "logs" / f"{worker}_{arm or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    record = OUTPUT / "jobs" / f"{worker}_{arm or 'all'}.json"
    _write_json(record, {"pid": process.pid, "worker": worker, "experiment": arm,
                         "gpus": gpus, "command": command, "started_unix": time.time(),
                         "status": "running", "log": str(log)})
    process.record_path = record
    print(json.dumps({"started": worker, "experiment": arm, "pid": process.pid,
                      "gpus": gpus, "log": str(log)}), flush=True)
    return process


def wait(process):
    code = process.wait()
    record = json.loads(process.record_path.read_text())
    record.update(exit_code=code, status="complete" if code == 0 else "failed", finished_unix=time.time())
    _write_json(process.record_path, record)
    if code: raise RuntimeError(f"Worker failed; inspect {record['log']}")


def worker(name: str, arm: str | None, resume: bool):
    if name == "train": exp.run_train(OUTPUT / f"{arm}.yaml", resume=resume)
    elif name == "gpu-check":
        from tools.mean_budget_checks import gpu_checks
        gpu_checks()
    elif name == "infer":
        from tools import v3_compare_evaluation as base_evaluation
        previous = base_evaluation.exp; base_evaluation.exp = exp
        try: base_evaluation.evaluate_arm(arm, resume=resume)
        finally: base_evaluation.exp = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "infer", "report"), default="all")
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("train", "gpu-check", "infer"))
    args = parser.parse_args()
    try: parse_gpus(args.gpus)
    except ValueError as error: parser.error(str(error))
    selected = ARMS if args.experiment == "all" else (args.experiment,)
    if args.dry_run:
        print(json.dumps({"dry_run_no_changes": True, "output": str(OUTPUT),
                          "experiments": selected, "budgets": exp.BUDGETS,
                          "steps": 400, "global_batch": 8, "checkpoints": [200, "best", 400],
                          "new_predictions": 376}, indent=2, ensure_ascii=False)); return
    if args.worker:
        worker(args.worker, args.experiment, args.resume); return
    if args.stage in ("all", "preflight", "train"):
        prepare(resume=args.resume, gpu_request=args.gpus)
    if args.stage in ("all", "preflight", "train") and not (OUTPUT / "gpu_checks.json").exists():
        wait(launch("gpu-check", free_gpus(args.gpus, required=1)))
    if args.stage == "preflight": return
    if args.stage in ("all", "train"):
        if not json.loads((OUTPUT / "gpu_checks.json").read_text()).get("passed"):
            raise RuntimeError("Pre-training GPU checks did not pass")
        for arm in selected:
            if args.resume and (OUTPUT / arm / "training_complete.json").exists(): continue
            cards = training_gpus(arm, args.gpus, resume=args.resume)
            print(json.dumps({"training_resource_selection": arm, "gpus": cards,
                              "world_size": len(cards), "global_batch_size": 8}), flush=True)
            wait(launch("train", cards, arm=arm,
                        resume=args.resume and (OUTPUT / arm / "checkpoint_last.pt").exists()))
    if args.stage in ("all", "infer"):
        pending = [arm for arm in selected if not (args.resume and (OUTPUT / "evaluation" / arm / "complete.json").exists())]
        while pending:
            cards = free_gpus(args.gpus); batch = pending[:len(cards)]; pending = pending[len(batch):]
            jobs = [launch("infer", [card], arm=arm, resume=args.resume) for card, arm in zip(cards, batch)]
            for job in jobs: wait(job)
    if args.stage in ("all", "report"):
        from tools.mean_budget_report import build_report
        print(json.dumps(build_report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
