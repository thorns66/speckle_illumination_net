"""Run the dated Mean-RL3-anchor E3+mean<=100% experiment end to end."""
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
from tools import mean_anchor_experiment as exp
from tools import three_way_experiment as old
from tools import run_v3_compare as resources


OUTPUT = exp.OUTPUT
ARM = exp.ARM
GPU_POOL = (1, 2, 3, 4, 5)
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")


def write_json(path: Path, value: Any) -> None:
    old.write_json(path, value)


def config() -> dict:
    source = yaml.safe_load(
        (exp.REFERENCE_OUTPUT / "e3_mean100.yaml").read_text(encoding="utf-8")
    )
    value = copy.deepcopy(source)
    value["experiment"].update(
        name="v3_mean_anchor_e3_mean100", seed=20260901,
        output_dir=str(OUTPUT / ARM),
    )
    value["data"].update(root=str(exp.DATA), cache_dir=str(OUTPUT / "data_cache"), precompute_cache=False)
    value["model"]["reconstruction_anchor"] = "mean_rl3"
    value["optimization"].update(
        max_steps=400, global_batch_size=8, micro_batch_per_gpu=1,
        lr_network=1e-3, lr_beta=1e-4, validate_every=20, checkpoint_every=20,
    )
    value["runtime"].update(amp=False, tensorboard=True)
    value["loss"].update(lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5)
    value["three_way"].update(
        kind="e3", lr_gain=1e-4,
        selection_rule="normalized mean + normalized variance + TV object macro; bounded mean-shape term excluded",
    )
    value["v3_compare"].update(
        kind=ARM, shared_initial_state=str(exp.SHARED_INITIAL_STATE), common_seed=20260901,
        from_scratch=True, shape_gradient_budget=1.0, shape_gradient_eps=1e-12,
        shape_coefficient_cap=1.0, ramp_steps=50,
        gradient_limit_location="per-sample normalized reconstruction q",
        shape_mean_definition="SmoothL1(H(q)/mean(H(q)), mu90/mean(mu90))",
        lr_drop_after_steps=200, lr_network_after_drop=1e-4, lr_scalar_after_drop=1e-5,
        numerical_precision="full FP32; AMP and TF32 disabled", best_selection_uses_gt=False,
        beta_anchor="mean_rl3", beta_cache_key="mean_rl3",
    )
    return value


def source_paths() -> list[Path]:
    return [
        ROOT / "models/configurable_anchor_lfm_net.py",
        ROOT / "tools/mean_anchor_experiment.py",
        ROOT / "tools/mean_anchor_checks.py",
        ROOT / "tools/mean_anchor_analysis.py",
        ROOT / "tools/spinach_root_mean_anchor.py",
        ROOT / "tools/spinch_root_network_v2.py",
        Path(__file__).resolve(),
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "tools/v3_compare_evaluation.py",
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "tools/spinach_root_network_v2.py",
        ROOT / "matlab_code/real_data/spinach_root_gain_worker_v2.m",
    ]


def prepare(resume: bool) -> dict:
    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"{OUTPUT} exists; use --resume")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for path, digest in record["source_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen source changed: {path}")
        for path, digest in record["reference_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen reference changed: {path}")
        _, fingerprint = load_dataset_index(exp.DATA)
        if fingerprint != record["dataset_fingerprint"]:
            raise ValueError("Dataset fingerprint changed")
        return record

    dataset = resources._verify_dataset_hashes()
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(items) for split, items in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(f"Unexpected V3 split: {counts}")
    if fingerprint != dataset["declared_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the completed audit")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    current = config()
    exp.validate_config(current)
    config_path = OUTPUT / f"{ARM}.yaml"
    config_path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    model = exp.build_model(current, initial=True)
    common = {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
    initial_hash = old.state_hash(common)
    payload = torch.load(exp.SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
    if initial_hash != payload["common_state_sha256"]:
        raise ValueError("Mean-anchor common initialization differs from the baseline")
    del model, common, payload
    sources = source_paths()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    references = [
        exp.SHARED_INITIAL_STATE,
        exp.REFERENCE_OUTPUT / "final_acceptance.json",
        exp.REFERENCE_OUTPUT / "e3_mean100/checkpoint_last.pt",
        exp.REFERENCE_OUTPUT / "e3_mean100/checkpoint_best.pt",
        exp.REFERENCE_OUTPUT / "evaluation/e3_mean100/complete.json",
        ROOT / "outputs/spinach_root_exploratory_20260909_run01/complete.json",
    ]
    record = {
        "complete": True, "experiment": ARM, "output": str(OUTPUT),
        "dataset_root": str(exp.DATA), "dataset_fingerprint": fingerprint,
        "dataset_hash_verification": dataset, "split_counts": counts,
        "config_path": str(config_path), "config_sha256": old.sha256(config_path),
        "source_hashes": source_hashes,
        "reference_hashes": {str(path.resolve()): old.sha256(path) for path in references},
        "shared_initial_common_state_sha256": initial_hash,
        "reconstruction_anchor": "mean_rl3", "beta_anchor": "mean_rl3",
        "taylor_feature_representation": "sqrt", "set_branch": True,
        "gpu_pool": list(GPU_POOL), "gpu_sharing": False,
        "max_steps": 400, "global_batch_size": 8, "validation_interval": 20,
        "lr_schedule": {"steps_1_200": {"network": 1e-3, "beta_gamma": 1e-4}, "steps_201_400": {"network": 1e-4, "beta_gamma": 1e-5}},
        "no_training_gt": True, "prepared_unix": time.time(),
    }
    write_json(marker, record)
    print(json.dumps({"preflight": "complete", "output": str(OUTPUT), "artifacts": dataset["artifacts_verified"]}), flush=True)
    return record


def inventory() -> dict[int, dict]:
    return resources.inventory()


def available() -> list[int]:
    current = inventory()
    return [
        index for index in GPU_POOL
        if index in current and not current[index]["busy"]
        and current[index]["memory"] <= 1024 and current[index]["utilization"] <= 10
    ]


def wait_for_gpus(required: int = 1) -> list[int]:
    last_report = 0.0
    while True:
        cards = available()
        if len(cards) >= required:
            return cards
        now = time.time()
        if now - last_report >= 60:
            print(json.dumps({"waiting_for_idle_gpu": list(GPU_POOL), "available": cards}), flush=True)
            last_report = now
        time.sleep(15)


def training_gpus(resume: bool) -> list[int]:
    checkpoint = OUTPUT / ARM / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        world_size = int(payload["world_size"])
        del payload
        return wait_for_gpus(world_size)[:world_size]
    return wait_for_gpus(1)[:5]


def launch(worker: str, cards: list[int], *, resume: bool = False):
    current = inventory()
    for index in cards:
        info = current[index]
        if info["busy"] or info["memory"] > 1024 or info["utilization"] > 10:
            raise RuntimeError(f"GPU {index} became occupied: {info}")
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(current[index]["uuid"] for index in cards),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", SPECKLE_PHYSICAL_GPUS=",".join(map(str, cards)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
        MEAN_ANCHOR_OUTPUT=str(OUTPUT),
    )
    args = [str(Path(__file__).resolve()), "--worker", worker]
    if resume:
        args.append("--resume")
    command = [sys.executable, *args]
    if worker == "train" and len(cards) > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(cards)}", *args]
    log_path = OUTPUT / "logs" / f"{worker}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    record_path = OUTPUT / "jobs" / f"{worker}.json"
    write_json(record_path, {"pid": process.pid, "worker": worker, "gpus": cards, "command": command, "log": str(log_path), "status": "running", "started_unix": time.time()})
    process.record_path = record_path
    print(json.dumps({"started": worker, "pid": process.pid, "gpus": cards, "log": str(log_path)}), flush=True)
    return process


def wait(process) -> None:
    code = process.wait()
    record = json.loads(process.record_path.read_text(encoding="utf-8"))
    record.update(status="complete" if code == 0 else "failed", exit_code=code, finished_unix=time.time())
    write_json(process.record_path, record)
    if code:
        raise RuntimeError(f"Worker failed; inspect {record['log']}")


def worker(name: str, resume: bool) -> None:
    if name == "gpu-check":
        from tools.mean_anchor_checks import gpu_checks
        gpu_checks()
    elif name == "train":
        exp.run_train(OUTPUT / f"{ARM}.yaml", resume=resume)
    elif name == "infer":
        from tools import v3_compare_evaluation as generic
        previous = generic.exp; generic.exp = exp
        try:
            generic.evaluate_arm(ARM, resume=resume)
        finally:
            generic.exp = previous
        from tools.mean_anchor_analysis import augment_evaluation
        augment_evaluation()
    else:
        raise ValueError(name)


def run_spinach(resume: bool) -> None:
    destination = OUTPUT / "spinach_root_transfer"
    if resume and (destination / "complete.json").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = OUTPUT / ARM / "checkpoint_last.pt"
    cfg = {
        **source_manifest,
        "experiment": "spinach_root_mean_anchor_transfer",
        "checkpoint": str(checkpoint), "checkpoint_sha256": old.sha256(checkpoint),
        "network_base_mat": str(destination / "network_base.mat"),
        "network_final_mat": str(destination / "mean_anchor_e3_mean100.mat"),
        "network_record": str(destination / "network_record.json"),
    }
    card = wait_for_gpus(1)[0]
    current = inventory(); cfg["gpu_uuid"] = current[card]["uuid"]
    manifest = destination / "manifest.json"; write_json(manifest, cfg)
    environment = os.environ.copy(); environment.update(
        CUDA_VISIBLE_DEVICES=current[card]["uuid"], PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        MEAN_ANCHOR_OUTPUT=str(OUTPUT), MPLCONFIGDIR=str(destination / "mpl_cache"),
    )
    with (destination / "network.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, str(ROOT / "tools/spinach_root_mean_anchor.py"), "network", str(manifest)], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Spinach Mean-anchor network inference failed")
    matlab_paths = [ROOT / "matlab_code/real_data", ROOT / "matlab_code/pilot_dataset", ROOT / "matlab_code/Util", ROOT / "matlab_code/Solver"]
    quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in matlab_paths)
    expression = f"addpath({quoted});spinach_root_gain_worker_v2('{manifest}');"
    environment["MATLAB_PREFDIR"] = str(destination / "matlab_preferences")
    with (destination / "gain.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run([str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Spinach Mean-anchor global gain failed")
    from tools.spinach_root_mean_anchor import report
    report(manifest)
    write_json(destination / "complete.json", {"complete": True, "gpu": card, "checkpoint_step": 400, "methods": ["mean_rl3", "taylor_rl3_sqrt", "e3_mean100", ARM]})


def finalize() -> dict:
    from tools.mean_anchor_analysis import build_report
    analysis = build_report()
    complete = json.loads((OUTPUT / "evaluation" / ARM / "complete.json").read_text())
    training = json.loads((OUTPUT / ARM / "training_complete.json").read_text())
    result = {"complete": True, "experiment": ARM, "training": training, "evaluation": complete, "analysis": analysis, "spinach_root": True, "gpu_pool": list(GPU_POOL), "finished_unix": time.time()}
    write_json(OUTPUT / "complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "infer", "report", "spinach"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("gpu-check", "train", "infer"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.resume); return
    if args.stage in ("all", "preflight", "train"):
        prepare(args.resume)
    if args.stage in ("all", "preflight", "train") and not (OUTPUT / "gpu_checks.json").exists():
        wait(launch("gpu-check", [wait_for_gpus(1)[0]]))
    if args.stage == "preflight":
        return
    if args.stage in ("all", "train") and not (args.resume and (OUTPUT / ARM / "training_complete.json").exists()):
        cards = training_gpus(args.resume)
        wait(launch("train", cards, resume=args.resume and (OUTPUT / ARM / "checkpoint_last.pt").exists()))
    if args.stage in ("all", "infer") and not (args.resume and (OUTPUT / "evaluation" / ARM / "complete.json").exists()):
        wait(launch("infer", [wait_for_gpus(1)[0]], resume=args.resume))
    if args.stage in ("all", "report"):
        from tools.mean_anchor_analysis import build_report
        build_report()
    if args.stage in ("all", "spinach"):
        run_spinach(args.resume)
    if args.stage == "all":
        print(json.dumps(finalize(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

