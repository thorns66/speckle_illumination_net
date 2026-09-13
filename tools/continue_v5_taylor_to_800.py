"""Queue an exact V5 Taylor-anchor continuation from step 600 to step 800.

The original 600-step experiment remains immutable.  Its optimizer, mixed-data
scheduler, RNG, and best-validation state are copied into a dated extension run.
"""
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
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import three_way_experiment as old
from tools import v5_mixed_real_anchor_experiment as experiment
from tools import run_v5_mixed_real_anchor_compare as runner


BASELINE_VALIDATE_CONFIG = experiment.validate_config
ARM = "taylor_anchor_e3_mean100"
SOURCE = ROOT / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run02"
DESTINATION = ROOT / "outputs/v5_sim_real_no_p12_taylor_anchor_e3_mean100_800_20260911_run01"
SOURCE_ARM = SOURCE / ARM
DESTINATION_ARM = DESTINATION / ARM
CONFIG = DESTINATION / f"{ARM}.yaml"
STATUS = DESTINATION / "extension_status.json"
TARGET_STEPS = 800


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def update_status(stage: str, **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    current.update(stage=stage, updated_unix=time.time(), **extra)
    write_json(STATUS, current)


def source_is_complete() -> bool:
    complete = SOURCE_ARM / "training_complete.json"
    if not complete.is_file():
        return False
    record = json.loads(complete.read_text(encoding="utf-8"))
    return bool(record.get("complete")) and int(record.get("completed_steps", -1)) == 600


def verify_frozen_sources(source_preflight: dict[str, Any]) -> None:
    for path, expected in source_preflight["source_hashes"].items():
        if old.sha256(path) != expected:
            raise ValueError(f"Frozen V5 source changed before continuation: {path}")


def prepare_extension() -> None:
    source_preflight_path = SOURCE / "preflight.json"
    source_checkpoint = SOURCE_ARM / "checkpoint_last.pt"
    if not source_is_complete():
        raise RuntimeError("The source Taylor run has not completed exactly 600 steps")
    source_preflight = json.loads(source_preflight_path.read_text(encoding="utf-8"))
    verify_frozen_sources(source_preflight)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if int(payload["completed_steps"]) != 600 or int(payload["world_size"]) != 6:
        raise ValueError("The continuation source must be the six-GPU step-600 checkpoint")

    DESTINATION_ARM.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((SOURCE / f"{ARM}.yaml").read_text(encoding="utf-8"))
    config["experiment"]["output_dir"] = str(DESTINATION_ARM)
    config["optimization"]["max_steps"] = TARGET_STEPS
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    required = [
        "checkpoint_last.pt",
        "checkpoint_best.pt",
        "checkpoint_step_000200.pt",
        "checkpoint_step_000400.pt",
        "checkpoint_step_000600.pt",
        "training_metrics.csv",
        "validation_metrics.csv",
    ]
    for name in required:
        source = SOURCE_ARM / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, DESTINATION_ARM / name)
    for pattern in (
        "validation_step_*.json",
        "domain_training_rank*.jsonl",
        "gradient_diagnostics_rank*.jsonl",
        "learning_rate_rank*.jsonl",
        "preflight_rank_*.json",
    ):
        for source in SOURCE_ARM.glob(pattern):
            shutil.copy2(source, DESTINATION_ARM / source.name)
    shutil.copy2(
        SOURCE_ARM / "training_complete.json",
        DESTINATION_ARM / "training_complete_step_000600.json",
    )
    if (SOURCE_ARM / "test_metrics.csv").is_file():
        shutil.copy2(
            SOURCE_ARM / "test_metrics.csv",
            DESTINATION_ARM / "test_metrics_step_000600.csv",
        )

    extension_preflight = copy.deepcopy(source_preflight)
    extension_preflight.update(
        plan="Exact Taylor-anchor continuation from step 600 to step 800; Mean remains a 600-step control",
        output=str(DESTINATION),
        prepared_unix=time.time(),
        config_hashes={str(CONFIG.resolve()): old.sha256(CONFIG)},
        max_steps_per_arm=TARGET_STEPS,
        checkpoint_steps=[200, 400, 600, 800],
        lr_schedule={
            "1-200": {"network": 1e-3, "beta_gain": 1e-4},
            "201-800": {"network": 1e-4, "beta_gain": 1e-5},
        },
        continuation={
            "source_output": str(SOURCE),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": old.sha256(source_checkpoint),
            "source_completed_steps": 600,
            "target_completed_steps": TARGET_STEPS,
            "resume_contract": "model + optimizer + 6:1:1 scheduler + per-rank RNG + best state",
            "comparison_policy": "step 600 remains the primary fair comparison; step 800 is supplemental",
        },
    )
    write_json(DESTINATION / "preflight.json", extension_preflight)

    for source in runner.source_paths() + [Path(__file__).resolve()]:
        destination = DESTINATION / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    update_status(
        "prepared",
        source_checkpoint=str(source_checkpoint),
        destination=str(DESTINATION),
        completed_steps=600,
        target_steps=TARGET_STEPS,
    )


def validate_extended_config(config: dict[str, Any]) -> None:
    if int(config["optimization"]["max_steps"]) != TARGET_STEPS:
        raise ValueError("Taylor continuation must end at step 800")
    baseline = copy.deepcopy(config)
    baseline["optimization"]["max_steps"] = 600
    BASELINE_VALIDATE_CONFIG(baseline)


def worker() -> None:
    experiment.validate_config = validate_extended_config
    experiment.OUTPUT = DESTINATION
    experiment.run_train(CONFIG, resume=True)
    if int(os.environ.get("RANK", "0")) == 0:
        payload = torch.load(
            DESTINATION_ARM / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        if int(payload["completed_steps"]) != TARGET_STEPS:
            raise ValueError("Continuation did not reach step 800")
        shutil.copy2(
            DESTINATION_ARM / "checkpoint_last.pt",
            DESTINATION_ARM / "checkpoint_step_000800.pt",
        )


def wait_for_cards(checkpoint: Path) -> tuple[list[int], dict[int, dict[str, Any]]]:
    while True:
        try:
            cards = runner.ordered_cards(resume_checkpoint=checkpoint)
            return cards, runner.inventory()
        except (RuntimeError, subprocess.CalledProcessError) as exception:
            update_status("waiting_for_gpu_memory", reason=str(exception))
            time.sleep(60)


def launch_extension() -> None:
    checkpoint = DESTINATION_ARM / "checkpoint_last.pt"
    cards, inventory = wait_for_cards(checkpoint)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(inventory[index]["uuid"] for index in cards),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SPECKLE_PHYSICAL_GPUS=",".join(map(str, cards)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        V5_MIXED_OUTPUT=str(DESTINATION),
        V5_REAL_DATA=str(experiment.REAL_DATA),
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=6",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    log_path = DESTINATION / "logs/train_taylor_extension_600_to_800.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_status(
        "running",
        command=command,
        gpus_in_rank_order=cards,
        started_unix=time.time(),
        log=str(log_path),
    )
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        update_status("failed", exit_code=result.returncode, finished_unix=time.time())
        raise RuntimeError(f"Taylor extension failed with exit code {result.returncode}")
    update_status("complete", exit_code=0, completed_steps=800, finished_unix=time.time())


def continue_mean_control() -> None:
    complete = SOURCE / "mean_anchor_e3_mean100/training_complete.json"
    if complete.is_file():
        update_status("complete", mean_control="already_complete")
        return
    checkpoint = DESTINATION_ARM / "checkpoint_last.pt"
    wait_for_cards(checkpoint)
    environment = os.environ.copy()
    environment.update(
        V5_MIXED_OUTPUT=str(SOURCE),
        V5_REAL_DATA=str(experiment.REAL_DATA),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
    )
    update_status("starting_mean_600_control")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_v5_mixed_real_anchor_compare.py"),
            "--stage",
            "train",
            "--experiment",
            "mean_anchor_e3_mean100",
            "--resume",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    update_status(
        "all_training_complete" if result.returncode == 0 else "mean_control_failed",
        mean_exit_code=result.returncode,
        finished_unix=time.time(),
    )
    if result.returncode:
        raise RuntimeError(f"Mean control failed with exit code {result.returncode}")


def queue() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=False)
    update_status(
        "waiting_for_source_step_600",
        source=str(SOURCE),
        destination=str(DESTINATION),
        target_steps=TARGET_STEPS,
        queued_unix=time.time(),
    )
    while not source_is_complete():
        time.sleep(30)
    time.sleep(30)
    prepare_extension()
    launch_extension()
    continue_mean_control()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        queue()


if __name__ == "__main__":
    main()
