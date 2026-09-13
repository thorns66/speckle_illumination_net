"""Resume V5 Taylor to 800, then train the matched Mean anchor for 800 steps.

This coordinator deliberately leaves the frozen 600-step experiment and its
launchers unchanged.  Taylor resumes from the latest exact checkpoint in its
dated extension directory.  Mean starts from the common random initialization
and uses the same 800-update mixed-data schedule and learning-rate policy.
"""
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
from tools import v5_mixed_real_anchor_experiment as experiment
from tools import run_v5_mixed_real_anchor_compare as runner
from tools.v5_mixed_dataset import MixedSixOneOneScheduler


BASELINE_VALIDATE_CONFIG = experiment.validate_config
SOURCE = ROOT / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run02"
TAYLOR_OUTPUT = ROOT / "outputs/v5_sim_real_no_p12_taylor_anchor_e3_mean100_800_20260911_run01"
TAYLOR_ARM = "taylor_anchor_e3_mean100"
TAYLOR_DIR = TAYLOR_OUTPUT / TAYLOR_ARM
TAYLOR_CHECKPOINT = TAYLOR_DIR / "checkpoint_last.pt"
TAYLOR_COMPLETE = TAYLOR_DIR / "training_complete.json"
TAYLOR_STATUS = TAYLOR_OUTPUT / "extension_status.json"
TAYLOR_WORKER = ROOT / "tools/continue_v5_taylor_to_800.py"

MEAN_OUTPUT = ROOT / "outputs/v5_sim_real_no_p12_mean_anchor_e3_mean100_800_20260911_run01"
MEAN_ARM = "mean_anchor_e3_mean100"
MEAN_DIR = MEAN_OUTPUT / MEAN_ARM
MEAN_CONFIG = MEAN_OUTPUT / f"{MEAN_ARM}.yaml"
MEAN_CHECKPOINT = MEAN_DIR / "checkpoint_last.pt"
MEAN_COMPLETE = MEAN_DIR / "training_complete.json"
QUEUE_STATUS = MEAN_OUTPUT / "queue_status.json"
TARGET_STEPS = 800


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def update_status(path: Path, stage: str, **extra: Any) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(stage=stage, updated_unix=time.time(), **extra)
    write_json(path, current)


def completed_exactly(marker: Path, steps: int) -> bool:
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return bool(record.get("complete")) and int(record.get("completed_steps", -1)) == steps


def checkpoint_steps(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("world_size", -1)) != 6:
        raise ValueError(f"Expected a six-rank checkpoint: {path}")
    return int(payload["completed_steps"])


def verify_frozen_sources(preflight: dict[str, Any]) -> None:
    for path, expected in preflight["source_hashes"].items():
        if old.sha256(path) != expected:
            raise ValueError(f"Frozen V5 source changed: {path}")
    manifest = Path(preflight["real_final_manifest"])
    if old.sha256(manifest) != preflight["real_final_manifest_sha256"]:
        raise ValueError(f"Frozen real-data manifest changed: {manifest}")


def schedule_hash(steps: int) -> str:
    scheduler = MixedSixOneOneScheduler(130, 8, 20260901)
    schedule = [scheduler.next_batch() for _ in range(steps)]
    for batch in schedule:
        if (
            len(batch) != 8
            or sum(index < 110 for index in batch) != 6
            or sum(110 <= index < 120 for index in batch) != 1
            or sum(index >= 120 for index in batch) != 1
        ):
            raise AssertionError("Mixed schedule violated 6+1+1")
    return hashlib.sha256(json.dumps(schedule).encode("utf-8")).hexdigest()


def validate_mean_800_config(config: dict[str, Any]) -> None:
    if int(config["optimization"]["max_steps"]) != TARGET_STEPS:
        raise ValueError("Mean control must train for exactly 800 optimizer updates")
    if config["v5_mixed"]["kind"] != MEAN_ARM:
        raise ValueError("The queued 800-step arm must be the Mean anchor")
    if config["model"]["reconstruction_anchor"] != "mean_rl3":
        raise ValueError("The queued Mean arm must use the Mean-RL3 reconstruction anchor")
    baseline = copy.deepcopy(config)
    baseline["optimization"]["max_steps"] = 600
    BASELINE_VALIDATE_CONFIG(baseline)


def prepare_mean() -> None:
    source_preflight = json.loads((SOURCE / "preflight.json").read_text(encoding="utf-8"))
    verify_frozen_sources(source_preflight)
    if MEAN_OUTPUT.exists():
        if not (MEAN_CONFIG.is_file() and (MEAN_OUTPUT / "preflight.json").is_file()):
            raise FileExistsError(f"Incomplete existing Mean-800 destination: {MEAN_OUTPUT}")
        existing = yaml.safe_load(MEAN_CONFIG.read_text(encoding="utf-8"))
        validate_mean_800_config(existing)
        return

    MEAN_DIR.mkdir(parents=True)
    config = yaml.safe_load((SOURCE / f"{MEAN_ARM}.yaml").read_text(encoding="utf-8"))
    config["experiment"]["output_dir"] = str(MEAN_DIR)
    config["optimization"]["max_steps"] = TARGET_STEPS
    validate_mean_800_config(config)
    MEAN_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    preflight = copy.deepcopy(source_preflight)
    source_paths = runner.source_paths() + [Path(__file__).resolve()]
    preflight.update(
        plan=(
            "P12 excluded; P01-P11 plus full-field real 45/55; Mean anchor trained "
            "from the common random initialization for 800 steps after Taylor reaches 800"
        ),
        output=str(MEAN_OUTPUT),
        prepared_unix=time.time(),
        config_hashes={str(MEAN_CONFIG.resolve()): old.sha256(MEAN_CONFIG)},
        source_hashes={str(path.resolve()): old.sha256(path) for path in source_paths},
        sample_schedule_sha256=schedule_hash(TARGET_STEPS),
        max_steps_per_arm=TARGET_STEPS,
        checkpoint_steps=[200, 400, 600, 800],
        lr_schedule={
            "1-200": {"network": 1e-3, "beta_gain": 1e-4},
            "201-800": {"network": 1e-4, "beta_gain": 1e-5},
        },
        queued_after={
            "arm": TAYLOR_ARM,
            "output": str(TAYLOR_OUTPUT),
            "required_completed_steps": TARGET_STEPS,
        },
        mean_800={
            "from_scratch": True,
            "shared_initial_state": str(experiment.SHARED_INITIAL_STATE),
            "common_seed": 20260901,
            "comparison_policy": "matched Taylor/Mean final checkpoints at step 800",
        },
    )
    write_json(MEAN_OUTPUT / "preflight.json", preflight)
    for source in source_paths:
        destination = MEAN_OUTPUT / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    update_status(
        QUEUE_STATUS,
        "queued_waiting_for_taylor_800",
        target_steps=TARGET_STEPS,
        mean_output=str(MEAN_OUTPUT),
        taylor_output=str(TAYLOR_OUTPUT),
        from_scratch=True,
        queued_unix=time.time(),
    )


def wait_for_cards(checkpoint: Path | None, status_stage: str) -> tuple[list[int], dict[int, dict[str, Any]]]:
    while True:
        try:
            cards = runner.ordered_cards(resume_checkpoint=checkpoint)
            return cards, runner.inventory()
        except (RuntimeError, subprocess.CalledProcessError) as exception:
            update_status(QUEUE_STATUS, status_stage, reason=str(exception))
            time.sleep(60)


def environment_for(cards: list[int], inventory: dict[int, dict[str, Any]], output: Path) -> dict[str, str]:
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
        V5_MIXED_OUTPUT=str(output),
        V5_REAL_DATA=str(experiment.REAL_DATA),
    )
    return environment


def run_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    stage: str,
    cards: list[int],
    inventory: dict[int, dict[str, Any]],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        update_status(
            QUEUE_STATUS,
            stage,
            pid=process.pid,
            command=command,
            gpus_in_rank_order=cards,
            inventory_at_launch={str(index): inventory[index] for index in cards},
            log=str(log_path),
            started_unix=time.time(),
        )
        return_code = process.wait()
    if return_code:
        update_status(
            QUEUE_STATUS,
            f"{stage}_failed",
            exit_code=return_code,
            finished_unix=time.time(),
        )
        raise RuntimeError(f"{stage} failed with exit code {return_code}; inspect {log_path}")


def finish_taylor() -> None:
    if completed_exactly(TAYLOR_COMPLETE, TARGET_STEPS):
        if checkpoint_steps(TAYLOR_CHECKPOINT) != TARGET_STEPS:
            raise ValueError("Taylor completion marker and checkpoint disagree")
        update_status(QUEUE_STATUS, "taylor_800_complete")
        return
    current_steps = checkpoint_steps(TAYLOR_CHECKPOINT)
    if not 600 <= current_steps < TARGET_STEPS or current_steps % 20:
        raise ValueError(f"Unexpected Taylor resume step: {current_steps}")
    cards, inventory = wait_for_cards(TAYLOR_CHECKPOINT, "waiting_for_gpu_memory_taylor")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=6",
        str(TAYLOR_WORKER),
        "--worker",
    ]
    update_status(
        TAYLOR_STATUS,
        "resuming_after_queue_update",
        resumed_from_step=current_steps,
        target_steps=TARGET_STEPS,
        queue_supervisor=str(Path(__file__).resolve()),
    )
    run_process(
        command,
        environment_for(cards, inventory, TAYLOR_OUTPUT),
        TAYLOR_OUTPUT / "logs/train_taylor_extension_600_to_800.log",
        "resuming_taylor_800",
        cards,
        inventory,
    )
    if not completed_exactly(TAYLOR_COMPLETE, TARGET_STEPS):
        raise ValueError("Taylor worker exited without an exact step-800 completion marker")
    if checkpoint_steps(TAYLOR_CHECKPOINT) != TARGET_STEPS:
        raise ValueError("Taylor worker exited without an exact step-800 checkpoint")
    update_status(
        TAYLOR_STATUS,
        "complete",
        completed_steps=TARGET_STEPS,
        exit_code=0,
        finished_unix=time.time(),
    )
    update_status(QUEUE_STATUS, "taylor_800_complete", taylor_finished_unix=time.time())


def mean_worker() -> None:
    experiment.validate_config = validate_mean_800_config
    experiment.OUTPUT = MEAN_OUTPUT
    experiment.run_train(MEAN_CONFIG, resume=False)
    if int(os.environ.get("RANK", "0")) == 0:
        if checkpoint_steps(MEAN_CHECKPOINT) != TARGET_STEPS:
            raise ValueError("Mean training did not reach step 800")
        shutil.copy2(MEAN_CHECKPOINT, MEAN_DIR / "checkpoint_step_000800.pt")


def train_mean() -> None:
    if completed_exactly(MEAN_COMPLETE, TARGET_STEPS):
        update_status(QUEUE_STATUS, "all_training_complete", completed_steps=TARGET_STEPS)
        return
    if MEAN_CHECKPOINT.exists():
        raise RuntimeError(
            "Mean-800 is required to start from the common initialization; refusing an implicit resume"
        )
    cards, inventory = wait_for_cards(None, "waiting_for_gpu_memory_mean")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=6",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    run_process(
        command,
        environment_for(cards, inventory, MEAN_OUTPUT),
        MEAN_OUTPUT / "logs/train_mean_anchor_e3_mean100_800.log",
        "running_mean_800",
        cards,
        inventory,
    )
    if not completed_exactly(MEAN_COMPLETE, TARGET_STEPS):
        raise ValueError("Mean worker exited without an exact step-800 completion marker")
    if checkpoint_steps(MEAN_CHECKPOINT) != TARGET_STEPS:
        raise ValueError("Mean worker exited without an exact step-800 checkpoint")
    update_status(
        QUEUE_STATUS,
        "all_training_complete",
        completed_steps=TARGET_STEPS,
        exit_code=0,
        finished_unix=time.time(),
    )


def queue() -> None:
    prepare_mean()
    finish_taylor()
    train_mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        mean_worker()
    else:
        queue()


if __name__ == "__main__":
    main()
