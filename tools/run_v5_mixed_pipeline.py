"""Finish real RL data, run both V5 arms, evaluate, report, and verify delivery."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "V5_MIXED_OUTPUT",
    ROOT / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run01",
))
REAL_DATA = Path(os.environ.get(
    "V5_REAL_DATA", ROOT / "data/spinach_real_mixed_training_20260910_run01",
))
PYTHON = Path(sys.executable).resolve()
ARMS = ("taylor_anchor_e3_mean100", "mean_anchor_e3_mean100")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def status(stage: str, **extra: Any) -> None:
    write_json(OUTPUT / "pipeline_status.json", {
        "stage": stage, "updated_unix": time.time(), **extra,
    })


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_logged(command: list[str], log_name: str, *, environment: dict[str, str] | None = None) -> None:
    log = OUTPUT / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def inventory() -> list[dict[str, Any]]:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    result = []
    for line in rows.splitlines():
        index, uuid, name, free, utilization = [part.strip() for part in line.split(",")]
        result.append({"index": int(index), "uuid": uuid, "name": name,
                       "free": int(free), "utilization": int(utilization)})
    return result


def wait_for_evaluation_gpu() -> dict[str, Any]:
    while True:
        cards = [row for row in inventory()
                 if row["index"] in range(6) and "A40" in row["name"].upper()
                 and row["free"] >= 24 * 1024]
        if cards:
            cards.sort(key=lambda row: (row["utilization"] > 5, -row["free"]))
            return cards[0]
        status("waiting_for_evaluation_gpu", requirement="one A40 with at least 24 GiB free")
        time.sleep(30)


def row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify() -> dict[str, Any]:
    arms = {}
    for arm in ARMS:
        folder = OUTPUT / arm
        current = {
            "training_rows": row_count(folder / "training_metrics.csv"),
            "validation_rows": row_count(folder / "validation_metrics.csv"),
            "test_rows": row_count(folder / "test_metrics.csv"),
            "checkpoint_200": (folder / "checkpoint_step_000200.pt").is_file(),
            "checkpoint_400": (folder / "checkpoint_step_000400.pt").is_file(),
            "checkpoint_600": (folder / "checkpoint_step_000600.pt").is_file(),
            "checkpoint_best": (folder / "checkpoint_best.pt").is_file(),
            "training_complete": (folder / "training_complete.json").is_file(),
        }
        if current != {
            "training_rows": 600, "validation_rows": 900, "test_rows": 30,
            "checkpoint_200": True, "checkpoint_400": True, "checkpoint_600": True,
            "checkpoint_best": True, "training_complete": True,
        }:
            raise RuntimeError(f"V5 arm acceptance failed: {arm}: {current}")
        arms[arm] = current
    prediction_markers = list((OUTPUT / "comparison").glob("**/complete.json"))
    if len(prediction_markers) != 300:
        raise RuntimeError(f"Expected 300 evaluated prediction cases, got {len(prediction_markers)}")
    if not (OUTPUT / "comparison/evaluation_complete.json").is_file():
        raise RuntimeError("Evaluation completion marker is missing")
    if not (OUTPUT / "comparison/report_complete.json").is_file():
        raise RuntimeError("Report completion marker is missing")
    return {"arms": arms, "evaluated_prediction_cases": len(prediction_markers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    try:
        if args.wait_pid is not None:
            status("waiting_for_existing_rl", pid=args.wait_pid)
            while pid_alive(args.wait_pid):
                time.sleep(30)
        status("finalizing_real_rl")
        run_logged([str(PYTHON), "tools/v5_mixed_real_data.py", "rl"], "pipeline_rl_finalize.log")
        status("training_two_arms")
        run_logged([
            str(PYTHON), "tools/run_v5_mixed_real_anchor_compare.py",
            "--stage", "all", "--experiment", "all", "--resume",
        ], "pipeline_training_launcher.log")
        card = wait_for_evaluation_gpu()
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=card["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        status("evaluating_300_cases", evaluation_gpu=card)
        run_logged([str(PYTHON), "tools/v5_mixed_real_evaluation.py"],
                   "pipeline_evaluation.log", environment=environment)
        report_environment = os.environ.copy()
        report_environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT),
                                  MPLCONFIGDIR="/tmp/mpl-v5-mixed-report")
        status("building_report")
        run_logged([str(PYTHON), "tools/v5_mixed_real_report.py"],
                   "pipeline_report.log", environment=report_environment)
        acceptance = verify()
        write_json(OUTPUT / "pipeline_complete.json", {
            "complete": True, "finished_unix": time.time(), **acceptance,
        })
        status("complete", **acceptance)
    except BaseException as exception:
        status("failed", error_type=type(exception).__name__, error=str(exception))
        raise


if __name__ == "__main__":
    main()
