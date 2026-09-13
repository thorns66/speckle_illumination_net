"""Compatibility-complete supervisor for the Mean-anchor experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import run_mean_anchor_experiment as runner
from tools import three_way_experiment as old


_prepare = runner.prepare


def prepare(resume: bool):
    record = _prepare(resume)
    config_path = Path(record["config_path"])
    changed = False
    if "config_hashes" not in record:
        record["config_hashes"] = {str(config_path.resolve()): old.sha256(config_path)}
        changed = True
    if "training_source_hashes" not in record:
        record["training_source_hashes"] = dict(record["source_hashes"])
        changed = True
    if changed:
        old.write_json(runner.OUTPUT / "preflight.json", record)
    return record


def run_spinach(resume: bool):
    complete = runner.OUTPUT / "spinach_root_transfer/complete.json"
    if resume and complete.exists():
        return
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        MEAN_ANCHOR_OUTPUT=str(runner.OUTPUT),
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/run_mean_anchor_spinach.py")],
        cwd=ROOT, env=environment,
    )
    if result.returncode:
        raise RuntimeError("Spinach Mean-anchor transfer failed")


runner.prepare = prepare
runner.run_spinach = run_spinach


if __name__ == "__main__":
    runner.main()

