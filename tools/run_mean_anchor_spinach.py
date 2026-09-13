"""Run the Mean-anchor spinach transfer on an idle GPU from physical cards 1-5."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import mean_anchor_experiment as exp
from tools import run_v3_compare as resources
from tools import three_way_experiment as old


DESTINATION = exp.OUTPUT / "spinach_root_transfer"
SOURCE = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")


def idle_card() -> tuple[int, str]:
    while True:
        current = resources.inventory()
        for index in (1, 2, 3, 4, 5):
            info = current[index]
            if not info["busy"] and info["memory"] <= 1024 and info["utilization"] <= 10:
                return index, info["uuid"]
        print(json.dumps({"waiting_for_idle_gpu": [1, 2, 3, 4, 5]}), flush=True)
        time.sleep(15)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = exp.OUTPUT / exp.ARM / "checkpoint_last.pt"
    index, uuid = idle_card()
    manifest = DESTINATION / "manifest.json"
    cfg = {
        **source,
        "experiment": "spinach_root_mean_anchor_transfer",
        "checkpoint": str(checkpoint), "checkpoint_sha256": old.sha256(checkpoint),
        "gpu_uuid": uuid,
        "network_base_mat": str(DESTINATION / "network_base.mat"),
        "network_final_mat": str(DESTINATION / "mean_anchor_e3_mean100.mat"),
        "network_record": str(DESTINATION / "network_record.json"),
    }
    old.write_json(manifest, cfg)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        MEAN_ANCHOR_OUTPUT=str(exp.OUTPUT),
        MPLCONFIGDIR=str(DESTINATION / "mpl_cache"),
    )
    with (DESTINATION / "network.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/spinach_root_mean_anchor.py"), "network", str(manifest)],
            cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("Spinach Mean-anchor network inference failed")
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    cfg["gpu_uuid"] = {"gain": uuid}
    old.write_json(manifest, cfg)
    paths = [ROOT / "matlab_code/real_data", ROOT / "matlab_code/pilot_dataset", ROOT / "matlab_code/Util", ROOT / "matlab_code/Solver"]
    quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    expression = f"addpath({quoted});spinach_root_gain_worker_v2('{manifest}');"
    environment["MATLAB_PREFDIR"] = str(DESTINATION / "matlab_preferences")
    with (DESTINATION / "gain.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression],
            cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("Spinach Mean-anchor global gain failed")
    from tools.spinach_root_mean_anchor import report
    report(manifest)
    old.write_json(DESTINATION / "complete.json", {
        "complete": True, "physical_gpu": index, "checkpoint_step": 400,
        "methods": ["mean_rl3", "taylor_rl3_sqrt", "e3_mean100", exp.ARM],
    })
    print(json.dumps({"complete": True, "output": str(DESTINATION), "gpu": index}), flush=True)


if __name__ == "__main__":
    main()

