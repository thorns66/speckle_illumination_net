"""Run both P12-trained anchor networks on the frozen spinach-root input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import three_way_experiment as old
from tools import v4_p12_anchor_compare_experiment as exp
from tools.run_v4_p12_anchor_compare import eligible_gpus, MIN_FREE_MIB_INFER, inventory
from tools.spinach_root_v4_anchor import report


DESTINATION = exp.OUTPUT / "spinach_root_transfer"
SOURCE = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for arm in exp.ARMS:
        folder = DESTINATION / arm
        complete = folder / "complete.json"
        if complete.exists() and json.loads(complete.read_text(encoding="utf-8")).get("complete"):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        card = eligible_gpus(minimum_free_mib=MIN_FREE_MIB_INFER, required=1)[0]
        current = inventory()
        uuid = current[card]["uuid"]
        checkpoint = exp.OUTPUT / arm / "checkpoint_last.pt"
        manifest = folder / "manifest.json"
        cfg = {
            **source,
            "experiment": f"spinach_root_v4_{arm}",
            "arm": arm,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": old.sha256(checkpoint),
            "checkpoint_role": "final",
            "checkpoint_step": 400,
            "gpu_uuid": {"network": uuid, "gain": uuid},
            "physical_gpu": card,
            "network_base_mat": str(folder / "network_base.mat"),
            "network_final_mat": str(folder / f"{arm}.mat"),
            "network_record": str(folder / "network_record.json"),
        }
        old.write_json(manifest, cfg)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=uuid,
            PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
            V4_ANCHOR_OUTPUT=str(exp.OUTPUT),
            MPLCONFIGDIR=str(folder / "mpl_cache"),
        )
        with (folder / "network.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/spinach_root_v4_anchor.py"), "network", str(manifest)],
                cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{arm} spinach network inference failed")
        matlab_paths = [
            ROOT / "matlab_code/real_data", ROOT / "matlab_code/pilot_dataset",
            ROOT / "matlab_code/Util", ROOT / "matlab_code/Solver",
        ]
        quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in matlab_paths)
        expression = f"addpath({quoted});spinach_root_gain_worker_v2('{manifest}');"
        environment["MATLAB_PREFDIR"] = str(folder / "matlab_preferences")
        with (folder / "gain.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression],
                cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{arm} spinach global E3 gain failed")
        old.write_json(complete, {
            "complete": True, "arm": arm, "physical_gpu": card,
            "checkpoint_step": 400, "anchor": exp.ANCHORS[arm],
        })
        print(json.dumps({"spinach_complete": arm, "gpu": card}), flush=True)
    result = report(DESTINATION)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
