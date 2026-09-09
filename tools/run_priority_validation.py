#!/usr/bin/env python3
"""Stable command-line entry for the 2026-09-07 priority validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import priority_validation as implementation
from priority_validation_common import REPO_ROOT, resolve_from_repo


def _source_paths(config_path: Path, config: dict[str, Any]) -> list[Path]:
    return [
        config_path,
        resolve_from_repo(config["experiment"]["checkpoint"]),
        REPO_ROOT / "infer_dataset.py",
        REPO_ROOT / "datasets" / "matlab_multivolume_dataset.py",
        REPO_ROOT / "models" / "variance_anchored_lfm_net.py",
        REPO_ROOT / "losses" / "self_supervised_losses.py",
        Path(__file__).resolve(),
        Path(implementation.__file__).resolve(),
        Path(__file__).with_name("priority_validation_common.py").resolve(),
        Path(__file__).with_name("priority_validation_analysis.py").resolve(),
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_make_truth.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_generate_illumination.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_combine_calibration.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_scene_task.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_loss_target_task.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_zero_task.m",
        REPO_ROOT / "matlab_code" / "cell_dataset" / "cell_generate_illumination_3d.m",
        REPO_ROOT / "matlab_code" / "cell_dataset" / "cell_forward_project_acc.m",
        REPO_ROOT / "matlab_code" / "pilot_dataset" / "pilot_reconstruct_volume.m",
        REPO_ROOT / "matlab_code" / "Solver" / "deconvRL.m",
    ]


def _run_generate(config: dict[str, Any], tasks: dict[str, list[dict[str, Any]]], output: Path,
                  gpus: list[int], inventory: dict[int, dict[str, Any]]) -> None:
    matlab = str(config["runtime"]["matlab"])
    workers = int(config["runtime"]["max_workers"])
    illumination = [
        implementation.matlab_job(
            matlab, f"illumination_repeat_{repeat:02d}",
            f"priority_validation_generate_illumination({implementation.quote_matlab(output)},'repeat',{repeat});",
        ) for repeat in range(1, 4)
    ]
    illumination.extend(
        implementation.matlab_job(
            matlab, f"calibration_batch_{batch:02d}",
            f"priority_validation_generate_illumination({implementation.quote_matlab(output)},'calibration',{batch});",
        ) for batch in range(1, 17)
    )
    implementation.command_runner(illumination, gpus, inventory, output, "illumination", min(3, workers))
    combine = implementation.matlab_job(
        matlab, "combine_calibration",
        f"priority_validation_combine_calibration({implementation.quote_matlab(output)});",
    )
    implementation.command_runner([combine], [gpus[0]], inventory, output, "calibration", 1)

    scene_commands = []
    for task in tasks["scenes"]:
        scene = task["scene"]
        depth = "[]" if scene.depth_um is None else str(scene.depth_um)
        expression = (
            f"priority_validation_scene_task({implementation.quote_matlab(REPO_ROOT)},"
            f"{implementation.quote_matlab(output)},{implementation.quote_matlab(scene.family)},"
            f"{depth},{task['repeat']});"
        )
        scene_commands.append(implementation.matlab_job(matlab, task["id"], expression))
    implementation.command_runner(scene_commands, gpus, inventory, output, "scenes", workers)
    target_commands = [
        implementation.matlab_job(
            matlab, task["id"],
            f"priority_validation_loss_target_task({implementation.quote_matlab(REPO_ROOT)},"
            f"{implementation.quote_matlab(output)},{implementation.quote_matlab(task['sample'])},"
            f"{task['repeat']});",
        ) for task in tasks["targets"]
    ]
    implementation.command_runner(target_commands, gpus, inventory, output, "loss_targets", workers)
    zero = implementation.matlab_job(
        matlab, "zero_control",
        f"priority_validation_zero_task({implementation.quote_matlab(output)});",
    )
    implementation.command_runner([zero], [gpus[0]], inventory, output, "zero", 1)


if __name__ == "__main__":
    implementation.source_paths = _source_paths
    implementation.run_generate = _run_generate
    implementation.main()
