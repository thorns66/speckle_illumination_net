#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from priority_validation_common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    check_source_manifest,
    json_dump,
    load_config,
    resolve_from_repo,
    scenes,
    select_gpus,
    source_manifest,
    task_id,
)


def quote_matlab(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the frozen high-priority validation suite")
    result.add_argument("--config", default=str(DEFAULT_CONFIG))
    result.add_argument("--stage", choices=["all", "generate", "infer", "loss", "report"], default="all")
    result.add_argument("--gpus", default="all", help="physical nvidia-smi indices, e.g. 1,2,3")
    result.add_argument("--output-dir")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--allow-busy-gpus", action="store_true")
    return result


def source_paths(config_path: Path, config: dict[str, Any]) -> list[Path]:
    return [
        config_path,
        resolve_from_repo(config["experiment"]["checkpoint"]),
        REPO_ROOT / "infer_dataset.py",
        REPO_ROOT / "datasets" / "matlab_multivolume_dataset.py",
        REPO_ROOT / "models" / "variance_anchored_lfm_net.py",
        REPO_ROOT / "losses" / "self_supervised_losses.py",
        Path(__file__).resolve(),
        Path(__file__).with_name("priority_validation_common.py").resolve(),
        Path(__file__).with_name("priority_validation_analysis.py").resolve(),
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_make_truth.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_generate_illumination.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_combine_calibration.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_scene_task.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_target_task.m",
        REPO_ROOT / "matlab_code" / "priority_validation" / "priority_validation_zero_task.m",
        REPO_ROOT / "matlab_code" / "cell_dataset" / "cell_generate_illumination_3d.m",
        REPO_ROOT / "matlab_code" / "cell_dataset" / "cell_forward_project_acc.m",
        REPO_ROOT / "matlab_code" / "pilot_dataset" / "pilot_reconstruct_volume.m",
        REPO_ROOT / "matlab_code" / "Solver" / "deconvRL.m",
    ]


def expected_checkpoint(config: dict[str, Any]) -> None:
    from priority_validation_common import sha256

    checkpoint = resolve_from_repo(config["experiment"]["checkpoint"])
    actual = sha256(checkpoint)
    expected = str(config["experiment"]["expected_checkpoint_sha256"])
    if actual != expected:
        raise ValueError(f"Checkpoint hash changed: {actual}, expected {expected}")


def build_tasks(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scene_tasks = []
    for scene in scenes(config):
        for repeat, seed in enumerate(config["acquisition"]["repeat_seeds"], start=1):
            scene_tasks.append({"scene": scene, "repeat": repeat, "seed": int(seed), "id": task_id(scene, repeat)})
    target_tasks = [
        {"sample": sample, "repeat": repeat, "id": f"target_{sample}_r{repeat:02d}"}
        for sample in config["calibration"]["diagnostic_objects"]
        for repeat in range(1, 4)
    ]
    inference = [
        {"sample_dir": None, "subset": 1, "id": task["id"], "kind": "generated"}
        for task in scene_tasks
    ]
    inference.append({"sample_dir": None, "subset": 1, "id": "zero_control", "kind": "generated"})
    data = REPO_ROOT / "data" / "matlab_cells_pilot_v2_r04"
    for sample in config["calibration"]["diagnostic_objects"]:
        for subset in range(1, 11):
            inference.append(
                {"sample_dir": data / sample, "subset": subset, "id": f"{sample}_subset_{subset:02d}", "kind": "existing"}
            )
    return {"scenes": scene_tasks, "targets": target_tasks, "inference": inference}


def command_runner(
    commands: list[dict[str, Any]],
    gpus: list[int],
    inventory: dict[int, dict[str, Any]],
    output: Path,
    label: str,
    max_workers: int,
) -> None:
    logs = output / "logs" / label
    logs.mkdir(parents=True, exist_ok=True)

    def worker(gpu: int, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for job in jobs:
            log = logs / f"{job['id']}.log"
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = str(inventory[gpu]["uuid"])
            environment["SPECKLE_PHYSICAL_GPUS"] = str(gpu)
            matlab_prefs = output / "matlab_prefs" / label / job["id"]
            matlab_prefs.mkdir(parents=True, exist_ok=True)
            environment["MATLAB_PREFDIR"] = str(matlab_prefs)
            mpl_cache = output / "analysis" / "mpl_cache"
            mpl_cache.mkdir(parents=True, exist_ok=True)
            environment["MPLCONFIGDIR"] = str(mpl_cache)
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "tools")
            started = time.monotonic()
            with log.open("a", encoding="utf-8") as stream:
                run = subprocess.run(job["command"], cwd=REPO_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            record = {"id": job["id"], "gpu": gpu, "returncode": run.returncode,
                      "seconds": time.monotonic() - started, "log": str(log)}
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if run.returncode:
                raise RuntimeError(f"{job['id']} failed; inspect {log}")
            results.append(record)
        return results

    assignments = [[] for _ in gpus]
    for position, command in enumerate(commands):
        assignments[position % len(gpus)].append(command)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), max_workers)) as executor:
        futures = [executor.submit(worker, gpu, jobs) for gpu, jobs in zip(gpus, assignments) if jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def matlab_job(matlab: str, task_name: str, expression: str) -> dict[str, Any]:
    folder = REPO_ROOT / "matlab_code" / "priority_validation"
    value = f"addpath({quote_matlab(folder)}); {expression}"
    return {"id": task_name, "command": [matlab, "-batch", value]}


def run_generate(config: dict[str, Any], tasks: dict[str, list[dict[str, Any]]], output: Path,
                 gpus: list[int], inventory: dict[int, dict[str, Any]]) -> None:
    matlab = str(config["runtime"]["matlab"])
    workers = int(config["runtime"]["max_workers"])
    illumination = [
        matlab_job(matlab, f"illumination_repeat_{repeat:02d}",
                   f"priority_validation_generate_illumination({quote_matlab(output)},'repeat',{repeat});")
        for repeat in range(1, 4)
    ]
    illumination.extend(
        matlab_job(matlab, f"calibration_batch_{batch:02d}",
                   f"priority_validation_generate_illumination({quote_matlab(output)},'calibration',{batch});")
        for batch in range(1, 17)
    )
    command_runner(illumination, gpus, inventory, output, "illumination", min(3, workers))
    combine = matlab_job(matlab, "combine_calibration",
                         f"priority_validation_combine_calibration({quote_matlab(output)});")
    command_runner([combine], [gpus[0]], inventory, output, "calibration", 1)

    scene_commands = []
    for task in tasks["scenes"]:
        scene = task["scene"]
        depth = "[]" if scene.depth_um is None else str(scene.depth_um)
        expression = (
            f"priority_validation_scene_task({quote_matlab(REPO_ROOT)},{quote_matlab(output)},"
            f"{quote_matlab(scene.family)},{depth},{task['repeat']});"
        )
        scene_commands.append(matlab_job(matlab, task["id"], expression))
    command_runner(scene_commands, gpus, inventory, output, "scenes", workers)
    target_commands = [
        matlab_job(matlab, task["id"],
                   f"priority_validation_target_task({quote_matlab(REPO_ROOT)},{quote_matlab(output)},"
                   f"{quote_matlab(task['sample'])},{task['repeat']});")
        for task in tasks["targets"]
    ]
    command_runner(target_commands, gpus, inventory, output, "loss_targets", workers)
    zero = matlab_job(matlab, "zero_control", f"priority_validation_zero_task({quote_matlab(output)});")
    command_runner([zero], [gpus[0]], inventory, output, "zero", 1)


def run_infer(config: dict[str, Any], tasks: dict[str, list[dict[str, Any]]], output: Path,
              gpus: list[int], inventory: dict[int, dict[str, Any]]) -> None:
    checkpoint = resolve_from_repo(config["experiment"]["checkpoint"])
    python = sys.executable
    commands = []
    for task in tasks["inference"]:
        if task["kind"] == "generated":
            sample_dir = output / "generated" / task["id"]
            destination = sample_dir / "network"
        else:
            sample_dir = Path(task["sample_dir"])
            destination = output / "existing_inference" / task["id"]
        contract = destination / "inference_contract.json"
        if contract.is_file():
            continue
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Partial inference output requires manual inspection: {destination}")
        command = [python, str(REPO_ROOT / "infer_dataset.py"), "--worker", "--checkpoint", str(checkpoint),
                   "--sample-dir", str(sample_dir), "--subset", str(task["subset"]), "--gpu", "0",
                   "--output-dir", str(destination)]
        commands.append({"id": f"infer_{task['id']}", "command": command})
    if commands:
        command_runner(commands, gpus, inventory, output, "inference",
                       int(config["runtime"]["max_workers"]))


def run_analysis(stage: str, config_path: Path, output: Path, gpu: int | None,
                 inventory: dict[int, dict[str, Any]] | None) -> None:
    command = [sys.executable, str(Path(__file__).with_name("priority_validation_analysis.py")),
               "--stage", stage, "--config", str(config_path), "--output-dir", str(output)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "tools")
    if gpu is not None and inventory is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(inventory[gpu]["uuid"])
        environment["SPECKLE_PHYSICAL_GPUS"] = str(gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def main() -> None:
    args = parser().parse_args()
    config_path, config = load_config(args.config)
    if args.output_dir:
        output = Path(args.output_dir).expanduser().resolve()
    else:
        output = resolve_from_repo(config["experiment"]["output_dir"])
    expected_checkpoint(config)
    tasks = build_tasks(config)
    planned = {"output": str(output), "stage": args.stage, "scene_repeats": len(tasks["scenes"]),
               "loss_targets": len(tasks["targets"]), "inference_tasks": len(tasks["inference"])}
    if args.dry_run:
        print(json.dumps(planned, indent=2, ensure_ascii=False)); return

    gpus: list[int] = []
    inventory: dict[int, dict[str, Any]] = {}
    if args.stage in {"all", "generate", "infer", "loss"}:
        gpus, inventory = select_gpus(args.gpus, config, allow_busy=args.allow_busy_gpus)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume for existing run: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = check_source_manifest(manifest["source_sha256"])
        if changed:
            raise RuntimeError(f"Source files changed since run start: {changed}")
    else:
        manifest = {"status": "running", "started_unix": time.time(), "protocol": planned,
                    "selected_gpus": gpus, "gpu_inventory": inventory,
                    "source_sha256": source_manifest(source_paths(config_path, config))}
        json_dump(manifest_path, manifest)

    stages = ["generate", "infer", "loss", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"START {stage}", flush=True)
        if stage == "generate":
            run_generate(config, tasks, output, gpus, inventory)
        elif stage == "infer":
            run_infer(config, tasks, output, gpus, inventory)
        elif stage == "loss":
            run_analysis("loss", config_path, output, gpus[0], inventory)
        else:
            run_analysis("report", config_path, output, None, None)
        manifest.setdefault("completed_stages", []).append(stage)
        manifest["completed_stages"] = list(dict.fromkeys(manifest["completed_stages"]))
        json_dump(manifest_path, manifest)
    if set(manifest.get("completed_stages", [])) >= {"generate", "infer", "loss", "report"}:
        manifest["status"] = "complete"; manifest["finished_unix"] = time.time(); json_dump(manifest_path, manifest)


if __name__ == "__main__":
    main()
