"""Audited six-A40 continuation; immutable numerical protocol and old code."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import h5py

from tools.dataset_v3_parallel_common import (
    GPU_INDICES,
    missing_jobs,
    partition_jobs,
    validate_gpu,
)
from tools.run_dataset_v3 import MATLAB, now, save, verify_code, verify_sources
from tools.run_dataset_v3_preview import (
    REPO,
    cpu_environment,
    matlab_string,
    read_json,
    sha256,
)
from tools.run_t04_axial_preview import summarize_unittest
from utils.dataset_splits import V3_SPLITS
from utils.experiment_paths import next_experiment_path


def gpu_inventory() -> list[dict]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    result = []
    for row in rows:
        index, uuid, name, used, total, utilization = [
            v.strip() for v in row.split(",")
        ]
        result.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "used_mib": int(used),
                "free_mib": int(total) - int(used),
                "utilization_percent": int(utilization),
                "compute_processes": [a for a in apps if a.startswith(uuid)],
            }
        )
    if [g["physical_index"] for g in result] != list(GPU_INDICES) or any(
        "A40" not in g["name"] for g in result
    ):
        raise RuntimeError("Expected exactly the six authorized A40 identities")
    return result


def alive(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(")", 1)[1].strip().split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def update(root: Path, control: Path, stage: str, **extra) -> None:
    record = {
        "stage": stage,
        "dataset_complete": stage == "complete",
        "worker_pid": os.getpid(),
        "updated_beijing": now(),
        "control_root": str(control),
        "physical_gpus": list(GPU_INDICES),
        **extra,
    }
    save(root / "run_status.json", record)
    save(control / "status.json", record)
    print(f"{now()} | {stage} | {extra}", flush=True)


def verify_runtime(root: Path, control: Path) -> None:
    verify_code(root)
    for name, digest in read_json(control / "authorization.json")[
        "code_sha256"
    ].items():
        if sha256(REPO / name) != digest:
            raise RuntimeError(f"Parallel runtime changed: {name}")


def launch_matlab(
    root: Path, control: Path, label: str, expression: str, uuid=""
) -> subprocess.Popen:
    verify_runtime(root, control)
    env = cpu_environment(control)
    env["CUDA_VISIBLE_DEVICES"] = uuid
    # Independent preferences avoid MATLAB workers writing a shared settings file.
    env["MATLAB_PREFDIR"] = str(control / "runtime" / label)
    paths = [
        REPO / "matlab_code" / n
        for n in ("dataset_v3", "cell_dataset", "pilot_dataset", "Util", "Solver")
    ]
    expression = (
        "addpath(" + ",".join(matlab_string(p) for p in paths) + "); " + expression
    )
    log = control / "logs" / f"{label}.log"
    with log.open("x") as stream:
        process = subprocess.Popen(
            [str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression],
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    save(
        control / "processes" / f"{label}.json",
        {
            "pid": process.pid,
            "label": label,
            "gpu_uuid": uuid,
            "started_beijing": now(),
        },
    )
    return process


def wait_process(control: Path, label: str, process: subprocess.Popen) -> None:
    code = process.wait()
    path = control / "processes" / f"{label}.json"
    record = read_json(path)
    record.update(exit_code=code, finished_beijing=now())
    save(path, record)
    if code:
        raise RuntimeError(f"{label} failed with exit {code}; inspect its log")


def run_cpu(root: Path, control: Path, label: str, expression: str) -> None:
    wait_process(control, label, launch_matlab(root, control, label, expression))


def cutover(root: Path, control: Path) -> None:
    parent = read_json(root / "worker_launch.json")["pid"]
    child_record = read_json(root / "active_child.json")
    child = child_record["pid"]
    if (
        child_record["stage"] != "T04_reconstruct"
        or not alive(parent)
        or not alive(child)
    ):
        raise RuntimeError("Serial task changed stage/state; re-audit before cutover")
    parent_cmd = (
        Path(f"/proc/{parent}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    )
    child_cmd = (
        Path(f"/proc/{child}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    )
    if (
        "tools.run_dataset_v3 --worker" not in parent_cmd
        or str(root) not in parent_cmd
        or "MATLAB" not in child_cmd
        or str(root) not in child_cmd
    ):
        raise RuntimeError("Refusing to signal unverified process identities")
    update(
        root,
        control,
        "parallel_cutover_waiting_for_frame_checkpoint",
        old_parent=parent,
        old_child=child,
    )
    before = len(list((root / "T04/recon_frames").glob("frame_*.mat")))
    # Stop only our serial scheduler so it cannot launch another stage.
    os.kill(parent, signal.SIGTERM)
    deadline = time.monotonic() + 180
    while alive(child) and time.monotonic() < deadline:
        if len(list((root / "T04/recon_frames").glob("frame_*.mat"))) > before:
            break
        time.sleep(1)
    if alive(child):
        os.kill(child, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while (alive(parent) or alive(child)) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(parent) or alive(child):
        raise RuntimeError(
            "Owned serial processes did not exit; no parallel writes started"
        )
    preserved = {}
    for sample in ("T04", "V03"):
        for folder in ("recon_frames", "subsets"):
            for p in (root / sample / folder).glob("*.mat"):
                prefix = "frame_" if folder == "recon_frames" else "subset_"
                if not p.name.startswith(prefix):
                    if not p.name.startswith("tp"):
                        raise RuntimeError(
                            f"Unexpected non-job MAT file; left untouched: {p}"
                        )
                    target = (
                        control / "interrupted_temporary" / sample / folder / p.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    p.rename(
                        target
                    )  # Recoverable quarantine of our unpublished atomic-save temporary.
                    continue
                with h5py.File(p, "r") as handle:
                    if int(handle["iterations"][()].item()) != 3:
                        raise RuntimeError(f"Wrong completed RL iteration count: {p}")
                preserved[str(p.relative_to(root))] = sha256(p)
    save(control / "preserved_reconstruction_sha256.json", preserved)
    save(
        control / "cutover.json",
        {
            "complete": True,
            "stopped_owned_pids": [parent, child],
            "stopped_at_beijing": now(),
            "T04_completed_frames": len(
                list((root / "T04/recon_frames").glob("frame_*.mat"))
            ),
            "preserved_completed_mat_count": len(preserved),
            "other_processes_signaled": False,
        },
    )


def make_plan(root: Path, control: Path, sample: str, stage: str) -> Path:
    authorization = read_json(control / "authorization.json")
    shards = partition_jobs(missing_jobs(root / sample, stage))
    workers = [
        {
            "physical_index": i,
            "gpu_uuid": authorization["gpus"][i]["uuid"],
            "jobs": shard,
        }
        for i, shard in enumerate(shards)
    ]
    plan = {
        "version": 1,
        "authorized": True,
        "allowed_physical_gpu_indices": list(GPU_INDICES),
        "dataset_root": str(root),
        "control_root": str(control),
        "sample_id": sample,
        "stage": stage,
        "workers": workers,
        "authorization_file": str(control / "authorization.json"),
        "created_beijing": now(),
    }
    path = control / f"{sample}_{stage}_jobs.json"
    if path.exists():
        raise RuntimeError("Refusing to overwrite a dispatched immutable job plan")
    save(path, plan)
    return path


def run_shards(root: Path, control: Path, sample: str, stage: str) -> None:
    plan_path = make_plan(root, control, sample, stage)
    plan = read_json(plan_path)
    if stage == "reconstruct":
        expected = read_json(control / "authorization.json")["gpus"]
        while True:
            inventory = gpu_inventory()
            save(control / f"{sample}_gpu_gate.json", inventory)
            if all(validate_gpu(g, e) for g, e in zip(inventory, expected)):
                break
            update(root, control, "parallel_waiting_for_gpu_capacity", sample=sample)
            time.sleep(30)
    processes = []
    try:
        for slot, worker in enumerate(plan["workers"]):
            if not worker["jobs"]:
                continue
            label = f"{sample}_{stage}_slot{slot}"
            expression = f"dataset_v3_run_shard({matlab_string(plan_path)},{slot});"
            processes.append(
                (
                    label,
                    launch_matlab(
                        root,
                        control,
                        label,
                        expression,
                        worker["gpu_uuid"] if stage == "reconstruct" else "",
                    ),
                )
            )
        # On failure stop only sibling processes from this Popen registry.
        pending = dict(processes)
        while pending:
            for label, process in list(pending.items()):
                if process.poll() is not None:
                    wait_process(control, label, process)
                    del pending[label]
            time.sleep(2)
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                process.wait(timeout=45)
        raise


def validate_sample(root: Path, control: Path, sample: str) -> None:
    run_cpu(
        root,
        control,
        f"{sample}_validate",
        f"dataset_v3_run_sample({matlab_string(sample)},{matlab_string(root)},'validate');",
    )
    for stage in ("sensor", "reconstruct", "validate"):
        save(
            root / "progress" / f"{sample}_{stage}_complete.json",
            {
                "complete": True,
                "finished_beijing": now(),
                "parallel_control": str(control),
            },
        )


def publish(root: Path, control: Path) -> None:
    from datasets.matlab_multivolume_dataset import (
        DatasetItemKey,
        _read_input,
        _read_targets,
        load_dataset_index,
    )
    from tools.audit_dataset_manifest import audit

    verify_runtime(root, control)
    verify_sources(root)
    changed = [
        name
        for name, digest in read_json(
            control / "preserved_reconstruction_sha256.json"
        ).items()
        if sha256(root / name) != digest
    ]
    save(
        control / "preserved_results_verification.json",
        {"unchanged": not changed, "changed_files": changed},
    )
    if changed:
        raise RuntimeError("A previously completed reconstruction was modified")
    rows = []
    for split, ids in V3_SPLITS.items():
        for sample in ids:
            p = root / sample
            v = read_json(p / "validation_manifest.json")
            if (
                not v["complete"]
                or v["sample_id"] != sample
                or v["frame_count"] != 100
                or v["subset_count"] != 10
            ):
                raise RuntimeError(f"Incomplete validation: {sample}")
            rows.append(
                {
                    "sample_id": sample,
                    "object_group_id": sample,
                    "split": split,
                    "sample_dir": str(p),
                    "validation_manifest": str(p / "validation_manifest.json"),
                    "artifact_count": v["artifact_count"],
                }
            )
    counts = {k: len(v) for k, v in V3_SPLITS.items()}
    final = {
        "version": 3,
        "producer": "run_dataset_v3_parallel/1",
        "complete": True,
        "dataset_root": str(root),
        "samples": rows,
        "counts": {
            **counts,
            "objects": 17,
            "frames_per_object": 100,
            "subsets_per_object": 10,
        },
        "execution_resources": str(control / "authorization.json"),
        "physical_gpu_indices": list(GPU_INDICES),
        "training_started": False,
        "completed_beijing": now(),
    }
    splits = {
        "version": 3,
        "dataset_complete": True,
        "counts": counts,
        "samples": rows,
        "split_unit": "entire_object_including_all_frames_subsets_and_derivatives",
    }
    report = audit(root, workers=2, split_record=splits, final_record=final)
    save(root / "final_hash_audit.json", report)
    if not report["complete"]:
        raise RuntimeError(f"Dataset hash audit failed: {report['errors'][:5]}")
    for row in rows:
        for i in range(1, 11):
            key = DatasetItemKey(
                row["sample_id"], i, row["split"], Path(row["sample_dir"])
            )
            _read_input(key)
            targets = _read_targets(key, include_ground_truth=key.split != "train")
            if key.split == "train" and "ground_truth" in targets:
                raise RuntimeError("Training GT leakage")
    save(root / "FINAL_DATASET_MANIFEST.json", final)
    save(root / "dataset_splits.json", splits)
    try:
        indexed, fingerprint = load_dataset_index(root)
    except BaseException:
        final["complete"] = False
        splits["dataset_complete"] = False
        save(root / "FINAL_DATASET_MANIFEST.json", final)
        save(root / "dataset_splits.json", splits)
        raise
    save(
        root / "python_reader_verification.json",
        {
            "complete": True,
            "fingerprint": fingerprint,
            "counts": {k: len(v) for k, v in indexed.items()},
            "all_170_items_read": True,
        },
    )
    (root / "report_zh.md").write_text(
        f"# 新版 17 对象数据集完成\n\n训练 P01–P11；验证 V01/V02/V03；测试 T02/T03/T04。\n\n全部 100 帧、10 个 10/90 子集、RL3、Mean/Taylor raw 与 sqrt、逐帧重建及 AlgoRIM 输入堆栈已完成并验收。全 170 项 Python 输入/目标读取、MATLAB 数值验证及全部输出文件哈希已通过。\n\n资源在 {read_json(control / 'authorization.json')['authorized_beijing']} 按用户要求由单卡扩展至物理 0–5 号 A40；0 号允许原有空闲上下文共存，1–5 号仅在空闲时启动。初始单卡配置作为历史保留，当前资源授权见 `{control / 'authorization.json'}`。\n\n仅做完整帧/子集分片，未改变 RL3 算法、PSF、随机种子、10/90 划分或真值。既有重建哈希不变；六卡参考重建一致性记录在并行目录 progress/gpu*_parity.json。原始数据及旧预览未修改。\n\nT04 含 10/20/30/40 μm 中心间距；10 μm 原生相邻层无谷值采样，不能插值宣称光学分辨率。未训练网络，保留有 Set 基线；未执行外部 Windows AlgoRIM。\n",
        encoding="utf-8",
    )
    update(root, control, "complete", fingerprint=fingerprint)


def worker(root: Path, control: Path) -> None:
    with (control / ".controller.lock").open("a+") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_runtime(root, control)
            cutover(root, control)
            with (root / ".worker.lock").open("a+") as dataset_lock:
                fcntl.flock(dataset_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                update(root, control, "parallel_T04_reconstruction_and_V03_preparation")

                def prepare_v03():
                    expression = f"cfg=dataset_v3_simulation_config('V03',{matlab_string(root)}); dataset_v3_prepare_sample(cfg);"
                    run_cpu(root, control, "V03_prepare", expression)
                    run_shards(root, control, "V03", "sensor")

                with ThreadPoolExecutor(max_workers=1) as executor:
                    sensor_future = executor.submit(prepare_v03)
                    run_shards(root, control, "T04", "reconstruct")
                    validate_sample(root, control, "T04")
                    update(root, control, "parallel_waiting_for_V03_sensors")
                    sensor_future.result()
                update(root, control, "parallel_V03_reconstruction")
                run_shards(root, control, "V03", "reconstruct")
                validate_sample(root, control, "V03")
                update(root, control, "parallel_final_dataset_audit")
                publish(root, control)
        except BaseException as error:
            update(root, control, "failed", error=str(error))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.control:
        worker(root, args.control.resolve())
        return
    current_controller = root / "parallel_controller.json"
    if current_controller.exists() and alive(read_json(current_controller)["pid"]):
        raise RuntimeError(
            "A parallel controller is already running; refusing duplicate launch"
        )
    verify_code(root)
    inventory = gpu_inventory()
    # GPU0's currently running owned serial process is allowed before cutover.
    if not all(validate_gpu(g, g) for g in inventory[1:]):
        raise RuntimeError("A GPU among 1..5 is busy; no cutover performed")
    control = next_experiment_path(root, "parallel_dispatch")
    control.mkdir()
    for folder in (
        "logs",
        "progress",
        "processes",
        "runtime/matlab",
        "runtime/matplotlib",
        "source_snapshot",
    ):
        (control / folder).mkdir(parents=True)
    names = [
        "tools/run_dataset_v3_parallel.py",
        "tools/dataset_v3_parallel_common.py",
        "matlab_code/dataset_v3/dataset_v3_run_shard.m",
        "tests/test_dataset_v3_parallel.py",
    ]
    hashes = {name: sha256(REPO / name) for name in names}
    for name in names:
        target = control / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    save(
        control / "authorization.json",
        {
            "authorized": True,
            "authorized_beijing": now(),
            "physical_gpu_indices": list(GPU_INDICES),
            "allow_sharing_gpu0": True,
            "allow_sharing_gpu1_to_5": False,
            "gpus": inventory,
            "user_authorization": "把1-5号A40放开，把任务分到6张卡",
            "supersedes_gpu_restriction_only": str(root / "SIMULATION_APPROVED.json"),
            "frozen_numerical_protocol_unchanged": True,
            "code_sha256": hashes,
        },
    )
    env = cpu_environment(control)
    with (control / "logs/python_unittest.log").open("x") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("Full CPU Python tests failed; serial task remains running")
    save(
        control / "test_summary.json",
        summarize_unittest((control / "logs/python_unittest.log").read_text()),
    )
    expression = "issues=checkcode('matlab_code/dataset_v3/dataset_v3_run_shard.m','-id'); assert(~any(strcmp({issues.id},'PARSE'))); disp('Parallel MATLAB parse check passed');"
    run_cpu(root, control, "matlab_parse_check", expression)
    save(
        root / "EXECUTION_RESOURCES.json",
        {
            "physical_gpu_indices": list(GPU_INDICES),
            "authorization": str(control / "authorization.json"),
            "overrides_initial_hardware_restriction_only": True,
            "numerical_protocol_unchanged": True,
            "allow_sharing_gpu0": True,
            "allow_sharing_gpu1_to_5": False,
            "updated_beijing": now(),
        },
    )
    with (control / "launcher.log").open("x") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "tools.run_dataset_v3_parallel",
                "--root",
                str(root),
                "--control",
                str(control),
            ],
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        root / "parallel_controller.json",
        {"pid": process.pid, "control_root": str(control), "started_beijing": now()},
    )
    print(f"PARALLEL CONTROLLER pid={process.pid} control={control}", flush=True)


if __name__ == "__main__":
    main()
