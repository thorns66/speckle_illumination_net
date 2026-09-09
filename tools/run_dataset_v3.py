"""Build the approved 17-object revision without changing historical data."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from tools.run_dataset_v3_preview import (
    REPO,
    audit_source,
    cpu_environment,
    matlab_string,
    read_json,
    sha256,
    tree_signature,
)
from tools.run_t04_axial_preview import summarize_unittest
from utils.dataset_splits import V3_SPLITS
from utils.experiment_paths import next_experiment_path

NEW_IDS = ("T03", "T04", "V03")
SOURCE = REPO / "data" / "speckle_data_now"
PREVIEW = REPO / "data" / "speckle_dataset_v3_20260907_run01"
AXIAL = REPO / "data" / "t04_axial_preview_20260907_run02"
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")
MUTABLE = {
    "validation_manifest.mat",
    "validation_manifest.json",
    "full_run_report.mat",
    "full_run_report.json",
}


def save(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def matlab_text(dataset: h5py.Dataset) -> str:
    return "".join(chr(int(v)) for v in dataset[()].reshape(-1))


def rewrite_json(
    value,
    old_id: str,
    new_id: str,
    split: str,
    old_roots: list[str],
    root: str,
    field="",
):
    if isinstance(value, dict):
        return {
            k: rewrite_json(v, old_id, new_id, split, old_roots, root, k)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            rewrite_json(v, old_id, new_id, split, old_roots, root, field)
            for v in value
        ]
    if not isinstance(value, str):
        return value
    if field == "split":
        return split
    if value == old_id:
        return new_id
    for old in old_roots:
        if value == old or value.startswith(old + "/"):
            suffix = value[len(old) :]
            if suffix == "/" + old_id or suffix.startswith("/" + old_id + "/"):
                suffix = "/" + new_id + suffix[len(old_id) + 1 :]
            return root + suffix
    return value


def gpu_snapshot() -> dict:
    raw = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(raw) != 1:
        raise RuntimeError("Physical GPU 0 was not uniquely resolved")
    index, uuid, name, used, total, utilization = [x.strip() for x in raw[0].split(",")]
    if index != "0" or "A40" not in name:
        raise RuntimeError("Only physical 0 A40 is authorized")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return {
        "physical_index": 0,
        "uuid": uuid,
        "name": name,
        "used_mib": int(used),
        "total_mib": int(total),
        "free_mib": int(total) - int(used),
        "utilization_percent": int(utilization),
        "compute_processes": [s for s in apps if s.startswith(uuid)],
        "recorded_beijing": now(),
    }


def gpu_ready(snapshot: dict, uuid: str) -> bool:
    if (
        snapshot["physical_index"] != 0
        or snapshot["uuid"] != uuid
        or "A40" not in snapshot["name"]
    ):
        raise RuntimeError("GPU identity changed; refusing fallback")
    # Existing contexts may remain, as explicitly authorized by the user.
    return snapshot["free_mib"] >= 24 * 1024 and snapshot["utilization_percent"] <= 5


def state(root: Path, stage: str, **extra) -> None:
    save(
        root / "run_status.json",
        {
            "stage": stage,
            "dataset_complete": stage == "complete",
            "worker_pid": os.getpid(),
            "updated_beijing": now(),
            **extra,
        },
    )
    print(f"{now()} | {stage} | {extra}", flush=True)


def verify_code(root: Path) -> None:
    changed = [
        name
        for name, digest in read_json(root / "run_contract.json")["code_sha256"].items()
        if not (REPO / name).is_file() or sha256(REPO / name) != digest
    ]
    if changed:
        raise RuntimeError(f"Frozen runtime source changed: {changed}")


def run_matlab(root: Path, label: str, expression: str, gpu=False) -> None:
    verify_code(root)
    env = cpu_environment(root)
    if gpu:
        approval = read_json(root / "SIMULATION_APPROVED.json")
        while True:
            snapshot = gpu_snapshot()
            save(root / "progress" / "gpu0_gate.json", snapshot)
            if gpu_ready(snapshot, approval["gpu_uuid"]):
                break
            state(root, "waiting_for_gpu0_capacity", next_stage=label, gpu=snapshot)
            time.sleep(30)
        env["CUDA_VISIBLE_DEVICES"] = approval["gpu_uuid"]
    paths = [
        REPO / "matlab_code" / name
        for name in ("dataset_v3", "cell_dataset", "pilot_dataset", "Util", "Solver")
    ]
    expression = (
        "addpath(" + ",".join(matlab_string(p) for p in paths) + "); " + expression
    )
    state(root, label, device="physical_gpu0" if gpu else "CPU")
    logs = root / "logs"
    attempt = 1
    while (logs / f"{label}_attempt{attempt:02d}.log").exists():
        attempt += 1
    with (logs / f"{label}_attempt{attempt:02d}.log").open("x") as stream:
        process = subprocess.Popen(
            [str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression],
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        save(
            root / "active_child.json",
            {"pid": process.pid, "stage": label, "gpu": gpu, "started_beijing": now()},
        )
        code = process.wait()
    save(
        root / "active_child.json",
        {
            "pid": process.pid,
            "stage": label,
            "exit_code": code,
            "finished_beijing": now(),
        },
    )
    if code:
        raise RuntimeError(
            f"MATLAB stage {label} failed, exit {code}; no protocol changes or retries performed"
        )


def preview_sources() -> dict[str, Path]:
    return {"T03": PREVIEW / "T03", "T04": AXIAL / "T04", "V03": PREVIEW / "V03"}


def create_run() -> Path:
    # Do not mutate any old preview's pending-review flags.
    snapshot = gpu_snapshot()
    if not SOURCE.is_dir() or not MATLAB.is_file():
        raise ValueError("Missing historical source or MATLAB")
    truth_sources = {}
    for sample, folder in preview_sources().items():
        cfg = read_json(folder / "config.json")
        if cfg["sample_id"] != sample:
            raise ValueError("Wrong preview owner")
        if sample == "T04" and (
            cfg.get("geometry_revision") != "t04_axial_line_pairs_v2"
            or cfg["axial_board"]["separations_um"] != [10, 20, 30, 40]
        ):
            raise ValueError("T04 must be the reviewed 10/20/30/40 um revision")
        with h5py.File(folder / "truth.mat", "r") as handle:
            geometry_hash = matlab_text(handle["source_sha256"])
        truth_sources[sample] = {
            "source_dir": str(folder),
            "truth_mat_sha256": sha256(folder / "truth.mat"),
            "geometry_source_sha256": geometry_hash,
        }
    while True:
        root = next_experiment_path(REPO / "data", "speckle_dataset_v3_full")
        try:
            root.mkdir()
            break
        except FileExistsError:
            continue
    for folder in (
        "logs",
        "progress",
        "runtime/matlab",
        "runtime/matplotlib",
        "source_snapshot",
    ):
        (root / folder).mkdir(parents=True)
    paths = []
    for folder in ("dataset_v3", "cell_dataset", "pilot_dataset", "Util", "Solver"):
        paths += sorted((REPO / "matlab_code" / folder).glob("*.m"))
    paths += [
        REPO / p
        for p in (
            "tools/run_dataset_v3.py",
            "tools/run_dataset_v3_preview.py",
            "tools/run_t04_axial_preview.py",
            "tools/audit_dataset_manifest.py",
            "utils/dataset_splits.py",
            "utils/experiment_paths.py",
            "datasets/matlab_multivolume_dataset.py",
            "tests/test_dataset_v3_build.py",
        )
    ]
    hashes = {str(p.relative_to(REPO)): sha256(p) for p in paths}
    for relative in hashes:
        target = root / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    contract = {
        "version": 3,
        "created_beijing": now(),
        "source_root": str(SOURCE),
        "output_root": str(root),
        "dataset_complete": False,
        "network_training_authorized": False,
        "code_sha256": hashes,
        "baseline_config_sha256": sha256(REPO / "configs/multivolume_n10_no_mean.yaml"),
    }
    save(root / "run_contract.json", contract)
    save(
        root / "SIMULATION_APPROVED.json",
        {
            "approved": True,
            "approved_new_sample_ids": list(NEW_IDS),
            "truth_sources": truth_sources,
            "allowed_physical_gpu_indices": [0],
            "required_gpu_model": "A40",
            "gpu_uuid": snapshot["uuid"],
            "allow_sharing": True,
            "allow_fallback_gpu": False,
            "minimum_free_mib_before_gpu_stage": 24 * 1024,
            "authorization": "User approved reviewed truth and original dataset plan, then explicitly allowed physical GPU 0 despite four idle compute contexts",
            "approved_beijing": now(),
            "no_existing_process_termination": True,
            "training_authorized": False,
        },
    )
    save(root / "initial_gpu0.json", snapshot)
    state(root, "created")
    return root


def prepare_copies(root: Path) -> None:
    state(root, "source_audit_and_independent_copies")
    audit = audit_source(SOURCE)
    save(root / "historical_source_audit.json", audit)
    protected_roots = (SOURCE, PREVIEW, AXIAL)
    save(
        root / "protected_tree_metadata.json",
        {str(p): tree_signature(p) for p in protected_roots},
    )
    protected = {}
    for parent in protected_roots:
        for path in sorted(parent.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"Source symlink requires explicit review: {path}")
            if path.is_file():
                protected[str(path)] = sha256(path)
    save(root / "protected_content_sha256.json", protected)
    psf = sorted((REPO / "psf").glob("*.mat"))
    if len(psf) != 1:
        raise RuntimeError("PSF is not uniquely resolved")
    with h5py.File(psf[0], "r") as handle:
        depths = handle["x3objspace"][()].reshape(-1) * 1e6
        if any(
            np.count_nonzero(np.abs(depths - z) < 1e-4) != 1 for z in range(10, 101, 10)
        ):
            raise RuntimeError("PSF lacks the exact ten depth planes")
        psf_record = {
            "path": str(psf[0]),
            "sha256": sha256(psf[0]),
            "depths_um": depths.tolist(),
            "H_hdf5_shape": list(handle["H"].shape),
            "Ht_hdf5_shape": list(handle["Ht"].shape),
        }
    save(root / "psf_preflight.json", psf_record)
    jobs = []
    for split, objects in V3_SPLITS.items():
        for sample in objects:
            if sample in NEW_IDS:
                source = preview_sources()[sample]
            else:
                source = SOURCE / ("T01" if sample == "P11" else sample)
            target = root / sample
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite partial copy: {target}")
            shutil.copytree(
                source,
                target,
                ignore=lambda folder, names: [n for n in names if n in MUTABLE],
            )
            copied = 0
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    original = source / path.relative_to(target)
                    if sha256(path) != protected[str(original)]:
                        raise RuntimeError(f"Copy differs from frozen source: {path}")
                    if path.stat().st_ino == original.stat().st_ino:
                        raise RuntimeError("Hard-linked mutable copy is forbidden")
                    copied += 1
            if sample not in NEW_IDS:
                with h5py.File(source / "prepared.mat", "r") as handle:
                    recorded = matlab_text(handle["cfg"]["output_root"])
                old_roots = sorted({str(SOURCE), recorded}, key=len, reverse=True)
                job = {
                    "sample_id": sample,
                    "source_sample_id": source.name,
                    "split": split,
                    "source_dir": str(source),
                    "target_dir": str(target),
                    "recorded_roots": old_roots,
                    "copied_file_count": copied,
                    "copy_sha256_exact": True,
                }
                jobs.append(job)
                for path in target.rglob("*.json"):
                    value = read_json(path)
                    revised = rewrite_json(
                        value, source.name, sample, split, old_roots, str(root)
                    )
                    if value != revised:
                        save(path, revised)
            state(root, "copy_verified", sample=sample, files=copied)
    save(root / "migration_jobs.json", {"samples": jobs})
    save(root / "copies_complete.json", {"complete": True, "objects": 17})


def cpu_tests(root: Path) -> None:
    env = cpu_environment(root)
    with (root / "logs" / "python_unittest.log").open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("Python CPU unittest failed")
    python_tests = summarize_unittest(
        (root / "logs" / "python_unittest.log").read_text()
    )
    expression = "r=runtests({'matlab_code/dataset_v3/test_dataset_v3_migration.m','matlab_code/dataset_v3/test_t04_axial_truth.m','matlab_code/dataset_v3/test_dataset_v3_truth.m','matlab_code/cell_dataset/test_cell_truth.m'}); disp(table(r)); "
    expression += "s=struct('total',numel(r),'passed',nnz([r.Passed]),'failed',nnz([r.Failed]),'incomplete',nnz([r.Incomplete])); "
    expression += f"cell_write_json({matlab_string(root / 'logs/matlab_test_summary.json')},s); assert(all([r.Passed]),'CPU tests failed');"
    run_matlab(root, "cpu_tests", expression)
    save(
        root / "test_summary.json",
        {
            "python": python_tests,
            "matlab": read_json(root / "logs/matlab_test_summary.json"),
        },
    )


def verify_sources(root: Path) -> None:
    changed = [
        p
        for p, digest in read_json(root / "protected_content_sha256.json").items()
        if not Path(p).is_file() or sha256(Path(p)) != digest
    ]
    trees = read_json(root / "protected_tree_metadata.json")
    changed_trees = [
        p for p, metadata in trees.items() if tree_signature(Path(p)) != metadata
    ]
    baseline = (
        sha256(REPO / "configs/multivolume_n10_no_mean.yaml")
        == read_json(root / "run_contract.json")["baseline_config_sha256"]
    )
    save(
        root / "source_preservation.json",
        {
            "all_hashed_source_files_unchanged": not changed,
            "changed_files": changed,
            "changed_trees": changed_trees,
            "set_baseline_config_unchanged": baseline,
        },
    )
    if changed or changed_trees or not baseline:
        raise RuntimeError("Protected historical data, preview, or baseline changed")


def publish(root: Path) -> None:
    from datasets.matlab_multivolume_dataset import (
        DatasetItemKey,
        _read_input,
        _read_targets,
        load_dataset_index,
    )
    from tools.audit_dataset_manifest import audit

    samples = []
    for split, objects in V3_SPLITS.items():
        for sample in objects:
            folder = root / sample
            validation = read_json(folder / "validation_manifest.json")
            if not validation["complete"] or validation["sample_id"] != sample:
                raise RuntimeError(f"Incomplete sample {sample}")
            if validation["frame_count"] != 100 or validation["subset_count"] != 10:
                raise RuntimeError(f"Unexpected counts in {sample}")
            samples.append(
                {
                    "sample_id": sample,
                    "object_group_id": sample,
                    "split": split,
                    "sample_dir": str(folder),
                    "validation_manifest": str(folder / "validation_manifest.json"),
                    "artifact_count": validation["artifact_count"],
                    "reused": sample not in NEW_IDS,
                }
            )
    verify_sources(root)
    verify_code(root)
    counts = {k: len(v) for k, v in V3_SPLITS.items()}
    final = {
        "version": 3,
        "producer": "run_dataset_v3/1",
        "complete": True,
        "dataset_root": str(root),
        "counts": {
            **counts,
            "objects": 17,
            "frames_per_object": 100,
            "subsets_per_object": 10,
        },
        "samples": samples,
        "completed_beijing": now(),
        "training_started": False,
        "algorim_status": "20 input stacks per object; external Windows AlgoRIM not executed",
    }
    splits = {
        "version": 3,
        "dataset_complete": True,
        "counts": counts,
        "samples": samples,
        "split_unit": "entire_object_including_all_frames_subsets_and_derivatives",
    }
    try:
        report = audit(root, workers=2, split_record=splits, final_record=final)
        save(root / "final_hash_audit.json", report)
        if not report["complete"]:
            raise RuntimeError(f"Final hash audit failed: {report['errors'][:5]}")
        loaded = {}
        for split, objects in V3_SPLITS.items():
            seen = set()
            for sample in objects:
                for subset in range(1, 11):
                    key = DatasetItemKey(sample, subset, split, root / sample)
                    item = _read_input(key)
                    targets = _read_targets(key, include_ground_truth=split != "train")
                    if split == "train" and (
                        "ground_truth" in item or "ground_truth" in targets
                    ):
                        raise RuntimeError("Training GT leakage")
                    seen.add(sample)
            loaded[split] = {"items": len(objects) * 10, "objects": sorted(seen)}
        # No complete manifest exists until hashes and every real data item pass.
        save(root / "FINAL_DATASET_MANIFEST.json", final)
        save(root / "dataset_splits.json", splits)
        indexed, fingerprint = load_dataset_index(root)
        save(
            root / "python_reader_verification.json",
            {
                "complete": True,
                "fingerprint": fingerprint,
                "counts": {k: len(v) for k, v in indexed.items()},
                "loaded": loaded,
            },
        )
    except BaseException:
        final["complete"] = False
        splits["dataset_complete"] = False
        save(root / "FINAL_DATASET_MANIFEST.json", final)
        save(root / "dataset_splits.json", splits)
        raise
    text = f"# 新版 17 对象数据集完成\n\n输出：`{root}`\n\n训练 P01–P11（P07/P09 回归，旧 T01 副本改为 P11）；验证 V01/V02/V03；测试 T02/T03/T04。\n\nT03 是分辨率板；T04 是含 10/20/30/40 μm 中心间距的双层短线板；V03 为 60 颗微球。\n\n每对象 100 帧、10 个 10/90 子集、RL3。已完成逐帧 legacy/physics 重建、子集 Mean/Taylor raw 与 sqrt、AlgoRIM 输入堆栈、MATLAB 深度校验、全文件哈希及 Python 全 170 项读取。\n\n旧数据/预览独立保留；迁移只改文本元数据，数值数组与种子经校验不变。GPU 仅物理 0 A40，按用户最新授权允许与原有空闲上下文共存。没有启动网络训练，没有运行外部 AlgoRIM。\n\n10 μm 双层在原生 10 μm 网格相邻，不能通过插值谷值宣称光学分辨率。新旧划分的模型指标不能直接混算；有 Set 基线不替换。\n"
    (root / "report_zh.md").write_text(text, encoding="utf-8")
    state(root, "complete", fingerprint=fingerprint)


def worker(root: Path) -> None:
    lock = (root / ".worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        verify_code(root)
        if not (root / "test_summary.json").exists():
            cpu_tests(root)
        if not (root / "copies_complete.json").exists():
            prepare_copies(root)
        if not (root / "migration_complete.json").exists():
            expression = ""
            for job in read_json(root / "migration_jobs.json")["samples"]:
                marker = root / "progress" / f"migration_{job['sample_id']}.json"
                if not marker.exists():
                    expression += f"dataset_v3_migrate_sample({matlab_string(root)},{matlab_string(job['sample_id'])}); "
            if expression:
                run_matlab(root, "migrate_legacy", expression)
            save(root / "migration_complete.json", {"complete": True, "objects": 14})
        for sample in NEW_IDS:
            for stage in ("sensor", "reconstruct", "validate"):
                marker = root / "progress" / f"{sample}_{stage}_complete.json"
                if marker.exists():
                    continue
                run_matlab(
                    root,
                    f"{sample}_{stage}",
                    f"dataset_v3_run_sample({matlab_string(sample)},{matlab_string(root)},{matlab_string(stage)});",
                    gpu=stage == "reconstruct",
                )
                save(marker, {"complete": True, "finished_beijing": now()})
        publish(root)
    except BaseException as error:
        state(root, "failed", error=str(error))
        raise
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if bool(args.start) == bool(args.worker):
        parser.error("Choose --start (new dated run) or --worker ROOT (resume)")
    if args.worker:
        worker(args.worker.resolve())
        return
    root = create_run()
    with (root / "launcher.log").open("x") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "tools.run_dataset_v3", "--worker", str(root)],
            cwd=REPO,
            env=cpu_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        root / "worker_launch.json",
        {"pid": process.pid, "root": str(root), "started_beijing": now()},
    )
    print(f"STARTED pid={process.pid} root={root}", flush=True)


if __name__ == "__main__":
    main()
