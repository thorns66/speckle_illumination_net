"""Simulate the reviewed P12 truth and publish it as an additive V4 dataset."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat

from datasets.matlab_multivolume_dataset import (
    DatasetItemKey,
    _read_input,
    _read_targets,
    load_dataset_index,
)
from tools.audit_dataset_manifest import audit
from utils.dataset_splits import V4_SPLITS


REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/speckle_dataset_v3_full_20260907_run01"
PREVIEW_ROOT = REPO / "data/root_cell_P12_pixel_preview_20260909_run02"
PREVIEW = PREVIEW_ROOT / "P12"
APPROVED_PREVIEW_TRUTH_SHA256 = (
    "cf5bffa74e845d968b26cafcc521a24117180e0fd55f748b02830ddc1218cb8a"
)
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")
GPU_UUID = "GPU-c0e00376-9bc0-1308-2ef6-018f6561eaf7"
MATLAB_PATHS = tuple(
    REPO / "matlab_code" / name
    for name in ("dataset_v3", "cell_dataset", "pilot_dataset", "Util", "Solver")
)
TOP_LEVEL_HISTORY_FILES = (
    "dataset_splits.json",
    "FINAL_DATASET_MANIFEST.json",
    "final_hash_audit.json",
    "python_reader_verification.json",
    "report_zh.md",
    "run_status.json",
    "SIMULATION_APPROVED.json",
)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def matlab_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def cpu_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["MPLCONFIGDIR"] = str(DATASET / "runtime/p12_matplotlib")
    env["MATLAB_PREFDIR"] = str(DATASET / "runtime/p12_matlab")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MATLAB_PREFDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def status(stage: str, **extra: object) -> None:
    record = {
        "stage": stage,
        "sample_id": "P12",
        "updated_beijing": now(),
        "dataset_complete": stage == "complete",
        **extra,
    }
    save_json(DATASET / "P12_EXTENSION_STATUS.json", record)
    print(f"{record['updated_beijing']} | {stage} | {extra}", flush=True)


def gpu_snapshot() -> dict:
    row = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    index, uuid, name, used, total, utilization = [v.strip() for v in row.split(",")]
    if index != "0" or uuid != GPU_UUID or "A40" not in name:
        raise RuntimeError("Physical GPU 0 identity changed; refusing any fallback")
    processes = subprocess.check_output(
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
        "compute_processes": [line for line in processes if line.startswith(uuid)],
        "recorded_beijing": now(),
    }


def run_matlab(label: str, expression: str, *, gpu: bool = False) -> None:
    env = cpu_environment()
    device = "CPU"
    if gpu:
        while True:
            snapshot = gpu_snapshot()
            save_json(DATASET / "progress/P12_gpu0_gate.json", snapshot)
            if snapshot["free_mib"] >= 24 * 1024:
                break
            status("waiting_for_gpu0_capacity", free_mib=snapshot["free_mib"])
            time.sleep(30)
        env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        device = "physical_gpu0"
    paths = ",".join(matlab_string(path) for path in MATLAB_PATHS)
    command = f"addpath({paths}); {expression}"
    logs = DATASET / "P12/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{label}.log"
    status(label, device=device, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(MATLAB),
                "-singleCompThread",
                "-softwareopengl",
                "-batch",
                command,
            ],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"MATLAB stage {label} failed with exit code {code}")


def verify_reviewed_truth(path: Path) -> dict:
    truth = loadmat(path, simplify_cells=True)
    cfg = truth["cfg"]
    metrics = truth["metrics"]
    if (
        cfg["sample_id"] != "P12"
        or cfg["split"] != "train"
        or cfg["geometry_kind"] != "nonoverlapping_deformed_cell_section"
        or cfg["geometry_version"] != "root_native_pixel_v4"
        or int(cfg["cell_count"]) != 18
    ):
        raise RuntimeError("P12 truth owner or reviewed geometry changed")
    expected_layers = np.array([40, 50, 60])
    mass = np.asarray(truth["ground_truth"]).sum(axis=(0, 1))
    occupied = np.asarray(cfg["z_um"])[mass > 0]
    if not np.array_equal(occupied, expected_layers):
        raise RuntimeError(f"P12 occupies unexpected layers: {occupied.tolist()}")
    required_zero = (
        int(metrics["filled_cell_overlap_subpixels"]) == 0
        and int(metrics["projected_cell_overlap_subpixels"]) == 0
        and int(metrics["maximum_cell_ownership"]) == 1
        and bool(metrics["all_cells_and_lumens_connected"])
        and bool(metrics["all_individual_lumens_dark"])
        and int(metrics["out_of_allowed_layers_nonzero_voxels"]) == 0
    )
    if not required_zero:
        raise RuntimeError("P12 reviewed non-overlap or depth audit changed")
    calibration = metrics["calibration"]
    if (
        bool(calibration["real_resizing_applied"])
        or not bool(calibration["no_forward_or_network_run"])
        or int(calibration["accepted_real_object_ring_count"]) != 25
        or not np.isclose(calibration["real_object_diameter_px"]["median"], 44.0)
        or not np.isclose(
            calibration["synthetic_wall_peak_diameter_px"]["median"], 44.0
        )
    ):
        raise RuntimeError("P12 native-pixel size calibration changed")
    generator = REPO / "tools/run_root_pixel_preview.py"
    if truth["source_sha256"] != sha256(generator):
        raise RuntimeError("P12 generator source hash no longer matches truth.mat")
    if path.resolve() == (PREVIEW / "truth.mat").resolve() and (
        sha256(path) != APPROVED_PREVIEW_TRUTH_SHA256
    ):
        raise RuntimeError("P12 truth is not the exact preview approved by the user")
    return {
        "geometry_source_sha256": truth["source_sha256"],
        "truth_mat_sha256": sha256(path),
        "occupied_z_um": occupied.astype(int).tolist(),
        "cell_count": int(cfg["cell_count"]),
        "touching_pairs": int(metrics["touching_pairs"]),
        "mass_fraction": np.asarray(metrics["mass_fraction"]).tolist(),
    }


def patch_copied_truth(source_truth: Path, target_truth: Path) -> None:
    temporary = target_truth.with_name("truth.approved.tmp.mat")
    numeric_cfg_fields = (
        "schema_version",
        "geometry_seed",
        "illumination_seed",
        "subset_seed",
        "image_size",
        "object_pixel_pitch_um",
        "x_um",
        "y_um",
        "z_um",
        "fine_dz_um",
        "fine_z_um",
        "frame_count",
        "input_frames",
        "holdout_frames",
        "subset_count",
        "iterations",
        "allowed_physical_gpu_indices",
        "lateral_quadrature_samples_per_pixel",
        "wall_sigma_range_um",
        "cell_count",
        "allowed_truth_z_um",
        "local_slice_full_support_thickness_um",
    )
    matlab_fields = "{" + ",".join(matlab_string(name) for name in numeric_cfg_fields) + "}"
    expression = (
        f"s=load({matlab_string(source_truth)}); "
        f"names={matlab_fields}; for k=1:numel(names), "
        "s.cfg.(names{k})=double(s.cfg.(names{k})); end; "
        "s.input_indices=double(s.input_indices); "
        "s.holdout_indices=double(s.holdout_indices); "
        f"s.cfg.output_root={matlab_string(DATASET)}; "
        f"s.cfg.sample_dir={matlab_string(DATASET / 'P12')}; "
        "s.cfg.stage='approved_full_simulation'; "
        "s.cfg.morphology_approved=true; "
        "s.approval_status='approved_by_user_for_full_simulation_20260909'; "
        f"save({matlab_string(temporary)},'-struct','s','-v7'); "
        f"movefile({matlab_string(temporary)},{matlab_string(target_truth)},'f');"
    )
    run_matlab("P12_approve_truth", expression)
    before = loadmat(source_truth)
    after = loadmat(target_truth)
    for name in ("ground_truth", "ground_truth_fine", "input_indices", "holdout_indices"):
        if not np.array_equal(before[name], after[name]):
            raise RuntimeError(f"Approved truth rewrite changed numeric array {name}")


def prepare() -> None:
    destination = DATASET / "P12"
    source_truth = PREVIEW / "truth.mat"
    reviewed = verify_reviewed_truth(source_truth)
    if destination.exists():
        if not (DATASET / "P12_EXTENSION_CONTRACT.json").is_file():
            raise RuntimeError("P12 exists without this extension's provenance record")
        sensor_files = list((destination / "sensor_frames").glob("frame_*.mat"))
        marker = DATASET / "progress/P12_sensor_complete.json"
        if not sensor_files and not marker.exists():
            for relative in (
                "prepared.mat",
                "simulation_config.json",
                "previews/ground_truth_float.tif",
            ):
                partial = destination / relative
                if partial.exists():
                    partial.unlink()
            patch_copied_truth(source_truth, destination / "truth.mat")
            approved = verify_reviewed_truth(destination / "truth.mat")
            approval = read_json(DATASET / "SIMULATION_APPROVED.json")
            approval["truth_sources"]["P12"]["truth_mat_sha256"] = approved[
                "truth_mat_sha256"
            ]
            save_json(DATASET / "SIMULATION_APPROVED.json", approval)
            approval_record = read_json(destination / "approval_record.json")
            approval_record["approved_truth_sha256"] = approved["truth_mat_sha256"]
            approval_record["numeric_config_repaired_beijing"] = now()
            save_json(destination / "approval_record.json", approval_record)
            contract = read_json(DATASET / "P12_EXTENSION_CONTRACT.json")
            contract["approved_copy"] = approved
            contract["source_sha256"]["tools/run_p12_dataset_extension.py"] = sha256(
                REPO / "tools/run_p12_dataset_extension.py"
            )
            contract["numeric_config_repaired_beijing"] = now()
            save_json(DATASET / "P12_EXTENSION_CONTRACT.json", contract)
        status("prepare_resume", destination=str(destination))
        return
    split = read_json(DATASET / "dataset_splits.json")
    final = read_json(DATASET / "FINAL_DATASET_MANIFEST.json")
    if (
        split.get("version") != 3
        or final.get("version") != 3
        or not split.get("dataset_complete")
        or not final.get("complete")
    ):
        raise RuntimeError("Expected the complete 17-object V3 dataset before adding P12")
    indexed, parent_fingerprint = load_dataset_index(DATASET)
    if {name: len(items) for name, items in indexed.items()} != {
        "train": 110,
        "validation": 30,
        "test": 30,
    }:
        raise RuntimeError("Parent dataset item counts changed")
    old_reader = read_json(DATASET / "python_reader_verification.json")
    if old_reader.get("fingerprint") != parent_fingerprint:
        raise RuntimeError("Parent dataset fingerprint does not match its recorded audit")
    history = DATASET / "history/pre_P12_large_v3_20260909"
    history.mkdir(parents=True, exist_ok=False)
    historical_hashes = {}
    for name in TOP_LEVEL_HISTORY_FILES:
        source = DATASET / name
        if source.exists():
            shutil.copy2(source, history / name)
            historical_hashes[name] = sha256(source)
    save_json(history / "sha256.json", historical_hashes)
    shutil.copytree(PREVIEW, destination)
    patch_copied_truth(source_truth, destination / "truth.mat")
    approved = verify_reviewed_truth(destination / "truth.mat")
    approval = read_json(DATASET / "SIMULATION_APPROVED.json")
    ids = list(approval["approved_new_sample_ids"])
    if "P12" not in ids:
        ids.append("P12")
    approval["approved_new_sample_ids"] = ids
    approval["truth_sources"]["P12"] = {
        "source_dir": str(PREVIEW),
        "source_preview_truth_mat_sha256": reviewed["truth_mat_sha256"],
        "truth_mat_sha256": approved["truth_mat_sha256"],
        "geometry_source_sha256": approved["geometry_source_sha256"],
    }
    approval["authorization_p12"] = (
        "User approved the revised 18-cell native-pixel-calibrated preview and "
        "explicitly requested full simulation as P12 on physical GPU 0"
    )
    approval["approved_p12_beijing"] = now()
    save_json(DATASET / "SIMULATION_APPROVED.json", approval)
    gpu = gpu_snapshot()
    contract = {
        "schema_version": 1,
        "dataset_revision_before": 3,
        "dataset_revision_after": 4,
        "sample_id": "P12",
        "split": "train",
        "parent_fingerprint": parent_fingerprint,
        "reviewed_preview": str(PREVIEW),
        "reviewed": reviewed,
        "approved_copy": approved,
        "physical_gpu_index": 0,
        "gpu": gpu,
        "sensor_forward_device": "CPU",
        "rl3_reconstruction_device": "physical_gpu0",
        "frame_count": 100,
        "input_frames": 10,
        "holdout_frames": 90,
        "subset_count": 10,
        "iterations": 3,
        "user_authorization": (
            "可以，执行完整仿真"
        ),
        "created_beijing": now(),
        "source_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                REPO / "tools/run_root_pixel_preview.py",
                REPO / "tools/run_root_thin_preview.py",
                REPO / "tools/run_root_cell_preview.py",
                REPO / "tools/run_p12_dataset_extension.py",
                REPO / "utils/dataset_splits.py",
                REPO / "datasets/matlab_multivolume_dataset.py",
                REPO / "matlab_code/dataset_v3/dataset_v3_validate_truth.m",
                REPO / "matlab_code/dataset_v3/dataset_v3_run_sample.m",
                REPO / "matlab_code/dataset_v3/dataset_v3_sensor_p12_tail.m",
                REPO / "matlab_code/dataset_v3/dataset_v3_reconstruct_p12_tail.m",
            )
        },
        "parent_manifest_sha256": historical_hashes,
    }
    save_json(DATASET / "P12_EXTENSION_CONTRACT.json", contract)
    save_json(
        destination / "approval_record.json",
        {
            "approved": True,
            "sample_id": "P12",
            "split": "train",
            "reviewed_preview": str(PREVIEW),
            "reviewed_preview_truth_sha256": reviewed["truth_mat_sha256"],
            "approved_truth_sha256": approved["truth_mat_sha256"],
            "geometry_source_sha256": approved["geometry_source_sha256"],
            "approved_beijing": now(),
        },
    )
    status("prepared", reviewed=approved)


def stage(name: str) -> None:
    marker = DATASET / f"progress/P12_{name}_complete.json"
    if marker.is_file():
        status(f"{name}_already_complete")
        return
    run_matlab(
        f"P12_{name}",
        f"dataset_v3_run_sample('P12',{matlab_string(DATASET)},'{name}');",
        gpu=name == "reconstruct",
    )
    save_json(marker, {"complete": True, "sample_id": "P12", "finished_beijing": now()})


def publish() -> None:
    validation_path = DATASET / "P12/validation_manifest.json"
    if not validation_path.is_file() or not read_json(validation_path).get("complete"):
        raise RuntimeError("P12 cannot be published before complete MATLAB validation")
    rows = []
    for split, samples in V4_SPLITS.items():
        for sample in samples:
            folder = DATASET / sample
            validation = read_json(folder / "validation_manifest.json")
            if (
                not validation.get("complete")
                or validation.get("sample_id") != sample
                or validation.get("frame_count") != 100
                or validation.get("subset_count") != 10
            ):
                raise RuntimeError(f"Incomplete dataset object: {sample}")
            rows.append(
                {
                    "sample_id": sample,
                    "object_group_id": sample,
                    "split": split,
                    "sample_dir": str(folder),
                    "validation_manifest": str(folder / "validation_manifest.json"),
                    "artifact_count": int(validation["artifact_count"]),
                }
            )
    counts = {name: len(samples) for name, samples in V4_SPLITS.items()}
    final = {
        "version": 4,
        "producer": "run_p12_dataset_extension/1",
        "complete": True,
        "dataset_root": str(DATASET),
        "samples": rows,
        "counts": {
            **counts,
            "objects": 18,
            "frames_per_object": 100,
            "subsets_per_object": 10,
        },
        "extension_contract": str(DATASET / "P12_EXTENSION_CONTRACT.json"),
        "physical_gpu_indices_for_P12": [0],
        "training_started_by_extension": False,
        "completed_beijing": now(),
    }
    splits = {
        "version": 4,
        "dataset_complete": True,
        "counts": counts,
        "samples": rows,
        "split_unit": "entire_object_including_all_frames_subsets_and_derivatives",
        "revision_note": "V4 adds reviewed P12 to train; V3 split meaning remains archived",
    }
    report = audit(DATASET, workers=4, split_record=splits, final_record=final)
    if not report["complete"]:
        raise RuntimeError(f"Full dataset hash audit failed: {report['errors'][:5]}")
    save_json(DATASET / "final_hash_audit.json", report)
    save_json(DATASET / "FINAL_DATASET_MANIFEST.json", final)
    save_json(DATASET / "dataset_splits.json", splits)
    indexed, fingerprint = load_dataset_index(DATASET)
    loaded = {}
    for split, items in indexed.items():
        objects = set()
        for key in items:
            _read_input(key)
            targets = _read_targets(key, include_ground_truth=split != "train")
            if split == "train" and "ground_truth" in targets:
                raise RuntimeError("Training GT leakage")
            objects.add(key.sample_id)
        loaded[split] = {"items": len(items), "objects": sorted(objects)}
    if {name: len(items) for name, items in indexed.items()} != {
        "train": 120,
        "validation": 30,
        "test": 30,
    }:
        raise RuntimeError("Published item counts are not 120/30/30")
    save_json(
        DATASET / "python_reader_verification.json",
        {
            "complete": True,
            "fingerprint": fingerprint,
            "counts": {name: len(items) for name, items in indexed.items()},
            "loaded": loaded,
            "all_180_items_read": True,
        },
    )
    contract = read_json(DATASET / "P12_EXTENSION_CONTRACT.json")
    lineage = {
        "revision": 4,
        "parent_revision": 3,
        "parent_fingerprint": contract["parent_fingerprint"],
        "current_fingerprint": fingerprint,
        "additive_change": "P12 added to train",
        "unchanged_validation": list(V4_SPLITS["validation"]),
        "unchanged_test": list(V4_SPLITS["test"]),
        "published_beijing": now(),
    }
    save_json(DATASET / "dataset_lineage.json", lineage)
    (DATASET / "report_zh.md").write_text(
        "# 18 对象数据集（第 4 版）完成\n\n"
        "训练集为 P01–P12，共 120 个十帧子集；验证集 V01–V03 和测试集 "
        "T02–T04 保持不变，各 30 个子集。\n\n"
        "P12 是用户确认的薄层、互不重叠细胞样本：18 个大圆环细胞，其壁间典型直径约 "
        "44 个原生物方像素，仅 40/50/60 μm "
        "三层有真值。已生成 100 帧传感器数据、100 份逐帧 RL3、10 个严格不重叠的 "
        "10/90 子集及 Mean/Taylor 重建。传感器前向沿用现有 CPU 实现，全部 RL3 "
        "重建固定使用物理 0 号 A40。\n\n"
        "P12 和全数据集均通过 MATLAB 数值复算、10/90 帧划分、轴序、Taylor 开方、"
        "文件哈希和 Python 全 180 项读取检查。此次扩展没有启动训练。第 3 版清单与"
        "指纹已保存在 history/pre_P12_large_v3_20260909，旧实验仍对应第 3 版。\n",
        encoding="utf-8",
    )
    status(
        "complete",
        fingerprint=fingerprint,
        counts={name: len(items) for name, items in indexed.items()},
        p12_artifacts=int(read_json(validation_path)["artifact_count"]),
    )


def run(selected_stage: str) -> None:
    prepare()
    order = ("sensor", "reconstruct", "validate")
    if selected_stage in order:
        stage(selected_stage)
        return
    if selected_stage == "publish":
        publish()
        return
    if selected_stage == "prepare":
        return
    for name in order:
        stage(name)
    publish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "sensor", "reconstruct", "validate", "publish"),
        default="all",
    )
    args = parser.parse_args()
    DATASET.mkdir(parents=True, exist_ok=True)
    with (DATASET / ".p12_extension.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.stage)
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
