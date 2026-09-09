"""Dated CPU truth preview only; never migrates old data or launches acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.experiment_paths import next_experiment_path


REPO = Path(__file__).resolve().parents[1]
OLD_SPLITS = {
    "train": ["P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10"],
    "validation": ["P09", "V01", "V02"],
    "test": ["P07", "T01", "T02"],
}
NEW_SPLITS = {
    "train": [f"P{i:02d}" for i in range(1, 12)],
    "validation": ["V01", "V02", "V03"],
    "test": ["T02", "T03", "T04"],
}
NEW_OBJECTS = ("T03", "T04", "V03")
GPU_POLICY = {
    "allowed_physical_gpu_indices": [0],
    "required_gpu_model": "A40",
    "allow_sharing": False,
    "allow_fallback_gpu": False,
    "when_busy": "wait_for_physical_gpu_0",
    "future_binding": "verify physical index 0 is A40 and idle; bind its UUID; MATLAB logical device 1",
    "preview_compute": "CPU_ONLY",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_signature(root: Path) -> dict:
    """Metadata scan, not a full content hash of the historical data tree."""
    result = {}
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in sorted(
            files + [name for name in folders if (Path(directory) / name).is_symlink()]
        ):
            path = Path(directory) / name
            stat = path.lstat()
            result[str(path.relative_to(root))] = [
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_mode,
            ]
    return result


def split_plan(source: Path, output: Path) -> dict:
    old = {sample: split for split, samples in OLD_SPLITS.items() for sample in samples}
    rows = []
    for split, objects in NEW_SPLITS.items():
        for sample in objects:
            origin = "T01" if sample == "P11" else sample
            is_new = sample in NEW_OBJECTS
            rows.append(
                {
                    "sample_id": sample,
                    "object_group_id": sample,
                    "split": split,
                    "source_sample_id": None if is_new else origin,
                    "source_split": None if is_new else old[origin],
                    "source_dir": None if is_new else str(source / origin),
                    "planned_sample_dir": str(output / sample),
                    "stage": "truth_preview_only"
                    if is_new
                    else "reuse_planned_not_migrated",
                }
            )
    return {
        "version": 3,
        "dataset_complete": False,
        "morphology_approved": False,
        "split_unit": "entire_object_including_all_frames_subsets_and_derivatives",
        "counts": {split: len(samples) for split, samples in NEW_SPLITS.items()},
        "samples": rows,
        "migration_executed": False,
        "note": "Draft only. Old T01 stays untouched until approved migration to P11; no new T01.",
    }


def audit_source(source: Path) -> dict:
    import h5py
    import numpy as np

    final_path = source / "FINAL_DATASET_MANIFEST.json"
    split_path = source / "dataset_splits.json"
    final, splits = read_json(final_path), read_json(split_path)
    if not final.get("complete") or not splits.get("dataset_complete"):
        raise ValueError("Historical source is not marked complete")
    rows = splits["samples"]
    actual = {
        split: [r["sample_id"] for r in rows if r["split"] == split]
        for split in OLD_SPLITS
    }
    if actual != OLD_SPLITS or len(rows) != 14 or len(final["samples"]) != 14:
        raise ValueError(f"Unexpected historical split: {actual}")
    final_map = {r["sample_id"]: r for r in final["samples"]}
    if set(final_map) != {r["sample_id"] for r in rows}:
        raise ValueError("Historical manifests disagree on object IDs")
    selected = {str(p): sha256(p) for p in (final_path, split_path)}
    audit_rows = []
    stale = []
    for row in rows:
        sample = row["sample_id"]
        folder = source / sample
        if final_map[sample]["split"] != row["split"]:
            raise ValueError(f"Contradictory authoritative split for {sample}")
        for record in (row, final_map[sample]):
            if Path(record["sample_dir"]).resolve() != folder.resolve():
                stale.append(
                    {
                        "sample_id": sample,
                        "recorded": record["sample_dir"],
                        "actual": str(folder),
                    }
                )
        validation_path = folder / "validation_manifest.json"
        validation = read_json(validation_path)
        if not validation.get("complete") or validation["sample_id"] != sample:
            raise ValueError(f"Invalid source validation for {sample}")
        if validation["frame_count"] != 100 or validation["subset_count"] != 10:
            raise ValueError(f"Unexpected source counts for {sample}")
        artifacts = validation["artifacts"]
        if len(artifacts) != validation["artifact_count"]:
            raise ValueError(f"Wrong artifact count for {sample}")
        by_name = {item["relative_path"]: item for item in artifacts}
        if len(by_name) != len(artifacts):
            raise ValueError(f"Duplicate artifact in {sample}")
        for item in artifacts:
            path = folder / item["relative_path"]
            if not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f"Artifact leaves sample directory: {path}")
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Missing or size-mismatched source artifact: {path}")
        selected[str(validation_path)] = sha256(validation_path)
        for name in ("prepared.mat", "truth.mat"):
            path = folder / name
            if name == "prepared.mat" and not path.is_file():
                raise FileNotFoundError(path)
            if path.is_file():
                digest = sha256(path)
                if name in by_name and by_name[name]["sha256"] != digest:
                    raise ValueError(f"Source SHA-256 mismatch: {path}")
                selected[str(path)] = digest
        subset_paths = sorted((folder / "subsets").glob("subset_*.mat"))
        frame_paths = sorted((folder / "sensor_frames").glob("frame_*.mat"))
        if len(subset_paths) != 10 or len(frame_paths) != 100:
            raise ValueError(f"Missing source frames or subsets for {sample}")
        if sample == "T01":
            for index, path in enumerate(subset_paths, 1):
                with h5py.File(path, "r") as handle:
                    owner = "".join(
                        chr(int(v)) for v in handle["sample_id"][()].reshape(-1)
                    )
                    if (
                        owner != "T01"
                        or int(np.asarray(handle["subset_index"][()]).item()) != index
                    ):
                        raise ValueError(f"Unexpected T01 subset owner: {path}")
                selected[str(path)] = sha256(path)
        audit_rows.append(
            {
                "sample_id": sample,
                "source_split": row["split"],
                "artifact_count": len(artifacts),
                "artifact_existence_and_sizes_passed": True,
            }
        )
    return {
        "source_root": str(source),
        "source_objects": audit_rows,
        "selected_content_sha256": selected,
        "stale_sample_dir_references": stale,
        "recorded_dataset_root": final["dataset_root"],
        "hash_coverage": "Metadata, prepared/truth MATs, and ten T01 subset MATs; NOT all historical frame payloads.",
        "action": "read_only; stale references recorded, not rewritten",
    }


def cpu_environment(output: Path) -> dict:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(output / "runtime" / "matplotlib"),
        MATLAB_PREFDIR=str(output / "runtime" / "matlab"),
    )
    return env


def matlab_string(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_logged(command: list[str], log: Path, env: dict) -> None:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=REPO / "data" / "speckle_data_now"
    )
    parser.add_argument("--output-parent", type=Path, default=REPO / "data")
    parser.add_argument(
        "--matlab", type=Path, default=Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")
    )
    args = parser.parse_args()
    source, parent = args.source_root.resolve(), args.output_parent.resolve()
    if (
        parent.is_relative_to(source)
        or not source.is_dir()
        or not args.matlab.is_file()
    ):
        raise ValueError(
            "Invalid source, MATLAB binary, or output parent inside protected source"
        )
    print("AUDIT historical source read-only; no GPU initialization", flush=True)
    before = tree_signature(source)
    source_audit = audit_source(source)
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        output = next_experiment_path(parent, "speckle_dataset_v3")
        try:
            output.mkdir()
            break
        except FileExistsError:
            continue
    print(f"OUTPUT {output}", flush=True)
    for folder in ("logs", "runtime/matlab", "runtime/matplotlib"):
        (output / folder).mkdir(parents=True)
    code_paths = sorted((REPO / "matlab_code" / "dataset_v3").glob("*.m"))
    code_paths += [
        REPO / "RUN_DATASET_V3_PREVIEW.m",
        Path(__file__).resolve(),
        REPO / "tools" / "report_dataset_v3_preview.py",
        REPO / "utils" / "experiment_paths.py",
    ]
    for folder, names in {
        "cell_dataset": (
            "cell_make_truth.m",
            "cell_plot_truth.m",
            "cell_draw_surface.m",
            "cell_write_json.m",
        ),
        "pilot_dataset": ("pilot_atomic_save.m", "pilot_write_tiff.m"),
    }.items():
        code_paths += [REPO / "matlab_code" / folder / name for name in names]
    code_hashes = {str(p.relative_to(REPO)): sha256(p) for p in sorted(code_paths)}
    geometry_hash = hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode()
    ).hexdigest()
    # This digest deliberately covers code actually used by the preview, not
    # the old two-file approval hash. No legacy approval is invalidated.
    contract = {
        "stage": "truth_preview_only",
        "status": "running",
        "dataset_complete": False,
        "morphology_approved": False,
        "migration_executed": False,
        "forward_started": False,
        "rl_started": False,
        "training_started": False,
        "gpu_used": False,
        **GPU_POLICY,
        "started_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_root": str(source),
        "output_root": str(output),
        "new_objects": list(NEW_OBJECTS),
        "geometry_source_sha256": geometry_hash,
        "code_sha256": code_hashes,
        "baseline": "sqrt + Mean + Set branch + Gate; unchanged",
        "future_protocol": {
            "frames": 100,
            "subsets": 10,
            "input_frames": 10,
            "holdout_frames": 90,
            "rl_iterations": 3,
        },
    }
    write_json(output / "preview_contract.json", contract)
    write_json(output / "source_audit.json", source_audit)
    write_json(output / "source_tree_before.json", before)
    draft = split_plan(source, output)
    write_json(output / "dataset_splits_preview.json", draft)
    with (output / "migration_plan.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(draft["samples"][0]))
        writer.writeheader()
        writer.writerows(draft["samples"])
    write_json(
        output / "MORPHOLOGY_REVIEW_REQUIRED.json",
        {
            "approved": False,
            "stage": "truth_preview_only",
            "required_sample_ids": list(NEW_OBJECTS),
            "message": "STOP after previews. Explicit user morphology confirmation required before migration or simulation.",
            **GPU_POLICY,
        },
    )
    env = cpu_environment(output)
    try:
        folders = [
            REPO / "matlab_code" / name
            for name in ("dataset_v3", "cell_dataset", "pilot_dataset")
        ]
        expression = "addpath(" + ",".join(matlab_string(p) for p in folders) + "); "
        expression += "results=runtests({'matlab_code/dataset_v3/test_dataset_v3_truth.m','matlab_code/cell_dataset/test_cell_truth.m'}); "
        expression += "disp(table(results)); assert(all([results.Passed]),'CPU geometry tests failed'); "
        expression += f"RUN_DATASET_V3_PREVIEW({matlab_string(output)});"
        print("RUN MATLAB CPU geometry tests, then three truth previews", flush=True)
        run_logged(
            [
                str(args.matlab),
                "-singleCompThread",
                "-softwareopengl",
                "-batch",
                expression,
            ],
            output / "logs" / "matlab_cpu.log",
            env,
        )
        print("RUN complete Python unittest on CPU", flush=True)
        run_logged(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            output / "logs" / "python_unittest.log",
            env,
        )
        changed_code = [
            p for p, digest in code_hashes.items() if sha256(REPO / p) != digest
        ]
        if changed_code:
            raise RuntimeError(
                f"Preview code changed during generation: {changed_code}"
            )
        after = tree_signature(source)
        changed = sorted(
            k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
        )
        hashes_changed = [
            p
            for p, digest in source_audit["selected_content_sha256"].items()
            if not Path(p).is_file() or sha256(Path(p)) != digest
        ]
        preservation = {
            "tree_metadata_unchanged": not changed,
            "changed_paths": changed,
            "selected_content_hashes_unchanged": not hashes_changed,
            "changed_hashed_paths": hashes_changed,
            "tree_file_count": len(before),
            "selected_hash_count": len(source_audit["selected_content_sha256"]),
        }
        write_json(output / "source_preservation.json", preservation)
        if changed or hashes_changed:
            raise RuntimeError(
                "Historical source changed during preview; inspect preservation record"
            )
        print("RENDER normalized overview, depth maps and review report", flush=True)
        run_logged(
            [
                sys.executable,
                "-m",
                "tools.report_dataset_v3_preview",
                "--root",
                str(output),
            ],
            output / "logs" / "report.log",
            env,
        )
        contract.update(
            status="preview_complete_awaiting_user_review",
            finished_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )
        write_json(output / "preview_contract.json", contract)
        print(
            f"COMPLETE CPU previews; human review required: {output / 'report_zh.md'}",
            flush=True,
        )
    except BaseException as error:
        contract.update(status="failed", error=str(error))
        write_json(output / "preview_contract.json", contract)
        raise


if __name__ == "__main__":
    main()
