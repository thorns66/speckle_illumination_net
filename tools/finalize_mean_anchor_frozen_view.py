"""Finalize the isolated V3 dataset view used by the Mean-anchor A/B run."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from datasets.matlab_multivolume_dataset import load_dataset_index


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01"
SOURCE = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
VIEW = OUTPUT / "dataset_v3_frozen_view"
CONFIG = OUTPUT / "mean_anchor_e3_mean100.yaml"
PREFLIGHT = OUTPUT / "preflight.json"
OBJECTS = [
    *(f"P{i:02d}" for i in range(1, 12)),
    "V01", "V02", "V03", "T02", "T03", "T04",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def linked_files() -> tuple[int, int]:
    count = 0
    bytes_total = 0
    for sample_id in OBJECTS:
        source_files = sorted(path for path in (SOURCE / sample_id).rglob("*") if path.is_file())
        view_files = sorted(path for path in (VIEW / sample_id).rglob("*") if path.is_file())
        if len(source_files) != len(view_files):
            raise ValueError(f"Hard-link inventory differs for {sample_id}")
        for left, right in zip(source_files, view_files, strict=True):
            if left.relative_to(SOURCE / sample_id) != right.relative_to(VIEW / sample_id):
                raise ValueError(f"Hard-link path mismatch for {sample_id}")
            left_stat = left.stat()
            right_stat = right.stat()
            if (left_stat.st_dev, left_stat.st_ino) != (right_stat.st_dev, right_stat.st_ino):
                raise ValueError(f"Frozen view file is not a hard link: {right}")
            count += 1
            bytes_total += right_stat.st_size
    return count, bytes_total


def manifests() -> None:
    current = json.loads((SOURCE / "dataset_splits.json").read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in current["samples"]}
    samples = []
    for sample_id in OBJECTS:
        item = dict(by_id[sample_id])
        sample_dir = VIEW / sample_id
        item["sample_dir"] = str(sample_dir)
        item["validation_manifest"] = str(sample_dir / "validation_manifest.json")
        samples.append(item)
    split_manifest = {
        "version": 3,
        "dataset_complete": True,
        "counts": {"train": 11, "validation": 3, "test": 3},
        "samples": samples,
        "split_unit": "entire_object_including_all_frames_subsets_and_derivatives",
        "revision_note": (
            "Frozen V3 comparison view: P01-P11/V01-V03/T02-T04 only; "
            "objects are hard-linked and tensor-equivalent to the pre-P12 baseline data"
        ),
    }
    final_manifest = {
        "version": 3,
        "producer": "finalize_mean_anchor_frozen_view/1",
        "complete": True,
        "dataset_root": str(VIEW),
        "samples": samples,
        "counts": {
            "train": 11,
            "validation": 3,
            "test": 3,
            "objects": 17,
            "frames_per_object": 100,
            "subsets_per_object": 10,
        },
        "training_started_by_view_builder": False,
        "completed_unix": time.time(),
    }
    write_json(VIEW / "dataset_splits.json", split_manifest)
    write_json(VIEW / "FINAL_DATASET_MANIFEST.json", final_manifest)


def update_contract(fingerprint: str, linked_count: int, linked_bytes: int) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["data"]["root"] = str(VIEW)
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    contract = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    contract.update(
        dataset_root=str(VIEW),
        dataset_fingerprint=fingerprint,
        split_counts={"train": 110, "validation": 30, "test": 30},
        gpu_pool=[0, 1, 2, 3, 4, 5],
        gpu_sharing=True,
    )
    config_hash = sha256(CONFIG)
    contract["config_sha256"] = config_hash
    contract["config_hashes"] = {str(CONFIG): config_hash}
    contract["frozen_dataset_view"] = {
        "source_root": str(SOURCE),
        "view_root": str(VIEW),
        "excluded_concurrent_extension": ["P12"],
        "hard_linked_files": linked_count,
        "logical_bytes": linked_bytes,
        "baseline_cache_comparison": {
            "cache_fingerprint": "d2984a7ed7b98f6d",
            "items": 170,
            "arrays_and_fields": 2270,
            "bitwise_mismatches": 0,
        },
        "reason": "preserve the original 110/30/30 V3 A/B split after the shared root became V4",
        "created_unix": time.time(),
    }
    worker = ROOT / "tools/mean_anchor_frozen_v3_worker.py"
    source_hashes = dict(contract["source_hashes"])
    source_hashes[str(worker)] = sha256(worker)
    for name in list(source_hashes):
        path = Path(name)
        if path.is_file():
            source_hashes[name] = sha256(path)
    contract["source_hashes"] = source_hashes
    contract["training_source_hashes"] = dict(source_hashes)
    contract.setdefault("source_change_audit", []).append(
        {
            "path": str(SOURCE / "dataset_splits.json"),
            "reason": "shared dataset was concurrently extended from V3 to V4 with P12",
            "resolution": "isolated hard-linked V3 view excluding P12",
            "core_tensor_comparison": "170 items, 2270 arrays/fields, 0 bitwise mismatches",
            "accepted_unix": time.time(),
        }
    )
    write_json(PREFLIGHT, contract)

    snapshot = OUTPUT / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in source_hashes:
        path = Path(name)
        if path.is_file() and path.is_relative_to(ROOT):
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    write_json(
        VIEW / "frozen_view_audit.json",
        {
            "complete": True,
            "source_root": str(SOURCE),
            "view_root": str(VIEW),
            "objects": OBJECTS,
            "excluded": ["P12"],
            "hard_linked_files": linked_count,
            "logical_bytes": linked_bytes,
            "dataset_fingerprint": fingerprint,
            "baseline_cache_comparison": {
                "items": 170,
                "arrays_and_fields": 2270,
                "bitwise_mismatches": 0,
            },
        },
    )


def main() -> None:
    if not VIEW.is_dir():
        raise FileNotFoundError(VIEW)
    manifests()
    indexed, fingerprint = load_dataset_index(VIEW)
    counts = {name: len(items) for name, items in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(counts)
    linked_count, linked_bytes = linked_files()
    update_contract(fingerprint, linked_count, linked_bytes)
    # Recheck after all top-level audit files have been written. Only the two
    # authoritative manifests participate in the loader fingerprint.
    _indexed, verified = load_dataset_index(VIEW)
    if verified != fingerprint:
        raise ValueError("Frozen dataset fingerprint changed during finalization")
    print(json.dumps({
        "dataset_root": str(VIEW),
        "fingerprint": fingerprint,
        "counts": counts,
        "hard_linked_files": linked_count,
        "logical_bytes": linked_bytes,
        "config_sha256": sha256(CONFIG),
    }, indent=2))


if __name__ == "__main__":
    main()
