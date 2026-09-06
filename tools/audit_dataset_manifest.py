from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


EXPECTED_SPLITS = {
    "train": ["P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10"],
    "validation": ["P09", "V01", "V02"],
    "test": ["P07", "T01", "T02"],
}
MUTABLE_METADATA = {
    "validation_manifest.mat",
    "validation_manifest.json",
    "full_run_report.mat",
    "full_run_report.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_unhashed(relative: str) -> bool:
    portable = relative.replace("\\", "/")
    return portable in MUTABLE_METADATA or portable.startswith("logs/")


def audit(root: Path, workers: int) -> dict[str, object]:
    root = root.expanduser().resolve()
    split_record = json.loads((root / "dataset_splits.json").read_text(encoding="utf-8"))
    final_record = json.loads(
        (root / "FINAL_DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    if not split_record.get("dataset_complete") or not final_record.get("complete"):
        raise ValueError("Dataset is not declared complete")
    actual_splits = {
        split: [item["sample_id"] for item in split_record["samples"] if item["split"] == split]
        for split in EXPECTED_SPLITS
    }
    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected object splits: {actual_splits}")

    jobs: list[tuple[str, Path, int, str]] = []
    coverage_errors: list[str] = []
    sample_counts: dict[str, int] = {}
    for sample in final_record["samples"]:
        sample_id = sample["sample_id"]
        sample_dir = root / sample_id
        manifest = json.loads(
            (sample_dir / "validation_manifest.json").read_text(encoding="utf-8")
        )
        if not manifest.get("complete") or manifest["sample_id"] != sample_id:
            coverage_errors.append(f"{sample_id}: invalid validation manifest")
            continue
        artifacts = manifest["artifacts"]
        if len(artifacts) != int(manifest["artifact_count"]):
            coverage_errors.append(f"{sample_id}: artifact count mismatch")
        declared = {item["relative_path"].replace("\\", "/") for item in artifacts}
        if len(declared) != len(artifacts):
            coverage_errors.append(f"{sample_id}: duplicate artifact paths")
        forbidden = sorted(path for path in declared if _allowed_unhashed(path))
        if forbidden:
            coverage_errors.append(f"{sample_id}: mutable paths were hashed: {forbidden[:3]}")
        actual = {
            path.relative_to(sample_dir).as_posix()
            for path in sample_dir.rglob("*")
            if path.is_file() and not _allowed_unhashed(path.relative_to(sample_dir).as_posix())
        }
        missing_coverage = sorted(actual - declared)
        stale_declarations = sorted(declared - actual)
        if missing_coverage:
            coverage_errors.append(
                f"{sample_id}: unhashed immutable files: {missing_coverage[:3]}"
            )
        if stale_declarations:
            coverage_errors.append(
                f"{sample_id}: declared files absent: {stale_declarations[:3]}"
            )
        sample_counts[sample_id] = len(artifacts)
        for item in artifacts:
            relative = item["relative_path"].replace("\\", "/")
            jobs.append((sample_id, sample_dir / relative, int(item["bytes"]), item["sha256"]))

    def verify(job: tuple[str, Path, int, str]) -> str | None:
        sample_id, path, expected_bytes, expected_hash = job
        if not path.is_file():
            return f"{sample_id}: missing {path.name}"
        if path.stat().st_size != expected_bytes:
            return f"{sample_id}: byte mismatch {path.name}"
        if _sha256(path) != expected_hash:
            return f"{sample_id}: hash mismatch {path.name}"
        return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        hash_errors = [error for error in executor.map(verify, jobs) if error is not None]
    errors = coverage_errors + hash_errors
    return {
        "complete": not errors,
        "dataset_root": str(root),
        "objects": len(sample_counts),
        "immutable_artifacts": len(jobs),
        "sample_artifact_counts": sample_counts,
        "splits": actual_splits,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the complete MATLAB dataset manifests")
    parser.add_argument("--root", default="data/matlab_cells_pilot_v2_r04")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(Path(args.root), args.workers)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
