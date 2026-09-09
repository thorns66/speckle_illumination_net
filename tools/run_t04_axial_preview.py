"""Generate the replacement T04 on CPU; keep all earlier previews immutable."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

from tools.run_dataset_v3_preview import (
    GPU_POLICY,
    REPO,
    cpu_environment,
    matlab_string,
    read_json,
    run_logged,
    sha256,
    tree_signature,
    write_json,
)
from utils.experiment_paths import next_experiment_path

REVISION = "t04_axial_line_pairs_v2"


def summarize_unittest(text: str) -> dict:
    total = re.search(r"^Ran (\d+) tests? in .+$", text, re.MULTILINE)
    ok = re.search(r"^OK(?: \(([^)]+)\))?$", text, re.MULTILINE)
    if total is None or ok is None:
        raise ValueError("Python unittest did not finish successfully")
    count = int(total.group(1))
    skipped_match = re.search(r"skipped=(\d+)", ok.group(1) or "")
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    skipped_lines = [line for line in text.splitlines() if " ... skipped " in line]
    if len(skipped_lines) != skipped:
        raise ValueError("Unittest skip summary and individual records disagree")
    return {
        "total": count,
        "passed": count - skipped,
        "failed": 0,
        "skipped": skipped,
        "skipped_test_records": skipped_lines,
    }


def replacement_plan(
    previous: Path, output: Path, previous_axial: Path | None = None
) -> dict:
    return {
        "dataset_complete": False,
        "morphology_approved": False,
        "migration_executed": False,
        "replacement_sample_id": "T04",
        "geometry_revision": REVISION,
        "new_truth_dir": str(output / "T04"),
        "rejected_mesh_preserved_at": str(previous / "T04"),
        "previous_axial_preview_preserved_at": (
            str(previous_axial) if previous_axial is not None else None
        ),
        "retained_previews": {
            sample: str(previous / sample) for sample in ("T03", "V03")
        },
        "future_protocol": {
            "frames": 100,
            "subsets": 10,
            "input_frames": 10,
            "holdout_frames": 90,
            "rl_iterations": 3,
        },
        "baseline": "sqrt + Mean + Set branch + Gate; unchanged",
        **GPU_POLICY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-preview",
        type=Path,
        default=REPO / "data" / "speckle_dataset_v3_20260907_run01",
    )
    parser.add_argument("--output-parent", type=Path, default=REPO / "data")
    parser.add_argument(
        "--previous-axial-preview",
        type=Path,
        default=REPO / "data" / "t04_axial_preview_20260907_run01",
    )
    parser.add_argument(
        "--matlab", type=Path, default=Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")
    )
    args = parser.parse_args()
    previous, parent = args.reference_preview.resolve(), args.output_parent.resolve()
    previous_axial = args.previous_axial_preview.resolve()
    historical = REPO / "data" / "speckle_data_now"
    if not previous.is_dir() or not historical.is_dir() or not args.matlab.is_file():
        raise ValueError("Missing historical data, reference preview or MATLAB binary")
    if not previous_axial.is_dir() or not read_json(
        previous_axial / "preview_report.json"
    ).get("preview_complete"):
        raise ValueError("Missing or incomplete previous axial preview")
    if any(parent.is_relative_to(p) for p in (previous, previous_axial, historical)):
        raise ValueError("Output must not be inside any protected source")
    prior = read_json(previous / "preview_report.json")
    if not prior.get("preview_complete"):
        raise ValueError("Reference preview was not completed")
    for sample in ("T03", "T04", "V03"):
        if read_json(previous / sample / "config.json")["sample_id"] != sample:
            raise ValueError(f"Invalid reference owner {sample}")
    print(
        "Protect old previews by full hashes; historical data by metadata scan",
        flush=True,
    )
    previous_tree = tree_signature(previous)
    previous_axial_tree = tree_signature(previous_axial)
    historical_tree = tree_signature(historical)
    protected_hashes = {
        str(p): sha256(p)
        for source in (previous, previous_axial)
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    baseline = REPO / "configs" / "multivolume_n10_no_mean.yaml"
    protected_hashes[str(baseline)] = sha256(baseline)
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        output = next_experiment_path(parent, "t04_axial_preview")
        try:
            output.mkdir()
            break
        except FileExistsError:
            continue
    print(f"OUTPUT {output}", flush=True)
    for folder in ("logs", "runtime/matlab", "runtime/matplotlib", "source_snapshot"):
        (output / folder).mkdir(parents=True)
    code_paths = sorted((REPO / "matlab_code" / "dataset_v3").glob("*.m"))
    code_paths += [
        REPO / "matlab_code" / "cell_dataset" / name
        for name in (
            "cell_dataset_config.m",
            "cell_dataset_split.m",
            "cell_make_truth.m",
            "cell_validate_truth.m",
            "cell_plot_truth.m",
            "cell_draw_surface.m",
            "cell_write_json.m",
            "test_cell_truth.m",
        )
    ]
    code_paths += [
        REPO / "matlab_code" / "pilot_dataset" / name
        for name in ("pilot_atomic_save.m", "pilot_write_tiff.m")
    ]
    code_paths += [
        REPO / "tools" / name
        for name in (
            "run_t04_axial_preview.py",
            "report_t04_axial_preview.py",
            "run_dataset_v3_preview.py",
            "report_dataset_v3_preview.py",
        )
    ]
    code_paths += [
        REPO / "utils" / name
        for name in ("experiment_paths.py", "display_normalization.py")
    ]
    code_paths += [REPO / "tests" / "test_t04_axial_preview.py"]
    hashes = {str(p.relative_to(REPO)): sha256(p) for p in sorted(code_paths)}
    for rel in hashes:
        target = output / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, target)
    contract = {
        "stage": "truth_preview_only",
        "status": "running",
        "sample_id": "T04",
        "geometry_revision": REVISION,
        "morphology_approved": False,
        "dataset_complete": False,
        "forward_started": False,
        "rl_started": False,
        "training_started": False,
        "migration_executed": False,
        "gpu_used": False,
        **GPU_POLICY,
        "reference_preview": str(previous),
        "previous_axial_preview": str(previous_axial),
        "output_root": str(output),
        "started_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "geometry_source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "code_sha256": hashes,
    }
    write_json(output / "preview_contract.json", contract)
    write_json(
        output / "replacement_manifest.json",
        replacement_plan(previous, output, previous_axial),
    )
    write_json(output / "protected_source_sha256.json", protected_hashes)
    draft = read_json(previous / "dataset_splits_preview.json")
    for row in draft["samples"]:
        if row["sample_id"] == "T04":
            row.update(
                planned_sample_dir=str(output / "T04"), geometry_revision=REVISION
            )
    draft.update(
        dataset_complete=False, morphology_approved=False, migration_executed=False
    )
    write_json(output / "dataset_splits_preview.json", draft)
    write_json(
        output / "MORPHOLOGY_REVIEW_REQUIRED.json",
        {
            "approved": False,
            "stage": "truth_preview_only",
            "required_sample_ids": ["T04"],
            "geometry_revision": REVISION,
            **GPU_POLICY,
            "message": "Stop for morphology review; no simulation, RL, migration or training.",
        },
    )
    env = cpu_environment(output)
    try:
        folders = [
            REPO / "matlab_code" / name
            for name in ("dataset_v3", "cell_dataset", "pilot_dataset")
        ]
        expression = "addpath(" + ",".join(matlab_string(p) for p in folders) + "); "
        expression += "files=dir('matlab_code/dataset_v3/*.m'); for k=1:numel(files), issues=checkcode(fullfile(files(k).folder,files(k).name),'-id'); assert(~any(strcmp({issues.id},'PARSE')),'Parse error'); end; "
        expression += "r=runtests({'matlab_code/dataset_v3/test_t04_axial_truth.m','matlab_code/dataset_v3/test_dataset_v3_truth.m','matlab_code/cell_dataset/test_cell_truth.m'}); disp(table(r)); "
        expression += "s=struct('total',numel(r),'passed',nnz([r.Passed]),'failed',nnz([r.Failed]),'incomplete',nnz([r.Incomplete])); "
        expression += f"cell_write_json({matlab_string(output / 'logs' / 'matlab_test_summary.json')},s); assert(all([r.Passed]),'CPU geometry tests failed'); "
        expression += f"run_t04_axial_preview({matlab_string(output)});"
        print("RUN new and legacy MATLAB CPU tests; then T04 truth only", flush=True)
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
        print("RUN full Python unittest on CPU", flush=True)
        run_logged(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            output / "logs" / "python_unittest.log",
            env,
        )
        write_json(
            output / "test_summary.json",
            {
                "matlab": read_json(output / "logs" / "matlab_test_summary.json"),
                "python": summarize_unittest(
                    (output / "logs" / "python_unittest.log").read_text()
                ),
            },
        )
        print("RENDER local XZ sections, native depth profiles and report", flush=True)
        run_logged(
            [
                sys.executable,
                "-m",
                "tools.report_t04_axial_preview",
                "--root",
                str(output),
            ],
            output / "logs" / "report.log",
            env,
        )
        changed_hashes = [
            p
            for p, digest in protected_hashes.items()
            if not Path(p).is_file() or sha256(Path(p)) != digest
        ]
        changed_code = [
            rel for rel, digest in hashes.items() if sha256(REPO / rel) != digest
        ]
        protection = {
            "old_preview_tree_unchanged": tree_signature(previous) == previous_tree,
            "old_preview_tree_entries": len(previous_tree),
            "previous_axial_tree_unchanged": tree_signature(previous_axial)
            == previous_axial_tree,
            "previous_axial_tree_entries": len(previous_axial_tree),
            "historical_data_tree_unchanged": tree_signature(historical)
            == historical_tree,
            "historical_data_tree_entries": len(historical_tree),
            "protected_hash_count": len(protected_hashes),
            "changed_hashed_files": changed_hashes,
            "changed_source_code": changed_code,
            "hash_scope": "All original and previous axial preview files plus baseline config; historical training data metadata only",
        }
        write_json(output / "source_preservation.json", protection)
        if (
            changed_hashes
            or changed_code
            or not protection["old_preview_tree_unchanged"]
            or not protection["previous_axial_tree_unchanged"]
            or not protection["historical_data_tree_unchanged"]
        ):
            raise RuntimeError(
                "Protected source changed during preview; inspect source_preservation.json"
            )
        contract.update(
            status="preview_complete_awaiting_user_review",
            finished_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )
        write_json(output / "preview_contract.json", contract)
        print(
            f"COMPLETE: {output / 'report_zh.md'}; STOP for morphology confirmation",
            flush=True,
        )
    except BaseException as error:
        contract.update(status="failed", error=str(error))
        write_json(output / "preview_contract.json", contract)
        raise


if __name__ == "__main__":
    main()
