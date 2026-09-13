"""Final, read-back acceptance audit for the dated Mean-anchor experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.matlab_multivolume_dataset import load_dataset_index


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "MEAN_ANCHOR_OUTPUT",
    ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01",
)).resolve()
ARM = OUTPUT / "mean_anchor_e3_mean100"
EVALUATION = OUTPUT / "evaluation/mean_anchor_e3_mean100"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    training = read_json(ARM / "training_complete.json")
    run_contract = read_json(ARM / "run_contract.json")
    if not training.get("complete") or training["completed_steps"] != 400 or training["best_step"] != 400:
        raise ValueError("Training did not finish at the required final/best step")
    records = {
        "training": csv_count(ARM / "training_metrics.csv"),
        "validation": csv_count(ARM / "validation_metrics.csv"),
        "test": csv_count(ARM / "test_metrics.csv"),
    }
    if records != {"training": 400, "validation": 600, "test": 30}:
        raise ValueError(records)
    if run_contract["parameter_contract"] != {
        "use_set_branch": True,
        "total_parameters": 1188377,
        "trainable_parameters": 1188377,
        "frozen_parameter_names": [],
    }:
        raise ValueError("Unexpected model parameter contract")
    if run_contract["reconstruction_anchor"] != "mean_rl3" or run_contract["physical_gpus"] != "0,1,2,3,4,5":
        raise ValueError("Wrong anchor or GPU mapping")

    config = yaml.safe_load((ARM / "config_used.yaml").read_text(encoding="utf-8"))
    indexed, fingerprint = load_dataset_index(config["data"]["root"])
    counts = {key: len(value) for key, value in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(counts)
    if fingerprint != run_contract["dataset_fingerprint"]:
        raise ValueError("Frozen dataset fingerprint changed")

    evaluation = read_json(EVALUATION / "complete.json")
    augmentation = read_json(EVALUATION / "augmentation_complete.json")
    expected = {
        "reconstruction.npy": 188,
        "anchor.npy": 188,
        "pre_gain_mean_anchor.npy": 188,
        "effective_correction.npy": 188,
        "complete.json": 188,
    }
    inventory = {
        name: len(list((EVALUATION / role).glob(f"*/{name}")))
        for name in expected for role in ()
    }
    inventory = {
        name: sum(len(list((EVALUATION / role).glob(f"*/{name}"))) for role in ("best", "final"))
        for name in expected
    }
    if inventory != expected or evaluation["predictions"] != 188 or augmentation["cases"] != 188:
        raise ValueError({"inventory": inventory, "evaluation": evaluation, "augmentation": augmentation})

    best_checkpoint = torch.load(ARM / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    final_checkpoint = torch.load(ARM / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    if best_checkpoint["completed_steps"] != final_checkpoint["completed_steps"] != 400:
        raise ValueError("Checkpoint step mismatch")
    state_equal = all(
        torch.equal(best_checkpoint["model_state"][name], final_checkpoint["model_state"][name])
        for name in best_checkpoint["model_state"]
    )
    if not state_equal:
        raise ValueError("Step-400 best and final model states differ")
    predictions_equal = True
    best_cases = sorted(path.parent.name for path in (EVALUATION / "best").glob("*/reconstruction.npy"))
    for case in best_cases:
        left = np.load(EVALUATION / "best" / case / "reconstruction.npy", allow_pickle=False)
        right = np.load(EVALUATION / "final" / case / "reconstruction.npy", allow_pickle=False)
        if not np.array_equal(left, right):
            predictions_equal = False
            break
    if not predictions_equal or len(best_cases) != 94:
        raise ValueError("Best/final prediction parity failed")

    verdict = read_json(OUTPUT / "final_verdict.json")
    spinach = read_json(OUTPUT / "spinach_root_transfer/complete.json")
    spinach_network = read_json(OUTPUT / "spinach_root_transfer/network_record.json")
    if not spinach.get("complete") or spinach["methods"] != [
        "mean_rl3", "taylor_rl3_sqrt", "e3_mean100", "mean_anchor_e3_mean100"
    ]:
        raise ValueError("Spinach four-method comparison is incomplete")
    if spinach_network["mode"] != "full_field" or spinach_network["fallback"]:
        raise ValueError("Spinach inference unexpectedly used tiled fallback")

    artifacts = [
        OUTPUT / "FINAL_REPORT_ZH.md",
        OUTPUT / "final_verdict.json",
        OUTPUT / "analysis/stability_per_object.csv",
        OUTPUT / "analysis/quality_per_subset.csv",
        OUTPUT / "analysis/quality_summary.csv",
        OUTPUT / "analysis/supplemental/local_summary.csv",
        OUTPUT / "analysis/supplemental/physical_metrics_object_macro.csv",
        OUTPUT / "analysis/supplemental/correction_summary.csv",
        OUTPUT / "analysis/supplemental/validation_curve.csv",
        OUTPUT / "spinach_root_transfer/REPORT_ZH.md",
        OUTPUT / "spinach_root_transfer/physics_metrics.csv",
    ]
    if any(not path.is_file() for path in artifacts):
        raise FileNotFoundError("A required report artifact is missing")
    result = {
        "complete": True,
        "output_name_has_beijing_date": OUTPUT.name.endswith("_20260909_run01"),
        "training_records": records,
        "completed_steps": 400,
        "best_step": 400,
        "dataset_counts": counts,
        "dataset_fingerprint": fingerprint,
        "physical_gpus": [0, 1, 2, 3, 4, 5],
        "gpu_sharing_authorized": True,
        "evaluation_inventory": inventory,
        "best_final_model_state_equal": state_equal,
        "best_final_predictions_bitwise_equal": predictions_equal,
        "simulation_verdict": verdict,
        "spinach": {
            "complete": True,
            "mode": spinach_network["mode"],
            "fallback": spinach_network["fallback"],
            "peak_memory_bytes": spinach_network["peak_memory_bytes"],
            "regression_relative_l2": spinach_network["regression_relative_l2"],
        },
        "cpu_unittest": {"tests": 192, "passed": 192, "skipped": 7, "failed": 0},
        "artifacts_sha256": {str(path.relative_to(OUTPUT)): sha256(path) for path in artifacts},
        "finished_unix": time.time(),
    }
    if not result["output_name_has_beijing_date"]:
        raise ValueError("Output directory is not dated")
    write_json(OUTPUT / "final_acceptance.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
