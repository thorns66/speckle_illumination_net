"""CPU-only final acceptance after all five round-2 training/evaluation runs.

This entry never trains, performs GPU inference, or edits frozen experiment
sources. It writes final_acceptance.json and a successful complete.json only
after provenance, gradients, checkpoints, predictions and reports pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import struct
import sys
import time

REPO = Path(__file__).resolve().parents[1]
for location in (str(REPO), str(REPO / "tools")):
    if location not in sys.path:
        sys.path.insert(0, location)

import h5py
import numpy as np
import torch
import yaml

from tools import three_way_experiment as old
from tools.three_way_report import all_cases
from training.global_batch_schedule import FixedGlobalBatchScheduler, slots_for_rank

OUTPUT = REPO / "outputs/mean_refinement_round2_20260908"
ARMS = ("r0_continue", "r1_no_mean", "r2_input_scale", "r3_mean_shape_005", "r4_mean_shape_010")
ROLES = {"best": "checkpoint_best.pt", "final": "checkpoint_last.pt"}
TABLES = ("metrics", "brightness", "point_targets", "point_pairs", "lines", "depth_profiles", "axial_points")
BRIGHTNESS = ("points_z060_r01", "lines_z060_r01", "P09")
GAINS = (0., .1, .5, 1., 2.)
SHAPE = (10, 260, 260)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def json_file(path):
    path = Path(path)
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text())


def csv_file(path):
    path = Path(path)
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hash_file(path, expected=None):
    path = Path(path)
    require(path.is_file(), f"Missing file: {path}")
    digest = old.sha256(path)
    if expected is not None:
        require(digest == expected, f"SHA256 mismatch: {path}")
    return digest


def close(actual, expected, *, relative=2e-6, absolute=1e-8):
    return math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(float(actual), float(expected), rel_tol=relative, abs_tol=absolute)


def same_metrics(csv_row, stored, context):
    for key, expected in stored.items():
        require(key in csv_row, f"CSV lacks {key}: {context}")
        actual = csv_row[key]
        if expected is None:
            valid = actual == ""
        elif isinstance(expected, bool):
            valid = actual.lower() == str(expected).lower()
        elif isinstance(expected, (int, float)):
            valid = close(actual, expected, relative=1e-12, absolute=1e-14)
        else:
            valid = actual == str(expected)
        require(valid, f"CSV/record disagreement for {key}: {context}")


def audit_volume(path, expected_hash, *, exact_zero=False):
    digest = hash_file(path, expected_hash)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.shape == SHAPE, f"Volume shape {value.shape}, expected {SHAPE}: {path}")
    require(value.dtype == np.float32, f"Expected original FP32 predictions: {path}")
    require(bool(np.isfinite(value).all()) and bool((value >= 0).all()), f"Invalid volume values: {path}")
    maximum = float(value.max())
    total = float(value.sum(dtype=np.float64))
    if exact_zero:
        require(maximum == 0. and total == 0., f"Exact zero input produced nonzero output: {path}")
    del value
    return {"sha256": digest, "maximum": maximum, "sum_float64": total}


def audit_provenance():
    preflight = json_file(OUTPUT / "preflight.json")
    require(preflight.get("complete") is True, "Round-2 preflight incomplete")
    previous_path = old.OUTPUT / "preflight.json"
    hash_file(previous_path, preflight["previous_preflight_sha256"])
    previous = json_file(previous_path)
    require(len(previous["source_sha256"]) == 37, "Expected 37 original frozen source files")
    require(len(previous["dataset_content_sha256"]) == 1554, "Expected 1554 original hashed dataset files")
    for name, digest in previous["source_sha256"].items():
        hash_file(REPO / name, digest)
    for name, digest in previous["dataset_content_sha256"].items():
        hash_file(old.DATA / name, digest)
    for name, digest in preflight["source_hashes"].items():
        require(Path(name).is_absolute(), "New source hash keys must be absolute paths")
        hash_file(name, digest)
        snapshot = OUTPUT / "source_snapshot" / Path(name).relative_to(REPO)
        hash_file(snapshot, digest)
    require(set(preflight["config_hashes"]) == {str(OUTPUT / (arm + ".yaml")) for arm in ARMS}, "Config inventory changed")
    configs = {}
    for name, digest in preflight["config_hashes"].items():
        hash_file(name, digest)
        config = yaml.safe_load(Path(name).read_text())
        configs[config["round2"]["kind"]] = config
    indexed, fingerprint = old.relocated_index(old.DATA)
    require(fingerprint == preflight["dataset_fingerprint"] == previous["dataset_fingerprint"], "Dataset fingerprint changed")
    hash_file(preflight["source_checkpoint"], preflight["source_checkpoint_sha256"])
    hash_file(old.BASELINE / "checkpoint_best.pt", preflight["old_baseline_best_sha256"])
    hash_file(old.BASELINE / "checkpoint_last.pt", preflight["old_baseline_final_sha256"])
    hash_file(configs[ARMS[0]]["psf"]["H_path"], preflight["psf_sha256"])
    require(set(preflight["initial_state_hashes"]) == set(ARMS), "Initial-state inventory changed")
    require(len(set(preflight["initial_state_hashes"].values())) == 1, "Arms did not share one initial model state")
    require(preflight["all_arms_gamma_reset"] == 0. and preflight["global_batch"] == 8, "Wrong initial gain/global batch")
    checks = json_file(OUTPUT / "gpu_checks.json")
    require(checks.get("passed") is True and checks.get("all_initial_states_identical") is True, "Numerical acceptance did not pass")
    require(checks["initial_state_hashes"] == preflight["initial_state_hashes"], "GPU checks used different starting weights")
    require(checks.get("precision", {}).get("cudnn_allow_tf32") is False and checks.get("precision", {}).get("matmul_allow_tf32") is False, "GPU acceptance must use the final FP32 settings")
    require({row["kind"] for row in checks["rows"]} == set(ARMS), "GPU acceptance lacks an arm")
    require(all(row.get("passed") is True for row in checks["rows"]), "One numerical check failed")
    brightness = json_file(old.OUTPUT / "brightness_inputs/complete.json")
    require(brightness.get("complete") is True and len(brightness["files"]) == 12, "Brightness source inventory incomplete")
    for name, digest in brightness["files"].items():
        hash_file(old.OUTPUT / name, digest)
    require(json_file(old.OUTPUT / "brightness_inputs/verification.json").get("passed") is True, "MATLAB brightness verification did not pass")
    summary = {"old_source_files": 37, "old_dataset_files": 1554, "new_source_files": len(preflight["source_hashes"]),
               "configs": len(configs), "brightness_inputs": 12, "dataset_fingerprint": fingerprint,
               "preflight_sha256": hash_file(OUTPUT / "preflight.json"), "gpu_checks_sha256": hash_file(OUTPUT / "gpu_checks.json")}
    return preflight, configs, indexed, summary


def audit_gradients(arm, config, world_size, train_keys):
    directory = OUTPUT / arm
    paths = sorted(directory.glob("gradient_diagnostics_rank*.jsonl"))
    require(len(paths) == world_size, f"{arm}: missing or extra rank gradient logs")
    rows = []
    ignored = Counter()
    for path in paths:
        file_rank = int(re.search(r"rank(\d+)\.jsonl$", path.name).group(1))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                require(int(record["rank"]) == file_rank, f"{path}:{line_number}: rank mismatch")
                if record.get("phase") != "training":
                    require(record.get("phase") in ("gpu_checks", "preflight"), f"Unexpected diagnostic phase: {path}:{line_number}")
                    ignored[record["phase"]] += 1
                    continue
                rows.append(record)
    require(len(rows) == 200 * 8, f"{arm}: expected 1600 valid training microbatches, found {len(rows)}; repeated/resumed attempts must not be silently combined")
    by_step = defaultdict(list)
    for row in rows:
        by_step[int(row["update_step"])].append(row)
    require(set(by_step) == set(range(1, 201)), f"{arm}: incomplete optimizer-step gradient coverage")
    scheduler = FixedGlobalBatchScheduler(len(train_keys), 8, int(config["experiment"]["seed"]))
    maximum_ratio = 0.
    budget = float(config["round2"]["shape_gradient_budget"])
    eps = float(config["round2"].get("shape_gradient_eps", 1e-12))
    coefficient_cap = float(config["round2"]["shape_coefficient_cap"])
    for step in range(1, 201):
        selected = by_step[step]
        require(len(selected) == 8, f"{arm} step {step}: expected exactly eight training samples")
        batch = scheduler.next_batch()
        expected = []
        for rank in range(world_size):
            slots, factor = slots_for_rank(batch, world_size, rank)
            require(close(factor, world_size / 8., relative=0., absolute=0.), "Global-batch scaling changed")
            expected.extend((rank, train_keys[slot.dataset_index].sample_id, train_keys[slot.dataset_index].subset_index) for slot in slots if slot.valid)
        observed = [(int(r["rank"]), str(r["sample_id"]), int(r["subset_index"])) for r in selected]
        require(Counter(observed) == Counter(expected), f"{arm} step {step}: gradient samples do not match the frozen global batch")
        ramp = min(step / int(config["round2"]["ramp_steps"]), 1.)
        for row in selected:
            require(int(row["optimizer_completed_steps"]) == step - 1 and int(row["source_completed_steps"]) == 200, f"{arm} step {step}: incorrect optimizer ramp counter")
            require(close(row["ramp"], ramp, relative=0., absolute=1e-12), f"{arm} step {step}: ramp is not based on optimizer updates")
            require(close(row["shape_gradient_budget"], budget, relative=0., absolute=1e-12), f"{arm}: wrong declared gradient budget")
            require(close(row["effective_gradient_budget"], budget * ramp, relative=0., absolute=1e-12), f"{arm}: incorrect effective gradient budget")
            require(row["gradients_finite"] is True and row["budget_passed"] is True, f"{arm} step {step}: failed gradient assertion")
            nv, nm, ntv = (float(row[key]) for key in ("var_q_gradient_norm", "mean_shape_q_gradient_norm", "weighted_tv_q_gradient_norm"))
            coefficient, ratio, cosine = (float(row[key]) for key in ("shape_coefficient", "shape_gradient_ratio", "mean_var_gradient_cosine"))
            require(all(math.isfinite(x) for x in (nv, nm, ntv, coefficient, ratio, cosine)), f"{arm}: nonfinite gradient diagnostic")
            require(min(nv, nm, ntv, coefficient, ratio) >= 0. and abs(cosine) <= 1.00001, f"{arm}: invalid gradient diagnostic values")
            expected_coefficient = min(coefficient_cap, budget * ramp * nv / (nm + eps))
            require(close(coefficient, expected_coefficient, relative=3e-6, absolute=1e-8), f"{arm} step {step}: coefficient formula mismatch")
            require(close(ratio, coefficient * nm / (nv + eps), relative=3e-6, absolute=1e-8), f"{arm} step {step}: reported gradient ratio mismatch")
            require(ratio <= budget * ramp + max(1e-7, budget * ramp * 1e-5), f"{arm} step {step}: gradient budget exceeded")
            if budget == 0.:
                require(coefficient == 0. and ratio == 0., f"{arm}: control arm received mean-shape gradients")
            maximum_ratio = max(maximum_ratio, ratio)
    return {"valid_microbatches": len(rows), "updates": 200, "samples_per_update": 8, "ignored_phases": dict(ignored),
            "maximum_q_gradient_ratio": maximum_ratio, "budget": budget,
            "logs_sha256": {str(path): hash_file(path) for path in paths}}


def audit_training(arm, config, train_keys):
    directory = OUTPUT / arm
    complete = json_file(directory / "training_complete.json")
    require(complete.get("complete") is True and int(complete["completed_steps"]) == 200, f"Training incomplete: {arm}")
    require(config["optimization"]["global_batch_size"] == 8 and config["optimization"]["max_steps"] == 200, f"Wrong training duration/batch: {arm}")
    require(yaml.safe_load((directory / "config_used.yaml").read_text()) == config, f"Actual training configuration differs: {arm}")
    training_rows = csv_file(directory / "training_metrics.csv")
    require(len(training_rows) == 200 and [int(r["step"]) for r in training_rows] == list(range(1, 201)), f"Training step CSV incomplete/duplicated: {arm}")
    for row in training_rows:
        require(int(row["global_batch_size"]) == 8, f"Wrong effective batch: {arm} step {row['step']}")
        require(float(row["var_band_loss"]) == 0. and float(row["weighted_var_band_loss"]) == 0., f"Mislabelled shape loss as variance band: {arm}")
        actual = sum(float(row[key]) for key in ("weighted_mean_loss", "weighted_var_loss", "weighted_tv_loss", "weighted_mean_shape_loss"))
        require(close(row["total_loss"], actual), f"Training objective breakdown does not sum: {arm} step {row['step']}")
        if arm == "r1_no_mean":
            require(float(row["weighted_mean_loss"]) == 0., "R1 trained with a nonzero mean-gain loss")
    validation = csv_file(directory / "validation_metrics.csv")
    require(len(validation) == 300, f"Expected 30 validation items at ten checkpoints: {arm}")
    steps = range(20, 201, 20)
    scores = {}
    expected_items = {(sample, subset) for sample in ("P09", "V01", "V02") for subset in range(1, 11)}
    for step in steps:
        selected = [row for row in validation if int(row["step"]) == step]
        require(len(selected) == 30 and {(r["sample_id"], int(r["subset_index"])) for r in selected} == expected_items, f"Wrong validation object inventory: {arm} step {step}")
        objects = defaultdict(list)
        for row in selected:
            common = float(row["normalized_mean_loss"]) + float(row["normalized_var_loss"]) + float(row["weighted_tv_loss"])
            require(close(row["weighted_mean_loss"], row["normalized_mean_loss"]), f"Evaluation omitted mean: {arm}")
            require(close(row["selection_score"], common) and close(row["total_loss"], common), f"Best selection used a different score: {arm} step {step}")
            require(float(row["weighted_mean_shape_loss"]) == 0., f"Shape add-on entered checkpoint selection: {arm}")
            objects[row["sample_id"]].append(float(row["selection_score"]))
        scores[step] = float(np.mean([np.mean(values) for values in objects.values()]))
        require(close(json_file(directory / f"validation_step_{step:06d}.json")["selection_score"], scores[step]), f"Validation summary mismatch: {arm} step {step}")
    expected_best = min(scores, key=lambda step: (scores[step], step))
    require(int(complete["best_step"]) == expected_best, f"Best checkpoint is not the common-score minimum: {arm}")
    checkpoints = {}
    world_size = None
    for role, filename in ROLES.items():
        path = directory / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actual_step = int(checkpoint["completed_steps"])
        require(checkpoint["config"] == config, f"Checkpoint configuration mismatch: {path}")
        require(actual_step == (expected_best if role == "best" else 200), f"Wrong actual checkpoint step: {path}")
        require(int(checkpoint["best_step"]) == expected_best, f"Incorrect best-step metadata: {path}")
        require(close(checkpoint["best_validation_score"], scores[expected_best]), f"Best-score metadata mismatch: {path}")
        require(int(checkpoint["scheduler_state"]["global_batch_size"]) == 8, f"Wrong scheduler batch: {path}")
        ranks = int(checkpoint["world_size"])
        require(1 <= ranks <= 8, f"Invalid DDP rank count: {path}")
        require(world_size in (None, ranks), f"Best/final DDP size differs: {arm}")
        world_size = ranks
        gamma = float(checkpoint["model_state"]["mean_gain_gamma"])
        require(math.isfinite(gamma), f"Nonfinite gamma: {path}")
        if arm == "r1_no_mean":
            require(gamma == 0., f"R1 gamma changed despite being frozen: {path}")
        expected_rates = [1e-4, 1e-5] + ([] if arm == "r1_no_mean" else [1e-5])
        groups = checkpoint["optimizer_state"]["param_groups"]
        require(len(groups) == len(expected_rates), f"Wrong active optimizer groups: {path}")
        require(all(close(group["lr"], rate, relative=0., absolute=1e-15) for group, rate in zip(groups, expected_rates)), f"Unexpected learning rates: {path}")
        state_steps = [int(state["step"]) for state in checkpoint["optimizer_state"]["state"].values() if "step" in state]
        require(state_steps and all(step == actual_step for step in state_steps), f"Optimizer state does not match actual updates: {path}")
        checkpoints[role] = {"path": str(path), "sha256": hash_file(path), "additional_steps": actual_step,
                             "total_steps": 200 + actual_step, "gamma": gamma, "world_size": ranks}
        del checkpoint
    gradient = audit_gradients(arm, config, world_size, train_keys)
    return {"checkpoints": checkpoints, "best_additional_step": expected_best, "common_best_score": scores[expected_best],
            "gradients": gradient, "training_metrics_sha256": hash_file(directory / "training_metrics.csv"),
            "validation_metrics_sha256": hash_file(directory / "validation_metrics.csv")}


def case_fingerprint(case):
    subset = case["path"] / "subsets" / f"subset_{case['subset']:02d}.mat"
    with h5py.File(subset, "r") as handle:
        inputs = np.asarray(handle["input_indices"], dtype=np.int64).ravel().tolist()
        held = np.asarray(handle["holdout_indices"], dtype=np.int64).ravel().tolist()
    require(len(inputs) == 10 and len(held) == 90, f"Incorrect frame split: {subset}")
    require(not set(inputs) & set(held) and set(inputs) | set(held) == set(range(1, 101)), f"Overlapping or incomplete frame split: {subset}")
    return {"subset_path": str(subset), "subset_sha256": hash_file(subset), "prepared_sha256": hash_file(case["path"] / "prepared.mat")}, inputs


def audit_evaluation(arm, training, cases, fingerprints):
    directory = OUTPUT / "evaluation" / arm
    complete = json_file(directory / "complete.json")
    require(complete.get("complete") is True and complete.get("experiment") == arm, f"Evaluation incomplete: {arm}")
    require(complete["cases_per_checkpoint"] == 114 and complete["brightness_cases"] == 30 and complete["checkpoints"] == ["best", "final"], f"Evaluation count mismatch: {arm}")
    require(complete.get("actual_weight_steps_used") is True, f"Evaluation did not verify actual weight steps: {arm}")
    hash_file(REPO / "tools/mean_round2_evaluation.py", complete["evaluation_source_sha256"])
    require(complete.get("precision", {}).get("cudnn_allow_tf32") is False and complete.get("precision", {}).get("matmul_allow_tf32") is False, f"Evaluation precision mismatch: {arm}")
    require(set(complete["table_sha256"]) == set(TABLES), f"Incomplete table hash inventory: {arm}")
    tables = {}
    for name in TABLES:
        hash_file(directory / (name + ".csv"), complete["table_sha256"][name])
        tables[name] = csv_file(directory / (name + ".csv"))
        require(tables[name], f"Empty evaluation table: {arm}/{name}")
    require(len(tables["metrics"]) == 228 and len(tables["brightness"]) == 30, f"Incomplete metric rows: {arm}")
    lookup = {(r["method"], r["case_id"]): r for r in tables["metrics"]}
    require(len(lookup) == 228, f"Duplicate case metrics: {arm}")
    brightness_lookup = {(r["method"], r["sample_id"], float(r["gain"])): r for r in tables["brightness"]}
    require(len(brightness_lookup) == 30, f"Duplicate brightness metrics: {arm}")
    leak_checks = []
    case_markers = {}
    for role in ROLES:
        checkpoint = training["checkpoints"][role]
        method = f"{arm}_{role}"
        actual_case_dirs = {path.parent.name for path in (directory / role).glob("*/complete.json")}
        require(actual_case_dirs == {case["id"] for case in cases}, f"Unexpected/missing case directories: {arm}/{role}")
        for case in cases:
            folder = directory / role / case["id"]
            marker = folder / "complete.json"
            stored = json_file(marker)
            require(stored.get("complete") is True and stored.get("inference_target_or_gt_used") is False, f"Invalid prediction completion/GT use flag: {marker}")
            require(stored["checkpoint_sha256"] == checkpoint["sha256"] and stored["checkpoint_role"] == role, f"Prediction/checkpoint binding mismatch: {marker}")
            require(int(stored["additional_weight_steps"]) == checkpoint["additional_steps"] and int(stored["weight_step"]) == checkpoint["total_steps"], f"Prediction weight-step mismatch: {marker}")
            fingerprint, inputs = fingerprints[case["id"]]
            require(stored["input_fingerprint"] == fingerprint and stored["input_indices"] == inputs, f"Prediction/input binding mismatch: {marker}")
            prediction = audit_volume(folder / "reconstruction.npy", stored["prediction_sha256"], exact_zero=case["split"] == "zero")
            audit_volume(folder / "anchor.npy", stored["anchor_sha256"], exact_zero=case["split"] == "zero")
            metrics = stored["metrics"]
            require(close(metrics["output_max"], prediction["maximum"]) and close(metrics["output_sum"], prediction["sum_float64"]), f"Recorded output quantities do not match NPY: {marker}")
            require(int(metrics["weight_step"]) == checkpoint["total_steps"] and int(metrics["additional_weight_steps"]) == checkpoint["additional_steps"], f"Metric weight-step mismatch: {marker}")
            same_metrics(lookup[method, case["id"]], metrics, str(marker))
            if case["id"] == "P09_subset_01":
                errors = {name: float(metrics[name]) for name in ("same_input_repeat_relative_l2", "gt_target_contamination_relative_l2")}
                require(all(math.isfinite(value) and 0 <= value <= 1e-6 for value in errors.values()), f"GT/target contamination or repeatability failure: {marker}")
                leak_checks.append({"method": method, **errors})
            case_markers[str(marker.relative_to(OUTPUT))] = hash_file(marker)
        for sample in BRIGHTNESS:
            base = old.DATA / "P09" if sample == "P09" else old.PRIORITY / "generated" / sample
            destination = directory / role / "brightness" / sample
            reference = np.load(destination / "gain_1.npy", allow_pickle=False)
            for gain in GAINS:
                marker = destination / f"gain_{gain:g}.json"
                record = json_file(marker)
                source = old.OUTPUT / "brightness_inputs" / sample / f"gain_{gain:g}.mat" if gain else base / "subsets/subset_01.mat"
                require(record.get("complete") is True and record.get("inference_target_or_gt_used") is False, f"Invalid brightness inference flag: {marker}")
                require(record["checkpoint_sha256"] == checkpoint["sha256"] and float(record["gain"]) == gain and record["synthetic_exact_zero"] == (gain == 0.), f"Brightness source/checkpoint mismatch: {marker}")
                hash_file(source, record["input_sha256"])
                path = destination / f"gain_{gain:g}.npy"
                volume = audit_volume(path, record["prediction_sha256"], exact_zero=gain == 0.)
                require(close(record["metrics"]["output_max"], volume["maximum"]) and close(record["metrics"]["output_sum"], volume["sum_float64"]), f"Brightness output quantities mismatch: {marker}")
                if gain:
                    value = np.load(path, allow_pickle=False)
                    expected = reference * np.float32(gain)
                    error = float(np.linalg.norm((value - expected).astype(np.float64)) / max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30))
                    require(close(record["metrics"]["relative_scale_error"], error, relative=3e-5, absolute=1e-7), f"Brightness scaling error cannot be reproduced from saved NPY: {marker}")
                same_metrics(brightness_lookup[method, sample, gain], record["metrics"], str(marker))
                case_markers[str(marker.relative_to(OUTPUT))] = hash_file(marker)
    require(len(leak_checks) == 2, f"Missing best/final target-contamination checks: {arm}")
    return {"primary_predictions": 228, "physical_anchors": 228, "brightness_predictions": 30,
            "leakage_checks": leak_checks, "completion_records_sha256": case_markers,
            "table_sha256": complete["table_sha256"], "complete_sha256": hash_file(directory / "complete.json")}


def audit_png(path):
    require(Path(path).is_file(), f"Missing figure: {path}")
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"Invalid PNG header: {path}")
    require(min(struct.unpack(">II", header[16:24])) > 0, f"Invalid PNG dimensions: {path}")
    return hash_file(path)


def audit_reports(training=None):
    analysis = OUTPUT / "analysis"
    completion = json_file(OUTPUT / "evaluation_complete.json")
    require(completion.get("complete") is True and set(completion["experiments"]) == set(ARMS), "Report aggregation incomplete")
    methods = {f"{name}_{role}" for name in ("baseline", "e3", *ARMS) for role in ROLES}
    require(set(completion["evaluated_methods"]) == methods, "Report lacks best/final reference/new methods")
    summary = csv_file(analysis / "summary.csv")
    require(len(summary) == 14 and {row["method"] for row in summary} == methods, "Summary must contain 14 methods")
    sources = json_file(analysis / "source_sha256.json")
    expected_sources = {str((old.OUTPUT if name in ("baseline", "e3") else OUTPUT) / "evaluation" / name / (table + ".csv")) for name in ("baseline", "e3", *ARMS) for table in TABLES}
    reference_source_count = len(expected_sources)
    for arm in ARMS:
        directory = OUTPUT / arm
        if training is not None:
            world_size = int(training[arm]["checkpoints"]["final"]["world_size"])
        else:
            payload = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            world_size = int(payload["world_size"])
            del payload
        expected_sources.update(str(directory / f"gradient_diagnostics_rank{rank}.jsonl") for rank in range(world_size))
        expected_sources.update(str(directory / filename) for filename in ("checkpoint_best.pt", "checkpoint_last.pt", "training_metrics.csv", "validation_metrics.csv"))
    require(set(sources) == expected_sources, "Aggregate table/gradient source inventory changed")
    for path, digest in sources.items():
        hash_file(path, digest)
    gradient_summary = csv_file(analysis / "gradient_summary.csv")
    gradient_steps = csv_file(analysis / "gradient_by_step.csv")
    require(len(gradient_summary) == 5 and {row["experiment"] for row in gradient_summary} == set(ARMS), "Gradient summary must contain five arms")
    require(len(gradient_steps) == 1000, "Gradient report must contain 1000 optimizer-step rows")
    for arm in ARMS:
        config = yaml.safe_load((OUTPUT / (arm + ".yaml")).read_text())
        spec = config["round2"]
        selected = [row for row in gradient_steps if row["experiment"] == arm]
        require(len(selected) == 200 and {int(row["additional_step"]) for row in selected} == set(range(1, 201)), f"Gradient report has missing/duplicate steps: {arm}")
        for row in selected:
            step = int(row["additional_step"])
            require(row["phase"] == "training" and int(row["sample_count"]) == 8 and int(row["weight_step"]) == 200 + step, f"Invalid reported gradient step: {arm} step {step}")
            require(row["all_gradients_finite"].lower() == "true" and row["all_budgets_passed"].lower() == "true", f"Failed reported gradient check: {arm} step {step}")
            budget = float(spec["shape_gradient_budget"]) * min(step / int(spec["ramp_steps"]), 1.)
            require(close(row["effective_gradient_budget"], budget, relative=0., absolute=1e-12), f"Wrong reported gradient ramp: {arm} step {step}")
            require(0 <= float(row["shape_gradient_ratio_max"]) <= budget + max(1e-7, budget * 1e-5), f"Reported gradient budget exceeded: {arm} step {step}")
        aggregate = next(row for row in gradient_summary if row["experiment"] == arm)
        require(aggregate["phase"] == "training" and int(aggregate["optimizer_steps"]) == 200 and int(aggregate["sample_count"]) == 1600 and int(aggregate["expected_global_batch"]) == 8, f"Gradient summary counts incorrect: {arm}")
        require(aggregate["all_gradients_finite"].lower() == "true" and aggregate["all_budgets_passed"].lower() == "true", f"Gradient summary failed: {arm}")
        require(int(aggregate["final_additional_step"]) == 200 and int(aggregate["final_weight_step"]) == 400, f"Gradient summary final step incorrect: {arm}")
        require(close(aggregate["shape_gradient_ratio_max"], max(float(row["shape_gradient_ratio_max"]) for row in selected)), f"Gradient summary maximum inconsistent: {arm}")
        if training is not None:
            for role in ROLES:
                checkpoint = training[arm]["checkpoints"][role]
                require(int(aggregate[role + "_additional_step"]) == checkpoint["additional_steps"] and int(aggregate[role + "_weight_step"]) == checkpoint["total_steps"], f"Gradient summary checkpoint step incorrect: {arm}/{role}")
                require(close(aggregate[role + "_gamma"], checkpoint["gamma"], relative=1e-12, absolute=1e-14), f"Gradient summary gamma incorrect: {arm}/{role}")
    report = OUTPUT / "REPORT_ZH.md"
    atlas = analysis / "actual_reconstruction_comparison/README_ZH.md"
    conclusions = OUTPUT / "CONCLUSIONS_ZH.md"
    require(conclusions.is_file() and conclusions.stat().st_size > 1000, "Plain-language conclusions missing/empty")
    require(report.is_file() and report.stat().st_size > 1000, "Chinese report missing/empty")
    require(atlas.is_file() and atlas.stat().st_size > 200, "Reconstruction atlas missing/empty")
    artifact_hashes = {str(path.relative_to(OUTPUT)): hash_file(path) for path in (report, atlas, conclusions, analysis / "summary.csv", OUTPUT / "evaluation_complete.json")}
    for document in (report, atlas, conclusions):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            linked = (document.parent / target.split("#", 1)[0]).resolve()
            require(linked.is_file(), f"Broken report/atlas link: {document} -> {target}")
    figures = json_file(analysis / "figure_manifest.json")
    require(figures, "Figure manifest empty")
    inventory = Counter()
    for entry in figures:
        path = (OUTPUT / entry["path"]).resolve()
        require(path.is_relative_to(OUTPUT), "Figure escaped isolated output directory")
        require(entry["role"] in ROLES and entry["display"] in ("shape", "shared"), "Wrong comparison panel metadata")
        require(set(entry["methods"]) == {"baseline_final", "e3_final", *[f"{arm}_{entry['role']}" for arm in ARMS]}, "Comparison panel lacks required frozen methods")
        inventory[entry["case_id"], entry["role"], entry["display"]] += 1
        artifact_hashes[str(path.relative_to(OUTPUT))] = audit_png(path)
    represented = {key[0] for key in inventory}
    require({"P07_subset_01", "T01_subset_01", "T02_subset_01", "points_z060_r01", "lines_z060_r01", "axial_pairs_r01"} <= represented, "Atlas lacks required test/local scenes")
    require(all(inventory[case, role, display] == 1 for case in represented for role in ROLES for display in ("shape", "shared")), "Atlas lacks or duplicates a best/final/display panel")
    for name in ("comparison.png", "depth_following_all_repeats.png", "training_validation_gradient_trends.png"):
        artifact_hashes[str((analysis / name).relative_to(OUTPUT))] = audit_png(analysis / name)
    for name in (*TABLES, "per_object_summary", "per_repeat_summary", "line_quality_summary", "pair_resolution_summary", "failure_cases", "engineering_acceptance", "representative_fixed_profiles", "gradient_summary", "gradient_by_step"):
        path = analysis / (name + ".csv")
        require(path.is_file() and path.stat().st_size > 0, f"Missing analysis CSV: {path}")
        artifact_hashes[str(path.relative_to(OUTPUT))] = hash_file(path)
    return {"methods": 14, "comparison_panels": len(figures), "reference_source_tables": reference_source_count,
            "gradient_source_files": len(sources) - reference_source_count, "all_report_sources": len(sources),
            "gradient_summary_rows": 5, "gradient_step_rows": 1000, "artifact_sha256": artifact_hashes}


def run():
    started = time.monotonic()
    record = {"complete": False, "passed": False, "cpu_only": True,
              "verifier_source_sha256": hash_file(Path(__file__)), "started_unix": time.time()}
    try:
        preflight, configs, indexed, provenance = audit_provenance()
        record["provenance"] = provenance
        print(json.dumps({"accepted": "provenance", **provenance}), flush=True)
        training = {}
        for arm in ARMS:
            training[arm] = audit_training(arm, configs[arm], indexed["train"])
            print(json.dumps({"accepted": "training", "arm": arm, "best_step": training[arm]["best_additional_step"]}), flush=True)
        record["training"] = training
        cases = all_cases()
        require(len(cases) == 114 and len({case["id"] for case in cases}) == 114, "Frozen evaluation case inventory changed")
        fingerprints = {case["id"]: case_fingerprint(case) for case in cases}
        evaluations = {}
        for arm in ARMS:
            evaluations[arm] = audit_evaluation(arm, training[arm], cases, fingerprints)
            print(json.dumps({"accepted": "evaluation", "arm": arm, "predictions": 228, "brightness": 30}), flush=True)
        record["evaluation"] = evaluations
        record["reports"] = audit_reports(training)
        record.update(complete=True, passed=True, finished_unix=time.time(), seconds=time.monotonic() - started,
                      primary_predictions=1140, physical_anchors=1140, brightness_predictions=150,
                      training_updates=1000, valid_training_microbatches=8000,
                      note="Artifact/numerical acceptance does not require a method to outperform a control.")
        old.write_json(OUTPUT / "final_acceptance.json", record)
        old.write_json(OUTPUT / "complete.json", {"complete": True, "passed": True, "finished_unix": time.time(),
                       "final_acceptance_sha256": hash_file(OUTPUT / "final_acceptance.json"),
                       "report_sha256": hash_file(OUTPUT / "REPORT_ZH.md"), "preflight_sha256": provenance["preflight_sha256"],
                       "experiments": list(ARMS), "primary_predictions": 1140, "physical_anchors": 1140,
                       "brightness_predictions": 150, "best_and_final_verified": True})
        print(json.dumps({"final_acceptance": "passed", "path": str(OUTPUT / "final_acceptance.json")}), flush=True)
        return record
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error), finished_unix=time.time(), seconds=time.monotonic() - started)
        old.write_json(OUTPUT / "final_acceptance.json", record)
        if (OUTPUT / "complete.json").exists():
            old.write_json(OUTPUT / "complete.json", {"complete": False, "passed": False,
                           "error": str(error), "final_acceptance_sha256": hash_file(OUTPUT / "final_acceptance.json")})
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
