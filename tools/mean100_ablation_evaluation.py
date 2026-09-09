"""Evaluate the E3+100% step extension and no-Set ablation checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from losses.self_supervised_losses import TaylorH2VarianceModel
import tools.priority_validation_analysis as pa
from tools.priority_validation_common import load_config as load_priority_config
from tools.three_way_checks import input_item
from tools import mean100_ablation_experiment as exp
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as v3


ROLE_FILES = {
    "extend600": (("best", "checkpoint_best.pt"), ("step600", "checkpoint_last.pt")),
    "noset600": (
        ("best", "checkpoint_best.pt"),
        ("step400", "checkpoint_step_000400.pt"),
        ("step600", "checkpoint_last.pt"),
    ),
}


def _expected_step(role: str, actual: int) -> None:
    if role == "step400" and actual != 400:
        raise ValueError(f"step400 checkpoint contains step {actual}")
    if role == "step600" and actual != 600:
        raise ValueError(f"step600 checkpoint contains step {actual}")
    if role == "best" and (actual % 20 or not 20 <= actual <= 600):
        raise ValueError(f"Invalid best checkpoint step {actual}")


def evaluate_arm(arm: str, *, resume: bool = False) -> dict:
    if arm not in exp.ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    previous = v3.exp
    v3.exp = exp
    try:
        return _evaluate_arm(arm, resume=resume)
    finally:
        v3.exp = previous


def _evaluate_arm(arm: str, *, resume: bool) -> dict:
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    exp.configure_precision()
    cases = v3.all_cases()
    fingerprints = {case["id"]: v3._fingerprint(case) for case in cases}
    _priority_path, priority_config = load_priority_config()
    result_dir = exp.OUTPUT / "evaluation" / arm
    result_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in v3.TABLES}
    operator = None
    started = time.time()
    roles = ROLE_FILES[arm]
    for role, filename in roles:
        checkpoint_path = exp.OUTPUT / arm / filename
        checkpoint_hash = old.sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config.get("v3_compare", {}).get("kind") != "e3_mean100":
            raise ValueError(f"Checkpoint does not use E3+100%: {checkpoint_path}")
        if arm == "noset600" and bool(config["ablation"]["use_set_branch"]):
            raise ValueError("No-Set checkpoint unexpectedly enables SetBranch")
        if arm == "extend600" and role != "best" and not bool(config["ablation"]["use_set_branch"]):
            raise ValueError("Continuation checkpoint unexpectedly disables SetBranch")
        step = int(checkpoint["completed_steps"])
        _expected_step(role, step)
        method = f"{arm}_{role}"
        model = exp.build_model(config, initial=False).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        if operator is None:
            operator = exp.load_operator(config, device)
        variance_model = TaylorH2VarianceModel(operator, **config["noise"])
        for case in cases:
            destination = result_dir / role / case["id"]
            marker = destination / "complete.json"
            expected_record = {
                "checkpoint_sha256": checkpoint_hash,
                "input_fingerprint": fingerprints[case["id"]],
            }
            if resume and marker.exists():
                record = v3._resume_case(marker, expected_record, destination)
                prediction = np.load(destination / "reconstruction.npy", allow_pickle=False)
                row = record["metrics"]
                target = v3._targets(case, device)
                truth = target.get("ground_truth")
                truth_np = None if truth is None else truth[0, 0].cpu().numpy()
            else:
                item = input_item(case["path"], case["subset"], device)
                with torch.inference_mode():
                    output, beta0 = exp.forward(model, item, operator, config=config)
                    prediction_tensor = output.reconstruction
                    prediction = prediction_tensor[0, 0].float().cpu().numpy()
                    anchor = output._physical_anchor[0, 0].float().cpu().numpy()
                target = v3._targets(case, device)
                scoring = {**item, **target}
                with torch.inference_mode():
                    native = exp.loss(output, scoring, operator, variance_model, config)
                    common = v3._common_scores(
                        prediction_tensor,
                        target["measured_mean"],
                        target["measured_variance"],
                        operator,
                        variance_model,
                        config,
                    )
                row = {
                    "method": method,
                    "experiment": arm,
                    "checkpoint_role": role,
                    "weight_step": step,
                    "sample_id": case["sample"],
                    "case_id": case["id"],
                    "subset": case["subset"],
                    "split": case["split"],
                    "native_total_loss": float(native.total),
                    "native_mean_loss": float(native.normalized_mean),
                    "native_variance_loss": float(native.normalized_var),
                    "native_weighted_tv": float(native.weighted_tv),
                    "beta0": float(beta0),
                    "beta": float(output.beta),
                    "gain": float(output._gain[0]),
                    **common,
                }
                truth = target.get("ground_truth")
                truth_np = None if truth is None else truth[0, 0].cpu().numpy()
                if truth_np is not None:
                    row.update(v3._structure_row(prediction, truth_np))
                else:
                    row.update(
                        prediction_max=float(prediction.max()),
                        prediction_sum=float(prediction.sum()),
                    )
                if case["id"] == "V01_subset_01":
                    contaminated = dict(item)
                    contaminated.update(
                        ground_truth=torch.full_like(prediction_tensor, 999),
                        measured_mean=torch.full_like(item["input_mean"], 999),
                        measured_variance=torch.full_like(item["input_mean"], 999),
                    )
                    with torch.inference_mode():
                        repeated, _ = exp.forward(model, item, operator, beta0, config=config)
                        changed, _ = exp.forward(model, contaminated, operator, beta0, config=config)
                    denominator = prediction_tensor.norm().clamp_min(1e-30)
                    repeat_error = float((repeated.reconstruction - prediction_tensor).norm() / denominator)
                    contamination_error = float((changed.reconstruction - prediction_tensor).norm() / denominator)
                    if max(repeat_error, contamination_error) > 1e-4:
                        raise ValueError("Repeatability or GT/target isolation failed")
                    row.update(
                        same_input_repeat_relative_l2=repeat_error,
                        gt_target_contamination_relative_l2=contamination_error,
                    )
                record = {
                    **expected_record,
                    "checkpoint_role": role,
                    "weight_step": step,
                    "inference_target_or_gt_used": False,
                    "input_indices": np.asarray(item["input_indices"]).tolist(),
                    "metrics": row,
                    "source_checkpoint": str(checkpoint_path),
                }
                v3._save_case(destination, prediction, anchor, record)
            tables["metrics"].append(row)
            if case["sample"] == "T03":
                for threshold in (0.05, 0.10, 0.20):
                    tables["t03_lines"].extend(v3.t03_metrics(method, prediction, threshold))
            if case["sample"] == "T04":
                for threshold in (0.05, 0.10, 0.20):
                    tables["t04_axial"].extend(v3.t04_metrics(method, prediction, threshold))
            if case["sample"] == "V03":
                for threshold in (0.05, 0.10, 0.20):
                    tables["v03_beads"].extend(v3.v03_metrics(method, prediction, threshold))
            if case["split"] == "priority":
                if truth_np is None:
                    raise ValueError("Priority case unexpectedly lacks GT")
                methods = {method: prediction}
                if case["family"] == "points":
                    points, pairs, profiles = pa._point_metrics(
                        priority_config, case["scene_id"], case["repeat"], truth_np, methods
                    )
                    tables["point_targets"].extend(points)
                    tables["point_pairs"].extend(pairs)
                    tables["depth_profiles"].extend(profiles)
                elif case["family"] == "lines":
                    tables["priority_lines"].extend(
                        pa._line_metrics(priority_config, case["scene_id"], case["repeat"], methods)
                    )
                else:
                    axial, pairs = pa._axial_metrics(priority_config, case["repeat"], methods)
                    tables["axial_points"].extend(axial)
                    tables["point_pairs"].extend(pairs)
        del model, checkpoint
        torch.cuda.empty_cache()
    for name, rows in tables.items():
        v3.csv_write(result_dir / f"{name}.csv", rows)
    role_names = [role for role, _ in roles]
    record = {
        "complete": True,
        "experiment": arm,
        "cases_per_checkpoint": 94,
        "checkpoint_roles": role_names,
        "predictions": 94 * len(roles),
        "anchors": 94 * len(roles),
        "official_items_per_checkpoint": 60,
        "priority_items_per_checkpoint": 33,
        "zero_items_per_checkpoint": 1,
        "tables": {name: len(rows) for name, rows in tables.items()},
        "source_sha256": old.sha256(Path(__file__)),
        "finished_unix": time.time(),
        "seconds": time.time() - started,
    }
    old.write_json(result_dir / "complete.json", record)
    return record

