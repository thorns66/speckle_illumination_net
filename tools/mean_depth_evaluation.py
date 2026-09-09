"""Evaluate best/final checkpoints for the four mean/depth continuation arms."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from datasets.matlab_multivolume_dataset import load_inference_input
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean_depth_experiment as exp
from tools import three_way_experiment as old
import tools.v3_compare_evaluation as base
import tools.v3_compare_local_audit as local
import tools.priority_validation_analysis as pa
from tools.priority_validation_common import load_config as load_priority_config
from tools.three_way_checks import input_item


TABLES = (
    "metrics", "t02_tubes", "t03_lines", "t04_axial", "v03_beads",
    "point_targets", "point_pairs", "priority_lines", "depth_profiles", "axial_points",
)


def all_cases():
    previous = base.exp
    base.exp = exp
    try:
        return base.all_cases()
    finally:
        base.exp = previous


def evaluate_arm(arm: str, *, resume: bool = False) -> dict:
    if arm not in exp.ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    exp.configure_precision()
    cases = all_cases()
    fingerprints = {case["id"]: base._fingerprint(case) for case in cases}
    _priority_path, priority_config = load_priority_config()
    result_dir = exp.OUTPUT / "evaluation" / arm
    result_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in TABLES}
    operator = None
    started = time.time()
    for role, filename in (("best", "checkpoint_best.pt"), ("final", "checkpoint_last.pt")):
        checkpoint_path = exp.OUTPUT / arm / filename
        checkpoint_hash = old.sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config.get("mean_depth", {}).get("kind") != arm:
            raise ValueError(f"Checkpoint belongs to another experiment: {checkpoint_path}")
        extra_step = int(checkpoint["completed_steps"])
        if role == "final" and extra_step != 200:
            raise ValueError(f"Final checkpoint is additional step {extra_step}, expected 200")
        if role == "best" and (extra_step % 20 or not 20 <= extra_step <= 200):
            raise ValueError(f"Best checkpoint is not a fixed validation step: {extra_step}")
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
            expected = {"checkpoint_sha256": checkpoint_hash, "input_fingerprint": fingerprints[case["id"]]}
            if resume and marker.exists():
                record = base._resume_case(marker, expected, destination)
                prediction = np.load(destination / "reconstruction.npy", allow_pickle=False)
                row = record["metrics"]
                target = base._targets(case, device)
                truth = target.get("ground_truth")
                truth_np = None if truth is None else truth[0, 0].cpu().numpy()
            else:
                item = input_item(case["path"], case["subset"], device)
                exp._STATE.update(phase="evaluation", evaluation=True, teacher=None, teacher_device=None)
                with torch.inference_mode():
                    output, beta0 = exp.forward(model, item, operator, config=config)
                    prediction_tensor = output.reconstruction
                    prediction = prediction_tensor[0, 0].float().cpu().numpy()
                    anchor = output._physical_anchor[0, 0].float().cpu().numpy()
                target = base._targets(case, device)
                scoring = {**item, **target}
                with torch.inference_mode():
                    native = exp.loss(output, scoring, operator, variance_model, config)
                    common = base._common_scores(
                        prediction_tensor, target["measured_mean"], target["measured_variance"],
                        operator, variance_model, config,
                    )
                row = {
                    "method": method,
                    "experiment": arm,
                    "checkpoint_role": role,
                    "additional_weight_step": extra_step,
                    "total_weight_step": 400 + extra_step,
                    "sample_id": case["sample"], "case_id": case["id"],
                    "subset": case["subset"], "split": case["split"],
                    "native_total_loss": float(native.total),
                    "native_mean_loss": float(native.normalized_mean),
                    "native_variance_loss": float(native.normalized_var),
                    "native_weighted_tv": float(native.weighted_tv),
                    "beta0": float(beta0), "beta": float(output.beta), "gain": float(output._gain[0]),
                    **common,
                }
                truth = target.get("ground_truth")
                truth_np = None if truth is None else truth[0, 0].cpu().numpy()
                if truth_np is not None:
                    row.update(base._structure_row(prediction, truth_np))
                else:
                    row.update(prediction_max=float(prediction.max()), prediction_sum=float(prediction.sum()))
                if case["id"] == "V01_subset_01":
                    poisoned = dict(item)
                    poisoned.update(
                        ground_truth=torch.full_like(prediction_tensor, 999),
                        measured_mean=torch.full_like(item["input_mean"], 999),
                        measured_variance=torch.full_like(item["input_mean"], 999),
                    )
                    with torch.inference_mode():
                        repeated, _ = exp.forward(model, item, operator, beta0, config=config)
                        changed, _ = exp.forward(model, poisoned, operator, beta0, config=config)
                    denominator = prediction_tensor.norm().clamp_min(1e-30)
                    repeat_error = float((repeated.reconstruction - prediction_tensor).norm() / denominator)
                    contamination_error = float((changed.reconstruction - prediction_tensor).norm() / denominator)
                    if max(repeat_error, contamination_error) > 1e-4:
                        raise ValueError("Same-input repeatability or target/GT isolation failed")
                    row.update(same_input_repeat_relative_l2=repeat_error, gt_target_contamination_relative_l2=contamination_error)
                record = {
                    **expected,
                    "checkpoint_role": role,
                    "additional_weight_step": extra_step,
                    "total_weight_step": 400 + extra_step,
                    "inference_target_or_gt_used": False,
                    "input_indices": np.asarray(item["input_indices"]).tolist(),
                    "metrics": row,
                    "source_checkpoint": str(checkpoint_path),
                }
                base._save_case(destination, prediction, anchor, record)
            tables["metrics"].append(row)
            metadata = {
                "method": method, "case_id": case["id"], "subset": case["subset"],
                "sample_id": case["sample"], "split": case["split"],
            }
            if case["sample"] in ("T02", "T03", "T04", "V03"):
                if truth_np is None:
                    raise ValueError("Official structural diagnostic lacks GT")
                table_name, function = {
                    "T02": ("t02_tubes", local.t02),
                    "T03": ("t03_lines", local.t03),
                    "T04": ("t04_axial", local.t04),
                    "V03": ("v03_beads", local.v03),
                }[case["sample"]]
                rows, _profiles = function(prediction, truth_np, metadata)
                tables[table_name].extend(rows)
            if case["split"] == "priority":
                if truth_np is None:
                    raise ValueError("Priority diagnostic lacks GT")
                methods = {method: prediction}
                if case["family"] == "points":
                    points, pairs, profiles = pa._point_metrics(priority_config, case["scene_id"], case["repeat"], truth_np, methods)
                    tables["point_targets"].extend(points)
                    tables["point_pairs"].extend(pairs)
                    tables["depth_profiles"].extend(profiles)
                elif case["family"] == "lines":
                    tables["priority_lines"].extend(pa._line_metrics(priority_config, case["scene_id"], case["repeat"], methods))
                else:
                    axial, pairs = pa._axial_metrics(priority_config, case["repeat"], methods)
                    tables["axial_points"].extend(axial)
                    tables["point_pairs"].extend(pairs)
            print(json.dumps({"evaluated": method, "case": case["id"], "total_step": 400 + extra_step}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    for name, rows in tables.items():
        base.csv_write(result_dir / f"{name}.csv", rows)
    result = {
        "complete": True, "experiment": arm,
        "cases_per_checkpoint": 94, "checkpoint_roles": ["best", "final"],
        "predictions": 188, "anchors": 188,
        "official_items_per_checkpoint": 60, "priority_items_per_checkpoint": 33,
        "zero_items_per_checkpoint": 1,
        "tables": {name: len(rows) for name, rows in tables.items()},
        "source_sha256": old.sha256(Path(__file__)),
        "seconds": time.time() - started, "finished_unix": time.time(),
    }
    old.write_json(result_dir / "complete.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=exp.ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_arm(args.experiment, resume=args.resume), indent=2))
