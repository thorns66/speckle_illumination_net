"""Best/final inference and fixed structural metrics for the V3 comparison."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
for _search_path in (str(_REPO), str(_REPO / "tools")):
    if _search_path not in sys.path:
        sys.path.insert(0, _search_path)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.optimize import linear_sum_assignment

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets, load_dataset_index
from losses.self_supervised_losses import TaylorH2VarianceModel, total_variation_3d
import tools.priority_validation_analysis as pa
from tools.priority_validation_common import load_config as load_priority_config, scenes
from tools.three_way_checks import input_item
from tools import three_way_experiment as old
from tools import v3_compare_experiment as exp


TABLES = (
    "metrics",
    "t03_lines",
    "t04_axial",
    "v03_beads",
    "point_targets",
    "point_pairs",
    "priority_lines",
    "depth_profiles",
    "axial_points",
)
Z_UM = np.arange(10.0, 101.0, 10.0)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def all_cases() -> list[dict[str, Any]]:
    indexed, _fingerprint = load_dataset_index(exp.DATA)
    result = []
    for split in ("validation", "test"):
        for key in indexed[split]:
            result.append(
                {
                    "id": f"{key.sample_id}_subset_{key.subset_index:02d}",
                    "path": key.sample_dir,
                    "subset": key.subset_index,
                    "split": split,
                    "sample": key.sample_id,
                }
            )
    _path, priority_config = load_priority_config()
    priority_root = old.PRIORITY
    for scene in scenes(priority_config):
        for repeat in range(1, 4):
            name = f"{scene.scene_id}_r{repeat:02d}"
            result.append(
                {
                    "id": name,
                    "path": priority_root / "generated" / name,
                    "subset": 1,
                    "split": "priority",
                    "sample": name,
                    "family": scene.family,
                    "scene_id": scene.scene_id,
                    "repeat": repeat,
                    "truth_depth_um": scene.depth_um,
                }
            )
    result.append(
        {
            "id": "zero_control",
            "path": priority_root / "generated/zero_control",
            "subset": 1,
            "split": "zero",
            "sample": "zero_control",
        }
    )
    if len(result) != 94 or len({case["id"] for case in result}) != 94:
        raise ValueError(f"Expected 94 unique evaluation cases, found {len(result)}")
    return result


def _targets(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    raw = _read_targets(key, include_ground_truth=case["split"] != "zero")
    return {
        name: torch.from_numpy(value).float().unsqueeze(0).to(device)
        if name in ("measured_mean", "measured_variance", "ground_truth")
        else value
        for name, value in raw.items()
    }


def _fingerprint(case: dict[str, Any]) -> dict[str, str]:
    subset = case["path"] / "subsets" / f"subset_{case['subset']:02d}.mat"
    result = {"subset_path": str(subset), "subset_sha256": old.sha256(subset)}
    prepared = case["path"] / "prepared.mat"
    if prepared.is_file():
        result["prepared_sha256"] = old.sha256(prepared)
    return result


def _save_case(destination: Path, prediction: np.ndarray, anchor: np.ndarray, record: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "reconstruction.npy", prediction.astype(np.float32, copy=False))
    np.save(destination / "anchor.npy", anchor.astype(np.float32, copy=False))
    record.update(
        complete=True,
        prediction_sha256=old.sha256(destination / "reconstruction.npy"),
        anchor_sha256=old.sha256(destination / "anchor.npy"),
    )
    old.write_json(destination / "complete.json", record)


@torch.inference_mode()
def _common_scores(prediction, measured_mean, measured_variance, operator, variance_model, config):
    eps = float(config["loss"]["var_log_eps"])
    predicted_mean = operator(prediction)
    predicted_variance = variance_model(prediction, measured_mean)
    mean_scale = measured_mean.abs().mean().clamp_min(1e-8)
    mean_loss = F.smooth_l1_loss(predicted_mean / mean_scale, measured_mean / mean_scale)
    absolute_var = F.smooth_l1_loss(
        torch.log(predicted_variance.clamp_min(0) + eps),
        torch.log(measured_variance.clamp_min(0) + eps),
    )
    q = prediction / prediction.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-30)
    shape_variance = variance_model(q, measured_mean)
    normalized_prediction = shape_variance / shape_variance.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    normalized_target = measured_variance / measured_variance.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    shape_var = F.smooth_l1_loss(
        torch.log(normalized_prediction.clamp_min(0) + eps),
        torch.log(normalized_target.clamp_min(0) + eps),
    )
    tv = total_variation_3d(prediction, z_weight=float(config["loss"]["lambda_tv_z"]))
    weighted_tv = tv * float(config["loss"]["lambda_tv"])
    return {
        "common_normalized_mean_loss": float(mean_loss),
        "common_absolute_log_variance_loss": float(absolute_var),
        "common_normalized_shape_variance_loss": float(shape_var),
        "common_weighted_tv": float(weighted_tv),
        "common_baseline_score": float(absolute_var + weighted_tv),
        "common_e3_score": float(mean_loss + shape_var + weighted_tv),
    }


def _structure_row(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    row = pa._score_structure_metrics(prediction, truth)
    support = binary_dilation(
        truth.max(axis=0) > 0.1 * max(float(truth.max()), 1e-30), iterations=2
    )
    positive = np.maximum(prediction, 0)
    row["background_xy_mass_fraction"] = float(
        positive[:, ~support].sum() / max(float(positive.sum()), 1e-30)
    )
    row["prediction_max"] = float(prediction.max())
    row["prediction_sum"] = float(prediction.sum())
    row["truth_max"] = float(truth.max())
    row["truth_sum"] = float(truth.sum())
    return row


def _clip_bounds(bounds: list[int], shape=(260, 260)) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(value) - 1 for value in bounds]
    return max(0, x0), max(0, y0), min(shape[1] - 1, x1), min(shape[0] - 1, y1)


def t03_metrics(method: str, prediction: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    geometry = json.loads((exp.DATA / "T03/geometry.json").read_text(encoding="utf-8"))["geometry"]
    layer = int(np.argmin(np.abs(Z_UM - 50)))
    image = np.maximum(prediction[max(0, layer - 1) : min(10, layer + 2)], 0).max(axis=0)
    global_threshold = float(prediction.max()) * threshold
    rows = []
    for index, group in enumerate(geometry, 1):
        orientation = group["orientation"]
        bars = [_clip_bounds(bounds) for bounds in group["bars_xy_one_based"]]
        if orientation == "horizontal":
            x0 = max(bounds[0] for bounds in bars)
            x1 = min(bounds[2] for bounds in bars)
            trim = int(math.floor((x1 - x0 + 1) * 0.2))
            xa, xb = x0 + trim, x1 - trim
            profile = image[:, xa : xb + 1].mean(axis=1)
            intervals = [(bounds[1], bounds[3]) for bounds in bars]
            longitudinal = [image[bounds[1] : bounds[3] + 1, xa : xb + 1].mean(axis=0) for bounds in bars]
        else:
            y0 = max(bounds[1] for bounds in bars)
            y1 = min(bounds[3] for bounds in bars)
            trim = int(math.floor((y1 - y0 + 1) * 0.2))
            ya, yb = y0 + trim, y1 - trim
            profile = image[ya : yb + 1].mean(axis=0)
            intervals = [(bounds[0], bounds[2]) for bounds in bars]
            longitudinal = [image[ya : yb + 1, bounds[0] : bounds[2] + 1].mean(axis=1) for bounds in bars]
        peaks = [float(profile[start : end + 1].max()) for start, end in intervals]
        localized = [peak >= global_threshold for peak in peaks]
        valleys = []
        ratios = []
        for left, right in zip(intervals[:-1], intervals[1:]):
            between = profile[left[1] + 1 : right[0]]
            valley = float(between.min()) if between.size else float("nan")
            weak_peak = min(peaks[len(valleys)], peaks[len(valleys) + 1])
            valleys.append(valley)
            ratios.append(valley / max(weak_peak, 1e-30))
        false_breaks = sum(bool(np.any(track < global_threshold)) for track in longitudinal)
        rows.append(
            {
                "method": method,
                "group": index,
                "orientation": orientation,
                "width_px": group["width_px"],
                "width_um": group["width_um"],
                "threshold_fraction": threshold,
                "all_three_localized": all(localized),
                "valley_ratio_1": ratios[0],
                "valley_ratio_2": ratios[1],
                "separated": all(localized) and max(ratios) <= 0.8,
                "false_break_count": false_breaks,
            }
        )
    return rows


def _um_slice(bounds: list[float], pitch: float, shape=(260, 260)) -> tuple[slice, slice]:
    x0, y0, x1, y1 = bounds
    ix0, iy0 = int(np.floor(x0 / pitch)), int(np.floor(y0 / pitch))
    ix1, iy1 = int(np.ceil(x1 / pitch)), int(np.ceil(y1 / pitch))
    return slice(max(0, iy0), min(shape[0], iy1 + 1)), slice(max(0, ix0), min(shape[1], ix1 + 1))


def _profile_peaks(profile: np.ndarray, threshold: float) -> np.ndarray:
    local = profile == maximum_filter(profile, size=3, mode="nearest")
    return np.flatnonzero(local & (profile >= threshold))


def t04_metrics(method: str, prediction: np.ndarray, threshold_fraction: float) -> list[dict[str, Any]]:
    geometry = json.loads((exp.DATA / "T04/geometry.json").read_text(encoding="utf-8"))["geometry"]
    pitch = float(json.loads((exp.DATA / "T04/simulation_config.json").read_text(encoding="utf-8"))["object_pixel_pitch_um"])
    rows = []
    for region in geometry:
        ys, xs = _um_slice(region["roi_bounds_xy_um"], pitch)
        profile = np.maximum(prediction[:, ys, xs], 0).sum(axis=(1, 2))
        threshold = threshold_fraction * max(float(profile.max()), 1e-30)
        candidates = _profile_peaks(profile, threshold)
        expected_um = region["z_um"] if isinstance(region["z_um"], list) else [region["z_um"]]
        expected = np.array([int(np.argmin(np.abs(Z_UM - depth))) for depth in expected_um])
        # One-to-one nearest peak matching with the fixed one-layer tolerance.
        matched = []
        used = set()
        for target in expected:
            available = [(abs(int(candidate) - int(target)), int(candidate)) for candidate in candidates if int(candidate) not in used]
            if available and min(available)[0] <= 1:
                distance, candidate = min(available)
                used.add(candidate)
                matched.append((candidate, distance))
            else:
                matched.append((None, None))
        localized = all(value[0] is not None for value in matched)
        separation = region["separation_um"] if region["separation_um"] != [] else None
        valley_ratio = float("nan")
        separated: bool | str = "not_applicable"
        if len(expected) == 2 and separation != 10 and localized:
            first, second = sorted(int(value[0]) for value in matched)
            interior = profile[first + 1 : second]
            if interior.size:
                valley = float(interior.min())
                valley_ratio = valley / max(min(float(profile[first]), float(profile[second])), 1e-30)
                separated = valley_ratio <= 0.8
        elif len(expected) == 2 and separation != 10:
            separated = False
        expected_energy = float(profile[expected].sum() / max(float(profile.sum()), 1e-30))
        weak_ratio = float("nan")
        if len(expected) == 2:
            weak_ratio = float(profile[expected[1]] / max(float(profile[expected[0]]), 1e-30))
        rows.append(
            {
                "method": method,
                "region_id": region["region_id"],
                "family": region["family"],
                "separation_um": separation,
                "threshold_fraction": threshold_fraction,
                "all_layers_localized": localized,
                "mean_z_error_layers": float(np.mean([value[1] for value in matched if value[1] is not None])) if any(value[1] is not None for value in matched) else float("nan"),
                "valley_ratio": valley_ratio,
                "separated": separated,
                "expected_layer_energy_fraction": expected_energy,
                "deep_to_shallow_energy_ratio": weak_ratio,
                "false_peak_count": max(0, len(candidates) - len(used)),
            }
        )
    return rows


def v03_metrics(method: str, prediction: np.ndarray, threshold_fraction: float) -> list[dict[str, Any]]:
    geometry = json.loads((exp.DATA / "V03/geometry.json").read_text(encoding="utf-8"))["geometry"]
    beads = [entry for entry in geometry if entry.get("kind") == "solid_sphere"]
    pitch = float(json.loads((exp.DATA / "V03/simulation_config.json").read_text(encoding="utf-8"))["object_pixel_pitch_um"])
    threshold = threshold_fraction * max(float(prediction.max()), 1e-30)
    local = prediction == maximum_filter(prediction, size=(3, 5, 5), mode="nearest")
    candidates = np.argwhere(local & (prediction >= threshold))
    expected = np.array(
        [[int(np.argmin(np.abs(Z_UM - entry["center_xyz_um"][2]))),
          int(round(entry["center_xyz_um"][1] / pitch)),
          int(round(entry["center_xyz_um"][0] / pitch))] for entry in beads], dtype=float
    )
    if len(candidates):
        dz = np.abs(expected[:, None, 0] - candidates[None, :, 0])
        dxy = np.sqrt(((expected[:, None, 1:] - candidates[None, :, 1:]) ** 2).sum(axis=2))
        cost = dxy + 10 * dz
        invalid = (dz > 1) | (dxy > 2)
        cost[invalid] = 1e6
        rows_idx, cols_idx = linear_sum_assignment(cost)
        matches = {int(row): int(col) for row, col in zip(rows_idx, cols_idx) if cost[row, col] < 1e6}
    else:
        matches = {}
    rows = []
    for index, entry in enumerate(beads):
        match = matches.get(index)
        if match is None:
            z_error = xy_error = float("nan")
        else:
            candidate = candidates[match]
            z_error = float(abs(candidate[0] - expected[index, 0]))
            xy_error = float(np.linalg.norm(candidate[1:] - expected[index, 1:]))
        rows.append(
            {
                "method": method,
                "bead_index": entry["bead_index"],
                "threshold_fraction": threshold_fraction,
                "detected": match is not None,
                "z_error_layers": z_error,
                "xy_error_pixels": xy_error,
                "global_false_peak_count": max(0, len(candidates) - len(matches)),
            }
        )
    return rows


def _resume_case(marker: Path, expected: dict, destination: Path):
    record = json.loads(marker.read_text(encoding="utf-8"))
    if not record.get("complete"):
        raise ValueError(f"Incomplete evaluation marker: {marker}")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Resume fingerprint mismatch for {marker}: {key}")
    if old.sha256(destination / "reconstruction.npy") != record["prediction_sha256"]:
        raise ValueError(f"Prediction changed on resume: {destination}")
    if old.sha256(destination / "anchor.npy") != record["anchor_sha256"]:
        raise ValueError(f"Anchor changed on resume: {destination}")
    return record


def evaluate_arm(arm: str, *, resume: bool = False) -> dict:
    if arm not in exp.ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    exp.configure_precision()
    cases = all_cases()
    fingerprints = {case["id"]: _fingerprint(case) for case in cases}
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
        if config.get("v3_compare", {}).get("kind") != arm:
            raise ValueError(f"Checkpoint belongs to another arm: {checkpoint_path}")
        step = int(checkpoint["completed_steps"])
        if role == "final" and step != 400:
            raise ValueError(f"Final checkpoint is step {step}, expected 400")
        if role == "best" and (step % 20 or not 20 <= step <= 400):
            raise ValueError(f"Best checkpoint is not one of the fixed validation steps: {step}")
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
                record = _resume_case(marker, expected_record, destination)
                prediction = np.load(destination / "reconstruction.npy", allow_pickle=False)
                row = record["metrics"]
                target = _targets(case, device)
                truth = target.get("ground_truth")
                truth_np = None if truth is None else truth[0, 0].cpu().numpy()
            else:
                item = input_item(case["path"], case["subset"], device)
                with torch.inference_mode():
                    output, beta0 = exp.forward(model, item, operator, config=config)
                    prediction_tensor = output.reconstruction
                    prediction = prediction_tensor[0, 0].float().cpu().numpy()
                    anchor_tensor = output._physical_anchor
                    anchor = anchor_tensor[0, 0].float().cpu().numpy()
                # Holdout statistics and GT are intentionally loaded only after inference.
                target = _targets(case, device)
                scoring = {**item, **target}
                with torch.inference_mode():
                    native = exp.loss(output, scoring, operator, variance_model, config)
                    common = _common_scores(
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
                    row.update(_structure_row(prediction, truth_np))
                else:
                    row.update(prediction_max=float(prediction.max()), prediction_sum=float(prediction.sum()))
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
                        raise ValueError("Same-input repeatability or target/GT isolation failed")
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
                _save_case(destination, prediction, anchor, record)
            tables["metrics"].append(row)
            if case["sample"] == "T03":
                for threshold in (0.05, 0.10, 0.20):
                    tables["t03_lines"].extend(t03_metrics(method, prediction, threshold))
            if case["sample"] == "T04":
                for threshold in (0.05, 0.10, 0.20):
                    tables["t04_axial"].extend(t04_metrics(method, prediction, threshold))
            if case["sample"] == "V03":
                for threshold in (0.05, 0.10, 0.20):
                    tables["v03_beads"].extend(v03_metrics(method, prediction, threshold))
            if case["split"] == "priority":
                if truth_np is None:
                    raise ValueError("Priority case unexpectedly lacks ground truth")
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
            print(json.dumps({"evaluated": method, "case": case["id"], "step": step}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    for name, rows in tables.items():
        csv_write(result_dir / f"{name}.csv", rows)
    record = {
        "complete": True,
        "experiment": arm,
        "cases_per_checkpoint": 94,
        "checkpoint_roles": ["best", "final"],
        "predictions": 188,
        "anchors": 188,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=exp.ARMS)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(evaluate_arm(arguments.experiment, resume=arguments.resume), indent=2))
