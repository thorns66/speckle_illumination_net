#!/usr/bin/env python3
"""Analysis and reporting for the frozen 2026-09-07 priority validation.

This module deliberately keeps structural measurements separate from display
normalization.  Detection thresholds use each complete prediction's maximum;
candidate loss scores always retain their physical scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib
import numpy as np
import torch
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from priority_validation_common import REPO_ROOT, json_dump, load_config, resolve_from_repo, scenes
from utils.reconstruction_metrics import reconstruction_metrics


METHODS = ("mean_rl3", "taylor_rl3", "anchor", "mean_rl5", "taylor_rl5", "network")
METHOD_LABEL = {
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3 (sqrt once)",
    "anchor": "initial anchor",
    "mean_rl5": "Mean-RL5",
    "taylor_rl5": "Taylor-RL5 (sqrt once)",
    "network": "frozen network, step 160",
}
Z_UM = np.arange(10.0, 101.0, 10.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or report the priority validation")
    parser.add_argument("--stage", choices=("loss", "report"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _mat_yx(handle: h5py.File, name: str) -> np.ndarray:
    value = np.asarray(handle[name][()])
    if value.ndim != 2:
        raise ValueError(f"{name}: expected MATLAB YX, got {value.shape}")
    return np.ascontiguousarray(value.T)


def _mat_yxz(handle: h5py.File, name: str) -> np.ndarray:
    value = np.asarray(handle[name][()])
    if value.ndim != 3:
        raise ValueError(f"{name}: expected MATLAB YXZ, got {value.shape}")
    return np.ascontiguousarray(value.transpose(0, 2, 1))


def _mat_yxzt(handle: h5py.File, name: str) -> np.ndarray:
    value = np.asarray(handle[name][()])
    if value.ndim != 4:
        raise ValueError(f"{name}: expected MATLAB YXZT, got {value.shape}")
    return np.ascontiguousarray(value.transpose(0, 1, 3, 2))


def _read_yx(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return _mat_yx(handle, name).astype(np.float32, copy=False)


def _read_yxz(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return _mat_yxz(handle, name).astype(np.float32, copy=False)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(first, np.float64) - second).ravel()) /
                 max(np.linalg.norm(np.asarray(second, np.float64).ravel()), 1e-12))


def _scale_aligned_nrmse(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    p = np.maximum(np.asarray(prediction, np.float64), 0)
    t = np.maximum(np.asarray(truth, np.float64), 0)
    gain = float(np.vdot(p.ravel(), t.ravel()).real / max(np.vdot(p.ravel(), p.ravel()).real, 1e-12))
    error = float(np.linalg.norm((gain * p - t).ravel()) / max(np.linalg.norm(t.ravel()), 1e-12))
    return gain, error


def axial_blur(volume: np.ndarray, passes: int = 1) -> np.ndarray:
    """Reflect-padded [0.25, 0.5, 0.25] blur, preserving total mass."""
    result = np.asarray(volume, np.float64).copy()
    original_mass = float(result.sum())
    for _ in range(int(passes)):
        padded = np.pad(result, ((1, 1), (0, 0), (0, 0)), mode="reflect")
        result = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        mass = float(result.sum())
        if mass > 0:
            result *= original_mass / mass
    return result.astype(np.float32)


def axial_shift(volume: np.ndarray, layers: int) -> np.ndarray:
    """Shift in Z without circular wrap; positive layers move to larger Z."""
    source = np.asarray(volume)
    result = np.zeros_like(source)
    if layers > 0:
        result[layers:] = source[:-layers]
    elif layers < 0:
        result[:layers] = source[-layers:]
    else:
        result[...] = source
    return result


def candidate_variants(truth: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "truth": np.asarray(truth, np.float32),
        "axial_blur_1": axial_blur(truth, 1),
        "axial_blur_2": axial_blur(truth, 2),
        "shift_up_1": axial_shift(truth, -1),
        "shift_down_1": axial_shift(truth, 1),
    }


def _torch_volume(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.float32)[None, None]


@torch.inference_mode()
def _score_volume(
    volume: np.ndarray,
    measured_mean: np.ndarray,
    measured_variance: np.ndarray,
    operator: torch.nn.Module,
    variance_model: torch.nn.Module,
    loss_config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    from losses.self_supervised_losses import compute_self_supervised_loss

    reconstruction = _torch_volume(volume, device)
    mean = torch.from_numpy(np.ascontiguousarray(measured_mean)).to(device)[None, None]
    variance = torch.from_numpy(np.ascontiguousarray(measured_variance)).to(device)[None, None]
    options = dict(loss_config)
    options.setdefault("lambda_var_band", 0.0)
    options.setdefault("var_band_levels", 2)
    breakdown = compute_self_supervised_loss(
        reconstruction, mean, variance, operator, variance_model,
        physics_use_checkpoint=False, **options,
    )
    return (
        breakdown.scalar_metrics(),
        breakdown.predicted_mean[0, 0].float().cpu().numpy(),
        breakdown.predicted_variance[0, 0].float().cpu().numpy(),
    )


def _load_operator(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    from losses.self_supervised_losses import TaylorH2VarianceModel
    from training.multivolume_trainer import _load_operator as make_operator
    from training.multivolume_trainer import _prepare_psf_cache

    synthetic_config = REPO_ROOT / "configs" / "checkpoint_config.yaml"
    cache = _prepare_psf_cache(config, synthetic_config, 0, 1)
    operator = make_operator(config, synthetic_config, device, selected_h_cache=cache)
    variance_model = TaylorH2VarianceModel(
        operator,
        alpha_noise=float(config["noise"]["alpha_noise"]),
        sigma_read=float(config["noise"]["sigma_read"]),
    )
    return operator, variance_model


def _forward_consistency(output: Path, operator: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    sample = output / "generated" / "points_z020_r01"
    truth = _read_yxz(sample / "prepared.mat", "ground_truth")
    with h5py.File(output / "shared_illumination" / "repeat_01.mat", "r") as handle:
        illumination = _mat_yxzt(handle, "illumination_raw")[0]
    matlab = _read_yx(sample / "sensor_frames" / "frame_001.mat", "sensor_pre_detector")
    with torch.inference_mode():
        predicted = operator(_torch_volume(truth * illumination, device))[0, 0].float().cpu().numpy()
    relative = _relative_l2(predicted, matlab)
    record = {
        "sample_id": sample.name,
        "frame": 1,
        "dtype": "FP32",
        "relative_l2": relative,
        "investigation_threshold": 1e-4,
        "passed": relative <= 1e-4,
    }
    json_dump(output / "analysis" / "python_matlab_forward_check.json", record)
    if not record["passed"]:
        raise RuntimeError(f"Python/MATLAB forward relative L2 {relative:.3g} exceeds 1e-4")
    return record


def _load_calibration(output: Path) -> tuple[np.ndarray, np.ndarray]:
    path = output / "calibration" / "illumination_moments_1024.mat"
    with h5py.File(path, "r") as handle:
        count = int(np.asarray(handle["frame_count"][()]).item())
        mean = _mat_yxz(handle, "illumination_mean").astype(np.float32)
        variance = _mat_yxz(handle, "illumination_variance_nminus1").astype(np.float32)
    if count != 1024 or mean.shape != (10, 260, 260) or variance.shape != mean.shape:
        raise ValueError("Invalid 1024-frame illumination calibration")
    return mean, variance


def _target_statistics(output: Path, sample: str, repeat: int) -> tuple[np.ndarray, np.ndarray]:
    path = output / "loss_targets" / sample / f"repeat_{repeat:02d}" / "holdout_statistics.mat"
    with h5py.File(path, "r") as handle:
        return _mat_yx(handle, "holdout_mean").astype(np.float32), _mat_yx(
            handle, "holdout_variance_nminus1"
        ).astype(np.float32)


def _load_existing_candidates(output: Path, sample: str, subset: int, operator: torch.nn.Module,
                              device: torch.device) -> dict[str, np.ndarray]:
    from datasets.matlab_multivolume_dataset import load_inference_input
    from training.multivolume_trainer import _analytic_beta0

    sample_dir = REPO_ROOT / "data" / "matlab_cells_pilot_v2_r04" / sample
    subset_path = sample_dir / "subsets" / f"subset_{subset:02d}.mat"
    with h5py.File(subset_path, "r") as handle:
        mean = _mat_yxz(handle, "physics_mean_raw").astype(np.float32)
        taylor = _mat_yxz(handle, "physics_taylor_sqrt_float").astype(np.float32)
    raw = load_inference_input(sample_dir, subset, var_feature_representation="sqrt")
    f_var = torch.from_numpy(raw["f_var"]).to(device=device, dtype=torch.float32)[None]
    input_mean = torch.from_numpy(raw["input_mean"]).to(device=device, dtype=torch.float32)[None]
    with torch.inference_mode():
        beta0 = float(_analytic_beta0(operator, f_var, input_mean).item())
    network = np.load(output / "existing_inference" / f"{sample}_subset_{subset:02d}" /
                      "reconstruction.npy", allow_pickle=False).astype(np.float32)
    return {"mean_rl3": mean, "taylor_rl3": taylor, "anchor": beta0 * taylor, "network": network}


def _score_structure_metrics(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    metrics = reconstruction_metrics(candidate, truth, Z_UM)
    local_mask = truth.max(axis=0) > 0.1 * max(float(truth.max()), 1e-12)
    pred_local = np.maximum(candidate[:, local_mask], 0).sum(axis=1)
    truth_local = np.maximum(truth[:, local_mask], 0).sum(axis=1)
    pred_fraction = pred_local / max(float(pred_local.sum()), 1e-12)
    truth_fraction = truth_local / max(float(truth_local.sum()), 1e-12)
    metrics["local_axial_w1_um"] = float(np.abs(np.cumsum(pred_fraction) -
                                                   np.cumsum(truth_fraction)).sum() * 10.0)
    metrics["mass_ratio_to_truth"] = float(np.maximum(candidate, 0).sum() /
                                            max(float(np.maximum(truth, 0).sum()), 1e-12))
    return metrics


def _loss_stage(config: dict[str, Any], output: Path) -> None:
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(resolve_from_repo(config["experiment"]["checkpoint"]),
                            map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    if int(checkpoint["best_step"]) != int(config["experiment"]["expected_best_step"]):
        raise ValueError("Frozen checkpoint best_step changed")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    operator, variance_model = _load_operator(train_config, device)
    forward_check = _forward_consistency(output, operator, device)
    illum_mean, illum_variance = _load_calibration(output)

    score_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    data_root = REPO_ROOT / "data" / "matlab_cells_pilot_v2_r04"
    for sample in config["calibration"]["diagnostic_objects"]:
        truth = _read_yxz(data_root / sample / "prepared.mat", "ground_truth")
        mean_units = truth * illum_mean
        variance_units = truth * np.sqrt(np.maximum(illum_variance, 0))
        unit_gain, unit_error = _scale_aligned_nrmse(variance_units, mean_units)
        unit_rows.append({
            "sample_id": sample,
            "mean_unit_mass": float(mean_units.sum()),
            "variance_unit_mass": float(variance_units.sum()),
            "gain_variance_to_mean_units": unit_gain,
            "aligned_nrmse_between_units": unit_error,
        })
        for name, density in candidate_variants(truth).items():
            volume = density * np.sqrt(np.maximum(illum_variance, 0))
            structure = _score_structure_metrics(density, truth)
            for repeat in range(1, 4):
                measured_mean, measured_variance = _target_statistics(output, sample, repeat)
                scores, predicted_mean, predicted_variance = _score_volume(
                    volume, measured_mean, measured_variance, operator, variance_model,
                    train_config["loss"], device,
                )
                score_rows.append({
                    "sample_id": sample, "candidate_group": "controlled_gt",
                    "candidate": name, "input_subset": "", "target_repeat": repeat,
                    "unit_rule": "density_times_sqrt_1024_frame_illumination_variance",
                    **scores, **structure,
                })
                if name == "truth":
                    for gain in config["calibration"]["gain_scan"]:
                        scaled_mean = predicted_mean * float(gain)
                        scaled_variance = predicted_variance * float(gain) ** 2
                        mean_scale = max(float(np.mean(np.abs(measured_mean))), 1e-8)
                        mean_norm = torch.nn.functional.smooth_l1_loss(
                            torch.from_numpy(scaled_mean / mean_scale),
                            torch.from_numpy(measured_mean / mean_scale),
                        ).item()
                        var_norm = torch.nn.functional.smooth_l1_loss(
                            torch.log(torch.from_numpy(np.maximum(scaled_variance, 0)) +
                                      float(train_config["loss"]["var_log_eps"])),
                            torch.log(torch.from_numpy(measured_variance) +
                                      float(train_config["loss"]["var_log_eps"])),
                        ).item()
                        gain_rows.append({"sample_id": sample, "target_repeat": repeat,
                                          "gain": float(gain), "normalized_mean_loss": mean_norm,
                                          "normalized_variance_loss": var_norm})

        for subset in range(1, 11):
            candidates = _load_existing_candidates(output, sample, subset, operator, device)
            for method, volume in candidates.items():
                structure = _score_structure_metrics(volume, truth)
                for repeat in range(1, 4):
                    measured_mean, measured_variance = _target_statistics(output, sample, repeat)
                    scores, _, _ = _score_volume(volume, measured_mean, measured_variance,
                                                  operator, variance_model, train_config["loss"], device)
                    score_rows.append({
                        "sample_id": sample, "candidate_group": "existing_ten_frame",
                        "candidate": method, "input_subset": subset, "target_repeat": repeat,
                        "unit_rule": "retained_existing_scale_no_refit", **scores, **structure,
                    })

    _write_csv(analysis / "loss_candidate_scores.csv", score_rows)
    _write_csv(analysis / "gain_scan.csv", gain_rows)
    _write_csv(analysis / "brightness_unit_check.csv", unit_rows)
    _write_checkpoint_trajectory(config, analysis)
    _write_cross_layer_covariance(output, analysis)
    summary = _summarize_loss(score_rows, gain_rows)
    summary["forward_consistency"] = forward_check
    summary["brightness_units"] = unit_rows
    json_dump(analysis / "loss_summary.json", summary)


def _write_checkpoint_trajectory(config: dict[str, Any], analysis: Path) -> None:
    source = REPO_ROOT / "outputs" / "multivolume_n10_no_mean_run01" / "validation_metrics.csv"
    desired = {int(value) for value in config["evaluation"]["validation_steps"]}
    rows = [row for row in _read_csv(source) if int(row["step"]) in desired]
    output_rows: list[dict[str, Any]] = []
    numeric = ("selection_score", "gt_scale_aligned_nrmse", "gt_axial_w1_um",
               "gt_support_outside_pm10_mass", "gt_xy_mip_ssim", "raw_mean_loss",
               "normalized_mean_loss", "normalized_var_loss", "tv_loss")
    for (step, sample), group in _group_rows(rows, lambda row: (int(row["step"]), row["sample_id"])):
        output_rows.append({"step": step, "sample_id": sample, "items": len(group),
                            **{name: float(np.mean([float(row[name]) for row in group]))
                               for name in numeric}})
    _write_csv(analysis / "validation_checkpoint_by_object.csv", output_rows)


def _write_cross_layer_covariance(output: Path, analysis: Path) -> None:
    rows = []
    for repeat in range(1, 4):
        path = output / "generated" / f"axial_pairs_r{repeat:02d}" / "subsets" / "subset_01.mat"
        with h5py.File(path, "r") as handle:
            total = _mat_yx(handle, "holdout_physics_variance_nminus1_float")
            layers = _mat_yxz(handle, "holdout_layer_sensor_variance_nminus1")
        independent = layers.sum(axis=0)
        cross = total - independent
        rows.append({
            "repeat": repeat,
            "relative_l2_total_minus_layer_sum": _relative_l2(independent, total),
            "integrated_cross_fraction": float(cross.sum() / max(float(total.sum()), 1e-12)),
            "absolute_cross_fraction": float(np.abs(cross).sum() / max(float(total.sum()), 1e-12)),
            "positive_cross_fraction": float(np.maximum(cross, 0).sum() / max(float(total.sum()), 1e-12)),
            "negative_cross_fraction": float(np.maximum(-cross, 0).sum() / max(float(total.sum()), 1e-12)),
        })
    _write_csv(analysis / "axial_pair_cross_layer_covariance.csv", rows)


def _summarize_loss(rows: Sequence[dict[str, Any]], gains: Sequence[dict[str, Any]]) -> dict[str, Any]:
    controlled = [row for row in rows if row["candidate_group"] == "controlled_gt"]
    ranking: dict[str, Any] = {}
    for sample in sorted({row["sample_id"] for row in controlled}):
        by_candidate = []
        for candidate in sorted({row["candidate"] for row in controlled if row["sample_id"] == sample}):
            selected = [row for row in controlled if row["sample_id"] == sample and row["candidate"] == candidate]
            by_candidate.append({"candidate": candidate,
                                 "total_loss": float(np.mean([row["total_loss"] for row in selected])),
                                 "normalized_variance_loss": float(np.mean([
                                     row["normalized_var_loss"] for row in selected])),
                                 "local_axial_w1_um": float(np.mean([
                                     row["local_axial_w1_um"] for row in selected]))})
        ranking[sample] = sorted(by_candidate, key=lambda row: row["total_loss"])
    optimum: dict[str, Any] = {}
    for sample in sorted({row["sample_id"] for row in gains}):
        selected = [row for row in gains if row["sample_id"] == sample]
        grouped = []
        for gain in sorted({row["gain"] for row in selected}):
            group = [row for row in selected if row["gain"] == gain]
            grouped.append((gain, np.mean([row["normalized_mean_loss"] for row in group]),
                            np.mean([row["normalized_variance_loss"] for row in group])))
        optimum[sample] = {
            "mean_best_gain": float(min(grouped, key=lambda item: item[1])[0]),
            "variance_best_gain": float(min(grouped, key=lambda item: item[2])[0]),
        }
    return {"controlled_candidate_ranking": ranking, "gain_optima": optimum}


def _group_rows(rows: Iterable[Any], key: Any) -> Iterable[tuple[Any, list[Any]]]:
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return sorted(grouped.items(), key=lambda item: item[0])


def point_targets(config: dict[str, Any], depth_um: float) -> list[dict[str, Any]]:
    cells = [(x, y) for y in config["scenes"]["point_grid_y_um"]
             for x in config["scenes"]["point_grid_x_um"]]
    targets = []
    for cell in range(4):
        targets.append({"cell_id": cell + 1, "target_id": "single", "x_um": cells[cell][0],
                        "y_um": cells[cell][1], "z_um": depth_um,
                        "amplitude": 1.0 - 0.5 * ((cell + 1) % 2), "kind": "single"})
    combinations = [(sep, axis, ratio) for sep in config["scenes"]["lateral_pair_separations_um"]
                    for axis in config["scenes"]["lateral_pair_axes"]
                    for ratio in config["scenes"]["pair_intensity_ratios"]]
    for offset, (separation, axis, ratio) in enumerate(combinations, start=5):
        cx, cy = cells[offset - 1]
        direction = (1.0, 0.0) if axis == "x" else (0.0, 1.0)
        for role, sign, amplitude in (("a", -1, 1.0), ("b", 1, ratio)):
            targets.append({"cell_id": offset, "target_id": role,
                            "x_um": cx + sign * direction[0] * separation / 2,
                            "y_um": cy + sign * direction[1] * separation / 2,
                            "z_um": depth_um, "amplitude": amplitude, "kind": "lateral_pair",
                            "separation_um": separation, "axis": axis, "ratio": ratio})
    return targets


def axial_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [(x, y) for y in config["scenes"]["axial_grid_y_um"]
             for x in config["scenes"]["axial_grid_x_um"]]
    targets = []
    cell = 0
    for deep in config["scenes"]["axial_deep_um"]:
        for ratio in config["scenes"]["pair_intensity_ratios"]:
            cx, cy = cells[cell]; cell += 1
            targets.extend([
                {"cell_id": cell, "target_id": "a", "x_um": cx, "y_um": cy, "z_um": 30,
                 "amplitude": 1.0, "kind": "axial_pair", "separation_um": deep - 30, "ratio": ratio},
                {"cell_id": cell, "target_id": "b", "x_um": cx, "y_um": cy, "z_um": deep,
                 "amplitude": ratio, "kind": "axial_pair", "separation_um": deep - 30, "ratio": ratio},
            ])
    for depth in (30, 50, 60, 70):
        cx, cy = cells[cell]; cell += 1
        targets.append({"cell_id": cell, "target_id": "single", "x_um": cx, "y_um": cy,
                        "z_um": depth, "amplitude": 1.0, "kind": "single_control"})
    return targets


def _to_index(target: dict[str, Any], pitch: float) -> np.ndarray:
    return np.asarray([(float(target["z_um"]) - 10.0) / 10.0,
                       float(target["y_um"]) / pitch, float(target["x_um"]) / pitch])


def find_peaks(volume: np.ndarray, threshold_fraction: float) -> np.ndarray:
    value = np.maximum(np.asarray(volume, np.float32), 0)
    threshold = float(value.max()) * float(threshold_fraction)
    if threshold <= 0:
        return np.empty((0, 4), dtype=np.float64)
    maxima = maximum_filter(value, size=(3, 3, 3), mode="nearest")
    locations = np.argwhere((value == maxima) & (value >= threshold))
    if len(locations) == 0:
        return np.empty((0, 4), dtype=np.float64)
    strengths = value[tuple(locations.T)]
    order = np.argsort(strengths)[::-1]
    return np.column_stack((locations[order], strengths[order])).astype(np.float64)


def match_points(targets: Sequence[dict[str, Any]], peaks: np.ndarray, pitch: float,
                 xy_limit: float, z_limit: float, roi_half: float = 18.0) -> tuple[list[dict[str, Any]], set[int]]:
    target_indices = np.asarray([_to_index(target, pitch) for target in targets])
    if len(peaks) == 0:
        return [{"matched": False} for _ in targets], set()
    relevant = np.asarray([
        any(abs(peak[1] - target[1]) <= roi_half and abs(peak[2] - target[2]) <= roi_half
            for target in target_indices) for peak in peaks
    ])
    candidate_ids = np.flatnonzero(relevant)
    candidates = peaks[candidate_ids, :3]
    if len(candidates) == 0:
        return [{"matched": False} for _ in targets], set()
    cost = np.full((len(targets), len(candidates)), 1e6, dtype=np.float64)
    for i, target in enumerate(target_indices):
        dz = np.abs(candidates[:, 0] - target[0])
        dxy = np.linalg.norm(candidates[:, 1:3] - target[1:3], axis=1)
        allowed = (dz <= z_limit) & (dxy <= xy_limit)
        cost[i, allowed] = dxy[allowed] + 0.25 * dz[allowed]
    row_ids, col_ids = linear_sum_assignment(cost)
    matches = [{"matched": False} for _ in targets]
    used: set[int] = set()
    for row, col in zip(row_ids, col_ids):
        if cost[row, col] >= 1e5:
            continue
        peak_id = int(candidate_ids[col]); peak = peaks[peak_id]
        target = target_indices[row]
        matches[row] = {"matched": True, "peak_id": peak_id, "pred_z_layer": int(peak[0]),
                        "pred_y_pixel": float(peak[1]), "pred_x_pixel": float(peak[2]),
                        "peak_value": float(peak[3]),
                        "xy_error_pixels": float(np.linalg.norm(peak[1:3] - target[1:3])),
                        "z_error_layers": float(abs(peak[0] - target[0]))}
        used.add(peak_id)
    return matches, used


def _local_depth(volume: np.ndarray, target: dict[str, Any], pitch: float, radius: int = 3) -> dict[str, float]:
    index = _to_index(target, pitch); y, x = int(round(index[1])), int(round(index[2]))
    y0, y1 = max(0, y - radius), min(volume.shape[1], y + radius + 1)
    x0, x1 = max(0, x - radius), min(volume.shape[2], x + radius + 1)
    profile = np.maximum(volume[:, y0:y1, x0:x1], 0).sum(axis=(1, 2)).astype(np.float64)
    fraction = profile / max(float(profile.sum()), 1e-12)
    target_layer = int(round(index[0]))
    delta = np.zeros_like(fraction); delta[target_layer] = 1
    return {"local_depth_w1_um": float(np.abs(np.cumsum(fraction) - np.cumsum(delta)).sum() * 10),
            "tail_mass_outside_pm1": float(fraction[np.abs(np.arange(10) - target_layer) > 1].sum()),
            "profile_peak_z_um": float(Z_UM[int(np.argmax(profile))]) if profile.sum() > 0 else math.nan,
            "local_profile_mass": float(profile.sum())}


def _fwhm(profile: np.ndarray, center: int, pitch: float) -> float:
    values = np.maximum(np.asarray(profile, np.float64), 0)
    if not 0 <= center < len(values) or values[center] <= 0:
        return math.nan
    half = values[center] / 2
    left = center
    while left > 0 and values[left - 1] >= half:
        left -= 1
    right = center
    while right + 1 < len(values) and values[right + 1] >= half:
        right += 1
    return float((right - left + 1) * pitch)


def _pair_valley(volume: np.ndarray, first: dict[str, Any], second: dict[str, Any]) -> float:
    if not first.get("matched") or not second.get("matched"):
        return math.nan
    a = np.asarray([first["pred_z_layer"], first["pred_y_pixel"], first["pred_x_pixel"]])
    b = np.asarray([second["pred_z_layer"], second["pred_y_pixel"], second["pred_x_pixel"]])
    coordinates = a[:, None] + (b - a)[:, None] * np.linspace(0.1, 0.9, 41)[None]
    valley = float(map_coordinates(volume, coordinates, order=1, mode="nearest").min())
    return valley / max(min(float(first["peak_value"]), float(second["peak_value"])), 1e-12)


def _load_methods(sample_dir: Path) -> dict[str, np.ndarray]:
    values = {}
    for name in ("mean_rl3", "taylor_rl3", "mean_rl5", "taylor_rl5"):
        values[name] = _read_yxz(sample_dir / "baselines" / name / "reconstruction.mat", "reconstruction")
    network_dir = sample_dir / "network"
    values["network"] = np.load(network_dir / "reconstruction.npy", allow_pickle=False).astype(np.float32)
    f_var = np.load(network_dir / "input_f_var.npy", allow_pickle=False).astype(np.float32)
    contract = json.loads((network_dir / "inference_contract.json").read_text(encoding="utf-8"))
    values["anchor"] = float(contract["beta0"]) * f_var
    for name, value in values.items():
        if value.shape != (10, 260, 260) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"Invalid {name} prediction in {sample_dir}")
    return values


def _point_metrics(config: dict[str, Any], scene_id: str, repeat: int, truth: np.ndarray,
                   predictions: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                                list[dict[str, Any]]]:
    depth = int(scene_id.split("z")[-1])
    targets = point_targets(config, depth)
    pitch = float(config["acquisition"]["object_pixel_pitch_um"])
    point_rows, pair_rows, profiles = [], [], []
    for method, volume in predictions.items():
        for threshold in config["evaluation"]["thresholds"]:
            peaks = find_peaks(volume, threshold)
            matches, used = match_points(targets, peaks, pitch,
                                         float(config["evaluation"]["xy_match_pixels"]),
                                         float(config["evaluation"]["z_match_layers"]))
            for target, match in zip(targets, matches):
                local = _local_depth(volume, target, pitch)
                row = {"scene_id": scene_id, "repeat": repeat, "method": method,
                       "threshold": threshold, **target, **match, **local}
                if match.get("matched") and target["kind"] == "single":
                    z, y, x = int(match["pred_z_layer"]), int(round(match["pred_y_pixel"])), int(round(match["pred_x_pixel"]))
                    row["fwhm_x_um"] = _fwhm(volume[z, y], x, pitch)
                    row["fwhm_y_um"] = _fwhm(volume[z, :, x], y, pitch)
                else:
                    row["fwhm_x_um"] = math.nan; row["fwhm_y_um"] = math.nan
                point_rows.append(row)
            cells = sorted({target["cell_id"] for target in targets if target["kind"] == "lateral_pair"})
            for cell in cells:
                selected = [(target, match) for target, match in zip(targets, matches) if target["cell_id"] == cell]
                valley = _pair_valley(volume, selected[0][1], selected[1][1])
                success = all(match.get("matched", False) for _, match in selected) and valley <= float(
                    config["evaluation"]["pair_valley_max_fraction"])
                pair_rows.append({"scene_id": scene_id, "repeat": repeat, "method": method,
                                  "threshold": threshold, "cell_id": cell,
                                  "separation_um": selected[0][0]["separation_um"],
                                  "axis": selected[0][0]["axis"], "ratio": selected[0][0]["ratio"],
                                  "both_localized": all(match.get("matched", False) for _, match in selected),
                                  "valley_to_weaker_peak": valley, "separated": success})
            # False peaks are counted only inside the twenty predefined cells.
            target_indices = [_to_index(target, pitch) for target in targets]
            relevant = {index for index, peak in enumerate(peaks) if any(
                abs(peak[1] - target[1]) <= 18 and abs(peak[2] - target[2]) <= 18
                for target in target_indices)}
            for row in point_rows[-len(targets):]:
                row["unmatched_peaks_in_test_cells"] = len(relevant - used)
        primary = float(config["evaluation"]["primary_threshold"])
        for target in targets:
            index = _to_index(target, pitch); y, x = int(round(index[1])), int(round(index[2]))
            for z_index, z_um in enumerate(Z_UM):
                profiles.append({"scene_id": scene_id, "repeat": repeat, "method": method,
                                 "cell_id": target["cell_id"], "target_id": target["target_id"],
                                 "truth_z_um": target["z_um"], "z_um": z_um,
                                 "value": float(np.maximum(volume[z_index, max(0,y-2):y+3,
                                                                      max(0,x-2):x+3], 0).sum()),
                                 "threshold": primary})
    return point_rows, pair_rows, profiles


def _axial_metrics(config: dict[str, Any], repeat: int, predictions: dict[str, np.ndarray]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    targets = axial_targets(config)
    pitch = float(config["acquisition"]["object_pixel_pitch_um"])
    rows, pair_rows = [], []
    for method, volume in predictions.items():
        for threshold in config["evaluation"]["thresholds"]:
            peaks = find_peaks(volume, threshold)
            matches, used = match_points(targets, peaks, pitch,
                                         float(config["evaluation"]["xy_match_pixels"]),
                                         float(config["evaluation"]["z_match_layers"]), roi_half=24)
            for target, match in zip(targets, matches):
                rows.append({"scene_id": "axial_pairs", "repeat": repeat, "method": method,
                             "threshold": threshold, **target, **match,
                             **_local_depth(volume, target, pitch)})
            for cell in range(1, 7):
                selected = [(target, match) for target, match in zip(targets, matches)
                            if target["cell_id"] == cell]
                valley = _pair_valley(volume, selected[0][1], selected[1][1])
                pair_rows.append({"scene_id": "axial_pairs", "repeat": repeat, "method": method,
                                  "threshold": threshold, "cell_id": cell,
                                  "separation_um": selected[0][0]["separation_um"],
                                  "ratio": selected[0][0]["ratio"],
                                  "both_localized": all(match.get("matched", False) for _, match in selected),
                                  "valley_to_weaker_peak": valley,
                                  "separated": all(match.get("matched", False) for _, match in selected) and
                                               valley <= float(config["evaluation"]["pair_valley_max_fraction"]),
                                  "unmatched_peaks_in_test_cells": len(set(range(len(peaks))) - used)})
    return rows, pair_rows


def line_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [(x, y) for y in config["scenes"]["line_grid_y_um"]
             for x in config["scenes"]["line_grid_x_um"]]
    combinations = [(gap, angle, amplitude) for gap in config["scenes"]["line_gaps_um"]
                    for angle in config["scenes"]["line_angles_deg"]
                    for amplitude in config["scenes"]["line_amplitudes"]]
    return [{"cell_id": index + 1, "x_um": cell[0], "y_um": cell[1], "gap_um": values[0],
             "angle_deg": values[1], "amplitude": values[2]}
            for index, (cell, values) in enumerate(zip(cells, combinations))]


def _longest_false_run(mask: np.ndarray, spacing: float) -> float:
    best = current = 0
    for false_value in ~mask:
        current = current + 1 if false_value else 0
        best = max(best, current)
    return float(best * spacing)


def _line_metrics(config: dict[str, Any], scene_id: str, repeat: int,
                  predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    depth = int(scene_id.split("z")[-1]); z_index = int(depth / 10 - 1)
    pitch = float(config["acquisition"]["object_pixel_pitch_um"])
    sample_um = np.linspace(-16, 16, 129); spacing = float(sample_um[1] - sample_um[0])
    rows = []
    for method, volume in predictions.items():
        global_max = max(float(volume.max()), 1e-12)
        for threshold_fraction in config["evaluation"]["thresholds"]:
            threshold = threshold_fraction * global_max
            for line in line_definitions(config):
                theta = math.radians(float(line["angle_deg"]))
                x = (float(line["x_um"]) + np.cos(theta) * sample_um) / pitch
                y = (float(line["y_um"]) + np.sin(theta) * sample_um) / pitch
                layer_values = []
                for layer in range(max(0, z_index - 1), min(10, z_index + 2)):
                    coords = np.vstack((np.full_like(x, layer), y, x))
                    layer_values.append(map_coordinates(volume, coords, order=1, mode="constant", cval=0))
                profile = np.max(np.asarray(layer_values), axis=0)
                detected = profile >= threshold
                gap = float(line["gap_um"])
                end_mask = np.abs(sample_um) >= max(gap / 2 + 1, 4)
                end_level = float(np.median(profile[end_mask])) if np.any(end_mask) else 0.0
                center_mask = np.abs(sample_um) <= max(gap / 2, spacing)
                center_level = float(np.mean(profile[center_mask]))
                endpoint = (np.abs(sample_um) >= 13) & (np.abs(sample_um) <= 16)
                rows.append({
                    "scene_id": scene_id, "repeat": repeat, "method": method,
                    "threshold": threshold_fraction, **line,
                    "coverage_fraction": float(detected.mean()),
                    "longest_below_threshold_run_um": _longest_false_run(detected, spacing),
                    "gap_center_to_segment_ratio": center_level / max(end_level, 1e-12),
                    "bridged": bool(gap > 0 and center_level / max(end_level, 1e-12) >
                                    float(config["evaluation"]["pair_valley_max_fraction"])),
                    "endpoint_retention_fraction": float(detected[endpoint].mean()),
                    "false_gap": bool(gap == 0 and np.any(~detected[np.abs(sample_um) <= 14])),
                })
    return rows


def _verification(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for scene in scenes(config):
        for repeat in range(1, 4):
            sample = output / "generated" / f"{scene.scene_id}_r{repeat:02d}"
            sum_input = np.zeros((260, 260), np.float64); sum_input2 = sum_input.copy()
            sum_holdout = np.zeros((260, 260), np.float64); sum_holdout2 = sum_holdout.copy()
            for frame in range(1, 101):
                value = _read_yx(sample / "sensor_frames" / f"frame_{frame:03d}.mat",
                                 "sensor_pre_detector").astype(np.float64)
                if frame <= 10:
                    sum_input += value; sum_input2 += value * value
                else:
                    sum_holdout += value; sum_holdout2 += value * value
            input_mean = sum_input / 10
            input_var = (sum_input2 - 10 * input_mean * input_mean) / 9
            holdout_mean = sum_holdout / 90
            holdout_var = (sum_holdout2 - 90 * holdout_mean * holdout_mean) / 89
            subset = sample / "subsets" / "subset_01.mat"
            with h5py.File(subset, "r") as handle:
                indices_input = np.asarray(handle["input_indices"][()]).reshape(-1).astype(int)
                indices_holdout = np.asarray(handle["holdout_indices"][()]).reshape(-1).astype(int)
                saved_input_mean = _mat_yx(handle, "input_physics_mean_float")
                saved_input_var = _mat_yx(handle, "input_physics_variance_nminus1_float")
                saved_holdout_mean = _mat_yx(handle, "holdout_physics_mean_float")
                saved_holdout_var = _mat_yx(handle, "holdout_physics_variance_nminus1_float")
                raw_taylor = _mat_yxz(handle, "physics_taylor_raw")
                sqrt_taylor = _mat_yxz(handle, "physics_taylor_sqrt_float")
            row = {
                "sample_id": sample.name,
                "partition_disjoint": not bool(set(indices_input) & set(indices_holdout)),
                "partition_complete": set(indices_input) | set(indices_holdout) == set(range(1, 101)),
                "input_mean_relative_l2": _relative_l2(saved_input_mean, input_mean),
                "input_variance_relative_l2": _relative_l2(saved_input_var, input_var),
                "holdout_mean_relative_l2": _relative_l2(saved_holdout_mean, holdout_mean),
                "holdout_variance_relative_l2": _relative_l2(saved_holdout_var, holdout_var),
                "taylor_sqrt_relative_l2": _relative_l2(sqrt_taylor, np.sqrt(np.maximum(raw_taylor, 0))),
            }
            row["passed"] = bool(row["partition_disjoint"] and row["partition_complete"] and
                                 max(row[name] for name in row if name.endswith("relative_l2")) <= 2e-6)
            rows.append(row)
    _write_csv(output / "analysis" / "data_integrity_per_scene.csv", rows)

    # Reproduce one validation item from the saved step-160 metrics.
    sample = "P09"; subset = 1
    prediction = np.load(output / "existing_inference" / f"{sample}_subset_{subset:02d}" /
                         "reconstruction.npy", allow_pickle=False)
    truth = _read_yxz(REPO_ROOT / "data" / "matlab_cells_pilot_v2_r04" / sample / "prepared.mat",
                      "ground_truth")
    current = reconstruction_metrics(prediction, truth, Z_UM)
    historical = next(row for row in _read_csv(REPO_ROOT / "outputs" /
                      "multivolume_n10_no_mean_run01" / "validation_metrics.csv")
                      if int(row["step"]) == 160 and row["sample_id"] == sample and
                      int(row["subset_index"]) == subset)
    comparison = {name: {"current": current[name], "historical": float(historical[name]),
                         "absolute_error": abs(current[name] - float(historical[name]))}
                  for name in ("gt_scale_aligned_nrmse", "gt_axial_w1_um",
                               "gt_support_outside_pm10_mass", "gt_xy_mip_ssim")}
    reproduced = max(item["absolute_error"] for item in comparison.values()) <= 1e-5
    result = {"all_scene_integrity_passed": all(row["passed"] for row in rows),
              "existing_P09_subset01_step160_reproduced": reproduced,
              "reproduction_metrics": comparison,
              "inference_did_not_read_gt_or_target": all(json.loads(path.read_text(encoding="utf-8"))[
                  "target_or_ground_truth_read"] is False for path in output.glob("**/network/inference_contract.json")),
              "taylor_sqrt_applied_exactly_once": max(row["taylor_sqrt_relative_l2"] for row in rows) <= 2e-6}
    json_dump(output / "analysis" / "verification.json", result)
    if not all((result["all_scene_integrity_passed"], reproduced,
                result["inference_did_not_read_gt_or_target"], result["taylor_sqrt_applied_exactly_once"])):
        raise RuntimeError("A mandatory data or inference reproduction check failed")
    return result


def _report_stage(config: dict[str, Any], output: Path) -> None:
    analysis = output / "analysis"; figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    required_loss = analysis / "loss_summary.json"
    if not required_loss.is_file():
        raise FileNotFoundError("Run --stage loss before --stage report")
    verification = _verification(config, output)
    volume_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    axial_rows: list[dict[str, Any]] = []
    for scene in scenes(config):
        for repeat in range(1, 4):
            sample = output / "generated" / f"{scene.scene_id}_r{repeat:02d}"
            truth = _read_yxz(sample / "prepared.mat", "ground_truth")
            predictions = _load_methods(sample)
            for method, prediction in predictions.items():
                volume_rows.append({"scene_id": scene.scene_id, "family": scene.family,
                                    "truth_depth_um": scene.depth_um, "repeat": repeat,
                                    "method": method, **reconstruction_metrics(prediction, truth, Z_UM)})
            if scene.family == "points":
                points, pairs, profiles = _point_metrics(config, scene.scene_id, repeat, truth, predictions)
                point_rows.extend(points); pair_rows.extend(pairs); profile_rows.extend(profiles)
            elif scene.family == "lines":
                line_rows.extend(_line_metrics(config, scene.scene_id, repeat, predictions))
            else:
                points, pairs = _axial_metrics(config, repeat, predictions)
                axial_rows.extend(points); pair_rows.extend(pairs)
    _write_csv(analysis / "volume_metrics.csv", volume_rows)
    _write_csv(analysis / "point_target_metrics.csv", point_rows)
    _write_csv(analysis / "point_pair_metrics.csv", pair_rows)
    _write_csv(analysis / "line_metrics.csv", line_rows)
    _write_csv(analysis / "fixed_position_depth_profiles.csv", profile_rows)
    _write_csv(analysis / "axial_point_metrics.csv", axial_rows)
    summary = _summarize_structure(config, point_rows, pair_rows, line_rows, axial_rows)
    summary["zero_control"] = _zero_control(output)
    summary["verification"] = verification
    json_dump(analysis / "structure_summary.json", summary)
    _make_figures(config, analysis, point_rows, pair_rows, line_rows)
    _write_chinese_report(config, output, summary,
                          json.loads(required_loss.read_text(encoding="utf-8")))
    json_dump(analysis / "report_contract.json", {
        "complete": True, "checkpoint_step": 160, "scene_repeats": 33,
        "methods": list(METHODS), "thresholds": config["evaluation"]["thresholds"],
        "per_repeat_results_retained": True, "display_per_layer_rescaling": False,
    })


def _summarize_structure(config: dict[str, Any], points: Sequence[dict[str, Any]],
                         pairs: Sequence[dict[str, Any]], lines: Sequence[dict[str, Any]],
                         axial: Sequence[dict[str, Any]]) -> dict[str, Any]:
    primary = float(config["evaluation"]["primary_threshold"])
    result: dict[str, Any] = {"primary_threshold": primary, "methods": {}}
    for method in METHODS:
        p = [row for row in points if row["method"] == method and float(row["threshold"]) == primary]
        singles = [row for row in p if row["kind"] == "single"]
        lateral = [row for row in pairs if row["method"] == method and
                   float(row["threshold"]) == primary and row["scene_id"].startswith("points")]
        axial_pair = [row for row in pairs if row["method"] == method and
                      float(row["threshold"]) == primary and row["scene_id"] == "axial_pairs"]
        continuous = [row for row in lines if row["method"] == method and
                      float(row["threshold"]) == primary and float(row["gap_um"]) == 0]
        broken = [row for row in lines if row["method"] == method and
                  float(row["threshold"]) == primary and float(row["gap_um"]) > 0]
        local = [row for row in p + list(axial) if row["method"] == method and
                 float(row["threshold"]) == primary]
        result["methods"][method] = {
            "single_localization_rate": float(np.mean([bool(row["matched"]) for row in singles])),
            "single_z_mae_um_detected": float(np.mean([row["z_error_layers"] * 10 for row in singles
                                                        if row.get("matched")])) if any(
                                                            row.get("matched") for row in singles) else math.nan,
            "local_depth_w1_um": float(np.mean([row["local_depth_w1_um"] for row in local])),
            "tail_mass_outside_pm1": float(np.mean([row["tail_mass_outside_pm1"] for row in local])),
            "lateral_pair_separation_rate": float(np.mean([bool(row["separated"]) for row in lateral])),
            "axial_pair_separation_rate": float(np.mean([bool(row["separated"]) for row in axial_pair])),
            "continuous_line_false_gap_rate": float(np.mean([bool(row["false_gap"]) for row in continuous])),
            "broken_line_bridge_rate": float(np.mean([bool(row["bridged"]) for row in broken])),
            "broken_line_endpoint_retention": float(np.mean([row["endpoint_retention_fraction"] for row in broken])),
        }
    sensitivity = {}
    for threshold in config["evaluation"]["thresholds"]:
        sensitivity[str(threshold)] = {method: {
            "point_detection_rate": float(np.mean([bool(row["matched"]) for row in points
                                                    if row["method"] == method and
                                                    float(row["threshold"]) == float(threshold)])),
            "pair_separation_rate": float(np.mean([bool(row["separated"]) for row in pairs
                                                   if row["method"] == method and
                                                   float(row["threshold"]) == float(threshold)])),
        } for method in METHODS}
    result["threshold_sensitivity"] = sensitivity
    return result


def _zero_control(output: Path) -> dict[str, float]:
    folder = output / "generated" / "zero_control" / "network"
    network = np.load(folder / "reconstruction.npy", allow_pickle=False)
    anchor = np.load(folder / "input_f_var.npy", allow_pickle=False) * float(json.loads(
        (folder / "inference_contract.json").read_text(encoding="utf-8"))["beta0"])
    return {"network_max": float(network.max()), "network_sum": float(network.sum()),
            "anchor_max": float(anchor.max()), "anchor_sum": float(anchor.sum())}


def _make_figures(config: dict[str, Any], analysis: Path, points: Sequence[dict[str, Any]],
                  pairs: Sequence[dict[str, Any]], lines: Sequence[dict[str, Any]]) -> None:
    figures = analysis / "figures"; primary = float(config["evaluation"]["primary_threshold"])
    plt.figure(figsize=(8, 6))
    for method in METHODS:
        values = []
        for depth in config["scenes"]["point_depths_um"]:
            selected = [row for row in points if row["method"] == method and
                        float(row["threshold"]) == primary and row["kind"] == "single" and
                        int(row["z_um"]) == int(depth) and row.get("matched")]
            values.append(np.mean([Z_UM[int(row["pred_z_layer"])] for row in selected]) if selected else np.nan)
        plt.plot(config["scenes"]["point_depths_um"], values, marker="o", label=METHOD_LABEL[method])
    plt.plot([10, 100], [10, 100], "k--", linewidth=1, label="ideal")
    plt.xlabel("True depth (um)"); plt.ylabel("Matched predicted depth (um)")
    plt.legend(fontsize=7); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(figures / "depth_follow.png", dpi=180); plt.close()

    labels = [METHOD_LABEL[name] for name in METHODS]
    lateral = [np.mean([bool(row["separated"]) for row in pairs if row["method"] == method and
                       float(row["threshold"]) == primary and row["scene_id"].startswith("points")])
               for method in METHODS]
    bridge = [np.mean([bool(row["bridged"]) for row in lines if row["method"] == method and
                      float(row["threshold"]) == primary and float(row["gap_um"]) > 0])
              for method in METHODS]
    x = np.arange(len(METHODS)); width = 0.36
    plt.figure(figsize=(10, 5)); plt.bar(x - width / 2, lateral, width, label="pair separated")
    plt.bar(x + width / 2, bridge, width, label="broken line bridged")
    plt.xticks(x, labels, rotation=20, ha="right"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "local_structure_summary.png", dpi=180); plt.close()

    gains = _read_csv(analysis / "gain_scan.csv")
    plt.figure(figsize=(9, 5))
    for sample in sorted({row["sample_id"] for row in gains}):
        selected = [row for row in gains if row["sample_id"] == sample]
        gain_values = sorted({float(row["gain"]) for row in selected})
        mean_loss = [np.mean([float(row["normalized_mean_loss"]) for row in selected
                             if float(row["gain"]) == gain]) for gain in gain_values]
        var_loss = [np.mean([float(row["normalized_variance_loss"]) for row in selected
                            if float(row["gain"]) == gain]) for gain in gain_values]
        plt.plot(gain_values, mean_loss, "--", label=f"{sample} mean")
        plt.plot(gain_values, var_loss, "-", label=f"{sample} variance")
    plt.xscale("log", base=2); plt.yscale("log"); plt.xlabel("shared candidate gain")
    plt.ylabel("normalized loss"); plt.legend(ncol=2, fontsize=7); plt.tight_layout()
    plt.savefig(figures / "gain_conflict.png", dpi=180); plt.close()

    trajectory = _read_csv(analysis / "validation_checkpoint_by_object.csv")
    plt.figure(figsize=(8, 5))
    for sample in sorted({row["sample_id"] for row in trajectory}):
        selected = sorted([row for row in trajectory if row["sample_id"] == sample],
                          key=lambda row: int(row["step"]))
        plt.plot([int(row["step"]) for row in selected],
                 [float(row["selection_score"]) for row in selected], marker="o", label=f"{sample} loss")
    plt.xlabel("training step"); plt.ylabel("validation physical score"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "checkpoint_physical_score.png", dpi=180); plt.close()


def _write_chinese_report(config: dict[str, Any], output: Path, structure: dict[str, Any],
                          loss: dict[str, Any]) -> None:
    methods = structure["methods"]
    network = methods["network"]
    mean5 = methods["mean_rl5"]
    rankings = loss["controlled_candidate_ranking"]
    wrong_wins = {sample: rows[0]["candidate"] for sample, rows in rankings.items()
                  if rows and rows[0]["candidate"] != "truth"}
    gain_optima = loss["gain_optima"]
    conflicts = {sample: value for sample, value in gain_optima.items()
                 if value["mean_best_gain"] != value["variance_best_gain"]}
    answer_depth = (
        f"网络在单点上的定位成功率为 {network['single_localization_rate']:.1%}，"
        f"成功定位目标的 Z 平均误差为 {network['single_z_mae_um_detected']:.2f} um，"
        f"局部深度分布 W1 为 {network['local_depth_w1_um']:.2f} um。"
    )
    answer_structure = (
        f"网络横向点对分开率 {network['lateral_pair_separation_rate']:.1%}，"
        f"轴向点对分开率 {network['axial_pair_separation_rate']:.1%}；"
        f"连续线假断率 {network['continuous_line_false_gap_rate']:.1%}，"
        f"断线被错误连上的比例 {network['broken_line_bridge_rate']:.1%}。"
    )
    answer_loss = (
        ("错误结构拿到更低总 loss：" + "、".join(f"{k}={v}" for k, v in wrong_wins.items()) + "。")
        if wrong_wins else "三个对象里，正确结构的平均总 loss 都不高于四种受控错误结构。"
    )
    answer_loss += ((" 均值项和方差项偏好的增益不同：" +
                     "、".join(f"{k}({v['mean_best_gain']} vs {v['variance_best_gain']})"
                                for k, v in conflicts.items()) + "。")
                    if conflicts else " 均值项和方差项在当前离散扫描中选择了相同增益。")
    zero = structure["zero_control"]
    text = f"""# 三项最高优先级验证报告

固定模型为第 160 步 `checkpoint_best.pt`。所有网络输入只有前 10 帧、Mean-RL3 和开方一次的 Taylor-RL3；后 90 帧只用于评分。这里的百分比均按三次采集分别计算后汇总，没有把三次预测平均成一幅图。

## 1. 物体换深度后，网络跟不跟

{answer_depth}

判断时只把 XY 误差不超过 2 个像素且 Z 误差不超过 1 层的峰算作成功。完整逐目标结果见 `point_target_metrics.csv`，固定位置深度曲线见 `fixed_position_depth_profiles.csv`，总图见 `figures/depth_follow.png`。如果曲线长期贴着同一深度而不跟随对角线，就是固定层偏置。

## 2. 点、线和断口有没有真的变好

{answer_structure}

点对必须两个峰都先定位成功，并且中间谷值不高于较弱峰的 80%，才算分开。假峰和漏检单独记账，未匹配峰的窄宽度不计作分辨率。作为参照，Mean-RL5 的横向点对分开率为 {mean5['lateral_pair_separation_rate']:.1%}，断线误连接率为 {mean5['broken_line_bridge_rate']:.1%}。5%、10%、20% 三个阈值的结果都保存在 `structure_summary.json`。

## 3. 当前 loss 会不会偏爱错结构，亮度单位是否冲突

{answer_loss}

正确、加厚一次、加厚两次、上移一层、下移一层全部沿用同一份 1024 帧标定，没有逐候选重新调亮度。GT 密度到方差模型输入的主换算为 `GT × sqrt(照明方差)`。`brightness_unit_check.csv` 同时记录了均值单位 `GT × 照明均值` 与它的差异；`gain_scan.csv` 只用于诊断，不改正式预测。

轴向双点的跨层相关项见 `axial_pair_cross_layer_covariance.csv`。第 140、160、180、200 步的验证集逐对象物理分数和结构指标见 `validation_checkpoint_by_object.csv`；第 160 步仍是本轮唯一正式模型。

## 必做核对

- 33 组原始帧均重新计算了前 10 帧和后 90 帧的均值、N-1 方差，结果：{'通过' if structure['verification']['all_scene_integrity_passed'] else '失败'}。
- Taylor 开方一次检查：{'通过' if structure['verification']['taylor_sqrt_applied_exactly_once'] else '失败'}。
- MATLAB/Python FP32 前向相对 L2 为 {loss['forward_consistency']['relative_l2']:.3g}，1e-4 排查线：{'通过' if loss['forward_consistency']['passed'] else '失败'}。
- 既有 P09 子集 01 的冻结网络第 160 步指标复现：{'通过' if structure['verification']['existing_P09_subset01_step160_reproduced'] else '失败'}。
- 推理记录确认没有读取 GT 或后 90 帧目标：{'通过' if structure['verification']['inference_did_not_read_gt_or_target'] else '失败'}。
- 全零输入网络输出最大值 {zero['network_max']:.6g}，总量 {zero['network_sum']:.6g}。

## 文件索引

- `volume_metrics.csv`：全部方法、全部场景、全部重复的三维指标。
- `point_target_metrics.csv`、`point_pair_metrics.csv`、`axial_point_metrics.csv`：定位、漏检、假峰、谷值和拖尾。
- `line_metrics.csv`：连续线假断、断口连接和端点保留。
- `loss_candidate_scores.csv`：每个候选、每个对象、每个独立目标重复的完整 loss 分解。
- `verification.json`、`data_integrity_per_scene.csv`：输入隔离、帧划分和数值复算。

图像只使用整幅体数据的统一阈值，没有逐层调亮度。原始预测、MATLAB 重建、输入帧、来源哈希和可恢复日志都保留在本实验目录。
"""
    (output / "REPORT_ZH.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    _, config = load_config(args.config)
    output = Path(args.output_dir).expanduser().resolve()
    if args.stage == "loss":
        _loss_stage(config, output)
    else:
        _report_stage(config, output)


if __name__ == "__main__":
    main()
