#!/usr/bin/env python3
"""Separate finite-frame estimator error from Taylor H^2 model mismatch.

The diagnostic uses a frozen 17x17 detector grid.  At every selected detector
pixel it evaluates the original shift-variant forward sum directly on sparse
ground truth, verifies that sum against the stored noiseless sensor frames,
and then decomposes empirical variance into diagonal and covariance terms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


OBJECTS = ("P09", "V01", "V02")
HALF_KERNEL = 98


def matlab_volume(handle: h5py.File, key: str) -> np.ndarray:
    return np.asarray(handle[key], dtype=np.float64).transpose(0, 2, 1)


def read_sensor_frames(sample_root: Path) -> np.ndarray:
    frames = []
    for frame_index in range(1, 101):
        with h5py.File(sample_root / "sensor_frames" / f"frame_{frame_index:03d}.mat") as handle:
            frames.append(np.asarray(handle["sensor_pre_detector"], dtype=np.float64).T)
    return np.stack(frames)


def optimal_gain(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(prediction, prediction))
    return float(np.dot(prediction, target) / max(denominator, np.finfo(float).tiny))


def relative_rms(error: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))) / max(np.sqrt(np.mean(np.square(target))), np.finfo(float).tiny))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def direct_sparse_forward(
    h: np.ndarray,
    truth: np.ndarray,
    illumination: np.ndarray,
    detector_y: np.ndarray,
    detector_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact sampled frames, H^2 g^2, and empirical diagonal variance."""

    z, y, x = np.nonzero(truth)
    g = truth[z, y, x]
    frames = np.empty((illumination.shape[0], len(detector_y)), dtype=np.float64)
    constant_diagonal = np.empty(len(detector_y), dtype=np.float64)
    empirical_diagonal = np.empty(len(detector_y), dtype=np.float64)
    for position, (output_y, output_x) in enumerate(zip(detector_y, detector_x)):
        support = (
            (np.abs(y - output_y) <= HALF_KERNEL)
            & (np.abs(x - output_x) <= HALF_KERNEL)
        )
        zz, yy, xx, gg = z[support], y[support], x[support], g[support]
        kernel_y = output_y + HALF_KERNEL - yy
        kernel_x = output_x + HALF_KERNEL - xx
        weights = (
            h[zz, yy % h.shape[1], xx % h.shape[2], kernel_y, kernel_x].astype(np.float64)
            * gg
        )
        local_illumination = illumination[:, zz, yy, xx]
        frames[:, position] = local_illumination @ weights
        squared_weights = np.square(weights)
        constant_diagonal[position] = squared_weights.sum()
        empirical_diagonal[position] = np.dot(
            local_illumination.var(axis=0, ddof=1), squared_weights
        )
    return frames, constant_diagonal, empirical_diagonal


def bootstrap_sampling_error(
    frames: np.ndarray, statistic: str, *, draws: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    target = frames.mean(0) if statistic == "mean" else frames.var(0, ddof=1)
    errors = []
    for _ in range(draws):
        sample = frames[rng.integers(0, len(frames), size=len(frames))]
        estimate = sample.mean(0) if statistic == "mean" else sample.var(0, ddof=1)
        errors.append(relative_rms(estimate - target, target))
    return tuple(float(v) for v in np.quantile(errors, [0.1, 0.5, 0.9]))


def fixed_subset_errors(sample_root: Path, frames: np.ndarray, statistic: str) -> list[float]:
    target = frames.mean(0) if statistic == "mean" else frames.var(0, ddof=1)
    errors = []
    for subset_index in range(1, 11):
        with h5py.File(sample_root / "subsets" / f"subset_{subset_index:02d}.mat") as handle:
            indices = np.asarray(handle["input_indices"]).reshape(-1).astype(int) - 1
        subset = frames[indices]
        estimate = subset.mean(0) if statistic == "mean" else subset.var(0, ddof=1)
        errors.append(relative_rms(estimate - target, target))
    return errors


def speckle_correlations(illumination: np.ndarray) -> dict[str, float]:
    # Correlations are across the 100 realizations, sampled away from borders.
    coordinates = np.arange(32, 228, 13)
    lateral, axial = [], []
    for z in range(illumination.shape[1]):
        for y in coordinates:
            for x in coordinates:
                lateral.append(correlation(illumination[:, z, y, x], illumination[:, z, y, x + 1]))
                if z + 1 < illumination.shape[1]:
                    axial.append(correlation(illumination[:, z, y, x], illumination[:, z + 1, y, x]))
    mean_map = illumination.mean(0)
    return {
        "adjacent_lateral_1px_correlation_median": float(np.nanmedian(lateral)),
        "adjacent_lateral_1px_correlation_mean": float(np.nanmean(lateral)),
        "adjacent_depth_10um_correlation_median": float(np.nanmedian(axial)),
        "adjacent_depth_10um_correlation_mean": float(np.nanmean(axial)),
        "illumination_mean_map_cv": float(mean_map.std() / mean_map.mean()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_root = repo / "data/matlab_cells_pilot_v2_r04"
    h_path = repo / "data/.cache/psf/selected_H_d57d0714b548a9d8c888.npy"
    h = np.load(h_path, mmap_mode="r", allow_pickle=False)
    assert h.shape == (10, 49, 49, 197, 197) and h.dtype == np.float32

    axis = np.linspace(16, 243, 17).round().astype(int)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    detector_y, detector_x = grid_y.ravel(), grid_x.ravel()
    rows = []
    detailed = {
        "complete": False,
        "detector_grid": {
            "axis_indices_zero_based": axis.tolist(),
            "count": int(len(detector_y)),
            "selection": "fixed 17x17 grid; no GT-based detector selection",
        },
        "bootstrap_draws": args.bootstrap_draws,
        "objects": {},
    }
    rng = np.random.default_rng(20260906)

    for sample in OBJECTS:
        sample_root = data_root / sample
        with h5py.File(sample_root / "prepared.mat") as handle:
            truth = matlab_volume(handle, "ground_truth")
        with h5py.File(sample_root / "illumination_3d.mat") as handle:
            illumination = np.asarray(handle["illumination_raw"], dtype=np.float64).transpose(0, 1, 3, 2)
        stored_frames = read_sensor_frames(sample_root)[:, detector_y, detector_x]
        predicted_frames, constant_diagonal, empirical_diagonal = direct_sparse_forward(
            h, truth, illumination, detector_y, detector_x
        )
        verification_rel = relative_rms(predicted_frames - stored_frames, stored_frames)
        verification_max = float(np.max(np.abs(predicted_frames - stored_frames)))
        assert verification_rel < 2e-6, (sample, verification_rel, verification_max)

        mean100 = stored_frames.mean(0)
        variance100 = stored_frames.var(0, ddof=1)
        exact_empirical_mean = predicted_frames.mean(0)
        mean_gain = optimal_gain(exact_empirical_mean, mean100)
        uniform_mean = np.empty_like(mean100)
        # Constant illumination mean is the direct forward sum of H*g.
        # Reuse the exact sampled mean divided by local empirical illumination
        # only through a dedicated all-ones call to avoid an invalid ratio.
        ones = np.ones((1, *truth.shape), dtype=np.float64)
        uniform_frame, _, _ = direct_sparse_forward(
            h, truth, ones, detector_y, detector_x
        )
        uniform_mean[:] = uniform_frame[0]
        uniform_gain = optimal_gain(uniform_mean, mean100)

        h2_gain = optimal_gain(constant_diagonal, variance100)
        empirical_diag_gain = optimal_gain(empirical_diagonal, variance100)
        mean_model_error = relative_rms(uniform_gain * uniform_mean - mean100, mean100)
        h2_model_error = relative_rms(h2_gain * constant_diagonal - variance100, variance100)
        diagonal_model_error = relative_rms(
            empirical_diag_gain * empirical_diagonal - variance100, variance100
        )
        exact_mean_error = relative_rms(mean_gain * exact_empirical_mean - mean100, mean100)
        off_diagonal = variance100 - empirical_diagonal
        off_diagonal_rel = relative_rms(off_diagonal, variance100)

        mean_boot = bootstrap_sampling_error(
            stored_frames, "mean", draws=args.bootstrap_draws, rng=rng
        )
        variance_boot = bootstrap_sampling_error(
            stored_frames, "variance", draws=args.bootstrap_draws, rng=rng
        )
        mean_n10 = fixed_subset_errors(sample_root, stored_frames, "mean")
        variance_n10 = fixed_subset_errors(sample_root, stored_frames, "variance")
        correlations = speckle_correlations(illumination)

        # Root-sum-square subtraction is an explicit approximation: it treats
        # estimator noise and deterministic model residual as independent.
        estimated_h2_excess = float(
            np.sqrt(max(h2_model_error**2 - variance_boot[1] ** 2, 0.0))
        )
        row = {
            "sample_id": sample,
            "forward_verification_relative_rms": verification_rel,
            "exact_empirical_mean_aligned_relative_rms": exact_mean_error,
            "uniform_mean_model_aligned_relative_rms": mean_model_error,
            "mean_n100_bootstrap_relative_rms_median": mean_boot[1],
            "mean_n10_vs_n100_relative_rms_mean": float(np.mean(mean_n10)),
            "h2_constant_model_aligned_relative_rms": h2_model_error,
            "h2_empirical_diagonal_aligned_relative_rms": diagonal_model_error,
            "off_diagonal_covariance_relative_rms": off_diagonal_rel,
            "variance_n100_bootstrap_relative_rms_median": variance_boot[1],
            "variance_n10_vs_n100_relative_rms_mean": float(np.mean(variance_n10)),
            "h2_residual_to_n100_sampling_ratio": h2_model_error / variance_boot[1],
            "estimated_h2_excess_model_relative_rms": estimated_h2_excess,
            **correlations,
        }
        rows.append(row)
        detailed["objects"][sample] = {
            **row,
            "forward_verification_max_abs": verification_max,
            "mean_n100_bootstrap_relative_rms_q10_median_q90": mean_boot,
            "variance_n100_bootstrap_relative_rms_q10_median_q90": variance_boot,
            "mean_fixed_n10_vs_n100_relative_rms": mean_n10,
            "variance_fixed_n10_vs_n100_relative_rms": variance_n10,
            "gains": {
                "uniform_mean_to_observed": uniform_gain,
                "exact_empirical_mean_to_observed": mean_gain,
                "constant_h2_to_observed_variance": h2_gain,
                "empirical_diagonal_h2_to_observed_variance": empirical_diag_gain,
            },
            "interpretation_limits": [
                "N=100 is the available finite-ensemble reference, not an infinite-frame population.",
                "Bootstrap quantifies estimator variability conditional on these 100 noiseless simulated frames.",
                "Empirical diagonal uses true object and local illumination variance; its remaining residual exposes omitted covariance terms on this finite ensemble.",
            ],
        }
        print(json.dumps(row), flush=True)

    detailed["complete"] = True
    detailed["aggregate"] = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
        if key != "sample_id"
    }
    write_csv(output / "statistics_diagnostic.csv", rows)
    (output / "statistics_diagnostic.json").write_text(json.dumps(detailed, indent=2) + "\n")
    print(f"COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
