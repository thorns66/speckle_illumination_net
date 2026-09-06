#!/usr/bin/env python3
"""Evaluate validation iteration trajectories with separate lateral/axial tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


OBJECTS = ("P09", "V01", "V02")
MODES = ("mean", "taylor")
ITERATIONS = (5, 10, 20, 50)
Z_UM = np.arange(10, 101, 10, dtype=float)
PITCH_UM = 220 / 49 / 4


def matlab_volume(handle: h5py.File, key: str) -> np.ndarray:
    return np.asarray(handle[key], dtype=np.float32).transpose(0, 2, 1).copy()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gradient_cosine(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred_grad = np.stack(np.gradient(prediction.astype(np.float64)))
    truth_grad = np.stack(np.gradient(truth.astype(np.float64)))
    numerator = float(np.sum(pred_grad * truth_grad))
    denominator = float(np.linalg.norm(pred_grad) * np.linalg.norm(truth_grad))
    return numerator / max(denominator, np.finfo(float).tiny)


def high_frequency_fraction(image: np.ndarray, threshold_cycles_per_pixel: float = 0.15) -> float:
    centered = image.astype(np.float64) - float(np.mean(image))
    spectrum = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    radial = np.sqrt(fx**2 + fy**2)
    return float(spectrum[radial >= threshold_cycles_per_pixel].sum() / max(spectrum.sum(), np.finfo(float).tiny))


def centroid_xy_um(volume: np.ndarray) -> tuple[float, float]:
    projection = volume.astype(np.float64).sum(0)
    mass = float(projection.sum())
    coordinates = np.arange(projection.shape[0], dtype=float) * PITCH_UM
    y = float(np.sum(projection * coordinates[:, None]) / max(mass, np.finfo(float).tiny))
    x = float(np.sum(projection * coordinates[None, :]) / max(mass, np.finfo(float).tiny))
    return y, x


def v01_profile_spec(data_root: Path) -> dict[str, object]:
    geometry = json.loads((data_root / "V01/geometry.json").read_text())
    first = np.asarray(geometry["geometry"][0]["control_xyz_um"], dtype=float)
    second = np.asarray(geometry["geometry"][1]["control_xyz_um"], dtype=float)
    assert len(first) == len(second) and np.all(first[:, 2] == 40) and np.all(second[:, 2] == 40)
    index = len(first) // 2
    center_a, center_b = first[index, :2], second[index, :2]
    midpoint = (center_a + center_b) / 2
    normal = (center_b - center_a) / np.linalg.norm(center_b - center_a)
    offsets = np.linspace(-12, 12, 241)
    xy = midpoint[None, :] + offsets[:, None] * normal[None, :]
    expected = np.array(
        [np.dot(center_a - midpoint, normal), np.dot(center_b - midpoint, normal)]
    )
    return {
        "center_a_xy_um": center_a,
        "center_b_xy_um": center_b,
        "midpoint_xy_um": midpoint,
        "normal_xy": normal,
        "offset_um": offsets,
        "sample_xy_um": xy,
        "expected_peak_offsets_um": expected,
        "true_separation_um": float(np.linalg.norm(center_b - center_a)),
        "depth_index": 3,
    }


def sample_v01_profile(volume: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    xy = np.asarray(spec["sample_xy_um"])
    return map_coordinates(
        volume[int(spec["depth_index"])],
        [xy[:, 1] / PITCH_UM, xy[:, 0] / PITCH_UM],
        order=1,
        mode="constant",
        cval=0,
    )


def v01_line_metrics(profile: np.ndarray, spec: dict[str, object]) -> dict[str, float]:
    offsets = np.asarray(spec["offset_um"])
    expected = np.asarray(spec["expected_peak_offsets_um"])
    peaks, positions = [], []
    for center in expected:
        window = np.abs(offsets - center) <= 2.0
        local_indices = np.flatnonzero(window)
        index = int(local_indices[np.argmax(profile[window])])
        peaks.append(float(profile[index]))
        positions.append(float(offsets[index]))
    valley = float(np.min(profile[np.abs(offsets) <= 0.8]))
    peak_mean = float(np.mean(peaks))
    contrast = (peak_mean - valley) / max(peak_mean + valley, np.finfo(float).tiny)
    return {
        "v01_two_line_valley_contrast": float(contrast),
        "v01_observed_peak_separation_um": float(abs(positions[1] - positions[0])),
        "v01_mean_abs_peak_localization_error_um": float(np.mean(np.abs(np.asarray(positions) - expected))),
        "v01_fixed_midpoint_to_peak_ratio": float(valley / max(peak_mean, np.finfo(float).tiny)),
    }


def metric_record(
    prediction: np.ndarray,
    truth: np.ndarray,
    reconstruction_metrics,
    ssim_2d,
    v01_spec: dict[str, object] | None,
) -> tuple[dict[str, float], np.ndarray | None]:
    assert prediction.shape == truth.shape == (10, 260, 260)
    assert np.isfinite(prediction).all() and np.all(prediction >= 0)
    prediction64, truth64 = prediction.astype(np.float64), truth.astype(np.float64)
    gain = float(np.sum(prediction64 * truth64) / max(np.sum(prediction64**2), np.finfo(float).tiny))
    aligned = prediction64 * gain
    metrics = reconstruction_metrics(prediction, truth, Z_UM)
    truth_mip, prediction_mip = truth64.max(0), aligned.max(0)
    occupied = truth64.sum((1, 2)) > 1e-8
    prediction_fraction = prediction64.sum((1, 2))
    prediction_fraction /= max(prediction_fraction.sum(), np.finfo(float).tiny)
    truth_fraction = truth64.sum((1, 2))
    truth_fraction /= max(truth_fraction.sum(), np.finfo(float).tiny)
    prediction_centroid = float(np.sum(Z_UM * prediction_fraction))
    truth_centroid = float(np.sum(Z_UM * truth_fraction))
    pred_y, pred_x = centroid_xy_um(prediction64)
    truth_y, truth_x = centroid_xy_um(truth64)
    metrics.update(
        evaluation_only_global_gain=gain,
        gt_xy_mip_ssim_global_gain_aligned=ssim_2d(truth_mip, prediction_mip),
        gt_true_depth_layers_nrmse=float(
            np.linalg.norm((aligned[occupied] - truth64[occupied]).ravel())
            / max(np.linalg.norm(truth64[occupied].ravel()), np.finfo(float).tiny)
        ),
        gt_xy_mip_gradient_cosine=gradient_cosine(prediction_mip, truth_mip),
        predicted_xy_mip_high_frequency_fraction=high_frequency_fraction(prediction_mip),
        truth_xy_mip_high_frequency_fraction=high_frequency_fraction(truth_mip),
        predicted_axial_std_um=float(
            np.sqrt(np.sum(np.square(Z_UM - prediction_centroid) * prediction_fraction))
        ),
        truth_axial_std_um=float(
            np.sqrt(np.sum(np.square(Z_UM - truth_centroid) * truth_fraction))
        ),
        predicted_mass_on_exact_truth_layers=float(prediction_fraction[occupied].sum()),
        gt_xy_centroid_error_um=float(np.hypot(pred_y - truth_y, pred_x - truth_x)),
    )
    profile = None
    if v01_spec is not None:
        profile = sample_v01_profile(aligned, v01_spec)
        metrics.update(v01_line_metrics(profile, v01_spec))
    else:
        metrics.update(
            v01_two_line_valley_contrast=float("nan"),
            v01_observed_peak_separation_um=float("nan"),
            v01_mean_abs_peak_localization_error_um=float("nan"),
            v01_fixed_midpoint_to_peak_ratio=float("nan"),
        )
    return metrics, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, audit_root, report = (
        args.repo.resolve(),
        args.audit_root.resolve(),
        args.report_dir.resolve(),
    )
    run_manifest = json.loads((audit_root / "run_manifest.json").read_text())
    assert run_manifest["status"] == "complete" and run_manifest["source_files_unchanged"]
    report.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(report / "mpl_cache"))
    sys.path.insert(0, str(repo))
    import torch
    from utils.reconstruction_metrics import reconstruction_metrics, _ssim_2d

    torch.set_num_threads(2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    data_root = repo / "data/matlab_cells_pilot_v2_r04"
    truths = {}
    for sample in OBJECTS:
        with h5py.File(data_root / sample / "prepared.mat") as handle:
            truths[sample] = matlab_volume(handle, "ground_truth")
    line_spec = v01_profile_spec(data_root)
    truth_profile = sample_v01_profile(truths["V01"], line_spec)
    truth_line_metrics = v01_line_metrics(truth_profile, line_spec)

    rows: list[dict[str, object]] = []
    profiles: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    volumes: dict[tuple[str, str, str, int, int, int], np.ndarray] = {}
    regression_checks = []

    def add_volume(
        *,
        sample: str,
        algorithm: str,
        mode: str,
        frames: int,
        subset: int,
        iteration: int,
        prediction: np.ndarray,
        source: Path,
    ) -> None:
        metrics, profile = metric_record(
            prediction,
            truths[sample],
            reconstruction_metrics,
            _ssim_2d,
            line_spec if sample == "V01" else None,
        )
        rows.append(
            {
                "sample_id": sample,
                "algorithm": algorithm,
                "mode": mode,
                "frames": frames,
                "subset_index": subset,
                "iteration": iteration,
                "source": str(source),
                **metrics,
            }
        )
        if profile is not None:
            profiles[algorithm, mode, frames, subset, iteration] = profile
        if subset in (0, 1) and iteration in (5, 50):
            volumes[sample, algorithm, mode, frames, subset, iteration] = prediction

    for sample in OBJECTS:
        truth = truths[sample]
        for mode in MODES:
            for subset in range(1, 11):
                folder = audit_root / "historical_isra" / sample / f"subset_{subset:02d}" / mode
                info = json.loads((folder / "complete.json").read_text())
                regression = info["iteration3_regression"]
                regression_checks.append(
                    {
                        "sample_id": sample,
                        "mode": mode,
                        "subset_index": subset,
                        **regression,
                    }
                )
                assert regression["relative_l2"] == 0 and regression["max_abs"] == 0
                with h5py.File(folder / "trajectory.mat") as handle:
                    for iteration in (3, *ITERATIONS):
                        add_volume(
                            sample=sample,
                            algorithm="historical_isra",
                            mode=mode,
                            frames=10,
                            subset=subset,
                            iteration=iteration,
                            prediction=matlab_volume(handle, f"reconstruction_iter_{iteration:03d}"),
                            source=folder / "trajectory.mat",
                        )
            folder = audit_root / "historical_isra" / sample / "full_100" / mode
            with h5py.File(folder / "trajectory.mat") as handle:
                for iteration in ITERATIONS:
                    add_volume(
                        sample=sample,
                        algorithm="historical_isra",
                        mode=mode,
                        frames=100,
                        subset=0,
                        iteration=iteration,
                        prediction=matlab_volume(handle, f"reconstruction_iter_{iteration:03d}"),
                        source=folder / "trajectory.mat",
                    )
            for tag, frames, subset in (("subset_01", 10, 1), ("full_100", 100, 0)):
                folder = audit_root / "standard_rl" / sample / tag / mode
                with h5py.File(folder / "trajectory.mat") as handle:
                    for iteration in ITERATIONS:
                        add_volume(
                            sample=sample,
                            algorithm="standard_rl",
                            mode=mode,
                            frames=frames,
                            subset=subset,
                            iteration=iteration,
                            prediction=matlab_volume(handle, f"reconstruction_iter_{iteration:03d}"),
                            source=folder / "trajectory.mat",
                        )

        network_root = repo / "outputs/multivolume_n10_no_mean_run01/validation/step_000160" / sample
        for subset in range(1, 11):
            path = network_root / f"subset_{subset:02d}/reconstruction.npy"
            add_volume(
                sample=sample,
                algorithm="network_step160",
                mode="network",
                frames=10,
                subset=subset,
                iteration=0,
                prediction=np.load(path),
                source=path,
            )

    assert len(rows) == 402 and len(regression_checks) == 60
    write_csv(report / "metrics_per_volume.csv", rows)
    write_csv(report / "iteration3_exact_regression.csv", regression_checks)

    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, np.floating)) and key != "iteration"
    ]
    object_rows = []
    group_keys = []
    for row in rows:
        key = (
            row["sample_id"],
            row["algorithm"],
            row["mode"],
            row["frames"],
            row["iteration"],
        )
        if key not in group_keys:
            group_keys.append(key)
    for sample, algorithm, mode, frames, iteration in group_keys:
        selected = [
            row
            for row in rows
            if (
                row["sample_id"],
                row["algorithm"],
                row["mode"],
                row["frames"],
                row["iteration"],
            )
            == (sample, algorithm, mode, frames, iteration)
        ]
        record = {
            "sample_id": sample,
            "algorithm": algorithm,
            "mode": mode,
            "frames": frames,
            "iteration": iteration,
            "items": len(selected),
        }
        for metric in numeric_keys:
            values = np.asarray([float(row[metric]) for row in selected])
            finite = values[np.isfinite(values)]
            record[metric] = float(np.mean(finite)) if len(finite) else float("nan")
            record[f"{metric}_subset_sd"] = float(np.std(finite)) if len(finite) > 1 else float("nan")
        object_rows.append(record)
    write_csv(report / "metrics_per_object.csv", object_rows)

    macro_rows = []
    macro_keys = []
    for row in object_rows:
        key = row["algorithm"], row["mode"], row["frames"], row["iteration"]
        if key not in macro_keys:
            macro_keys.append(key)
    for algorithm, mode, frames, iteration in macro_keys:
        selected = [
            row
            for row in object_rows
            if (row["algorithm"], row["mode"], row["frames"], row["iteration"])
            == (algorithm, mode, frames, iteration)
        ]
        record = {
            "algorithm": algorithm,
            "mode": mode,
            "frames": frames,
            "iteration": iteration,
            "objects": len(selected),
        }
        for metric in numeric_keys:
            finite = np.asarray([float(row[metric]) for row in selected])
            finite = finite[np.isfinite(finite)]
            record[metric] = float(np.mean(finite)) if len(finite) else float("nan")
        macro_rows.append(record)
    write_csv(report / "metrics_macro.csv", macro_rows)

    def macro(algorithm: str, mode: str, frames: int, iteration: int) -> dict[str, object]:
        selected = [
            row
            for row in macro_rows
            if (row["algorithm"], row["mode"], row["frames"], row["iteration"])
            == (algorithm, mode, frames, iteration)
        ]
        assert len(selected) == 1
        return selected[0]

    # Iteration curves, separating structural agreement from apparent sharpness.
    curve_metrics = [
        ("gt_scale_aligned_nrmse", "3D aligned NRMSE (lower)"),
        ("gt_axial_w1_um", "Axial W1, um (lower)"),
        ("gt_xy_mip_gradient_cosine", "XY gradient cosine (higher)"),
        ("predicted_xy_mip_high_frequency_fraction", "XY high-frequency fraction (sharpness only)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis_plot, (metric, title) in zip(axes.ravel(), curve_metrics):
        for mode, color in (("mean", "tab:blue"), ("taylor", "tab:orange")):
            for frames, linestyle, marker in ((10, "-", "o"), (100, "--", "s")):
                selected_iterations = (3, *ITERATIONS) if frames == 10 else ITERATIONS
                values = [float(macro("historical_isra", mode, frames, it)[metric]) for it in selected_iterations]
                axis_plot.plot(
                    selected_iterations,
                    values,
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    label=f"{mode.title()} N={frames}",
                )
        axis_plot.set(title=title, xlabel="Iterations")
        axis_plot.grid(alpha=0.25)
        axis_plot.legend(fontsize=8)
    fig.suptitle("Validation: historical update; object-equal macro averages")
    fig.savefig(report / "iteration_curves_historical.png", dpi=160)
    plt.close(fig)

    # Fixed V01 two-line profile; N=10 is descriptive mean +/- subset SD.
    offsets = np.asarray(line_spec["offset_um"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True, sharex=True)
    for row_index, mode in enumerate(MODES):
        for column, iteration in enumerate(ITERATIONS):
            axis_plot = axes[row_index, column]
            axis_plot.plot(offsets, truth_profile, "k-", lw=2, label="GT")
            subset_profiles = np.stack(
                [profiles["historical_isra", mode, 10, subset, iteration] for subset in range(1, 11)]
            )
            mean_profile, sd_profile = subset_profiles.mean(0), subset_profiles.std(0)
            axis_plot.plot(offsets, mean_profile, color="tab:blue", label="N=10 mean")
            axis_plot.fill_between(offsets, mean_profile - sd_profile, mean_profile + sd_profile, color="tab:blue", alpha=0.15)
            axis_plot.plot(
                offsets,
                profiles["historical_isra", mode, 100, 0, iteration],
                color="tab:red",
                linestyle="--",
                label="N=100",
            )
            for expected in np.asarray(line_spec["expected_peak_offsets_um"]):
                axis_plot.axvline(expected, color="gray", lw=0.7, alpha=0.5)
            axis_plot.set_title(f"{mode.title()}, iter {iteration}")
            axis_plot.grid(alpha=0.2)
            if row_index == 1:
                axis_plot.set_xlabel("Offset normal to lines (um)")
            if column == 0:
                axis_plot.set_ylabel("Globally aligned intensity")
            if row_index == 0 and column == 0:
                axis_plot.legend(fontsize=7)
    fig.suptitle(
        f"V01 fixed parallel-line pair, truth separation {line_spec['true_separation_um']:.2f} um at z=40 um"
    )
    fig.savefig(report / "V01_fixed_two_line_profiles.png", dpi=160)
    plt.close(fig)

    # Full-100 MIPs show how high-frequency appearance can diverge from fidelity.
    columns = (("GT", "", 0), ("Mean i5", "mean", 5), ("Mean i50", "mean", 50), ("Taylor i5", "taylor", 5), ("Taylor i50", "taylor", 50))
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), constrained_layout=True)
    for row_index, sample in enumerate(OBJECTS):
        truth = truths[sample]
        norm = PowerNorm(gamma=0.4, vmin=0, vmax=float(truth.max()))
        for column, (label, mode, iteration) in enumerate(columns):
            if label == "GT":
                volume = truth
            else:
                raw = volumes[sample, "historical_isra", mode, 100, 0, iteration]
                metric = next(
                    row
                    for row in rows
                    if row["sample_id"] == sample
                    and row["algorithm"] == "historical_isra"
                    and row["mode"] == mode
                    and row["frames"] == 100
                    and row["subset_index"] == 0
                    and row["iteration"] == iteration
                )
                volume = raw * float(metric["evaluation_only_global_gain"])
            axes[row_index, column].imshow(volume.max(0), cmap="magma", norm=norm)
            axes[row_index, column].set_title(f"{sample} | {label}", fontsize=9)
            axes[row_index, column].axis("off")
    fig.suptitle("N=100 historical update; one GT-fitted gain per whole volume; shared display scale per row")
    fig.savefig(report / "full100_mip_iter5_vs_iter50.png", dpi=160)
    plt.close(fig)

    # Full-100 axial distributions at early/late iterations.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis_plot, sample in zip(axes, OBJECTS):
        truth_mass = truths[sample].sum((1, 2)).astype(float)
        axis_plot.plot(Z_UM, truth_mass / truth_mass.sum(), "k-o", lw=2, label="GT")
        for mode, color in (("mean", "tab:blue"), ("taylor", "tab:orange")):
            for iteration, linestyle in ((5, "-"), (50, "--")):
                volume = volumes[sample, "historical_isra", mode, 100, 0, iteration]
                mass = volume.sum((1, 2)).astype(float)
                axis_plot.plot(Z_UM, mass / mass.sum(), color=color, linestyle=linestyle, marker="o", ms=3, label=f"{mode} i{iteration}")
        axis_plot.set(title=sample, xlabel="Depth (um)", ylabel="Mass fraction", ylim=(0, 1.02))
        axis_plot.grid(alpha=0.2)
        axis_plot.legend(fontsize=7)
    fig.savefig(report / "full100_axial_iter5_vs_iter50.png", dpi=160)
    plt.close(fig)

    operator_audit = json.loads((audit_root / "operator_audit/operator_audit.json").read_text())
    statistics = json.loads((audit_root / "statistics_model_audit/statistics_diagnostic.json").read_text())
    psf_normalization = json.loads((audit_root / "psf_normalization_audit/psf_depth_mass.json").read_text())
    assert statistics["complete"]
    assert psf_normalization["complete"]

    verification = {
        "complete": True,
        "metric_rows": len(rows),
        "historical_iteration3_exact_matches": len(regression_checks),
        "historical_iteration3_max_relative_l2": max(float(row["relative_l2"]) for row in regression_checks),
        "historical_iteration3_max_abs": max(float(row["max_abs"]) for row in regression_checks),
        "v01_profile": {
            "true_separation_um": line_spec["true_separation_um"],
            "expected_peak_offsets_um": np.asarray(line_spec["expected_peak_offsets_um"]).tolist(),
            "truth_metrics": truth_line_metrics,
            "selection": "first two geometry tubes, middle control point, fixed before reading reconstructions",
        },
        "operator_audit": operator_audit,
        "psf_depth_normalization_audit": psf_normalization,
        "statistics_model_audit": statistics,
        "metric_semantics": {
            "high_frequency_fraction": "apparent sharpness only; higher is not necessarily better",
            "gradient_cosine": "lateral edge agreement with truth; higher is better",
            "v01_two_line_valley_contrast": "fixed truth-defined profile; inspect with localization error and NRMSE",
            "global_gain": "evaluation only; one scalar per full 3D volume; no layer scaling or registration",
        },
    }
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    # Chinese summary generated from complete macro results.
    historical_table = []
    for frames in (10, 100):
        for iteration in ITERATIONS:
            for mode in MODES:
                historical_table.append(macro("historical_isra", mode, frames, iteration))
    best = {
        f"{mode}_{frames}": min(
            (macro("historical_isra", mode, frames, iteration) for iteration in ITERATIONS),
            key=lambda row: float(row["gt_scale_aligned_nrmse"]),
        )
        for mode in MODES
        for frames in (10, 100)
    }

    def update_audit_average(algorithm: str, mode: str, frames: int, iteration: int, metric: str) -> float:
        required_subset = 1 if frames == 10 else 0
        selected = [
            float(row[metric])
            for row in rows
            if row["algorithm"] == algorithm
            and row["mode"] == mode
            and row["frames"] == frames
            and row["subset_index"] == required_subset
            and row["iteration"] == iteration
        ]
        assert len(selected) == 3
        return float(np.mean(selected))
    lines = [
        "# 三项针对性检查：完整结果",
        "",
        "本报告使用验证集 P09/V01/V02。历史更新对全部 10 个固定 10 帧子集和完整 100 帧保存 5/10/20/50 次检查点；另用 subset_01 与 100 帧执行标准 Poisson-form RL 算法审计。网络参考固定为 step 160，不参与迭代次数选择。",
        "",
        "## 1. 迭代次数与过锐化扫描",
        "",
        "| 帧数 | 方法 | 迭代 | 3D NRMSE↓ | 轴向 W1 µm↓ | XY 梯度一致性↑ | 高频占比（仅锐度） |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in historical_table:
        lines.append(
            f"| {row['frames']} | {str(row['mode']).title()} | {row['iteration']} | "
            f"{float(row['gt_scale_aligned_nrmse']):.4f} | {float(row['gt_axial_w1_um']):.3f} | "
            f"{float(row['gt_xy_mip_gradient_cosine']):.4f} | {float(row['predicted_xy_mip_high_frequency_fraction']):.4f} |"
        )
    mean10_i5 = macro("historical_isra", "mean", 10, 5)
    taylor10_i5 = macro("historical_isra", "taylor", 10, 5)
    network_macro = macro("network_step160", "network", 10, 0)
    lines.extend(
        [
            "",
            "第 5 次的定量拆分：Taylor10 的高频占比为 "
            f"{float(taylor10_i5['predicted_xy_mip_high_frequency_fraction']):.4f}，Mean10 为 "
            f"{float(mean10_i5['predicted_xy_mip_high_frequency_fraction']):.4f}；但 Taylor10/Mean10 的 3D NRMSE 分别为 "
            f"{float(taylor10_i5['gt_scale_aligned_nrmse']):.4f}/{float(mean10_i5['gt_scale_aligned_nrmse']):.4f}，"
            f"轴向 W1 为 {float(taylor10_i5['gt_axial_w1_um']):.3f}/{float(mean10_i5['gt_axial_w1_um']):.3f} µm。高频更多与真值更准是两个问题。",
            "",
            "100 帧相对 10 帧（第 5 次、验证集宏平均）：",
        ]
    )
    for mode in MODES:
        ten = macro("historical_isra", mode, 10, 5)
        hundred = macro("historical_isra", mode, 100, 5)
        lines.append(
            f"- {mode.title()}: NRMSE {float(ten['gt_scale_aligned_nrmse']):.4f} → "
            f"{float(hundred['gt_scale_aligned_nrmse']):.4f}；W1 {float(ten['gt_axial_w1_um']):.3f} → "
            f"{float(hundred['gt_axial_w1_um']):.3f} µm；梯度一致性 "
            f"{float(ten['gt_xy_mip_gradient_cosine']):.4f} → {float(hundred['gt_xy_mip_gradient_cosine']):.4f}。"
        )
    lines.extend(
        [
            "",
            "固定网络 step160 的验证集参考（30 个 10 帧子集）："
            f"NRMSE={float(network_macro['gt_scale_aligned_nrmse']):.4f}，"
            f"W1={float(network_macro['gt_axial_w1_um']):.3f} µm，"
            f"梯度一致性={float(network_macro['gt_xy_mip_gradient_cosine']):.4f}。",
        ]
    )
    lines.extend(["", "按三维 NRMSE 的诊断性最优检查点（仅解释，不用于重新选网络或测试结果）："])
    for key, row in best.items():
        lines.append(
            f"- {key}: iter {row['iteration']}，NRMSE={float(row['gt_scale_aligned_nrmse']):.4f}，W1={float(row['gt_axial_w1_um']):.3f} µm。"
        )

    lines.extend(
        [
            "",
            "![迭代曲线](iteration_curves_historical.png)",
            "",
            "## 2. 更新式、伴随与归一化审计",
            "",
            "历史代码的更新是 `X <- X * B y / B A X`；Poisson-form 归一化更新是 `X <- X/B1 * B(y/(AX))`，其中 B 是现有 supplied-Ht 反投影。由于下表显示 B 并非严格 A^T，后者也是算法对照，而不是严格似然意义的标准 RL。本报告把二者分开，不再把历史式默认称为标准 RL。",
            "",
            "| 模式 | 伴随内积相对误差 | sensitivity CV | H 核质量 CV | Ht 核质量 CV |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        audit = operator_audit[mode]
        lines.append(
            f"| {mode.title()} | {float(audit['adjoint_relative_error']):.3e} | "
            f"{float(audit['sensitivity_cv']):.4f} | {float(audit['H_kernel_mass_cv']):.4f} | "
            f"{float(audit['Ht_kernel_mass_cv']):.4f} |"
        )
    h_depth = psf_normalization["aggregate"]["H"]
    h2_depth = psf_normalization["aggregate"]["H_squared"]
    lines.extend(
        [
            "",
            "逐深度核质量：普通 H 的首层/末层比为 "
            f"{float(h_depth['first_to_last_ratio']):.4f}；H² 为 {float(h2_depth['first_to_last_ratio']):.4f}。"
            f"H² 的深度质量 CV={float(h2_depth['depth_mass_cv']):.4f}，且 10–100 µm 严格单调下降={h2_depth['strictly_decreasing_with_depth']}。"
            "全局常数只影响尺度，但这种逐深度变化会影响未归一化迭代的轴向分配。",
        ]
    )
    lines.extend(
        [
            "",
            "同一输入下历史式与标准式的对象等权结果（10 帧固定为 subset_01）：",
            "",
            "| 帧数 | 方法 | 迭代 | 历史式 NRMSE | 标准式 NRMSE | 历史式 W1 | 标准式 W1 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for frames in (10, 100):
        for mode in MODES:
            for iteration in ITERATIONS:
                lines.append(
                    f"| {frames} | {mode.title()} | {iteration} | "
                    f"{update_audit_average('historical_isra', mode, frames, iteration, 'gt_scale_aligned_nrmse'):.4f} | "
                    f"{update_audit_average('standard_rl', mode, frames, iteration, 'gt_scale_aligned_nrmse'):.4f} | "
                    f"{update_audit_average('historical_isra', mode, frames, iteration, 'gt_axial_w1_um'):.3f} | "
                    f"{update_audit_average('standard_rl', mode, frames, iteration, 'gt_axial_w1_um'):.3f} |"
                )
    lines.extend(
        [
            "",
            "标准式数值结果见 `metrics_macro.csv` 中 `standard_rl` 行。Taylor 方差并非 Poisson 计数，因此这部分只用于识别更新式影响，不把它宣称为统计上正确的 Taylor 似然。",
            "",
            "## 3. 横向、轴向及采样—模型分解",
            "",
            f"V01 固定双线真值间距为 {float(line_spec['true_separation_um']):.3f} µm；真值固定剖面对比度为 {truth_line_metrics['v01_two_line_valley_contrast']:.4f}。逐体对比度、峰间距和定位误差均在 `metrics_per_volume.csv`。",
            "",
            "![V01 固定双线剖面](V01_fixed_two_line_profiles.png)",
            "",
            "轴向单独使用 W1、质量标准差、真值层质量占比和深度峰/质心；不以 XY 看起来更锐替代轴向正确性。",
            "",
            "![100 帧轴向分布](full100_axial_iter5_vs_iter50.png)",
            "",
            "### 受控统计诊断（三对象等权平均）",
            "",
        ]
    )
    aggregate = statistics["aggregate"]
    lines.extend(
        [
            f"- 直接前向复现保存传感器帧的相对 RMS 误差：{aggregate['forward_verification_relative_rms']:.3e}。",
            f"- 10 帧均值相对 100 帧参考的波动：{aggregate['mean_n10_vs_n100_relative_rms_mean']:.3f}；10 帧方差：{aggregate['variance_n10_vs_n100_relative_rms_mean']:.3f}。",
            f"- N=100 方差 bootstrap 采样误差中位数：{aggregate['variance_n100_bootstrap_relative_rms_median']:.3f}；H² 常方差模型对 N=100 方差的增益对齐残差：{aggregate['h2_constant_model_aligned_relative_rms']:.3f}。",
            f"- 只保留照明方差对角项后的残差：{aggregate['h2_empirical_diagonal_aligned_relative_rms']:.3f}；未对齐的协方差交叉项相对 RMS：{aggregate['off_diagonal_covariance_relative_rms']:.3f}。",
            f"- 相邻横向 1 像素散斑相关中位数：{aggregate['adjacent_lateral_1px_correlation_median']:.3f}；相邻 10 µm 深度：{aggregate['adjacent_depth_10um_correlation_median']:.4f}。",
            "",
            "限制：100 帧仍是有限总体参考，不是无限帧真值；bootstrap 是条件于现有 100 个无噪声仿真帧的估计。H² 与采样误差的残差不能简单相加，报告中的 excess-model 值仅作独立误差近似。",
            "",
            "## 可视证据",
            "",
            "![100 帧第5与50次 MIP](full100_mip_iter5_vs_iter50.png)",
            "",
            "所有原始逐体指标与审计证据：`metrics_per_volume.csv`、`metrics_per_object.csv`、`metrics_macro.csv`、`verification.json`。",
        ]
    )
    (report / "report_zh.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:25]), flush=True)
    print(f"COMPLETE report={report}", flush=True)


if __name__ == "__main__":
    main()
