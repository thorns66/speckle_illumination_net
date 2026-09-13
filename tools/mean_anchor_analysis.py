"""Evaluation augmentation, stability metrics, figures, and report for Mean anchor."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.matlab_multivolume_dataset import (
    DatasetItemKey,
    _read_targets,
    load_dataset_index,
    load_inference_input,
)
from tools import mean_anchor_experiment as exp
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as evaluation


BASELINE = exp.REFERENCE_OUTPUT
METHODS = ("mean_rl3", "taylor_rl3_sqrt", "e3_mean100", exp.ARM)
LABELS = {
    "mean_rl3": "Mean-RL3",
    "taylor_rl3_sqrt": "Taylor-RL3-sqrt",
    "e3_mean100": "Taylor anchor network",
    exp.ARM: "Mean anchor network",
    "ground_truth": "Ground truth",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def augment_evaluation() -> dict[str, Any]:
    rows = []
    cases = {case["id"]: case for case in evaluation.all_cases()}
    root = exp.OUTPUT / "evaluation" / exp.ARM
    for role in ("best", "final"):
        for case_id, case in cases.items():
            destination = root / role / case_id
            record_path = destination / "complete.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            inputs = load_inference_input(
                case["path"], case["subset"], var_feature_representation="sqrt"
            )
            g_mean = np.asarray(inputs["g_mean"][0], np.float32)
            beta = float(record["metrics"]["beta"])
            pre_gain_anchor = beta * g_mean
            prediction = np.load(destination / "reconstruction.npy", allow_pickle=False)
            physical_anchor = np.load(destination / "anchor.npy", allow_pickle=False)
            correction = prediction - physical_anchor
            np.save(destination / "pre_gain_mean_anchor.npy", pre_gain_anchor)
            np.save(destination / "effective_correction.npy", correction.astype(np.float32))
            ratio = float(np.linalg.norm(correction) / max(np.linalg.norm(physical_anchor), 1e-30))
            signed = float(correction.sum(dtype=np.float64) / max(physical_anchor.sum(dtype=np.float64), 1e-30))
            metrics = {
                "method": f"{exp.ARM}_{role}", "checkpoint_role": role,
                "case_id": case_id, "sample_id": case["sample"], "split": case["split"],
                "subset": case["subset"], "correction_to_anchor_l2": ratio,
                "signed_correction_mass_ratio": signed,
                "pre_gain_anchor_sha256": old.sha256(destination / "pre_gain_mean_anchor.npy"),
                "effective_correction_sha256": old.sha256(destination / "effective_correction.npy"),
            }
            record["mean_anchor_artifacts"] = metrics
            old.write_json(record_path, record)
            rows.append(metrics)
    write_csv(root / "correction_metrics.csv", rows)
    result = {"complete": True, "cases": len(rows), "expected": 188}
    if len(rows) != 188:
        raise ValueError(f"Expected 188 augmented evaluations, got {len(rows)}")
    old.write_json(root / "augmentation_complete.json", result)
    return result


def _case_volumes(sample: str, subset: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    indexed, _ = load_dataset_index(exp.DATA)
    key = next(
        item for item in indexed["test"]
        if item.sample_id == sample and item.subset_index == subset
    )
    raw = load_inference_input(key.sample_dir, subset, var_feature_representation="sqrt")
    target = _read_targets(key, include_ground_truth=True)
    volumes = {
        "mean_rl3": np.asarray(raw["g_mean"][0], np.float32),
        "taylor_rl3_sqrt": np.asarray(raw["f_var"][0], np.float32),
        "e3_mean100": np.load(
            BASELINE / "evaluation/e3_mean100/final" / f"{sample}_subset_{subset:02d}" / "reconstruction.npy",
            allow_pickle=False,
        ),
        exp.ARM: np.load(
            exp.OUTPUT / "evaluation" / exp.ARM / "final" / f"{sample}_subset_{subset:02d}" / "reconstruction.npy",
            allow_pickle=False,
        ),
    }
    return volumes, np.asarray(target["ground_truth"][0], np.float32)


def _normalized(volume: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(volume, np.float64), 0)
    return positive / max(positive.sum(), 1e-30)


def stability_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for sample in ("T02", "T03", "T04"):
        by_method = {method: [] for method in METHODS}
        truths = []
        for subset in range(1, 11):
            volumes, truth = _case_volumes(sample, subset)
            truths.append(truth)
            for method, volume in volumes.items():
                by_method[method].append(volume)
                score = evaluation._structure_row(volume, truth)
                quality_rows.append({"sample_id": sample, "subset": subset, "method": method, **score})
        for method, values in by_method.items():
            q = np.stack([_normalized(value) for value in values])
            center = q.mean(0)
            dispersion = float(np.sqrt(np.mean(np.sum((q - center) ** 2, axis=(1, 2, 3)))) / max(np.linalg.norm(center), 1e-30))
            profiles = q.sum((2, 3))
            pairwise = [
                float(np.abs(np.cumsum(profiles[a]) - np.cumsum(profiles[b])).sum() * 10)
                for a, b in itertools.combinations(range(10), 2)
            ]
            masses = np.asarray([np.maximum(value, 0).sum(dtype=np.float64) for value in values])
            errors = np.asarray([
                row["gt_scale_aligned_nrmse"] for row in quality_rows
                if row["sample_id"] == sample and row["method"] == method
            ])
            object_rows.append({
                "sample_id": sample, "method": method,
                "shape_relative_dispersion": dispersion,
                "pairwise_axial_w1_um": float(np.mean(pairwise)),
                "mass_coefficient_of_variation": float(masses.std(ddof=1) / max(masses.mean(), 1e-30)),
                "gt_scale_aligned_nrmse_std": float(errors.std(ddof=1)),
                "subsets": 10, "pair_count": 45,
            })
    for method in METHODS:
        selected = [row for row in object_rows if row["method"] == method]
        object_rows.append({
            "sample_id": "object_macro", "method": method,
            **{key: float(np.mean([row[key] for row in selected])) for key in (
                "shape_relative_dispersion", "pairwise_axial_w1_um",
                "mass_coefficient_of_variation", "gt_scale_aligned_nrmse_std",
            )},
            "subsets": 30, "pair_count": 135,
        })
    return object_rows, quality_rows


def _quality_summary(quality: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in ("T02", "T03", "T04", "object_macro"):
        for method in METHODS:
            selected = [row for row in quality if row["method"] == method and (sample == "object_macro" or row["sample_id"] == sample)]
            if sample == "object_macro":
                per_object = []
                for owner in ("T02", "T03", "T04"):
                    owned = [row for row in quality if row["method"] == method and row["sample_id"] == owner]
                    per_object.append({key: np.mean([float(row[key]) for row in owned]) for key in (
                        "gt_scale_aligned_nrmse", "gt_axial_w1_um", "gt_xy_mip_ssim", "background_xy_mass_fraction"
                    )})
                values = {key: float(np.mean([row[key] for row in per_object])) for key in per_object[0]}
            else:
                values = {key: float(np.mean([float(row[key]) for row in selected])) for key in (
                    "gt_scale_aligned_nrmse", "gt_axial_w1_um", "gt_xy_mip_ssim", "background_xy_mass_fraction"
                )}
            rows.append({"sample_id": sample, "method": method, **values})
    return rows


def _show(ax, image: np.ndarray, scale: float, title: str, aspect="equal") -> None:
    ax.imshow(np.clip(image / max(scale, 1e-30), 0, 1), cmap="magma", vmin=0, vmax=1, interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def figures() -> int:
    output = exp.OUTPUT / "analysis/figures"
    output.mkdir(parents=True, exist_ok=True)
    for sample in ("T02", "T03", "T04"):
        volumes, truth = _case_volumes(sample, 1)
        shown = {"ground_truth": truth, **volumes}
        fig, axes = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
        for column, method in enumerate(("ground_truth", *METHODS)):
            value = shown[method]; scale = float(np.quantile(value, .999))
            for row, view in enumerate((value.max(0), value.sum(1), value.sum(2))):
                _show(axes[row, column], view, scale if row == 0 else float(np.quantile(view, .999)), LABELS[method] + (" XY" if row == 0 else (" XZ" if row == 1 else " YZ")), "auto" if row else "equal")
        fig.savefig(output / f"{sample}_subset01_projections_shape_p999.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(10, 5, figsize=(14, 28), layout="constrained")
        for column, method in enumerate(("ground_truth", *METHODS)):
            value = shown[method]; scale = float(np.quantile(value, .999))
            for depth in range(10):
                _show(axes[depth, column], value[depth], scale, f"{LABELS[method]} · {10 * (depth + 1)} um")
        fig.savefig(output / f"{sample}_subset01_native_layers.png", dpi=110)
        plt.close(fig)

        shared = max(float(np.quantile(volumes[name], .999)) for name in ("e3_mean100", exp.ARM))
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
        for column, method in enumerate(("e3_mean100", exp.ARM)):
            _show(axes[0, column], volumes[method].max(0), shared, LABELS[method] + " · shared XY")
            _show(axes[1, column], volumes[method].sum(1), max(float(np.quantile(volumes[name].sum(1), .999)) for name in ("e3_mean100", exp.ARM)), LABELS[method] + " · shared XZ", "auto")
        fig.savefig(output / f"{sample}_subset01_network_shared.png", dpi=160)
        plt.close(fig)

        case = exp.OUTPUT / "evaluation" / exp.ARM / "final" / f"{sample}_subset_01"
        anchor = np.load(case / "anchor.npy", allow_pickle=False)
        correction = np.load(case / "effective_correction.npy", allow_pickle=False)
        prediction = np.load(case / "reconstruction.npy", allow_pickle=False)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, value, title in zip(axes, (anchor, correction, prediction), ("calibrated Mean anchor", "effective correction", "final")):
            scale = float(np.quantile(np.abs(value), .999))
            if title == "effective correction":
                ax.imshow(value.sum(0), cmap="coolwarm", vmin=-scale, vmax=scale)
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            else: _show(ax, value.max(0), scale, title)
        fig.savefig(output / f"{sample}_subset01_anchor_correction.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, sample in zip(axes, ("T02", "T03", "T04")):
        for method in ("e3_mean100", exp.ARM):
            profiles = []
            for subset in range(1, 11):
                volumes, _ = _case_volumes(sample, subset)
                profiles.append(_normalized(volumes[method]).sum((1, 2)))
            values = np.stack(profiles); mean = values.mean(0); std = values.std(0)
            z = np.arange(10, 101, 10); ax.plot(z, mean, label=LABELS[method]); ax.fill_between(z, mean - std, mean + std, alpha=.2)
        ax.set_title(sample); ax.set_xlabel("depth (um)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("mass fraction, mean ± std"); axes[-1].legend(fontsize=8)
    fig.savefig(output / "network_subset_axial_stability.png", dpi=170)
    plt.close(fig)
    return len(list(output.glob("*.png")))


def build_report() -> dict[str, Any]:
    stability, quality = stability_tables()
    quality_summary = _quality_summary(quality)
    analysis = exp.OUTPUT / "analysis"
    write_csv(analysis / "stability_per_object.csv", stability)
    write_csv(analysis / "quality_per_subset.csv", quality)
    write_csv(analysis / "quality_summary.csv", quality_summary)
    count = figures()
    stab = {(row["sample_id"], row["method"]): row for row in stability}
    qual = {(row["sample_id"], row["method"]): row for row in quality_summary}
    improved_objects = sum(
        stab[(sample, exp.ARM)]["shape_relative_dispersion"] < stab[(sample, "e3_mean100")]["shape_relative_dispersion"]
        for sample in ("T02", "T03", "T04")
    )
    stable = improved_objects >= 2 and stab[("object_macro", exp.ARM)]["shape_relative_dispersion"] < stab[("object_macro", "e3_mean100")]["shape_relative_dispersion"]
    quality_ok = all(
        qual[("object_macro", exp.ARM)][key] <= qual[("object_macro", "e3_mean100")][key]
        for key in ("gt_scale_aligned_nrmse", "gt_axial_w1_um", "background_xy_mass_fraction")
    )
    verdict = "supports_mean_anchor" if stable and quality_ok else ("mixed" if stable else "does_not_support_mean_anchor")
    lines = [
        "# Mean 主重建分支对照实验", "",
        "主比较为当前 Taylor-anchor E3＋mean≤100% 第400步 final 与新 Mean-anchor 第400步 final。", "",
        "| 方法 | 子集形状离散度↓ | 子集间轴向W1 μm↓ | aligned NRMSE↓ | GT轴向W1 μm↓ | 背景质量↓ |", "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        s = stab[("object_macro", method)]; q = qual[("object_macro", method)]
        lines.append(f"| {LABELS[method]} | {s['shape_relative_dispersion']:.4f} | {s['pairwise_axial_w1_um']:.2f} | {q['gt_scale_aligned_nrmse']:.4f} | {q['gt_axial_w1_um']:.2f} | {q['background_xy_mass_fraction']:.4f} |")
    lines += ["", f"稳定性改善对象数：{improved_objects}/3；预设判定：`{verdict}`。", "",
              "结论只针对当前单种子、10帧RL3和400步预算。图像按完整重建体统一尺度显示，原生层不逐层归一化。", "",
              "- `analysis/stability_per_object.csv`：逐对象稳定性。",
              "- `analysis/quality_per_subset.csv`：逐子集质量。",
              "- `analysis/figures/`：投影、原生层、共同尺度、基础体和修正图。"]
    (exp.OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"complete": True, "verdict": verdict, "stability_improved_objects": improved_objects, "stability_passed": stable, "quality_passed": quality_ok, "figures": count}
    old.write_json(exp.OUTPUT / "analysis_complete.json", result)
    return result

