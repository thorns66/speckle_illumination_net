"""CPU-only paired report for the frozen sqrt Set-branch ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch
import yaml

from tools.run_no_set_ablation import BASELINE, normalized_config
from train_volume import _model_from_config
from training.global_batch_schedule import FixedGlobalBatchScheduler
from utils.reconstruction_metrics import reconstruction_metrics
from utils.display_normalization import normalize_display_volume


METRICS = {
    "gt_raw_nrmse": "原始 3D NRMSE ↓",
    "gt_scale_aligned_nrmse": "尺度对齐 3D NRMSE ↓",
    "gt_axial_w1_um": "轴向 W1 (µm) ↓",
    "gt_axial_mass_l1": "轴向质量 L1 ↓",
    "gt_support_outside_pm10_mass": "支持区外质量比例 ↓",
    "gt_xy_mip_ssim": "XY MIP SSIM ↑",
    "selection_score": "物理选模分数 ↓",
}
IDS = {"step", "split", "sample_id", "subset_index"}
OBJECTS = {"validation": ("P09", "V01", "V02"), "test": ("P07", "T01", "T02")}


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in {"step", "subset_index"}:
                row[key] = int(value)
            elif key not in IDS:
                row[key] = float(value)
                if not np.isfinite(row[key]):
                    raise ValueError(f"Non-finite metric in {path}: {key}")
    return rows


def object_means(rows: list[dict]) -> dict[str, dict]:
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation")
    metrics = sorted(set(rows[0]) - IDS)
    result = {}
    for sample in sorted({row["sample_id"] for row in rows}):
        items = [row for row in rows if row["sample_id"] == sample]
        result[sample] = {
            key: float(np.mean([row[key] for row in items])) for key in metrics
        }
    return result


def macro(rows: list[dict]) -> dict:
    objects = object_means(rows)
    return {
        key: float(np.mean([row[key] for row in objects.values()]))
        for key in next(iter(objects.values()))
    }


def paired_rows(
    reference: list[dict], candidate: list[dict], comparison: str
) -> list[dict]:
    def indexed(rows):
        result = {(row["sample_id"], row["subset_index"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired evaluation key")
        return result

    left, right = indexed(reference), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Paired comparison has mismatched objects/subsets")
    result = []
    for key in sorted(left):
        row = {
            "comparison": comparison,
            "sample_id": key[0],
            "subset_index": key[1],
            "with_set_step": left[key]["step"],
            "no_set_step": right[key]["step"],
        }
        for metric in sorted(set(left[key]) - IDS):
            a, b = left[key][metric], right[key][metric]
            row.update(
                {
                    f"{metric}_with_set": a,
                    f"{metric}_no_set": b,
                    f"{metric}_delta_no_set_minus_with_set": b - a,
                }
            )
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_run(path: Path) -> dict:
    complete = json.loads((path / "training_complete.json").read_text())
    if not complete.get("complete") or complete["completed_steps"] != 200:
        raise ValueError(f"Incomplete 200-step run: {path}")
    config = yaml.safe_load((path / "config_used.yaml").read_text())
    contract = json.loads((path / "run_contract.json").read_text())
    validation = read_metrics(path / "validation_metrics.csv")
    test = read_metrics(path / "test_metrics.csv")
    with (path / "training_metrics.csv").open(newline="") as handle:
        training = list(csv.DictReader(handle))
    if [int(row["step"]) for row in training] != list(range(1, 201)):
        raise ValueError("Expected exactly 200 consecutive training rows")
    for split, rows, steps in (
        ("validation", validation, list(range(20, 201, 20))),
        ("test", test, [complete["best_step"]]),
    ):
        actual = {(row["step"], row["sample_id"], row["subset_index"]) for row in rows}
        expected = {
            (step, sample, subset)
            for step in steps
            for sample in OBJECTS[split]
            for subset in range(1, 11)
        }
        if (
            actual != expected
            or len(actual) != len(rows)
            or any(row["split"] != split for row in rows)
        ):
            raise ValueError(f"Incomplete or duplicate {split} evaluation")
        for step, sample, subset in sorted(actual):
            folder = path / split / f"step_{step:06d}" / sample / f"subset_{subset:02d}"
            volume = np.load(folder / "reconstruction.npy", allow_pickle=False)
            if (
                volume.shape != (10, 260, 260)
                or not np.isfinite(volume).all()
                or np.any(volume < 0)
            ):
                raise ValueError(f"Invalid reconstruction: {folder}")
            if not (folder / "reconstruction.tif").is_file():
                raise FileNotFoundError(folder / "reconstruction.tif")
    trajectory = {
        step: macro([row for row in validation if row["step"] == step])
        for step in range(20, 201, 20)
    }
    selected = min(trajectory, key=lambda step: trajectory[step]["selection_score"])
    if selected != complete["best_step"] or not np.isclose(
        trajectory[selected]["selection_score"],
        complete["best_validation_score"],
        rtol=1e-7,
    ):
        raise ValueError(
            "Best checkpoint does not follow the frozen physical selection rule"
        )
    for name, step in (("checkpoint_best.pt", selected), ("checkpoint_last.pt", 200)):
        checkpoint = torch.load(path / name, map_location="cpu", weights_only=False)
        if (
            checkpoint["completed_steps"] != step
            or checkpoint["dataset_fingerprint"] != contract["dataset_fingerprint"]
        ):
            raise ValueError(f"Checkpoint provenance mismatch: {name}")
        if checkpoint["config"]["ablation"] != config["ablation"]:
            raise ValueError("Checkpoint ablation flags differ from config_used")
        scheduler = FixedGlobalBatchScheduler(80, 8, int(config["experiment"]["seed"]))
        for _ in range(step):
            scheduler.next_batch()
        saved_schedule = checkpoint["scheduler_state"]
        if (
            scheduler.cursor != saved_schedule["cursor"]
            or scheduler.epoch != saved_schedule["epoch"]
            or not np.array_equal(scheduler.order, saved_schedule["order"])
            or scheduler.rng.bit_generator.state != saved_schedule["rng_state"]
        ):
            raise ValueError("Checkpoint does not match the frozen batch schedule")
        if not config["ablation"]["use_set_branch"]:
            torch.manual_seed(int(config["experiment"]["seed"]))
            initial = _model_from_config(config).state_dict()
            for parameter_name in contract["parameter_contract"][
                "frozen_parameter_names"
            ]:
                if not torch.equal(
                    initial[parameter_name], checkpoint["model_state"][parameter_name]
                ):
                    raise ValueError(
                        f"Bypassed parameter changed during training: {parameter_name}"
                    )
    return dict(
        path=path,
        complete=complete,
        config=config,
        contract=contract,
        validation=validation,
        test=test,
        trajectory=trajectory,
        training=training,
    )


def metric_table(left: dict, right: dict) -> list[str]:
    lines = ["|指标|有 Set|无 Set|无 Set − 有 Set|", "|---|---:|---:|---:|"]
    for key, label in METRICS.items():
        lines.append(
            f"|{label}|{left[key]:.6f}|{right[key]:.6f}|{right[key] - left[key]:+.6f}|"
        )
    return lines


def verdict(left: list[dict], right: list[dict]) -> dict:
    a, b = macro(left), macro(right)
    ao, bo = object_means(left), object_means(right)
    primary = ("gt_axial_w1_um", "gt_scale_aligned_nrmse")
    with_wins = sum(all(ao[obj][key] < bo[obj][key] for key in primary) for obj in ao)
    without_wins = sum(
        all(bo[obj][key] < ao[obj][key] for key in primary) for obj in ao
    )
    if all(a[key] < b[key] for key in primary) and with_wins >= 2:
        label = "本次单种子实验的两项主指标支持 Set 通路有正向作用"
    elif all(b[key] < a[key] for key in primary) and without_wins >= 2:
        label = "本次单种子实验的两项主指标支持无 Set；不据此自动替换基线"
    else:
        label = "本次结果为混合收益，不能认定 Set 整体改善重建"
    return {
        "label": label,
        "objects_both_metrics_favor_with_set": with_wins,
        "objects_both_metrics_favor_no_set": without_wins,
    }


def load_truth(root: Path, sample: str) -> np.ndarray:
    with h5py.File(root / sample / "prepared.mat", "r") as handle:
        return np.asarray(handle["ground_truth"][()], dtype=np.float32).transpose(
            0, 2, 1
        )


def load_prediction(run: dict, row: dict) -> np.ndarray:
    return np.load(
        run["path"]
        / row["split"]
        / f"step_{row['step']:06d}"
        / row["sample_id"]
        / f"subset_{row['subset_index']:02d}"
        / "reconstruction.npy",
        allow_pickle=False,
    )


def recheck_metrics(run: dict) -> dict:
    root = Path(run["contract"]["dataset_root"])
    truths = {
        sample: load_truth(root, sample)
        for samples in OBJECTS.values()
        for sample in samples
    }
    maximum_error = {}
    for row in run["validation"] + run["test"]:
        actual = reconstruction_metrics(
            load_prediction(run, row), truths[row["sample_id"]], np.arange(10, 101, 10)
        )
        for metric, value in actual.items():
            error = abs(value - row[metric])
            maximum_error[metric] = max(maximum_error.get(metric, 0), error)
            if not np.isclose(value, row[metric], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"Saved reconstruction/metric mismatch: {run['path']}, {row['sample_id']}, {metric}"
                )
    for metric, value in macro(run["test"]).items():
        if not np.isclose(value, run["complete"]["test"][metric], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Test summary is not object-macro: {metric}")
    return {"recomputed_volumes": 330, "maximum_metric_absolute_error": maximum_error}


def plot_trajectory(reference: dict, candidate: dict, destination: Path) -> None:
    panels = [
        ("selection_score", "Validation physical score"),
        ("gt_scale_aligned_nrmse", "Scale-aligned 3D NRMSE"),
        ("gt_axial_w1_um", "Axial W1 (um)"),
        ("gt_support_outside_pm10_mass", "Mass outside support +/-10 um"),
        ("gt_xy_mip_ssim", "XY MIP SSIM"),
        ("gt_raw_nrmse", "Raw 3D NRMSE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for run, label, color in (
            (reference, "With Set", "#2266aa"),
            (candidate, "No Set", "#cc7733"),
        ):
            trajectory = run["trajectory"]
            steps = sorted(trajectory)
            axis.plot(
                steps,
                [trajectory[step][metric] for step in steps],
                "o-",
                label=label,
                color=color,
                markersize=3,
            )
            best = run["complete"]["best_step"]
            axis.scatter(
                [best],
                [trajectory[best][metric]],
                s=95,
                marker="*",
                color=color,
                zorder=4,
            )
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(
        "Object-macro validation trajectory; stars = physical-selected checkpoints"
    )
    fig.savefig(destination / "validation_trajectory.png", dpi=160)
    plt.close(fig)


def plot_object(
    reference: dict,
    candidate: dict,
    sample: str,
    destination: Path,
    *,
    aligned: bool,
    normalized: bool = False,
    gamma: float = 1.0,
    subset_index: int = 1,
) -> dict:
    if aligned and normalized:
        raise ValueError(
            "GT-fitted alignment and independent display normalization are distinct modes"
        )
    if not np.isfinite(gamma) or gamma <= 0 or (not normalized and gamma != 1):
        raise ValueError(
            "Positive display gamma is supported only with normalized display"
        )
    truth = load_truth(Path(reference["contract"]["dataset_root"]), sample)
    arrays = [truth]
    gains = [1.0]
    for run in (reference, candidate):
        row = next(
            row
            for row in run["test"]
            if row["sample_id"] == sample and row["subset_index"] == subset_index
        )
        volume = load_prediction(run, row).astype(np.float64)
        gain = (
            float(
                np.vdot(volume.ravel(), truth.ravel())
                / max(np.vdot(volume.ravel(), volume.ravel()), 1e-12)
            )
            if aligned
            else 1.0
        )
        arrays.append(volume * gain)
        gains.append(gain)
    source_arrays = arrays
    maxima = None
    if normalized:
        normalized_pairs = [normalize_display_volume(value) for value in source_arrays]
        arrays = [item[0] for item in normalized_pairs]
        maxima = [item[1] for item in normalized_pairs]
        gains = [1.0 / value if value > 0 else 1.0 for value in maxima]
    vmax = max(float(value.max()) for value in arrays)
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=1) if normalized else None
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[3, 1.1, 1.1, 1.7])
    labels = [
        "Ground truth",
        f"With Set (step {reference['complete']['best_step']})",
        f"No Set (step {candidate['complete']['best_step']})",
    ]
    projection_axes = []
    for column, (volume, label) in enumerate(zip(arrays, labels)):
        for row, axis_to_max in enumerate((0, 1, 2)):
            axis = fig.add_subplot(grid[row, column])
            projection_axes.append(axis)
            projection = volume.max(axis=axis_to_max)
            kwargs = (
                {}
                if row == 0
                else {"extent": [0, projection.shape[1], 105, 5], "aspect": "auto"}
            )
            color_options = {"norm": norm} if normalized else {"vmin": 0, "vmax": vmax}
            plotted = axis.imshow(projection, cmap="magma", **color_options, **kwargs)
            if row == 0:
                axis.set_title(label)
            axis.set_ylabel(("XY: y (pixel)", "XZ: z (um)", "YZ: z (um)")[row])
            axis.set_xlabel("y (pixel)" if row == 2 else "x (pixel)")
    fig.colorbar(
        plotted,
        ax=projection_axes,
        shrink=0.6,
        label=f"Normalized intensity (display gamma={gamma:g})"
        if normalized
        else "Shared intensity scale",
    )
    axis = fig.add_subplot(grid[3, :])
    for volume, label, color in zip(
        source_arrays, labels, ("#333333", "#2266aa", "#cc7733")
    ):
        mass = volume.sum(axis=(1, 2))
        axis.plot(
            np.arange(10, 101, 10),
            mass / max(mass.sum(), 1e-12),
            "o-",
            label=label,
            color=color,
        )
    axis.set(xlabel="Depth (um)", ylabel="Fraction of total volume mass")
    axis.grid(alpha=0.25)
    axis.legend()
    mode = "scale_aligned" if aligned else "shared_intensity"
    subtitle = (
        "One GT-fitted scalar per WHOLE volume; no per-layer normalization"
        if aligned
        else "Native intensity; identical color scale across methods and projections"
    )
    if normalized:
        mode = (
            "normalized_linear"
            if gamma == 1
            else f"normalized_gamma{gamma:g}".replace(".", "p")
        )
        subtitle = f"Independent whole-volume max normalization; common display gamma={gamma:g}\nNo per-layer normalization; absolute brightness is not comparable"
    fig.suptitle(f"{sample}, subset {subset_index:02d}\n{subtitle}")
    filename = f"test_{sample}_subset{subset_index:02d}_{mode}.png"
    fig.savefig(destination / filename, dpi=160)
    plt.close(fig)
    return {
        "sample_id": sample,
        "subset_index": subset_index,
        "mode": mode,
        "gains_gt_with_set_no_set": gains,
        "vmax": vmax,
        "file": filename,
        "whole_volume_maxima": maxima,
        "display_gamma": gamma,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    run_path = args.run.resolve()
    if not re.fullmatch(r".+_\d{8}_run\d{2,}", run_path.name):
        raise ValueError("New experiment output must include YYYYMMDD and runNN")
    reference = validate_run(args.baseline.resolve())
    candidate = validate_run(run_path)
    if normalized_config(reference["config"]) != normalized_config(candidate["config"]):
        raise ValueError("Effective configurations differ beyond Set")
    if (
        not reference["config"]["ablation"]["use_set_branch"]
        or candidate["config"]["ablation"]["use_set_branch"]
    ):
        raise ValueError("Expected with-Set reference and no-Set candidate")
    for key in (
        "dataset_fingerprint",
        "selected_psf_cache",
        "phase_chunk_size",
        "train_objects",
        "validation_objects",
        "test_objects",
    ):
        if reference["contract"][key] != candidate["contract"][key]:
            raise ValueError(f"Run contract mismatch: {key}")
    protocol = json.loads((run_path / "comparison_protocol.json").read_text())
    for filename, expected in protocol["source_sha256"].items():
        actual = hashlib.sha256(
            (run_path / "source_snapshot" / filename).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Source snapshot changed: {filename}")
    audits = {}
    for label, run in (("with_set", reference), ("no_set", candidate)):
        print(f"Rechecking all 330 saved reconstruction metrics: {label}", flush=True)
        audits[label] = recheck_metrics(run)
    pairs = {
        "validation_physical_best": (
            [
                row
                for row in reference["validation"]
                if row["step"] == reference["complete"]["best_step"]
            ],
            [
                row
                for row in candidate["validation"]
                if row["step"] == candidate["complete"]["best_step"]
            ],
        ),
        "validation_step200": (
            [row for row in reference["validation"] if row["step"] == 200],
            [row for row in candidate["validation"] if row["step"] == 200],
        ),
        "test_physical_best": (reference["test"], candidate["test"]),
    }
    subset_rows, object_rows, macro_rows = [], [], []
    summaries = {}
    for label, (left, right) in pairs.items():
        subset_rows.extend(paired_rows(left, right, label))
        a, b = object_means(left), object_means(right)
        summaries[label] = {"with_set": macro(left), "no_set": macro(right)}
        for sample in a:
            object_rows.append(
                {
                    "comparison": label,
                    "sample_id": sample,
                    **{
                        f"{metric}_{method}": value
                        for metric in a[sample]
                        for method, value in (
                            ("with_set", a[sample][metric]),
                            ("no_set", b[sample][metric]),
                            (
                                "delta_no_set_minus_with_set",
                                b[sample][metric] - a[sample][metric],
                            ),
                        )
                    },
                }
            )
        for method in ("with_set", "no_set"):
            macro_rows.append(
                {"comparison": label, "method": method, **summaries[label][method]}
            )
        macro_rows.append(
            {
                "comparison": label,
                "method": "delta_no_set_minus_with_set",
                **{
                    key: summaries[label]["no_set"][key]
                    - summaries[label]["with_set"][key]
                    for key in summaries[label]["with_set"]
                },
            }
        )
    decision = verdict(reference["test"], candidate["test"])
    primary = ("gt_axial_w1_um", "gt_scale_aligned_nrmse")
    test_summary, final_summary = (
        summaries["test_physical_best"],
        summaries["validation_step200"],
    )
    conflicts = [
        key
        for key in primary
        if (test_summary["no_set"][key] - test_summary["with_set"][key])
        * (final_summary["no_set"][key] - final_summary["with_set"][key])
        < 0
    ]
    decision["fixed_step_validation_direction_conflicts"] = conflicts
    overall = decision["label"] if not conflicts else "主测试比较与固定第 200 步验证方向不一致，整体属于混合证据"
    decision["overall_conclusion"] = overall
    destination = run_path / "comparison_with_set_baseline"
    destination.mkdir(exist_ok=False)
    write_csv(destination / "paired_subsets.csv", subset_rows)
    write_csv(destination / "paired_objects.csv", object_rows)
    write_csv(destination / "macro_comparisons.csv", macro_rows)
    trajectory_rows = []
    for label, run in (("with_set", reference), ("no_set", candidate)):
        for step, values in run["trajectory"].items():
            trajectory_rows.append({"method": label, "step": step, **values})
    write_csv(destination / "validation_trajectory.csv", trajectory_rows)
    plot_trajectory(reference, candidate, destination)
    display = []
    for sample in OBJECTS["test"]:
        for aligned in (False, True):
            display.append(
                plot_object(reference, candidate, sample, destination, aligned=aligned)
            )
        for gamma in (1.0, 0.5):
            display.append(
                plot_object(
                    reference,
                    candidate,
                    sample,
                    destination,
                    aligned=False,
                    normalized=True,
                    gamma=gamma,
                )
            )
    parameter_contract = candidate["contract"]["parameter_contract"]
    parameter_reduction = 100 * (
        1
        - parameter_contract["trainable_parameters"]
        / parameter_contract["total_parameters"]
    )
    lines = [
        "# 无 Set 消融：与 sqrt＋Set 基线完整比较",
        "",
        "## 结论",
        "",
        f"**{overall}。**",
        "",
        "以上为预设主指标判定，不等同于分辨率或单层质量验收；基线选型以项目训练手册中的用户决定为准，不能仅凭主指标自动替换。",
        "",
        f"按各自物理最佳 checkpoint 的测试主比较：{decision['label']}。三个测试对象中，"
        f"{decision['objects_both_metrics_favor_with_set']}/3 在 W1 和对齐 NRMSE 上同时支持有 Set，"
        f"{decision['objects_both_metrics_favor_no_set']}/3 同时支持无 Set。",
        "",
        "## 实验口径",
        "",
        "- 两组均为 sqrt 特征＋sqrt 锚点；10 帧/RL3，90 帧方差约束，seed=20260901，global batch=8，200 步。",
        "- 仅关闭 Set 及其专属投影/门控修正，不改变 VAR、Mean、解码器和 β；公共初始权重及随机数状态一致。",
        f"- 有 Set 最佳 step={reference['complete']['best_step']}；无 Set 最佳 step={candidate['complete']['best_step']}。均按验证物理分数选择，不按 GT 或测试指标选择。",
        f"- 基线使用 {reference['contract']['world_size']} 张 GPU；本次使用 {candidate['contract']['world_size']} 张，物理编号 {candidate['contract']['physical_gpus']}。全局 batch 和样本顺序不变，但不同并行归约仍可能带来浮点差异。",
        f"- 有效训练参数：{parameter_contract['total_parameters']:,} → {parameter_contract['trainable_parameters']:,}（减少 {parameter_reduction:.2f}%）；未使用模块保留构造但不计算、不训练。",
        f"- 本次预检峰值显存 {candidate['contract']['preflight_peak_memory_gib']:.2f} GiB；卡数不同，不据此声称严格的速度收益。",
        "- 所有指标先对子集平均，再对对象等权平均；同一对象的十个子集不是十个独立对象。",
        "",
    ]
    titles = {
        "validation_physical_best": "物理最佳点：验证集",
        "validation_step200": "相同第 200 步：验证集",
        "test_physical_best": "物理最佳点：测试集",
    }
    for label, (left, right) in pairs.items():
        lines.extend(
            [
                f"## {titles[label]}",
                "",
                *metric_table(macro(left), macro(right)),
                "",
                "### 逐对象",
                "",
                "|对象|对齐 NRMSE：有→无|W1 µm：有→无|支持区外比例：有→无|SSIM：有→无|",
                "|---|---:|---:|---:|---:|",
            ]
        )
        a, b = object_means(left), object_means(right)
        for sample in a:
            values = [
                f"{a[sample][key]:.5f} → {b[sample][key]:.5f}"
                for key in (
                    "gt_scale_aligned_nrmse",
                    "gt_axial_w1_um",
                    "gt_support_outside_pm10_mass",
                    "gt_xy_mip_ssim",
                )
            ]
            lines.append("|" + "|".join([sample, *values]) + "|")
        lines.append("")
    lines.extend(
        [
            "## 验证轨迹",
            "",
            "星号表示物理选模点，其他 GT 曲线仅作诊断，不据此重选测试模型。",
            "",
            "![验证曲线](validation_trajectory.png)",
            "",
            "## 固定测试子集的可视化",
            "",
            "每个对象预先固定 subset_01，没有挑选最优子集。先显示全体积最大值归一化的清晰版（统一显示 gamma=0.5）及线性版；各方法各自归一化到0–1，不能比较绝对亮度。后面保留原始强度和 GT 尺度对齐审计图。均未逐层归一化，gamma仅作用于颜色映射，不改动重建体、深度曲线、输入或评价指标。",
            "",
        ]
    )
    for sample in OBJECTS["test"]:
        lines.extend(
            [
                f"### {sample}",
                "",
                f"![归一化清晰版](test_{sample}_subset01_normalized_gamma0p5.png)",
                "",
                f"![归一化线性版](test_{sample}_subset01_normalized_linear.png)",
                "",
                f"![原始强度](test_{sample}_subset01_shared_intensity.png)",
                "",
                f"![全体积尺度对齐](test_{sample}_subset01_scale_aligned.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 验收与限制",
            "",
            "- 两组各有 200 步训练、300 条验证和 30 条测试记录；330 个 NPY/TIFF 重建和最佳/最终权重完整。",
            "- 已从两组全部 660 个保存的 NPY 重算 GT 指标，并核对源代码快照哈希、数据指纹、PSF 及物理选模规则。",
            "- 全部指标（含各项物理损失）及差值在 paired_subsets.csv、paired_objects.csv、macro_comparisons.csv；轨迹在 validation_trajectory.csv。",
            "- 单种子、小样本结果不等于统计显著或普遍结论。只检验当前训练预算下整个 Set 通路的贡献，不能区分额外输入信息与模型容量的作用。",
            "- 测试对象已在前序实验中被查看，属于固定留出测试集复用，不应表述为全新、完全未触及的最终确认集。",
            "- 本轮不自动扩展 raw、RL5 或三随机种子实验。",
            "",
        ]
    )
    (destination / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "baseline": str(reference["path"]),
        "run": str(run_path),
        "decision": decision,
        "comparisons": summaries,
        "metric_rechecks": audits,
        "display_settings": display,
        "source_snapshot_verified": True,
        "report_generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), destination / "report_generator.py")
    print(
        json.dumps(
            {"report": str(destination / "report_zh.md"), "decision": decision},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
