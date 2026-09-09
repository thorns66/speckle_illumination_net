"""Aggregate V3 comparison results, draw actual reconstructions, and audit delivery."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets, load_inference_input
from tools import three_way_experiment as old
from tools import v3_compare_experiment as exp
from tools.v3_compare_evaluation import TABLES, all_cases, csv_write


LABELS = {
    "baseline": "A Baseline",
    "e3": "B E3",
    "e3_mean005": "C E3 + mean≤5%",
    "ground_truth": "GT",
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3",
}
ROLES = ("best", "final")
NETWORKS = tuple(exp.ARMS)
Z_UM = np.arange(10, 101, 10)


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _finite_mean(values) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan")


def _bool(value: Any) -> bool:
    return str(value).lower() in ("true", "1")


def _group_mean(rows: list[dict], keys: tuple[str, ...], metric: str) -> dict[tuple, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(_float(row, metric))
    return {key: _finite_mean(values) for key, values in groups.items()}


def aggregate_metrics(metrics: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    metric_names = (
        "gt_scale_aligned_nrmse",
        "gt_axial_w1_um",
        "gt_axial_mass_l1",
        "gt_support_outside_pm10_mass",
        "gt_xy_mip_ssim",
        "local_axial_w1_um",
        "background_xy_mass_fraction",
        "common_normalized_mean_loss",
        "common_absolute_log_variance_loss",
        "common_normalized_shape_variance_loss",
        "common_baseline_score",
        "common_e3_score",
        "prediction_max",
        "prediction_sum",
        "mass_ratio_to_truth",
    )
    official = [row for row in metrics if row["split"] in ("validation", "test")]
    object_rows = []
    for (method, split, sample), items in sorted(
        defaultdict(list, {
            key: [r for r in official if (r["method"], r["split"], r["sample_id"]) == key]
            for key in {(r["method"], r["split"], r["sample_id"]) for r in official}
        }).items()
    ):
        row = {"method": method, "split": split, "sample_id": sample, "subsets": len(items)}
        row.update({name: _finite_mean(_float(item, name) for item in items) for name in metric_names})
        object_rows.append(row)
    summary = []
    for method, split in sorted({(r["method"], r["split"]) for r in object_rows}):
        items = [row for row in object_rows if row["method"] == method and row["split"] == split]
        row = {"method": method, "split": split, "objects": len(items), "subsets": sum(int(x["subsets"]) for x in items)}
        row.update({name: _finite_mean(_float(item, name) for item in items) for name in metric_names})
        summary.append(row)
    return object_rows, summary


def _aggregate_local(tables: dict[str, list[dict[str, str]]]) -> list[dict]:
    rows = []
    t03 = [row for row in tables["t03_lines"] if abs(_float(row, "threshold_fraction") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in t03}):
        selected = [row for row in t03 if row["method"] == method]
        rows.append({
            "method": method, "diagnostic": "T03_three_lines", "items": len(selected),
            "localized_rate": _finite_mean(_bool(row["all_three_localized"]) for row in selected),
            "separated_rate": _finite_mean(_bool(row["separated"]) for row in selected),
            "false_break_rate": _finite_mean(int(float(row["false_break_count"])) > 0 for row in selected),
            "valley_ratio": _finite_mean(max(_float(row, "valley_ratio_1"), _float(row, "valley_ratio_2")) for row in selected),
        })
    t04 = [row for row in tables["t04_axial"] if abs(_float(row, "threshold_fraction") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in t04}):
        selected = [row for row in t04 if row["method"] == method]
        resolvable = [row for row in selected if 20 <= _float(row, "separation_um") <= 40 and row["family"] != "single_control"]
        rows.append({
            "method": method, "diagnostic": "T04_axial_pairs", "items": len(selected),
            "localized_rate": _finite_mean(_bool(row["all_layers_localized"]) for row in selected),
            "separated_rate_20_40um": _finite_mean(_bool(row["separated"]) for row in resolvable),
            "false_peaks_mean": _finite_mean(_float(row, "false_peak_count") for row in selected),
            "expected_layer_energy_fraction": _finite_mean(_float(row, "expected_layer_energy_fraction") for row in selected),
        })
    beads = [row for row in tables["v03_beads"] if abs(_float(row, "threshold_fraction") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in beads}):
        selected = [row for row in beads if row["method"] == method]
        rows.append({
            "method": method, "diagnostic": "V03_60_beads", "items": len(selected),
            "localized_rate": _finite_mean(_bool(row["detected"]) for row in selected),
            "mean_z_error_layers": _finite_mean(_float(row, "z_error_layers") for row in selected),
            "mean_xy_error_pixels": _finite_mean(_float(row, "xy_error_pixels") for row in selected),
            "global_false_peaks_mean": _finite_mean(_float(row, "global_false_peak_count") for row in selected),
        })
    points = [row for row in tables["point_targets"] if abs(_float(row, "threshold") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in points}):
        selected = [row for row in points if row["method"] == method]
        rows.append({
            "method": method, "diagnostic": "priority_points", "items": len(selected),
            "localized_rate": _finite_mean(_bool(row["matched"]) for row in selected),
            "mean_z_error_layers": _finite_mean(_float(row, "z_error_layers") for row in selected),
            "local_depth_w1_um": _finite_mean(_float(row, "local_depth_w1_um") for row in selected),
            "tail_mass": _finite_mean(_float(row, "tail_mass_outside_pm1") for row in selected),
        })
    pairs = [row for row in tables["point_pairs"] if abs(_float(row, "threshold") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in pairs}):
        selected = [row for row in pairs if row["method"] == method]
        rows.append({
            "method": method, "diagnostic": "priority_pairs", "items": len(selected),
            "localized_rate": _finite_mean(_bool(row["both_localized"]) for row in selected),
            "separated_rate": _finite_mean(_bool(row["separated"]) for row in selected),
            "valley_ratio": _finite_mean(_float(row, "valley_to_weaker_peak") for row in selected),
        })
    lines = [row for row in tables["priority_lines"] if abs(_float(row, "threshold") - 0.1) < 1e-9]
    for method in sorted({row["method"] for row in lines}):
        selected = [row for row in lines if row["method"] == method]
        continuous = [row for row in selected if _float(row, "gap_um") == 0]
        gapped = [row for row in selected if _float(row, "gap_um") > 0]
        rows.append({
            "method": method, "diagnostic": "priority_lines", "items": len(selected),
            "continuous_false_gap_rate": _finite_mean(_bool(row["false_gap"]) for row in continuous),
            "gapped_bridge_rate": _finite_mean(_bool(row["bridged"]) for row in gapped),
            "endpoint_retention": _finite_mean(_float(row, "endpoint_retention_fraction") for row in selected),
        })
    return rows


def _load_volume_set(case: dict, role: str) -> dict[str, np.ndarray]:
    raw = load_inference_input(case["path"], case["subset"], var_feature_representation="sqrt")
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    target = _read_targets(key, include_ground_truth=case["split"] != "zero")
    result = {
        "mean_rl3": raw["g_mean"][0],
        "taylor_rl3": raw["f_var"][0],
    }
    if "ground_truth" in target:
        result["ground_truth"] = target["ground_truth"][0]
    for arm in NETWORKS:
        result[arm] = np.load(
            exp.OUTPUT / "evaluation" / arm / role / case["id"] / "reconstruction.npy",
            allow_pickle=False,
        )
    return result


def _safe_normalize(volume: np.ndarray, scale: float | None = None) -> np.ndarray:
    value = np.maximum(np.asarray(volume, np.float32), 0)
    denominator = max(float(value.max()) if scale is None else float(scale), 1e-30)
    return np.clip(value / denominator, 0, 1)


def _draw_projection_panel(case: dict, role: str, display: str, destination: Path) -> dict:
    volumes = _load_volume_set(case, role)
    methods = [name for name in ("ground_truth", "mean_rl3", "taylor_rl3", *NETWORKS) if name in volumes]
    if display == "shape":
        shown = {name: _safe_normalize(volumes[name]) for name in methods}
        subtitle = "Each complete 3D volume uses one global maximum"
    else:
        prediction_methods = [name for name in methods if name != "ground_truth"]
        shared = max(float(np.maximum(volumes[name], 0).max()) for name in prediction_methods)
        shown = {name: _safe_normalize(volumes[name], None if name == "ground_truth" else shared) for name in methods}
        subtitle = "All predictions share one brightness scale; GT uses its own scale"
    figure, axes = plt.subplots(2, len(methods), figsize=(3.0 * len(methods), 6.1), constrained_layout=True)
    for column, name in enumerate(methods):
        xy = shown[name].max(axis=0)
        xz = shown[name].max(axis=1)
        axes[0, column].imshow(xy, cmap="magma", vmin=0, vmax=1, origin="lower")
        axes[1, column].imshow(xz, cmap="magma", vmin=0, vmax=1, origin="lower", aspect="auto",
                               extent=(0, xy.shape[1], 10, 100))
        axes[0, column].set_title(LABELS.get(name, name))
        axes[0, column].set_xticks([]); axes[0, column].set_yticks([])
        axes[1, column].set_xlabel("X pixel")
        if column == 0:
            axes[0, column].set_ylabel("Y pixel")
            axes[1, column].set_ylabel("Depth (µm)")
        else:
            axes[1, column].set_yticklabels([])
    figure.suptitle(f"{case['id']} · {role} · {display}\n{subtitle}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return {"case_id": case["id"], "role": role, "display": display,
            "path": str(destination.relative_to(exp.OUTPUT)), "methods": methods,
            "sha256": old.sha256(destination)}


def _draw_all_layers(case: dict, role: str, destination: Path) -> dict:
    volumes = _load_volume_set(case, role)
    methods = [name for name in ("ground_truth", "mean_rl3", "taylor_rl3", *NETWORKS) if name in volumes]
    shared = max(float(np.maximum(volumes[name], 0).max()) for name in methods if name != "ground_truth")
    figure, axes = plt.subplots(len(methods), 10, figsize=(22, 2.15 * len(methods)), constrained_layout=True)
    for row, name in enumerate(methods):
        normalized = _safe_normalize(volumes[name], None if name == "ground_truth" else shared)
        for layer in range(10):
            axes[row, layer].imshow(normalized[layer], cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[row, layer].set_xticks([]); axes[row, layer].set_yticks([])
            if row == 0:
                axes[row, layer].set_title(f"{Z_UM[layer]} µm")
            if layer == 0:
                axes[row, layer].set_ylabel(LABELS.get(name, name))
    figure.suptitle(f"{case['id']} · {role} · all ten native layers · shared prediction brightness")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)
    return {"case_id": case["id"], "role": role, "display": "all_layers_shared",
            "path": str(destination.relative_to(exp.OUTPUT)), "methods": methods,
            "sha256": old.sha256(destination)}


def _training_plot(destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for arm in NETWORKS:
        train = csv_read(exp.OUTPUT / arm / "training_metrics.csv")
        axes[0].plot([int(row["step"]) for row in train], [_float(row, "total_loss") for row in train], label=LABELS[arm], alpha=.85)
        validation = csv_read(exp.OUTPUT / arm / "validation_metrics.csv")
        grouped = defaultdict(list)
        for row in validation:
            grouped[int(row["step"])].append(_float(row, "selection_score"))
        axes[1].plot(sorted(grouped), [_finite_mean(grouped[step]) for step in sorted(grouped)], marker="o", ms=3, label=LABELS[arm])
    for arm, color in (("e3", "tab:orange"), ("e3_mean005", "tab:green")):
        records = []
        for path in sorted((exp.OUTPUT / arm).glob("gradient_diagnostics_rank*.jsonl")):
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        records = [row for row in records if row.get("phase") == "training"]
        grouped = defaultdict(list)
        for row in records:
            grouped[int(row["update_step"])].append(float(row["shape_gradient_ratio"]))
        if grouped:
            axes[2].plot(sorted(grouped), [_finite_mean(grouped[step]) for step in sorted(grouped)], label=LABELS[arm], color=color)
    axes[0].axvline(200, color="k", ls="--", lw=1); axes[1].axvline(200, color="k", ls="--", lw=1); axes[2].axvline(50, color="k", ls="--", lw=1)
    axes[0].set(title="Training objective", xlabel="Step", ylabel="Loss")
    axes[1].set(title="Validation selection score", xlabel="Step", ylabel="Object-macro score")
    axes[2].set(title="Actual mean/variance q-gradient ratio", xlabel="Step", ylabel="Ratio")
    for axis in axes: axis.grid(alpha=.25); axis.legend(fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _metric_plot(summary: list[dict], local: list[dict], destination: Path) -> None:
    methods = [f"{arm}_final" for arm in NETWORKS]
    test = {row["method"]: row for row in summary if row["split"] == "test"}
    t03 = {row["method"]: row for row in local if row["diagnostic"] == "T03_three_lines"}
    t04 = {row["method"]: row for row in local if row["diagnostic"] == "T04_axial_pairs"}
    values = [
        ("Test aligned NRMSE ↓", [_float(test[m], "gt_scale_aligned_nrmse") for m in methods]),
        ("Test axial W1 / 100 ↓", [_float(test[m], "gt_axial_w1_um") / 100 for m in methods]),
        ("T03 separated ↑", [_float(t03[m], "separated_rate") for m in methods]),
        ("T04 20–40 µm separated ↑", [_float(t04[m], "separated_rate_20_40um") for m in methods]),
    ]
    x = np.arange(len(values)); width = .24
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for index, method in enumerate(methods):
        axis.bar(x + (index - 1) * width, [entry[1][index] for entry in values], width, label=LABELS[method.rsplit("_", 1)[0]])
    axis.set_xticks(x, [entry[0] for entry in values], rotation=12, ha="right")
    axis.grid(axis="y", alpha=.25); axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _delta(value_c: float, value_b: float, *, higher: bool = False) -> tuple[float, str]:
    change = value_c - value_b
    good = change > 0 if higher else change < 0
    return change, "改善" if good else "退步" if change else "持平"


def _write_report(summary: list[dict], object_rows: list[dict], local: list[dict], figures: list[dict]) -> None:
    lookup = {(row["method"], row["split"]): row for row in summary}
    local_lookup = {(row["method"], row["diagnostic"]): row for row in local}
    lines = [
        "# V3 新数据集三组 400 步对比报告",
        "",
        "本轮三组都从同一份随机初始化开始，样本顺序、全局 batch=8、训练步数和学习率日程一致。主要比较同为第 400 步的 final；best 只说明按各自无 GT 验证分数选出的模型。",
        "",
        "## 最直接的结果",
        "",
    ]
    for role in ROLES:
        lines += [f"### {role}", "", "| 方法 | 测试 aligned NRMSE ↓ | 测试轴向 W1 (µm) ↓ | 测试 XY-SSIM ↑ | 背景占比 ↓ | T03 三线分开率 ↑ | T04 20–40 µm 分开率 ↑ |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in NETWORKS:
            method = f"{arm}_{role}"
            test = lookup[(method, "test")]
            line = local_lookup[(method, "T03_three_lines")]
            axial = local_lookup[(method, "T04_axial_pairs")]
            lines.append(
                f"| {LABELS[arm]} | {_float(test, 'gt_scale_aligned_nrmse'):.4f} | {_float(test, 'gt_axial_w1_um'):.2f} | {_float(test, 'gt_xy_mip_ssim'):.4f} | {_float(test, 'background_xy_mass_fraction'):.3f} | {_float(line, 'separated_rate'):.3f} | {_float(axial, 'separated_rate_20_40um'):.3f} |"
            )
        lines.append("")
    b = lookup[("e3_final", "test")]
    c = lookup[("e3_mean005_final", "test")]
    nrmse_delta, nrmse_word = _delta(_float(c, "gt_scale_aligned_nrmse"), _float(b, "gt_scale_aligned_nrmse"))
    w1_delta, w1_word = _delta(_float(c, "gt_axial_w1_um"), _float(b, "gt_axial_w1_um"))
    bg_delta, bg_word = _delta(_float(c, "background_xy_mass_fraction"), _float(b, "background_xy_mass_fraction"))
    t03_b = local_lookup[("e3_final", "T03_three_lines")]
    t03_c = local_lookup[("e3_mean005_final", "T03_three_lines")]
    t04_b = local_lookup[("e3_final", "T04_axial_pairs")]
    t04_c = local_lookup[("e3_mean005_final", "T04_axial_pairs")]
    lines += [
        "## C 相对 B 到底有没有额外收益",
        "",
        f"在 30 个正式测试子集按对象等权汇总后，C 的 aligned NRMSE 相对 B 变化 {nrmse_delta:+.4f}（{nrmse_word}），轴向 W1 变化 {w1_delta:+.2f} µm（{w1_word}），背景占比变化 {bg_delta:+.3f}（{bg_word}）。",
        "",
        f"局部结构方面，T03 三线分开率从 {_float(t03_b, 'separated_rate'):.3f} 变为 {_float(t03_c, 'separated_rate'):.3f}；T04 的 20/30/40 µm 轴向双层线分开率从 {_float(t04_b, 'separated_rate_20_40um'):.3f} 变为 {_float(t04_c, 'separated_rate_20_40um'):.3f}。10 µm 轴向点对没有原生层间谷值，报告只统计定位和能量，不把它算成“分开”。",
        "",
        "这些数字若有一项变好、另一项变差，应按取舍理解，不能只凭一个平均分宣布全面获胜。",
        "",
        "## checkpoint 怎么看",
        "",
    ]
    for arm in NETWORKS:
        complete = json.loads((exp.OUTPUT / arm / "training_complete.json").read_text(encoding="utf-8"))
        best = lookup[(f"{arm}_best", "test")]
        final = lookup[(f"{arm}_final", "test")]
        lines.append(
            f"- {LABELS[arm]}：best 在第 {complete['best_step']} 步；best/final 的测试 aligned NRMSE 分别是 {_float(best, 'gt_scale_aligned_nrmse'):.4f}/{_float(final, 'gt_scale_aligned_nrmse'):.4f}，轴向 W1 分别是 {_float(best, 'gt_axial_w1_um'):.2f}/{_float(final, 'gt_axial_w1_um'):.2f} µm。"
        )
    lines += [
        "",
        "best 的选择没有看 GT：A 用原方差+TV；B/C 用同一套 E3 的 mean+归一化方差+TV，C 的 5% 结构 mean 项没有进入选模分数。",
        "",
        "## 图和原始数据",
        "",
        "下列图都来自实际保存的三维预测。shape 图对每个完整三维体只做一次全局归一化；shared 图让 Mean-RL3、Taylor-RL3 和三组网络共用同一亮度上限。没有逐层调亮，也没有按 GT 重新拟合预测亮度。",
        "",
    ]
    for figure in figures:
        if figure["display"] in ("shape", "shared"):
            relative = Path(figure["path"]).relative_to("analysis") if figure["path"].startswith("analysis/") else Path(figure["path"])
            lines.append(f"- [{figure['case_id']} · {figure['role']} · {figure['display']}](analysis/{relative.as_posix()})")
    lines += [
        "",
        "完整逐项结果见 [summary.csv](analysis/summary.csv)、[per_object.csv](analysis/per_object.csv)、[local_summary.csv](analysis/local_summary.csv)。每份原始预测和 anchor 位于 `evaluation/<实验>/<best|final>/<case>/`。",
        "",
        "## 这轮能说明到什么程度",
        "",
        "这是一个共同随机种子的固定条件对比。三组都严格跑了 400 步，足以回答这一次初始化下 E3 和 5% mean 结构梯度的作用；如果 C 与 B 差距很小，下一步应优先增加训练种子，而不是继续微调 5% 这个数字。照明 NA=0.05 与检测 PSF NA=0.15 沿用数据集生成设置；真实系统 NA 的对应关系仍按你的决定留待单独核对。",
    ]
    (exp.OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_training(arm: str) -> dict:
    folder = exp.OUTPUT / arm
    complete = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
    if not complete.get("complete") or int(complete["completed_steps"]) != 400:
        raise ValueError(f"{arm} did not complete 400 steps")
    train = csv_read(folder / "training_metrics.csv")
    validation = csv_read(folder / "validation_metrics.csv")
    if len(train) != 400 or {int(row["step"]) for row in train} != set(range(1, 401)):
        raise ValueError(f"{arm} training step inventory is incomplete")
    if len(validation) != 20 * 30:
        raise ValueError(f"{arm} validation inventory must be 20 checkpoints × 30 subsets")
    checkpoints = {}
    for role, filename, expected in (("step200", "checkpoint_step_000200.pt", 200), ("final", "checkpoint_last.pt", 400), ("best", "checkpoint_best.pt", None)):
        path = folder / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload["completed_steps"])
        if expected is not None and step != expected:
            raise ValueError(f"{arm}/{role} is step {step}, expected {expected}")
        if role == "best" and step != int(complete["best_step"]):
            raise ValueError(f"{arm} best checkpoint and training record disagree")
        checkpoints[role] = {"step": step, "sha256": old.sha256(path)}
    lr = []
    for path in sorted(folder.glob("learning_rate_rank*.jsonl")):
        lr.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    by_rank = defaultdict(list)
    for row in lr: by_rank[int(row["rank"])].append(row)
    if not by_rank or any(len(rows) != 400 for rows in by_rank.values()):
        raise ValueError(f"{arm} learning-rate logs do not contain 400 steps per rank")
    for rows in by_rank.values():
        for row in rows:
            step = int(row["step"]); used = row["used_lrs"]
            expected_network = 1e-3 if step <= 200 else 1e-4
            expected_scalar = 1e-4 if step <= 200 else 1e-5
            if not math.isclose(float(used["network"]), expected_network, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{arm} wrong network LR at step {step}")
            if not math.isclose(float(used["beta"]), expected_scalar, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{arm} wrong beta LR at step {step}")
    gradient_count = 0
    gradient_max = 0.0
    if arm != "baseline":
        gradients = []
        for path in sorted(folder.glob("gradient_diagnostics_rank*.jsonl")):
            gradients.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        gradients = [row for row in gradients if row.get("phase") == "training"]
        gradient_count = len(gradients)
        if gradient_count != 400 * 8:
            raise ValueError(f"{arm} expected 3200 per-sample gradient records, found {gradient_count}")
        for row in gradients:
            step = int(row["update_step"])
            budget = (0.05 if arm == "e3_mean005" else 0.0) * min(step / 50, 1)
            ratio = float(row["shape_gradient_ratio"])
            if not row["gradients_finite"] or not row["budget_passed"] or ratio > budget + max(1e-7, budget * 1e-5):
                raise ValueError(f"{arm} bounded gradient failed at step {step}")
            gradient_max = max(gradient_max, ratio)
    return {"best_step": int(complete["best_step"]), "checkpoints": checkpoints,
            "training_rows": len(train), "validation_rows": len(validation),
            "world_size": len(by_rank), "gradient_records": gradient_count,
            "max_gradient_ratio": gradient_max}


def build_report() -> dict:
    started = time.time()
    for arm in NETWORKS:
        marker = exp.OUTPUT / "evaluation" / arm / "complete.json"
        if not marker.is_file() or not json.loads(marker.read_text(encoding="utf-8")).get("complete"):
            raise ValueError(f"Evaluation incomplete: {arm}")
    tables = {name: [] for name in TABLES}
    for arm in NETWORKS:
        for name in TABLES:
            tables[name].extend(csv_read(exp.OUTPUT / "evaluation" / arm / f"{name}.csv"))
    from tools.v3_compare_local_audit import run as audit_local
    corrected = audit_local()
    for name in ("t03_lines", "t04_axial", "v03_beads"):
        tables[name] = corrected[name]
    if len(tables["metrics"]) != 3 * 2 * 94:
        raise ValueError(f"Expected 564 primary metric rows, found {len(tables['metrics'])}")
    analysis = exp.OUTPUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items(): csv_write(analysis / f"{name}.csv", rows)
    object_rows, summary = aggregate_metrics(tables["metrics"])
    local = _aggregate_local(tables)
    csv_write(analysis / "per_object.csv", object_rows)
    csv_write(analysis / "summary.csv", summary)
    csv_write(analysis / "local_summary.csv", local)

    cases = {case["id"]: case for case in all_cases()}
    representative = (
        "T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01",
        "points_z060_r01", "lines_z060_r01", "axial_pairs_r01",
    )
    figure_manifest = []
    for case_id in representative:
        for role in ROLES:
            for display in ("shape", "shared"):
                destination = analysis / "actual_reconstruction_comparison" / case_id / f"{role}_{display}.png"
                figure_manifest.append(_draw_projection_panel(cases[case_id], role, display, destination))
    for case_id in ("T02_subset_01", "T03_subset_01", "T04_subset_01"):
        destination = analysis / "actual_reconstruction_comparison" / case_id / "final_all_layers_shared.png"
        figure_manifest.append(_draw_all_layers(cases[case_id], "final", destination))
    _training_plot(analysis / "training_validation_gradient_trends.png")
    _metric_plot(summary, local, analysis / "final_metric_comparison.png")
    old.write_json(analysis / "figure_manifest.json", figure_manifest)
    _write_report(summary, object_rows, local, figure_manifest)

    training = {arm: _audit_training(arm) for arm in NETWORKS}
    evaluation = {}
    for arm in NETWORKS:
        record = json.loads((exp.OUTPUT / "evaluation" / arm / "complete.json").read_text(encoding="utf-8"))
        if int(record["predictions"]) != 188 or int(record["anchors"]) != 188:
            raise ValueError(f"{arm} evaluation inventory is incomplete")
        prediction_files = list((exp.OUTPUT / "evaluation" / arm).glob("best/*/reconstruction.npy")) + list((exp.OUTPUT / "evaluation" / arm).glob("final/*/reconstruction.npy"))
        anchor_files = list((exp.OUTPUT / "evaluation" / arm).glob("best/*/anchor.npy")) + list((exp.OUTPUT / "evaluation" / arm).glob("final/*/anchor.npy"))
        if len(prediction_files) != 188 or len(anchor_files) != 188:
            raise ValueError(f"{arm} raw prediction/anchor file count is incomplete")
        evaluation[arm] = {"predictions": len(prediction_files), "anchors": len(anchor_files),
                           "complete_sha256": old.sha256(exp.OUTPUT / "evaluation" / arm / "complete.json")}
    artifacts = [exp.OUTPUT / "REPORT_ZH.md", analysis / "summary.csv", analysis / "per_object.csv",
                 analysis / "local_summary.csv", analysis / "figure_manifest.json",
                 analysis / "training_validation_gradient_trends.png", analysis / "final_metric_comparison.png"]
    acceptance = {
        "complete": True, "passed": True, "finished_unix": time.time(),
        "seconds": time.time() - started, "training": training, "evaluation": evaluation,
        "primary_predictions": 564, "physical_anchors": 564,
        "official_predictions": 360, "priority_predictions": 198, "zero_predictions": 6,
        "best_and_final_verified": True, "step200_checkpoints_verified": True,
        "data_artifacts_verified_preflight": 6926,
        "source_sha256": {"evaluation": old.sha256(Path(__file__).with_name("v3_compare_evaluation.py")),
                          "report": old.sha256(Path(__file__))},
        "artifact_sha256": {str(path.relative_to(exp.OUTPUT)): old.sha256(path) for path in artifacts},
        "note": "Passing artifact acceptance does not require any method to outperform another.",
    }
    old.write_json(exp.OUTPUT / "final_acceptance.json", acceptance)
    old.write_json(exp.OUTPUT / "complete.json", {
        "complete": True, "passed": True, "experiments": list(NETWORKS),
        "primary_predictions": 564, "best_and_final_verified": True,
        "report_sha256": old.sha256(exp.OUTPUT / "REPORT_ZH.md"),
        "acceptance_sha256": old.sha256(exp.OUTPUT / "final_acceptance.json"),
    })
    print(json.dumps({"report_complete": True, "output": str(exp.OUTPUT),
                      "primary_predictions": 564}, ensure_ascii=False), flush=True)
    from tools.v3_compare_delivery import enrich
    enrich(acceptance)
    return acceptance


if __name__ == "__main__":
    build_report()
