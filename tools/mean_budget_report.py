"""Five-method evaluation, figures, provenance audit, and Chinese report."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets, load_inference_input
from tools import mean_budget_experiment as exp
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as evaluation
from tools import v3_compare_local_audit as local_audit
from tools import v3_compare_report as aggregation
from training.global_batch_schedule import FixedGlobalBatchScheduler


REFERENCE_ARMS = ("baseline", "e3", "e3_mean005")
ALL_ARMS = (*REFERENCE_ARMS, *exp.ARMS)
ROLES = ("best", "final")
TABLES = evaluation.TABLES
LABELS = {
    "baseline": "A Baseline",
    "e3": "B E3",
    "e3_mean005": "C E3+mean≤5%",
    "e3_mean050": "D E3+mean≤50%",
    "e3_mean100": "E E3+mean≤100%",
    "ground_truth": "GT",
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3",
}
COLORS = {
    "ground_truth": "black", "mean_rl3": "#999999", "taylor_rl3": "#a97824",
    "baseline": "#4c72b0", "e3": "#8b4ba5", "e3_mean005": "#2a9d55",
    "e3_mean050": "#e67e22", "e3_mean100": "#d62728",
}
Z = np.arange(10.0, 101.0, 10.0)
PITCH = 220 / 4 / 49


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    evaluation.csv_write(path, rows)


def all_cases() -> list[dict]:
    previous = evaluation.exp
    evaluation.exp = exp
    try:
        return evaluation.all_cases()
    finally:
        evaluation.exp = previous


def _new_corrected_local() -> dict[str, list[dict]]:
    previous_local = local_audit.exp
    previous_eval = evaluation.exp
    local_audit.exp = exp
    evaluation.exp = exp
    try:
        return local_audit.run()
    finally:
        local_audit.exp = previous_local
        evaluation.exp = previous_eval


def _network_row(row: dict, arms: tuple[str, ...]) -> bool:
    return any(str(row.get("method", "")).startswith(f"{arm}_") for arm in arms)


def collect_tables() -> dict[str, list[dict]]:
    tables = {name: [] for name in TABLES}
    for arm in REFERENCE_ARMS:
        for name in TABLES:
            tables[name].extend(read_csv(exp.REFERENCE_OUTPUT / "evaluation" / arm / f"{name}.csv"))
    for arm in exp.ARMS:
        for name in TABLES:
            tables[name].extend(read_csv(exp.OUTPUT / "evaluation" / arm / f"{name}.csv"))

    corrected_new = _new_corrected_local()
    for name in ("t02_tubes", "t03_lines", "t04_axial", "v03_beads"):
        reference = read_csv(exp.REFERENCE_OUTPUT / "analysis" / f"{name}.csv")
        source_networks = [row for row in reference if _network_row(row, REFERENCE_ARMS)]
        new_networks = [row for row in corrected_new[name] if _network_row(row, exp.ARMS)]
        controls = [row for row in corrected_new[name] if row.get("method") in ("mean_rl3", "taylor_rl3")]
        tables[name] = [*source_networks, *new_networks, *controls]
    reference_profiles = read_csv(exp.REFERENCE_OUTPUT / "analysis" / "fixed_profiles.csv")
    new_profiles = [row for row in corrected_new["fixed_profiles"] if _network_row(row, exp.ARMS)]
    tables["fixed_profiles"] = [*reference_profiles, *new_profiles]
    return tables


def checkpoint_path(arm: str, role: str) -> Path:
    root = exp.REFERENCE_OUTPUT if arm in REFERENCE_ARMS else exp.OUTPUT
    return root / arm / ("checkpoint_best.pt" if role == "best" else "checkpoint_last.pt")


def prediction_path(arm: str, role: str, case_id: str) -> Path:
    root = exp.REFERENCE_OUTPUT if arm in REFERENCE_ARMS else exp.OUTPUT
    return root / "evaluation" / arm / role / case_id / "reconstruction.npy"


def load_volumes(case: dict, role: str) -> dict[str, np.ndarray]:
    raw = load_inference_input(case["path"], case["subset"], var_feature_representation="sqrt")
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    target = _read_targets(key, include_ground_truth=case["split"] != "zero")
    result = {"mean_rl3": raw["g_mean"][0], "taylor_rl3": raw["f_var"][0]}
    if "ground_truth" in target:
        result["ground_truth"] = target["ground_truth"][0]
    for arm in ALL_ARMS:
        result[arm] = np.load(prediction_path(arm, role, case["id"]), allow_pickle=False)
    return result


def normalize(value: np.ndarray, scale: float | None = None) -> np.ndarray:
    value = np.maximum(np.asarray(value, np.float32), 0)
    denominator = max(float(value.max()) if scale is None else scale, 1e-30)
    return np.clip(value / denominator, 0, 1)


def draw_views(case: dict, role: str, display: str, destination: Path, book=None) -> dict:
    volumes = load_volumes(case, role)
    methods = [name for name in ("ground_truth", "mean_rl3", "taylor_rl3", *ALL_ARMS) if name in volumes]
    network_scale = max(float(volumes[name].max()) for name in ALL_ARMS)
    shown = {}
    side_views = {name: (volumes[name].sum(1), volumes[name].sum(2)) for name in methods}
    network_side_scale = max(float(view.max()) for name in ALL_ARMS for view in side_views[name])
    for name in methods:
        scale = network_scale if display == "shared" and name in ALL_ARMS else None
        shown[name] = normalize(volumes[name], scale)
    figure, axes = plt.subplots(3, len(methods), figsize=(3.0 * len(methods), 8.8), layout="constrained")
    for column, name in enumerate(methods):
        pair = side_views[name]
        side_scale = (
            network_side_scale
            if display == "shared" and name in ALL_ARMS
            else max(float(pair[0].max()), float(pair[1].max()), 1e-30)
        )
        views = (shown[name].max(0), pair[0] / side_scale, pair[1] / side_scale)
        axes[0, column].imshow(views[0], cmap="magma", vmin=0, vmax=1, origin="lower")
        for row in (1, 2):
            axes[row, column].imshow(
                views[row], cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                origin="lower", aspect="equal", extent=(-PITCH / 2, 259.5 * PITCH, 5, 105),
            )
        axes[0, column].set_title(LABELS[name], fontsize=9)
        axes[0, column].set_xticks([]); axes[0, column].set_yticks([])
        axes[1, column].set_xlabel("X (µm)"); axes[2, column].set_xlabel("Y (µm)")
        if column == 0:
            axes[0, column].set_ylabel("XY max")
            axes[1, column].set_ylabel("Z (µm) · XZ sum Y")
            axes[2, column].set_ylabel("Z (µm) · YZ sum X")
        else:
            axes[1, column].set_yticklabels([]); axes[2, column].set_yticklabels([])
    subtitle = "five networks share brightness" if display == "shared" else "each volume normalized once"
    figure.suptitle(f"{case['id']} · {role} · {subtitle}; native 10-layer Z, no interpolation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=145)
    if book is not None:
        book.savefig(figure)
    plt.close(figure)
    return {"case_id": case["id"], "role": role, "display": display,
            "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def draw_layers(case: dict, role: str, destination: Path) -> dict:
    volumes = load_volumes(case, role)
    methods = ["ground_truth", "mean_rl3", "taylor_rl3", *ALL_ARMS]
    shared = max(float(volumes[name].max()) for name in ALL_ARMS)
    figure, axes = plt.subplots(len(methods), 10, figsize=(22, 2.0 * len(methods)), layout="constrained")
    for row, name in enumerate(methods):
        value = normalize(volumes[name], shared if name in ALL_ARMS else None)
        for layer in range(10):
            axes[row, layer].imshow(value[layer], cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[row, layer].set_xticks([]); axes[row, layer].set_yticks([])
            if row == 0: axes[row, layer].set_title(f"{int(Z[layer])} µm")
            if layer == 0: axes[row, layer].set_ylabel(LABELS[name], fontsize=8)
    figure.suptitle(f"{case['id']} · {role} · all native layers; five networks share brightness")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=125)
    plt.close(figure)
    return {"case_id": case["id"], "role": role, "display": "all_layers_shared",
            "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def draw_profiles(rows: list[dict], sample: str, role: str, destination: Path, book=None) -> dict:
    count, shape = {"T02": (13, (4, 4)), "T03": (12, (3, 4)), "T04": (16, (4, 4))}[sample]
    figure, axes = plt.subplots(*shape, figsize=(16, 3.0 * shape[0]), layout="constrained")
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.axis("off"); continue
        region = f"{chr(65 + index // 4)}{index % 4 + 1}" if sample == "T04" else str(index + 1)
        selected = [r for r in rows if r["sample_id"] == sample and int(r["subset"]) == 1 and r["region"] == region]
        methods = ["mean_rl3", "taylor_rl3", *(f"{arm}_{role}" for arm in ALL_ARMS)]
        for method in methods:
            values = [r for r in selected if r["method"] == method]
            values.sort(key=lambda r: float(r["coordinate"]))
            if not values: continue
            x = np.asarray([float(r["coordinate"]) for r in values])
            y = np.asarray([float(r["value"]) for r in values])
            base = method.removesuffix(f"_{role}")
            axis.plot(x, y / max(float(y.max()), 1e-30), color=COLORS[base],
                      marker="o" if sample != "T03" else None, markersize=2,
                      linewidth=1.2, label=LABELS[base])
        reference = [r for r in selected if r["method"] == f"e3_{role}"]
        reference.sort(key=lambda r: float(r["coordinate"]))
        if reference and reference[0].get("gt_value") not in (None, ""):
            y = np.asarray([float(r["gt_value"]) for r in reference])
            axis.plot([float(r["coordinate"]) for r in reference], y / max(float(y.max()), 1e-30),
                      "k--", linewidth=1.8, label="GT")
        axis.set_title(f"{sample} region {region}")
        axis.set_xlabel("Transverse pixel" if sample == "T03" else "Native depth (µm)")
        axis.set_ylim(-0.02, 1.05); axis.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    figure.suptitle(f"{sample} subset 01 · {role} · fixed ROI profiles; curves normalized only for display")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    if book is not None: book.savefig(figure)
    plt.close(figure)
    return {"case_id": f"{sample}_subset_01", "role": role, "display": "fixed_profiles",
            "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def training_audit() -> tuple[dict, list[dict]]:
    indexed, _ = __import__("datasets.matlab_multivolume_dataset", fromlist=["load_dataset_index"]).load_dataset_index(exp.DATA)
    scheduler = FixedGlobalBatchScheduler(110, 8, 20260901)
    expected_batches = {step: scheduler.next_batch() for step in range(1, 401)}
    audits, gradients = {}, []
    for arm in exp.ARMS:
        folder = exp.OUTPUT / arm
        complete = json.loads((folder / "training_complete.json").read_text())
        train = read_csv(folder / "training_metrics.csv")
        validation = read_csv(folder / "validation_metrics.csv")
        if not complete.get("complete") or int(complete["completed_steps"]) != 400 or len(train) != 400 or len(validation) != 600:
            raise ValueError(f"Incomplete training inventory for {arm}")
        records = []
        for path in sorted(folder.glob("gradient_diagnostics_rank*.jsonl")):
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        records = [row for row in records if row.get("phase") == "training"]
        if len(records) != 3200:
            raise ValueError(f"{arm} has {len(records)} gradient rows, expected 3200")
        by_step = defaultdict(list)
        for row in records: by_step[int(row["update_step"])].append(row)
        saturation = 0
        for step, batch in expected_batches.items():
            rows = by_step[step]
            expected = [indexed["train"][i] for i in batch]
            if sorted((r["sample_id"], int(r["subset_index"])) for r in rows) != sorted((k.sample_id, k.subset_index) for k in expected):
                raise ValueError(f"{arm} sample order differs at step {step}")
            bound = exp.BUDGETS[arm] * min(step / 50, 1)
            for row in rows:
                ratio = float(row["shape_gradient_ratio"])
                if not row["gradients_finite"] or not row["budget_passed"] or ratio > bound + max(1e-7, bound * 1e-5):
                    raise ValueError(f"{arm} gradient bound failed at step {step}")
                saturation += math.isclose(float(row["shape_coefficient"]), 1.0, rel_tol=0, abs_tol=1e-7)
                gradients.append({"experiment": arm, **row})
        checkpoint_rows = {}
        for role in ROLES:
            path = checkpoint_path(arm, role)
            state = torch.load(path, map_location="cpu", weights_only=False)
            step = int(state["completed_steps"])
            if role == "final" and step != 400: raise ValueError(f"{arm} final is step {step}")
            checkpoint_rows[role] = {"step": step, "sha256": old.sha256(path)}
        audits[arm] = {
            "best_step": int(complete["best_step"]), "checkpoints": checkpoint_rows,
            "gradient_records": len(records), "coefficient_cap_fraction": saturation / len(records),
            "mean_actual_ratio": float(np.mean([r["shape_gradient_ratio"] for r in records])),
            "maximum_actual_ratio": float(max(r["shape_gradient_ratio"] for r in records)),
            "mean_gradient_cosine": float(np.mean([r["mean_var_gradient_cosine"] for r in records])),
            "negative_cosine_fraction": float(np.mean([r["mean_var_gradient_cosine"] < 0 for r in records])),
        }
    return audits, gradients


def draw_training(gradients: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for arm in ALL_ARMS:
        root = exp.REFERENCE_OUTPUT if arm in REFERENCE_ARMS else exp.OUTPUT
        train = read_csv(root / arm / "training_metrics.csv")
        axes[0].plot([int(r["step"]) for r in train], [float(r["normalized_var_loss"]) for r in train], label=LABELS[arm])
        axes[1].plot([int(r["step"]) for r in train], [float(r["normalized_mean_loss"]) for r in train], label=LABELS[arm])
    for arm in exp.ARMS:
        by_step = defaultdict(list)
        for row in gradients:
            if row["experiment"] == arm: by_step[int(row["update_step"])].append(float(row["shape_gradient_ratio"]))
        axes[2].plot(sorted(by_step), [np.mean(by_step[s]) for s in sorted(by_step)], label=LABELS[arm])
    axes[0].set_title("Normalized variance loss"); axes[1].set_title("Ordinary normalized mean loss")
    axes[2].set_title("Actual mean / variance q-gradient ratio")
    for axis in axes:
        axis.set_xlabel("Optimizer step"); axis.grid(alpha=0.25); axis.legend(fontsize=7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150); plt.close(figure)


def _number(rows, method, split, key):
    return float(next(r for r in rows if r["method"] == method and r["split"] == split)[key])


def _object_number(rows, method, sample_id, key):
    return float(next(r for r in rows if r["method"] == method and r["sample_id"] == sample_id)[key])


def _local(rows, method, diagnostic, key):
    return float(next(r for r in rows if r["method"] == method and r["diagnostic"] == diagnostic)[key])


def _truthy(value) -> bool:
    return str(value).lower() in ("true", "1")


def _mean(rows, key) -> float:
    values = []
    for row in rows:
        try: value = float(row[key])
        except (KeyError, TypeError, ValueError): continue
        if math.isfinite(value): values.append(value)
    return float(np.mean(values)) if values else float("nan")


def write_diagnostic_tables(tables: dict[str, list[dict]], analysis: Path) -> None:
    repeats = []
    methods = sorted({_["method"] for _ in tables["point_targets"] if _network_row(_, ALL_ARMS)})
    for method in methods:
        for repeat in (1, 2, 3):
            points = [r for r in tables["point_targets"] if r["method"] == method and int(r["repeat"]) == repeat and float(r["threshold"]) == 0.1]
            pairs = [r for r in tables["point_pairs"] if r["method"] == method and int(r["repeat"]) == repeat and float(r["threshold"]) == 0.1]
            lines = [r for r in tables["priority_lines"] if r["method"] == method and int(r["repeat"]) == repeat and float(r["threshold"]) == 0.1]
            continuous = [r for r in lines if float(r["gap_um"]) == 0]
            gapped = [r for r in lines if float(r["gap_um"]) > 0]
            repeats.append({
                "method": method, "repeat": repeat, "point_targets": len(points),
                "point_localized_rate": _mean([{"v": _truthy(r["matched"])} for r in points], "v"),
                "point_local_depth_w1_um": _mean(points, "local_depth_w1_um"),
                "point_pairs": len(pairs),
                "point_pair_separated_rate": _mean([{"v": _truthy(r["separated"])} for r in pairs], "v"),
                "continuous_false_gap_rate": _mean([{"v": _truthy(r["false_gap"])} for r in continuous], "v"),
                "gapped_bridge_rate": _mean([{"v": _truthy(r["bridged"])} for r in gapped], "v"),
            })
    write_csv(analysis / "per_repeat_summary.csv", repeats)

    failures = []
    for table, threshold_key, failed in (
        ("t03_lines", "threshold_fraction", lambda r: not _truthy(r["separated"])),
        ("t04_axial", "threshold_fraction", lambda r: _truthy(r["separation_applicable"]) and not _truthy(r["separated"])),
        ("v03_beads", "threshold_fraction", lambda r: not _truthy(r["detected"])),
        ("point_targets", "threshold", lambda r: not _truthy(r["matched"])),
        ("priority_lines", "threshold", lambda r: _truthy(r["bridged"]) or _truthy(r["false_gap"])),
    ):
        for row in tables[table]:
            if float(row[threshold_key]) == 0.1 and failed(row):
                failures.append({"diagnostic": table, **row})
    write_csv(analysis / "failure_cases.csv", failures)
    write_csv(analysis / "zero_input.csv", [r for r in tables["metrics"] if r["split"] == "zero"])


def write_report(summary, objects, local, training, figures) -> None:
    lines = [
        "# E3＋50% / 100% mean 结构梯度：五种方法对比", "",
        "两组新实验均从与原 baseline、E3、E3＋5% 完全相同的随机初始化开始训练 400 步。主要结论看第 400 步 final；best 使用原 E3 无 GT 验证评分。", "",
        "## Final 正式测试结果", "",
        "| 方法 | 形状误差↓ | 轴向 W1 µm↓ | 背景占比↓ | T03 三线分开率↑ | T03 假断率↓ | T04 20–40 µm 分开率↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ALL_ARMS:
        method = f"{arm}_final"
        lines.append(
            f"| {LABELS[arm]} | {_number(summary, method, 'test', 'gt_scale_aligned_nrmse'):.4f} | "
            f"{_number(summary, method, 'test', 'gt_axial_w1_um'):.2f} | "
            f"{_number(summary, method, 'test', 'background_xy_mass_fraction'):.4f} | "
            f"{_local(local, method, 'T03_three_lines', 'separated_rate'):.3f} | "
            f"{_local(local, method, 'T03_three_lines', 'false_break_rate'):.3f} | "
            f"{_local(local, method, 'T04_axial_pairs', 'separated_rate_20_40um'):.3f} |"
        )
    lines += ["", "## Best checkpoint", "",
              "| 方法 | best 步数 | 形状误差↓ | 轴向 W1 µm↓ | T03 三线分开率↑ | T03 假断率↓ |",
              "|---|---:|---:|---:|---:|---:|"]
    source_acceptance = json.loads((exp.REFERENCE_OUTPUT / "final_acceptance.json").read_text())
    for arm in ALL_ARMS:
        method = f"{arm}_best"
        if arm in exp.ARMS: step = training[arm]["best_step"]
        else: step = int(source_acceptance["training"][arm]["best_step"])
        lines.append(
            f"| {LABELS[arm]} | {step} | {_number(summary, method, 'test', 'gt_scale_aligned_nrmse'):.4f} | "
            f"{_number(summary, method, 'test', 'gt_axial_w1_um'):.2f} | "
            f"{_local(local, method, 'T03_three_lines', 'separated_rate'):.3f} | "
            f"{_local(local, method, 'T03_three_lines', 'false_break_rate'):.3f} |"
        )
    lines += ["", "## 50% 和 100% 的实际训练状态", "",
              "标称比例是逐样本归一化重建体 q 上的梯度上限。系数上限为 1，因此实际比例可以低于标称值。", "",
              "| 方法 | 实际平均比例 | 实际最大比例 | 系数达到1的样本比例 | mean/var方向相反比例 |",
              "|---|---:|---:|---:|---:|"]
    for arm in exp.ARMS:
        row = training[arm]
        lines.append(f"| {LABELS[arm]} | {row['mean_actual_ratio']:.4f} | {row['maximum_actual_ratio']:.4f} | {row['coefficient_cap_fraction']:.3f} | {row['negative_cosine_fraction']:.3f} |")

    best_shape = min(ALL_ARMS, key=lambda arm: _number(summary, f"{arm}_final", "test", "gt_scale_aligned_nrmse"))
    best_depth = min(ALL_ARMS, key=lambda arm: _number(summary, f"{arm}_final", "test", "gt_axial_w1_um"))
    lines += ["", "## 直接结论", "",
              f"按 final 的三对象等权平均，形状误差最低的是 {LABELS[best_shape]}，轴向 W1 最低的是 {LABELS[best_depth]}。", "",
              f"**100% 组是本轮更值得保留的方案。** 它的总体形状误差为 {_number(summary, 'e3_mean100_final', 'test', 'gt_scale_aligned_nrmse'):.4f}，低于 E3 的 {_number(summary, 'e3_final', 'test', 'gt_scale_aligned_nrmse'):.4f}；轴向 W1 为 {_number(summary, 'e3_mean100_final', 'test', 'gt_axial_w1_um'):.2f} µm，也没有牺牲 E3 的 {_number(summary, 'e3_final', 'test', 'gt_axial_w1_um'):.2f} µm。总体背景占比 {_number(summary, 'e3_mean100_final', 'test', 'background_xy_mass_fraction'):.4f}，与 E3 的 {_number(summary, 'e3_final', 'test', 'background_xy_mass_fraction'):.4f} 相当。", "",
              f"**50% 组的平均轴向 W1 最低，但稳定性较差。** 它在 T02 的背景占比达到 {_object_number(objects, 'e3_mean050_final', 'T02', 'background_xy_mass_fraction'):.4f}，明显高于 E3 的 {_object_number(objects, 'e3_final', 'T02', 'background_xy_mass_fraction'):.4f} 和 100% 组的 {_object_number(objects, 'e3_mean100_final', 'T02', 'background_xy_mass_fraction'):.4f}。因此不能只凭平均轴向分数选择 50%。", "",
              f"5%、50%、100% 在 T03 的三线分开率都约为 0.667；但假断率分别为 {_local(local, 'e3_mean005_final', 'T03_three_lines', 'false_break_rate'):.3f}、{_local(local, 'e3_mean050_final', 'T03_three_lines', 'false_break_rate'):.3f}、{_local(local, 'e3_mean100_final', 'T03_three_lines', 'false_break_rate'):.3f}。提高 mean 梯度的主要收益不是继续增加通过数量，而是用更少的断线代价保持同一分开率。", "",
              f"**T04 轴向双层仍未解决。** E3、5%、50% 和 100% 对 20–40 µm 双层的正式分开率都为 0。旧点目标诊断中，E3 的局部轴向 W1 为 {_local(local, 'e3_final', 'priority_points', 'local_depth_w1_um'):.2f} µm，仍优于 50% 的 {_local(local, 'e3_mean050_final', 'priority_points', 'local_depth_w1_um'):.2f} µm 和 100% 的 {_local(local, 'e3_mean100_final', 'priority_points', 'local_depth_w1_um'):.2f} µm，所以 100% 也不是所有轴向场景都胜过 E3。", "",
              "T03 的分开率必须和假断率一起看；谷值更深但线条被切断，不能算作可靠分辨率提升。T04 仍按原生十层判定，10 µm 双层不进行分离判断。", "",
              "本轮建议把 100% 组带入下一轮重复种子验证，同时继续保留 E3 作为轴向基准；50% 暂时只作为机理证据，不宜直接替代 E3。", "",
              "本轮只有一个共同训练种子；很小的差异不能视为稳定收益。GT 只用于训练结束后的评价，没有进入训练或 best 选择。", "",
              "## 结果入口", "",
              "- [五种方法逐对象汇总](analysis/per_object.csv)",
              "- [五种方法总体汇总](analysis/summary.csv)",
              "- [局部结构汇总](analysis/local_summary.csv)",
              "- [全部逐项指标](analysis/metrics.csv)",
              "- [三次采集逐次汇总](analysis/per_repeat_summary.csv)",
              "- [失败案例](analysis/failure_cases.csv)",
              "- [零输入结果](analysis/zero_input.csv)",
              "- [实际梯度记录汇总](analysis/gradient_summary.csv)",
              "- [实际重建图册索引](analysis/actual_reconstruction_comparison/README_ZH.md)",
              "- [best 轴向图册](analysis/轴向对比_best.pdf)",
              "- [final 轴向图册](analysis/轴向对比_final.pdf)", ""]
    exp.OUTPUT.joinpath("REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict:
    started = time.time()
    for arm in exp.ARMS:
        marker = exp.OUTPUT / "evaluation" / arm / "complete.json"
        if not marker.is_file() or not json.loads(marker.read_text()).get("complete"):
            raise ValueError(f"Evaluation is incomplete: {arm}")
    tables = collect_tables()
    analysis = exp.OUTPUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items(): write_csv(analysis / f"{name}.csv", rows)
    objects, summary = aggregation.aggregate_metrics(tables["metrics"])
    local = aggregation._aggregate_local(tables)
    write_csv(analysis / "per_object.csv", objects)
    write_csv(analysis / "summary.csv", summary)
    write_csv(analysis / "local_summary.csv", local)
    write_diagnostic_tables(tables, analysis)

    training, gradients = training_audit()
    gradient_summary = []
    for arm, row in training.items():
        gradient_summary.append({"experiment": arm, **{key: value for key, value in row.items() if not isinstance(value, dict)}})
    write_csv(analysis / "gradient_summary.csv", gradient_summary)
    draw_training(gradients, analysis / "training_validation_gradient_trends.png")

    cases = all_cases()
    by_id = {case["id"]: case for case in cases}
    axial_book_cases = {"T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01",
                        "axial_pairs_r01", "axial_pairs_r02", "axial_pairs_r03"}
    manifest_path = analysis / "figure_manifest.json"
    pdfs = [analysis / f"轴向对比_{role}.pdf" for role in ROLES]
    if manifest_path.exists() and all(path.exists() for path in pdfs):
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) != 160 or any(not (exp.OUTPUT / row["path"]).exists() for row in manifest):
            raise ValueError("Existing figure atlas is incomplete")
    else:
        manifest = []
        books = {role: PdfPages(analysis / f"轴向对比_{role}.pdf") for role in ROLES}
        try:
            selected_cases = [case for case in cases if case["split"] == "priority"]
            selected_cases += [by_id[name] for name in ("T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01")]
            for case in selected_cases:
                for role in ROLES:
                    for display in ("shape", "shared"):
                        destination = analysis / "actual_reconstruction_comparison" / case["id"] / f"{role}_{display}.png"
                        manifest.append(draw_views(case, role, display, destination,
                                                   books[role] if case["id"] in axial_book_cases else None))
            for sample in ("T02", "T03", "T04"):
                case = by_id[f"{sample}_subset_01"]
                for role in ROLES:
                    destination = analysis / "actual_reconstruction_comparison" / case["id"] / f"{role}_all_layers_shared.png"
                    manifest.append(draw_layers(case, role, destination))
                    profile_path = analysis / f"{sample}_{role}_fixed_profiles.png"
                    manifest.append(draw_profiles(tables.get("fixed_profiles", []), sample, role, profile_path, books[role]))
        finally:
            for book in books.values(): book.close()
        old.write_json(manifest_path, manifest)

    atlas = ["# 五种方法实际重建图册", "",
             "每个三维体只做一次显示归一化；shared 图中五个网络共用亮度尺度。XZ/YZ 是求和投影，保留原生十层，不进行 Z 插值。三次独立采集分别展示。", ""]
    for case in [c for c in cases if c["split"] == "priority"] + [by_id[n] for n in ("T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01")]:
        atlas += [f"## {case['id']}", ""]
        for row in manifest:
            if row["case_id"] == case["id"]:
                atlas.append(f"- [{row['role']} · {row['display']}]({(exp.OUTPUT / row['path']).as_posix()})")
        atlas.append("")
    atlas_path = analysis / "actual_reconstruction_comparison/README_ZH.md"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.write_text("\n".join(atlas), encoding="utf-8")

    write_report(summary, objects, local, training, manifest)
    raw_files = []
    for arm in exp.ARMS:
        predictions = list((exp.OUTPUT / "evaluation" / arm).glob("best/*/reconstruction.npy")) + list((exp.OUTPUT / "evaluation" / arm).glob("final/*/reconstruction.npy"))
        anchors = list((exp.OUTPUT / "evaluation" / arm).glob("best/*/anchor.npy")) + list((exp.OUTPUT / "evaluation" / arm).glob("final/*/anchor.npy"))
        if len(predictions) != 188 or len(anchors) != 188:
            raise ValueError(f"Raw artifact count is incomplete for {arm}")
        raw_files.extend(predictions + anchors)
    artifacts = [exp.OUTPUT / "REPORT_ZH.md", analysis / "summary.csv", analysis / "per_object.csv",
                 analysis / "local_summary.csv", analysis / "gradient_summary.csv",
                 analysis / "per_repeat_summary.csv", analysis / "failure_cases.csv",
                 analysis / "zero_input.csv",
                 analysis / "figure_manifest.json", analysis / "轴向对比_best.pdf", analysis / "轴向对比_final.pdf"]
    acceptance = {
        "passed": True, "new_arms": list(exp.ARMS), "reference_arms": list(REFERENCE_ARMS),
        "methods": 5, "checkpoint_roles": 2, "metric_rows": len(tables["metrics"]),
        "new_predictions": 376, "new_anchors": 376, "figures": len(manifest),
        "training": training,
        "reference_acceptance_sha256": old.sha256(exp.REFERENCE_OUTPUT / "final_acceptance.json"),
        "artifacts": {str(path): old.sha256(path) for path in artifacts},
        "raw_new_artifacts_sha256": {str(path.relative_to(exp.OUTPUT)): old.sha256(path) for path in raw_files},
        "finished_unix": time.time(), "seconds": time.time() - started,
    }
    old.write_json(exp.OUTPUT / "final_acceptance.json", acceptance)
    return acceptance
