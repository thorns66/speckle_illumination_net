"""Audit and report the E3+100% continuation versus no-Set comparison."""
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

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets, load_dataset_index, load_inference_input
from training.global_batch_schedule import FixedGlobalBatchScheduler
from tools import mean100_ablation_experiment as exp
from tools import mean100_ablation_evaluation as evaluation
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as v3
from tools import v3_compare_local_audit as local_audit
from tools import v3_compare_report as aggregation


REFERENCE_EVAL = exp.BASELINE_OUTPUT / "evaluation/e3_mean100"
PRIMARY = ("baseline400", "extend600_step600", "noset600_step400", "noset600_step600")
ALL_METHODS = (
    "baseline400", "extend600_best", "extend600_step600",
    "noset600_best", "noset600_step400", "noset600_step600",
)
LABELS = {
    "ground_truth": "GT",
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3",
    "baseline400": "完整网络 400",
    "extend600_best": "完整网络 best",
    "extend600_step600": "完整网络 600",
    "noset600_best": "无Set best",
    "noset600_step400": "无Set 400",
    "noset600_step600": "无Set 600",
}
FIGURE_LABELS = {
    "ground_truth": "GT",
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3",
    "baseline400": "Full 400",
    "extend600_best": "Full best",
    "extend600_step600": "Full 600",
    "noset600_best": "NoSet best",
    "noset600_step400": "NoSet 400",
    "noset600_step600": "NoSet 600",
}
COLORS = {
    "mean_rl3": "#888888", "taylor_rl3": "#bbbbbb", "baseline400": "#1f77b4",
    "extend600_step600": "#ff7f0e", "noset600_step400": "#2ca02c", "noset600_step600": "#d62728",
}
Z = np.arange(10.0, 101.0, 10.0)
PITCH = 220 / 4 / 49


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cases() -> list[dict]:
    previous = v3.exp
    v3.exp = exp
    try:
        return v3.all_cases()
    finally:
        v3.exp = previous


def collect_tables() -> dict[str, list[dict]]:
    tables = {name: [] for name in v3.TABLES}
    for name in v3.TABLES:
        for row in read_csv(REFERENCE_EVAL / f"{name}.csv"):
            if row.get("method") == "e3_mean100_final":
                row = dict(row)
                row.update(method="baseline400", experiment="baseline400", checkpoint_role="step400")
                tables[name].append(row)
        for arm in exp.ARMS:
            tables[name].extend(read_csv(exp.OUTPUT / "evaluation" / arm / f"{name}.csv"))
    return tables


def training_audit() -> tuple[dict, list[dict]]:
    indexed, _ = load_dataset_index(exp.DATA)
    scheduler = FixedGlobalBatchScheduler(110, 8, 20260901)
    expected = {step: scheduler.next_batch() for step in range(1, 601)}
    result, all_gradients = {}, []
    initial = torch.load(exp.SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)["model_state"]
    frozen_names = [
        name for name in initial
        if name.startswith(("set_encoder.", "fusion.set_projection.", "fusion.gate."))
        or name == "fusion.raw_alpha"
    ]
    for arm in exp.ARMS:
        folder = exp.OUTPUT / arm
        complete = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
        train = read_csv(folder / "training_metrics.csv")
        validation = read_csv(folder / "validation_metrics.csv")
        steps = [int(row["step"]) for row in train]
        if not complete.get("complete") or int(complete["completed_steps"]) != 600:
            raise ValueError(f"Incomplete training marker for {arm}")
        if steps != list(range(1, 601)) or len(validation) != 900:
            raise ValueError(f"Incomplete or duplicated training/validation rows for {arm}")
        gradients = []
        for path in sorted(folder.glob("gradient_diagnostics_rank*.jsonl")):
            gradients.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        gradients = [row for row in gradients if row.get("phase") == "training"]
        if len(gradients) != 4800:
            raise ValueError(f"{arm} has {len(gradients)} gradient rows, expected 4800")
        by_step = defaultdict(list)
        for row in gradients:
            by_step[int(row["update_step"])].append(row)
        saturation = 0
        for step, batch in expected.items():
            rows = by_step[step]
            wanted = [indexed["train"][index] for index in batch]
            found = sorted((row["sample_id"], int(row["subset_index"])) for row in rows)
            target = sorted((key.sample_id, key.subset_index) for key in wanted)
            if found != target:
                raise ValueError(f"{arm} sample order differs at step {step}")
            bound = min(step / 50, 1.0)
            for row in rows:
                ratio = float(row["shape_gradient_ratio"])
                if not row["gradients_finite"] or not row["budget_passed"] or ratio > bound + 1e-5:
                    raise ValueError(f"{arm} gradient bound failed at step {step}")
                saturation += math.isclose(float(row["shape_coefficient"]), 1.0, abs_tol=1e-7)
                all_gradients.append({"experiment": arm, **row})
        if arm == "extend600":
            staged = folder / "checkpoint_step_000400.pt"
            if old.sha256(staged) != json.loads((exp.OUTPUT / "preflight.json").read_text())["continuation"]["source_sha256"]:
                raise ValueError("The staged step-400 continuation source changed")
        if arm == "noset600":
            for checkpoint_name in ("checkpoint_step_000400.pt", "checkpoint_last.pt"):
                state = torch.load(folder / checkpoint_name, map_location="cpu", weights_only=False)["model_state"]
                for name in frozen_names:
                    if not torch.equal(state[name], initial[name]):
                        raise ValueError(f"Frozen no-Set parameter changed: {name}")
        lr_rows = []
        for path in sorted(folder.glob("learning_rate_rank*.jsonl")):
            lr_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        low_step = 401 if arm == "extend600" else 201
        sample_lr = next(row for row in lr_rows if int(row["step"]) == low_step)
        if float(sample_lr["used_lrs"]["network"]) != 1e-4:
            raise ValueError(f"{arm} low learning rate boundary failed")
        best = torch.load(folder / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        final = torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        result[arm] = {
            "best_step": int(best["completed_steps"]),
            "final_step": int(final["completed_steps"]),
            "best_sha256": old.sha256(folder / "checkpoint_best.pt"),
            "final_sha256": old.sha256(folder / "checkpoint_last.pt"),
            "gradient_records": len(gradients),
            "mean_actual_ratio": float(np.mean([float(row["shape_gradient_ratio"]) for row in gradients])),
            "maximum_actual_ratio": float(max(float(row["shape_gradient_ratio"]) for row in gradients)),
            "coefficient_cap_fraction": saturation / len(gradients),
            "negative_cosine_fraction": float(np.mean([float(row["mean_var_gradient_cosine"]) < 0 for row in gradients])),
        }
    return result, all_gradients


def load_volumes(case: dict) -> dict[str, np.ndarray]:
    raw = load_inference_input(case["path"], case["subset"], var_feature_representation="sqrt")
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    target = _read_targets(key, include_ground_truth=case["split"] != "zero")
    result = {"mean_rl3": raw["g_mean"][0], "taylor_rl3": raw["f_var"][0]}
    if "ground_truth" in target:
        result["ground_truth"] = target["ground_truth"][0]
    result["baseline400"] = np.load(REFERENCE_EVAL / "final" / case["id"] / "reconstruction.npy")
    result["extend600_step600"] = np.load(exp.OUTPUT / "evaluation/extend600/step600" / case["id"] / "reconstruction.npy")
    result["noset600_step400"] = np.load(exp.OUTPUT / "evaluation/noset600/step400" / case["id"] / "reconstruction.npy")
    result["noset600_step600"] = np.load(exp.OUTPUT / "evaluation/noset600/step600" / case["id"] / "reconstruction.npy")
    return result


def normalize(value: np.ndarray, scale: float | None = None) -> np.ndarray:
    value = np.maximum(np.asarray(value, np.float32), 0)
    return np.clip(value / max(float(value.max()) if scale is None else scale, 1e-30), 0, 1)


def draw_views(case: dict, display: str, destination: Path, book=None) -> dict:
    volumes = load_volumes(case)
    methods = [name for name in ("ground_truth", "mean_rl3", "taylor_rl3", *PRIMARY) if name in volumes]
    network_scale = max(float(volumes[name].max()) for name in PRIMARY)
    sides = {name: (volumes[name].sum(1), volumes[name].sum(2)) for name in methods}
    network_side_scale = max(float(view.max()) for name in PRIMARY for view in sides[name])
    figure, axes = plt.subplots(3, len(methods), figsize=(3.0 * len(methods), 8.7), layout="constrained")
    for column, name in enumerate(methods):
        scale = network_scale if display == "shared" and name in PRIMARY else None
        volume = normalize(volumes[name], scale)
        side_scale = network_side_scale if display == "shared" and name in PRIMARY else max(
            float(sides[name][0].max()), float(sides[name][1].max()), 1e-30
        )
        views = (volume.max(0), sides[name][0] / side_scale, sides[name][1] / side_scale)
        axes[0, column].imshow(views[0], cmap="magma", vmin=0, vmax=1, origin="lower")
        for row in (1, 2):
            axes[row, column].imshow(
                views[row], cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                origin="lower", aspect="equal", extent=(-PITCH / 2, 259.5 * PITCH, 5, 105),
            )
        axes[0, column].set_title(FIGURE_LABELS[name], fontsize=9)
        axes[0, column].set_xticks([]); axes[0, column].set_yticks([])
        axes[1, column].set_xlabel("X (µm)"); axes[2, column].set_xlabel("Y (µm)")
        if column == 0:
            axes[0, column].set_ylabel("XY max")
            axes[1, column].set_ylabel("Z (µm) · XZ sum Y")
            axes[2, column].set_ylabel("Z (µm) · YZ sum X")
        else:
            axes[1, column].set_yticklabels([]); axes[2, column].set_yticklabels([])
    figure.suptitle(f"{case['id']} · {display}; native 10-layer Z, no interpolation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    if book is not None:
        book.savefig(figure)
    plt.close(figure)
    return {"case_id": case["id"], "display": display, "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def draw_layers(case: dict, destination: Path, book=None) -> dict:
    volumes = load_volumes(case)
    methods = ["ground_truth", "mean_rl3", "taylor_rl3", *PRIMARY]
    shared = max(float(volumes[name].max()) for name in PRIMARY)
    figure, axes = plt.subplots(len(methods), 10, figsize=(22, 2.0 * len(methods)), layout="constrained")
    for row, name in enumerate(methods):
        volume = normalize(volumes[name], shared if name in PRIMARY else None)
        for layer in range(10):
            axes[row, layer].imshow(volume[layer], cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[row, layer].set_xticks([]); axes[row, layer].set_yticks([])
            if row == 0:
                axes[row, layer].set_title(f"{int(Z[layer])} µm")
            if layer == 0:
                axes[row, layer].set_ylabel(FIGURE_LABELS[name], fontsize=8)
    figure.suptitle(f"{case['id']} · all native layers; four networks share brightness")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    if book is not None:
        book.savefig(figure)
    plt.close(figure)
    return {"case_id": case["id"], "display": "all_layers_shared", "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def fixed_profiles(case: dict, destination: Path, book=None) -> dict:
    volumes = load_volumes(case)
    truth = volumes["ground_truth"]
    function = {"T02": local_audit.t02, "T03": local_audit.t03, "T04": local_audit.t04}[case["sample"]]
    profile_rows = []
    previous = local_audit.exp
    local_audit.exp = exp
    try:
        for method in ("mean_rl3", "taylor_rl3", *PRIMARY):
            _rows, profiles = function(volumes[method], truth, {"method": method, "sample_id": case["sample"], "subset": 1})
            profile_rows.extend(profiles)
    finally:
        local_audit.exp = previous
    regions = sorted({str(row["region"]) for row in profile_rows}, key=lambda value: (len(value), value))
    columns = 4
    rows_count = math.ceil(len(regions) / columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(16, 3.0 * rows_count), layout="constrained", squeeze=False)
    for index, axis in enumerate(axes.ravel()):
        if index >= len(regions):
            axis.axis("off"); continue
        region = regions[index]
        selected = [row for row in profile_rows if str(row["region"]) == region]
        for method in ("mean_rl3", "taylor_rl3", *PRIMARY):
            values = sorted((row for row in selected if row["method"] == method), key=lambda row: float(row["coordinate"]))
            if not values:
                continue
            x = np.asarray([float(row["coordinate"]) for row in values])
            y = np.asarray([float(row["value"]) for row in values])
            axis.plot(x, y / max(float(y.max()), 1e-30), color=COLORS[method], lw=1.1, label=FIGURE_LABELS[method])
        gt_rows = sorted((row for row in selected if row.get("gt_value") not in (None, "")), key=lambda row: float(row["coordinate"]))
        if gt_rows:
            x = np.asarray([float(row["coordinate"]) for row in gt_rows])
            y = np.asarray([float(row["gt_value"]) for row in gt_rows])
            axis.plot(x, y / max(float(y.max()), 1e-30), "k--", lw=1.5, label="GT")
        axis.set_title(f"{case['sample']} region {region}")
        axis.set_xlabel("Transverse pixel" if case["sample"] == "T03" else "Native depth (µm)")
        axis.set_ylim(-0.02, 1.05); axis.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    figure.suptitle(f"{case['id']} · fixed ROI profiles; display-normalized curves")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=145)
    if book is not None:
        book.savefig(figure)
    plt.close(figure)
    return {"case_id": case["id"], "display": "fixed_profiles", "path": str(destination.relative_to(exp.OUTPUT)), "sha256": old.sha256(destination)}


def draw_training(gradients: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for arm in exp.ARMS:
        train = read_csv(exp.OUTPUT / arm / "training_metrics.csv")
        axes[0].plot([int(row["step"]) for row in train], [float(row["total_loss"]) for row in train], label=arm)
        validation = read_csv(exp.OUTPUT / arm / "validation_metrics.csv")
        grouped = defaultdict(list)
        for row in validation:
            grouped[int(row["step"])].append(float(row["selection_score"]))
        axes[1].plot(sorted(grouped), [np.mean(grouped[step]) for step in sorted(grouped)], label=arm)
        by_step = defaultdict(list)
        for row in gradients:
            if row["experiment"] == arm:
                by_step[int(row["update_step"])].append(float(row["shape_gradient_ratio"]))
        axes[2].plot(sorted(by_step), [np.mean(by_step[step]) for step in sorted(by_step)], label=arm)
    for axis in axes:
        axis.axvline(400, color="k", ls="--", lw=1); axis.grid(alpha=.25); axis.legend(fontsize=8); axis.set_xlabel("Step")
    axes[0].set_title("Training total loss"); axes[1].set_title("Validation selection score"); axes[2].set_title("Actual mean/variance gradient ratio")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150); plt.close(figure)


def number(rows, method, split, key) -> float:
    return float(next(row for row in rows if row["method"] == method and row["split"] == split)[key])


def local_number(rows, method, diagnostic, key) -> float:
    return float(next(row for row in rows if row["method"] == method and row["diagnostic"] == diagnostic)[key])


def object_number(rows, method, split, sample_id, key) -> float:
    return float(next(row for row in rows if row["method"] == method and row["split"] == split and row["sample_id"] == sample_id)[key])


def write_report(summary, per_object, local, audit) -> None:
    lines = [
        "# E3＋100% 延长训练与去 SetBranch 对比", "",
        "完整网络从第400步精确续训到600步；无Set网络从共同随机初始化训练600步。主要比较固定步数，best单独保留。", "",
        "| 方法 | 形状误差↓ | 轴向W1 µm↓ | 背景占比↓ | T03分开率↑ | T03假断率↓ | T04双层分开率↑ |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in PRIMARY:
        lines.append(
            f"| {LABELS[method]} | {number(summary, method, 'test', 'gt_scale_aligned_nrmse'):.4f} | "
            f"{number(summary, method, 'test', 'gt_axial_w1_um'):.2f} | {number(summary, method, 'test', 'background_xy_mass_fraction'):.4f} | "
            f"{local_number(local, method, 'T03_three_lines', 'separated_rate'):.3f} | "
            f"{local_number(local, method, 'T03_three_lines', 'false_break_rate'):.3f} | "
            f"{local_number(local, method, 'T04_axial_pairs', 'separated_rate_20_40um'):.3f} |"
        )
    lines += ["", "## Best", "", "| 训练 | best步数 | 形状误差↓ | 轴向W1 µm↓ |", "|---|---:|---:|---:|"]
    for arm in exp.ARMS:
        method = f"{arm}_best"
        lines.append(f"| {arm} | {audit[arm]['best_step']} | {number(summary, method, 'test', 'gt_scale_aligned_nrmse'):.4f} | {number(summary, method, 'test', 'gt_axial_w1_um'):.2f} |")
    extend_shape = number(summary, "extend600_step600", "test", "gt_scale_aligned_nrmse")
    base_shape = number(summary, "baseline400", "test", "gt_scale_aligned_nrmse")
    extend_depth = number(summary, "extend600_step600", "test", "gt_axial_w1_um")
    base_depth = number(summary, "baseline400", "test", "gt_axial_w1_um")
    base_background = number(summary, "baseline400", "test", "background_xy_mass_fraction")
    extend_background = number(summary, "extend600_step600", "test", "background_xy_mass_fraction")
    noset400_shape = number(summary, "noset600_step400", "test", "gt_scale_aligned_nrmse")
    noset600_shape = number(summary, "noset600_step600", "test", "gt_scale_aligned_nrmse")
    noset400_depth = number(summary, "noset600_step400", "test", "gt_axial_w1_um")
    noset600_depth = number(summary, "noset600_step600", "test", "gt_axial_w1_um")
    noset400_background = number(summary, "noset600_step400", "test", "background_xy_mass_fraction")
    t03_base_depth = object_number(per_object, "baseline400", "test", "T03", "gt_axial_w1_um")
    t03_extend_depth = object_number(per_object, "extend600_step600", "test", "T03", "gt_axial_w1_um")
    t02_base_depth = object_number(per_object, "baseline400", "test", "T02", "gt_axial_w1_um")
    t02_extend_depth = object_number(per_object, "extend600_step600", "test", "T02", "gt_axial_w1_um")
    t04_base_depth = object_number(per_object, "baseline400", "test", "T04", "gt_axial_w1_um")
    t04_extend_depth = object_number(per_object, "extend600_step600", "test", "T04", "gt_axial_w1_um")
    base_false_break = local_number(local, "baseline400", "T03_three_lines", "false_break_rate")
    noset400_false_break = local_number(local, "noset600_step400", "T03_three_lines", "false_break_rate")
    noset600_false_break = local_number(local, "noset600_step600", "T03_three_lines", "false_break_rate")
    base_v03_localized = local_number(local, "baseline400", "V03_60_beads", "localized_rate")
    noset400_v03_localized = local_number(local, "noset600_step400", "V03_60_beads", "localized_rate")
    base_v03_false = local_number(local, "baseline400", "V03_60_beads", "global_false_peaks_mean")
    noset400_v03_false = local_number(local, "noset600_step400", "V03_60_beads", "global_false_peaks_mean")
    lines += ["", "## 直接结论", "",
              f"**继续训练到600步没有形成全面优势。** 测试集平均形状误差只从 {base_shape:.4f} 降到 {extend_shape:.4f}，改善 {100.0 * (base_shape - extend_shape) / base_shape:.2f}%；平均轴向W1从 {base_depth:.3f} 变为 {extend_depth:.3f} µm，基本不变。背景占比从 {base_background:.4f} 降到 {extend_background:.4f}，减少 {100.0 * (base_background - extend_background) / base_background:.1f}%，这是最明确的收益。", "",
              f"600步对不同对象的影响方向不一致：T03轴向W1从 {t03_base_depth:.2f} 降到 {t03_extend_depth:.2f} µm，但T02从 {t02_base_depth:.2f} 升到 {t02_extend_depth:.2f} µm，T04从 {t04_base_depth:.2f} 升到 {t04_extend_depth:.2f} µm。T03分开率仍为 0.667，假断率仍为 {base_false_break:.3f}，因此不能说继续训练提高了总体横向分辨率。", "",
              f"**SetBranch应当保留。** 同为400步，去掉Set后测试形状误差从 {base_shape:.4f} 升到 {noset400_shape:.4f}，轴向W1从 {base_depth:.2f} 升到 {noset400_depth:.2f} µm，背景变为完整网络的 {noset400_background / base_background:.1f} 倍。T03分开率没有提高，假断率却从 {base_false_break:.3f} 升到 {noset400_false_break:.3f}。", "",
              f"无Set也有局部优点：V03微球定位率从 {base_v03_localized:.3f} 升到 {noset400_v03_localized:.3f}，平均假峰从 {base_v03_false:.1f} 降到 {noset400_v03_false:.1f}；T04单项轴向误差也较低。但这些收益没有扩展到T02、T03、点目标诊断和测试总体，所以只能视为对象偏向，不能据此删除SetBranch。", "",
              f"无Set从400继续到600后，轴向W1由 {noset400_depth:.2f} 降到 {noset600_depth:.2f} µm，T03假断率由 {noset400_false_break:.3f} 降到 {noset600_false_break:.3f}；形状误差仍为 {noset600_shape:.4f}，没有追回完整网络，背景也没有恢复。", "",
              "**基线决定：继续保留E3＋100% mean结构梯度的完整网络第400步。** 完整网络第600步可作为低背景、偏T03的备选检查点；无Set模型不升级为基线。best检查点仍单独保留，但它们按物理验证分数选出，不替代固定步数的主比较。", "",
              "T04继续使用原生十层；10 µm双层没有层间采样点，不做插值分离结论。所有结论来自一个共同训练种子。", "",
              "## 结果入口", "",
              "- [总体指标](analysis/summary.csv)", "- [逐对象指标](analysis/per_object.csv)", "- [局部结构指标](analysis/local_summary.csv)",
              "- [全部逐项指标](analysis/metrics.csv)", "- [训练和梯度审计](analysis/training_audit.json)",
              "- [实际重建图册](analysis/actual_reconstruction_comparison/README_ZH.md)", "- [轴向对比PDF](analysis/轴向主要对比.pdf)", ""]
    (exp.OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict:
    started = time.time()
    for arm in exp.ARMS:
        marker = json.loads((exp.OUTPUT / "evaluation" / arm / "complete.json").read_text())
        if not marker.get("complete"):
            raise ValueError(f"Incomplete evaluation: {arm}")
    tables = collect_tables()
    analysis = exp.OUTPUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        v3.csv_write(analysis / f"{name}.csv", rows)
    object_rows, summary = aggregation.aggregate_metrics(tables["metrics"])
    local = aggregation._aggregate_local(tables)
    v3.csv_write(analysis / "per_object.csv", object_rows)
    v3.csv_write(analysis / "summary.csv", summary)
    v3.csv_write(analysis / "local_summary.csv", local)
    audit, gradients = training_audit()
    old.write_json(analysis / "training_audit.json", audit)
    draw_training(gradients, analysis / "training_validation_gradient_trends.png")

    all_cases = cases()
    by_id = {case["id"]: case for case in all_cases}
    chosen = [case for case in all_cases if case["split"] == "priority"]
    chosen += [by_id[name] for name in ("T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01")]
    book_cases = {"T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01", "axial_pairs_r01", "axial_pairs_r02", "axial_pairs_r03"}
    manifest = []
    with PdfPages(analysis / "轴向主要对比.pdf") as book:
        for case in chosen:
            for display in ("shape", "shared"):
                destination = analysis / "actual_reconstruction_comparison" / case["id"] / f"{display}.png"
                manifest.append(draw_views(case, display, destination, book if case["id"] in book_cases else None))
        for sample in ("T02", "T03", "T04"):
            case = by_id[f"{sample}_subset_01"]
            destination = analysis / "actual_reconstruction_comparison" / case["id"] / "all_layers_shared.png"
            manifest.append(draw_layers(case, destination, book))
            profile = analysis / f"{sample}_fixed_profiles.png"
            manifest.append(fixed_profiles(case, profile, book))
    old.write_json(analysis / "figure_manifest.json", manifest)
    atlas = ["# 实际重建对比图册", "", "四个网络结果在shared图中使用共同亮度；XZ/YZ为求和投影，保留原生十层。", ""]
    for case in chosen:
        atlas += [f"## {case['id']}", ""]
        for row in manifest:
            if row["case_id"] == case["id"]:
                atlas.append(f"- [{row['display']}]({(exp.OUTPUT / row['path']).as_posix()})")
        atlas.append("")
    atlas_path = analysis / "actual_reconstruction_comparison/README_ZH.md"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.write_text("\n".join(atlas), encoding="utf-8")
    write_report(summary, object_rows, local, audit)

    raw = []
    for arm, roles in evaluation.ROLE_FILES.items():
        for role, _ in roles:
            predictions = list((exp.OUTPUT / "evaluation" / arm / role).glob("*/reconstruction.npy"))
            anchors = list((exp.OUTPUT / "evaluation" / arm / role).glob("*/anchor.npy"))
            if len(predictions) != 94 or len(anchors) != 94:
                raise ValueError(f"Raw artifact count failed: {arm}/{role}")
            raw.extend(predictions + anchors)
    artifacts = [
        exp.OUTPUT / "REPORT_ZH.md", analysis / "metrics.csv", analysis / "summary.csv",
        analysis / "per_object.csv", analysis / "local_summary.csv", analysis / "training_audit.json",
        analysis / "figure_manifest.json", analysis / "轴向主要对比.pdf", atlas_path,
    ]
    acceptance = {
        "passed": True,
        "methods": list(ALL_METHODS),
        "metric_rows": len(tables["metrics"]),
        "new_predictions": 470,
        "new_anchors": 470,
        "figures": len(manifest),
        "training": audit,
        "artifacts": {str(path): old.sha256(path) for path in artifacts},
        "raw_new_artifacts_sha256": {str(path.relative_to(exp.OUTPUT)): old.sha256(path) for path in raw},
        "reference_final_acceptance_sha256": old.sha256(exp.BASELINE_OUTPUT / "final_acceptance.json"),
        "finished_unix": time.time(),
        "seconds": time.time() - started,
    }
    old.write_json(exp.OUTPUT / "final_acceptance.json", acceptance)
    return acceptance
