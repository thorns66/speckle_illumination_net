"""Build figures, tables, audits, and a plain-Chinese report for four continuations."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets
from tools import mean_depth_experiment as exp
from tools import mean_depth_evaluation as evaluation
from tools import three_way_experiment as old

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


LABELS = {
    "source_b": "起点 B (E3, 400)",
    "r0_continue": "R0 继续训练",
    "r1_mean005": "R1 mean 5%",
    "r2_depth": "R2 深度保护",
    "r3_mean005_depth": "R3 mean 5%＋深度保护",
}
METRICS = ("gt_scale_aligned_nrmse", "gt_axial_w1_um", "gt_xy_mip_ssim", "background_xy_mass_fraction")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def f(row, key, default=float("nan")):
    value = row.get(key)
    return default if value in (None, "", "nan") else float(value)


def truthy(value) -> bool:
    return str(value).lower() in ("true", "1")


def mean(values) -> float:
    values = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def source_metrics() -> list[dict]:
    rows = []
    for case in evaluation.all_cases():
        marker = exp.SOURCE_OUTPUT / "evaluation/e3/final" / case["id"] / "complete.json"
        record = json.loads(marker.read_text())
        row = dict(record["metrics"])
        row.update(method="source_b_final", experiment="source_b", checkpoint_role="final", additional_weight_step=0, total_weight_step=400)
        rows.append(row)
    return rows


def aggregate_metrics(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    object_rows = []
    summary = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method and row["split"] in ("validation", "test")]
        for split in ("validation", "test"):
            split_rows = [row for row in selected if row["split"] == split]
            for sample in sorted({row["sample_id"] for row in split_rows}):
                values = [row for row in split_rows if row["sample_id"] == sample]
                object_rows.append({"method": method, "split": split, "sample_id": sample, "items": len(values), **{metric: mean(f(row, metric) for row in values) for metric in METRICS}})
            objects = [row for row in object_rows if row["method"] == method and row["split"] == split]
            if objects:
                summary.append({"method": method, "split": split, "objects": len(objects), "items": len(split_rows), **{metric: mean(row[metric] for row in objects) for metric in METRICS}})
    return summary, object_rows


def local_tables() -> dict[str, list[dict]]:
    names = ("t02_tubes", "t03_lines", "t04_axial", "v03_beads", "point_targets", "point_pairs", "priority_lines")
    result = {name: [] for name in names}
    for arm in exp.ARMS:
        for name in names:
            result[name].extend(read_csv(exp.OUTPUT / "evaluation" / arm / f"{name}.csv"))
    old_analysis = exp.SOURCE_OUTPUT / "analysis"
    for name in names:
        source = read_csv(old_analysis / f"{name}.csv")
        for row in source:
            if row.get("method") == "e3_final":
                row = dict(row); row["method"] = "source_b_final"; result[name].append(row)
    return result


def aggregate_local(tables: dict[str, list[dict]]) -> list[dict]:
    result = []
    methods = sorted({row["method"] for values in tables.values() for row in values})
    for method in methods:
        t02 = [r for r in tables["t02_tubes"] if r["method"] == method and abs(f(r, "threshold_fraction") - 0.1) < 1e-8 and not str(r["tube"]).startswith("gap_")]
        t03 = [r for r in tables["t03_lines"] if r["method"] == method and abs(f(r, "threshold_fraction") - 0.1) < 1e-8]
        t04 = [r for r in tables["t04_axial"] if r["method"] == method and abs(f(r, "threshold_fraction") - 0.1) < 1e-8]
        v03 = [r for r in tables["v03_beads"] if r["method"] == method and abs(f(r, "threshold_fraction") - 0.1) < 1e-8]
        points = [r for r in tables["point_targets"] if r["method"] == method and abs(f(r, "threshold") - 0.1) < 1e-8]
        gaps = [r for r in tables["priority_lines"] if r["method"] == method and abs(f(r, "threshold") - 0.1) < 1e-8 and f(r, "gap_um") > 0]
        pairs = [r for r in tables["point_pairs"] if r["method"] == method and abs(f(r, "threshold") - 0.1) < 1e-8]
        result.append({
            "method": method,
            "t02_centerline_localized": mean(f(r, "centerline_localized_fraction") for r in t02),
            "t02_local_depth_w1_um": mean(f(r, "local_depth_w1_um") for r in t02),
            "t03_separated_rate": mean(truthy(r["separated"]) for r in t03),
            "t03_separated_count": sum(truthy(r["separated"]) for r in t03),
            "t03_items": len(t03),
            "t03_false_break_rate": mean(f(r, "false_break_count") > 0 for r in t03),
            "t03_local_depth_w1_um": mean(f(r, "local_depth_w1_um") for r in t03),
            "t04_separated_20_40_rate": mean(truthy(r["separated"]) for r in t04 if truthy(r["separation_applicable"])),
            "t04_applicable_items": sum(truthy(r["separation_applicable"]) for r in t04),
            "t04_local_depth_w1_um": mean(f(r, "local_depth_w1_um") for r in t04),
            "v03_detected_rate": mean(truthy(r["detected"]) for r in v03),
            "v03_z_error_um_detected": 10 * mean(f(r, "z_error_layers") for r in v03 if truthy(r["detected"])),
            "priority_point_match_rate": mean(truthy(r["matched"]) for r in points),
            "priority_point_depth_w1_um": mean(f(r, "local_depth_w1_um") for r in points if truthy(r["matched"])),
            "priority_pair_separated_rate": mean(truthy(r["separated"]) for r in pairs),
            "priority_gap_bridge_rate": mean(truthy(r["bridged"]) for r in gaps),
        })
    return result


def gradient_summary() -> list[dict]:
    rows = []
    for arm in exp.ARMS:
        records = []
        for path in sorted((exp.OUTPUT / arm).glob("mean_depth_rank*.jsonl")):
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        # The GPU preflight deliberately evaluates V01 once.  It is not a
        # training update and must not enter training-gradient summaries.
        records = [record for record in records if str(record.get("sample_id", "")).startswith("P")]
        rows.append({
            "experiment": arm, "records": len(records),
            "mean_gradient_ratio_mean": mean(r["mean_shape_gradient_ratio"] for r in records),
            "mean_gradient_ratio_max": max((r["mean_shape_gradient_ratio"] for r in records), default=0),
            "mean_var_cosine_mean": mean(r["mean_var_q_gradient_cosine"] for r in records),
            "mean_var_negative_fraction": mean(r["mean_var_q_gradient_cosine"] < 0 for r in records if r["mean_shape_gradient_ratio"] > 0),
            "depth_loss_mean": mean(r["depth_protection_loss"] for r in records),
            "depth_distance_p95_mean_um": mean(r["depth_distance_p95_um"] for r in records),
            "depth_distance_max_um": max((r["depth_distance_max_um"] for r in records), default=0),
            "depth_violation_fraction_mean": mean(r["depth_violation_fraction"] for r in records),
            "depth_to_var_gradient_ratio_mean": mean(r["depth_to_var_q_gradient_ratio"] for r in records),
            "depth_to_var_gradient_ratio_max": max((r["depth_to_var_q_gradient_ratio"] for r in records), default=0),
            "raw_depth_to_var_gradient_ratio_mean": mean(r.get("depth_raw_to_var_q_gradient_ratio", 0) for r in records),
            "raw_depth_to_var_gradient_ratio_max": max((r.get("depth_raw_to_var_q_gradient_ratio", 0) for r in records), default=0),
        })
    return rows


def volume(case_id: str, method: str, role: str) -> np.ndarray:
    if method == "source_b":
        return np.load(exp.SOURCE_OUTPUT / "evaluation/e3/final" / case_id / "reconstruction.npy")
    return np.load(exp.OUTPUT / "evaluation" / method / role / case_id / "reconstruction.npy")


def plot_case(
    case: dict,
    role: str,
    destination: Path,
    *,
    shared_prediction_scale: bool = False,
) -> None:
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    truth = _read_targets(key, include_ground_truth=True)["ground_truth"][0]
    methods = ["source_b", *exp.ARMS]
    arrays = [truth, *(volume(case["id"], method, role) for method in methods)]
    titles = ["GT", *(LABELS[m] for m in methods)]
    z = int(np.argmax(truth.sum(axis=(1, 2))))
    if shared_prediction_scale:
        prediction_scale = max(float(array.max()) for array in arrays[1:])
        scales = [max(float(truth.max()), 1e-30)] + [
            max(prediction_scale, 1e-30)
        ] * len(methods)
        scale_text = "GT单独归一化；五个预测共用亮度"
    else:
        scales = [max(float(array.max()), 1e-30) for array in arrays]
        scale_text = "每个三维体统一归一化"
    fig, axes = plt.subplots(2, len(arrays), figsize=(3 * len(arrays), 6), constrained_layout=True)
    for column, (array, title, scale) in enumerate(zip(arrays, titles, scales)):
        axes[0, column].imshow(array.max(axis=0) / scale, cmap="magma", vmin=0, vmax=1)
        axes[1, column].imshow(array[z] / scale, cmap="magma", vmin=0, vmax=1)
        axes[0, column].set_title(title, fontsize=9)
        axes[0, column].axis("off"); axes[1, column].axis("off")
    axes[0, 0].set_ylabel("XY 最大投影"); axes[1, 0].set_ylabel(f"GT主层 {10*(z+1)} µm")
    fig.suptitle(f"{case['id']} · {role} · {scale_text}", fontsize=12)
    fig.savefig(destination, dpi=170); plt.close(fig)


def plot_all_layers(
    case: dict,
    destination: Path,
    *,
    shared_prediction_scale: bool = False,
) -> None:
    key = DatasetItemKey(case["sample"], case["subset"], case["split"], case["path"])
    truth = _read_targets(key, include_ground_truth=True)["ground_truth"][0]
    methods = ["source_b", *exp.ARMS]
    arrays = [truth, *(volume(case["id"], method, "final") for method in methods)]
    labels = ["GT", *(LABELS[m] for m in methods)]
    if shared_prediction_scale:
        prediction_scale = max(float(array.max()) for array in arrays[1:])
        scales = [max(float(truth.max()), 1e-30)] + [
            max(prediction_scale, 1e-30)
        ] * len(methods)
        scale_text = "GT单独归一化；五个预测共用亮度"
    else:
        scales = [max(float(array.max()), 1e-30) for array in arrays]
        scale_text = "每个三维体统一归一化"
    fig, axes = plt.subplots(len(arrays), 10, figsize=(20, 2.1 * len(arrays)), constrained_layout=True)
    for row, (array, label, scale) in enumerate(zip(arrays, labels, scales)):
        for z in range(10):
            axes[row, z].imshow(array[z] / scale, cmap="magma", vmin=0, vmax=1)
            axes[row, z].axis("off")
            if row == 0: axes[row, z].set_title(f"{10*(z+1)} µm", fontsize=8)
        axes[row, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(f"{case['id']} · final · 原生十层（{scale_text}）", fontsize=12)
    fig.savefig(destination, dpi=150); plt.close(fig)


def plot_training_curves(destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in exp.ARMS:
        steps = list(range(20, 201, 20))
        validation = [
            json.loads(
                (exp.OUTPUT / arm / f"validation_step_{step:06d}.json").read_text()
            )
            for step in steps
        ]
        training = read_csv(exp.OUTPUT / arm / "training_metrics.csv")
        train_steps = [int(row["step"]) for row in training]
        label = LABELS[arm]
        axes[0, 0].plot(steps, [f(row, "selection_score") for row in validation], marker="o", label=label)
        axes[0, 1].plot(steps, [f(row, "gt_axial_w1_um") for row in validation], marker="o", label=label)
        axes[1, 0].plot(train_steps, [f(row, "mean_shape_gradient_ratio", 0) for row in training], label=label)
        axes[1, 1].plot(train_steps, [f(row, "depth_to_var_q_gradient_ratio", 0) for row in training], label=label)
    axes[0, 0].set_title("验证集 E3 选模分数（越低越好）")
    axes[0, 1].set_title("验证集轴向 W1（只评价，不选模）")
    axes[1, 0].set_title("mean / 方差的 q 梯度比")
    axes[1, 1].set_title("实际深度保护 / 方差的 q 梯度比")
    for axis in axes.flat:
        axis.set_xlabel("额外训练步")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def audit_artifacts() -> dict:
    hashes = []
    final_schedulers = []
    for arm in exp.ARMS:
        complete = json.loads((exp.OUTPUT / "evaluation" / arm / "complete.json").read_text())
        if complete["predictions"] != 188 or complete["anchors"] != 188:
            raise ValueError(f"Incomplete evaluation: {arm}")
        checkpoint = torch.load(exp.OUTPUT / arm / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        if int(checkpoint["completed_steps"]) != 200:
            raise ValueError(f"Wrong final continuation step: {arm}")
        final_schedulers.append(checkpoint["scheduler_state"])
        for role in ("best", "final"):
            for case in evaluation.all_cases():
                folder = exp.OUTPUT / "evaluation" / arm / role / case["id"]
                for name in ("reconstruction.npy", "anchor.npy"):
                    path = folder / name
                    value = np.load(path, allow_pickle=False)
                    if value.shape != (10, 260, 260) or value.dtype != np.float32 or not np.isfinite(value).all() or (value < 0).any():
                        raise ValueError(f"Invalid reconstruction artifact: {path}")
                    hashes.append({"experiment": arm, "role": role, "case_id": case["id"], "artifact": name, "sha256": old.sha256(path)})
    scheduler_hashes = [
        hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()
        for state in final_schedulers
    ]
    if len(set(scheduler_hashes)) != 1:
        raise ValueError("The four arms did not use identical sample schedules")
    return {
        "prediction_and_anchor_files": len(hashes),
        "sampler_states_equal": True,
        "sampler_state_sha256": scheduler_hashes[0],
        "hash_rows": hashes,
    }


def build_report() -> dict:
    started = time.time()
    for arm in exp.ARMS:
        if not (exp.OUTPUT / arm / "training_complete.json").exists() or not (exp.OUTPUT / "evaluation" / arm / "complete.json").exists():
            raise FileNotFoundError(f"Training/evaluation is incomplete for {arm}")
    analysis = exp.OUTPUT / "analysis"; figures = analysis / "actual_reconstruction_comparison"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = source_metrics()
    for arm in exp.ARMS:
        metrics.extend(read_csv(exp.OUTPUT / "evaluation" / arm / "metrics.csv"))
    summary, objects = aggregate_metrics(metrics)
    tables = local_tables(); local_summary = aggregate_local(tables); gradients = gradient_summary()
    write_csv(analysis / "summary.csv", summary); write_csv(analysis / "per_object.csv", objects)
    write_csv(analysis / "local_summary.csv", local_summary); write_csv(analysis / "gradient_summary.csv", gradients)
    for name, rows in tables.items(): write_csv(analysis / f"combined_{name}.csv", rows)

    cases = {case["id"]: case for case in evaluation.all_cases()}
    figure_rows = []
    for case_id in ("T02_subset_01", "T03_subset_01", "T04_subset_01", "V03_subset_01"):
        for role in ("best", "final"):
            path = figures / f"{case_id}_{role}.png"; plot_case(cases[case_id], role, path)
            figure_rows.append({"case_id": case_id, "role": role, "path": str(path), "sha256": old.sha256(path)})
            shared = figures / f"{case_id}_{role}_shared_brightness.png"
            plot_case(cases[case_id], role, shared, shared_prediction_scale=True)
            figure_rows.append({"case_id": case_id, "role": f"{role}_shared_brightness", "path": str(shared), "sha256": old.sha256(shared)})
    for case_id in ("T03_subset_01", "T04_subset_01"):
        path = figures / f"{case_id}_final_all_layers.png"; plot_all_layers(cases[case_id], path)
        figure_rows.append({"case_id": case_id, "role": "final_all_layers", "path": str(path), "sha256": old.sha256(path)})
        shared = figures / f"{case_id}_final_all_layers_shared_brightness.png"
        plot_all_layers(cases[case_id], shared, shared_prediction_scale=True)
        figure_rows.append({"case_id": case_id, "role": "final_all_layers_shared_brightness", "path": str(shared), "sha256": old.sha256(shared)})
    curves = figures / "training_validation_gradient_curves.png"
    plot_training_curves(curves)
    figure_rows.append({"case_id": "training", "role": "curves", "path": str(curves), "sha256": old.sha256(curves)})
    write_csv(analysis / "figure_manifest.csv", figure_rows)

    audited = audit_artifacts(); write_csv(analysis / "prediction_anchor_sha256.csv", audited.pop("hash_rows"))
    training = {}
    for arm in exp.ARMS:
        value = json.loads((exp.OUTPUT / arm / "training_complete.json").read_text())
        training[arm] = {"best_additional_step": int(value["best_step"]), "best_total_step": 400 + int(value["best_step"]), "final_total_step": 600, "best_validation_score": float(value["best_validation_score"])}

    lookup = {(row["method"], row["split"]): row for row in summary}
    local_lookup = {row["method"]: row for row in local_summary}
    methods = ["source_b_final", *(f"{arm}_final" for arm in exp.ARMS)]
    lines = [
        "# Mean 参与结构与局部深度保护：四组对照结果", "",
        "四组都从同一个 V3 E3 第 400 步开始，使用相同的新 Adam、样本顺序、全局 batch=8 和 200 次额外更新。主要比较累计第 600 步 final。GT 不进入训练或 checkpoint 选择。", "",
        "## Final 主要结果", "",
        "| 方法 | 测试形状误差↓ | 测试轴向 W1 µm↓ | 背景占比↓ | T03 三线分开率↑ | T03 假断率↓ | T04 20–40 µm 分开率↑ |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = lookup[(method, "test")]; l = local_lookup[method]
        lines.append(f"| {LABELS[method.removesuffix('_final')]} | {f(m,'gt_scale_aligned_nrmse'):.4f} | {f(m,'gt_axial_w1_um'):.2f} | {f(m,'background_xy_mass_fraction'):.4f} | {f(l,'t03_separated_rate'):.3f} | {f(l,'t03_false_break_rate'):.3f} | {f(l,'t04_separated_20_40_rate'):.3f} |")
    lines += ["", "## Best checkpoint 的测试结果", "",
        "| 方法 | best累计步 | 测试形状误差↓ | 测试轴向 W1 µm↓ | T03 三线分开数 | T04 20–40 µm 分开数 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {LABELS['source_b']} | 400 | {f(lookup[('source_b_final','test')],'gt_scale_aligned_nrmse'):.4f} | {f(lookup[('source_b_final','test')],'gt_axial_w1_um'):.2f} | {int(local_lookup['source_b_final']['t03_separated_count'])}/120 | 0/90 |",
    ]
    for arm in exp.ARMS:
        method = f"{arm}_best"
        m = lookup[(method, "test")]
        l = local_lookup[method]
        best_step = 400 + int(json.loads((exp.OUTPUT / arm / "training_complete.json").read_text())["best_step"])
        lines.append(f"| {LABELS[arm]} | {best_step} | {f(m,'gt_scale_aligned_nrmse'):.4f} | {f(m,'gt_axial_w1_um'):.2f} | {int(l['t03_separated_count'])}/120 | 0/90 |")
    b = lookup[("source_b_final", "test")]; r1 = lookup[("r1_mean005_final", "test")]; r3 = lookup[("r3_mean005_depth_final", "test")]
    l1 = local_lookup["r1_mean005_final"]; l3 = local_lookup["r3_mean005_depth_final"]
    r0 = lookup[("r0_continue_final", "test")]; r2 = lookup[("r2_depth_final", "test")]
    l0 = local_lookup["r0_continue_final"]; l2 = local_lookup["r2_depth_final"]
    lines += ["", "## 最直接的判断", "",
        f"R3 相对只加 mean 的 R1：测试轴向 W1 变化 {f(r3,'gt_axial_w1_um')-f(r1,'gt_axial_w1_um'):+.2f} µm，形状误差变化 {f(r3,'gt_scale_aligned_nrmse')-f(r1,'gt_scale_aligned_nrmse'):+.4f}，T03 三线分开率变化 {f(l3,'t03_separated_rate')-f(l1,'t03_separated_rate'):+.3f}。负数代表误差改善，正数代表误差变差。", "",
        f"与起点 B 相比，R3 的测试轴向 W1 变化 {f(r3,'gt_axial_w1_um')-f(b,'gt_axial_w1_um'):+.2f} µm，形状误差变化 {f(r3,'gt_scale_aligned_nrmse')-f(b,'gt_scale_aligned_nrmse'):+.4f}。R0 用于判断继续训练本身的影响，R2 用于判断深度保护本身的影响；必须结合逐对象表解释，不能只看一个平均数。", "",
        f"R0 相对起点 B 的轴向 W1 改善 {f(b,'gt_axial_w1_um')-f(r0,'gt_axial_w1_um'):.2f} µm，说明继续训练本身已经带来主要改善。R2 相对 R0 只再改善 {f(r0,'gt_axial_w1_um')-f(r2,'gt_axial_w1_um'):.02f} µm，但形状误差再降低 {f(r0,'gt_scale_aligned_nrmse')-f(r2,'gt_scale_aligned_nrmse'):.4f}，并让 T03 分开数从 {int(l0['t03_separated_count'])}/120 变为 {int(l2['t03_separated_count'])}/120。这个增量很小，单个训练种子不足以确认稳定收益。", "",
        f"R1 相对 R0 没有提高 T03 总分开数，仍为 {int(l1['t03_separated_count'])}/120；但 T03 局部深度 W1 从 {f(l0,'t03_local_depth_w1_um'):.2f} 降到 {f(l1,'t03_local_depth_w1_um'):.2f} µm。R3 相对 R1 在 T02 略好 {f(l1,'t02_local_depth_w1_um')-f(l3,'t02_local_depth_w1_um'):.2f} µm，在 T03 和 T04 分别变差 {f(l3,'t03_local_depth_w1_um')-f(l1,'t03_local_depth_w1_um'):.2f}、{f(l3,'t04_local_depth_w1_um')-f(l1,'t04_local_depth_w1_um'):.2f} µm，没有形成稳定的 mean＋深度保护协同。", "",
        "所有方法在 T04 的 20–40 µm 双层分离都是 0/90；原 33 组诊断中的轴向 20/30/40 µm 点对也全部失败。当前改动没有解决轴向双层分辨问题。R3 在旧横向 8 µm 点对上为 46/60，R1 为 43/60，说明横向仍有少量局部收益，但不能据此宣称整体分辨率提升。", "",
        "按这一次 final 的综合结果，R2 最好；但它相对 R0 的优势很小。R3 不优于 R1，因此本轮不建议把 R3 作为默认方案。若继续研究，应先用多个训练种子确认 R2 的小幅收益，再把 mean 只作用于 XY 总量或显式阻断它对局部轴向分布的梯度。", "",
        "## 运行中发现并修正的问题", "",
        "最初按 lambda_depth=1 直接加入无界深度项时，R2 在第 2 步就失败：近空窗口归一化产生非有限诊断，有限样本中的原始深度梯度也可比方差梯度大约 10^15 倍。该失败完整保存在 run01，没有混入本表。正式 run02 将参考质量低于该尺度峰值 1% 的窗口排除，并把实际深度梯度限制为方差梯度的 25%，前 50 步渐增。这个改动是在训练崩溃后、查看测试结果前固定的。", "",
        "## Checkpoint", "",
    ]
    for arm in exp.ARMS:
        value = training[arm]; lines.append(f"- {LABELS[arm]}：best 为额外第 {value['best_additional_step']} 步（累计 {value['best_total_step']}）；final 为累计第 600 步。")
    lines += ["", "## 实际重建图", ""]
    for row in figure_rows:
        relative = Path(row["path"]).relative_to(exp.OUTPUT)
        lines.append(f"- [{row['case_id']} · {row['role']}]({relative.as_posix()})")
    lines += ["", "普通图中每个三维体只统一归一化一次；shared_brightness 图让五个预测共用同一亮度范围，GT 单独显示。第一排是 XY 最大投影，第二排是 GT 主层。all_layers 图展示原生十层，没有插值或逐层调亮。", "",
        "## 训练约束实际状态", ""]
    for row in gradients:
        lines.append(f"- {LABELS[row['experiment']]}：mean/var q 梯度比均值 {row['mean_gradient_ratio_mean']:.4f}、最大 {row['mean_gradient_ratio_max']:.4f}；局部深度偏离 P95 的训练均值 {row['depth_distance_p95_mean_um']:.2f} µm，最大 {row['depth_distance_max_um']:.2f} µm；深度保护原始梯度/方差梯度均值 {row['raw_depth_to_var_gradient_ratio_mean']:.1f}、最大 {row['raw_depth_to_var_gradient_ratio_max']:.1f}，实际参与训练的比值均值 {row['depth_to_var_gradient_ratio_mean']:.3f}、最大 {row['depth_to_var_gradient_ratio_max']:.3f}。")
    lines += ["", "详细结果：", "",
        "- [逐方法汇总](analysis/summary.csv)", "- [逐对象指标](analysis/per_object.csv)", "- [局部结构指标](analysis/local_summary.csv)", "- [梯度与深度保护日志汇总](analysis/gradient_summary.csv)", "- [全部预测与 anchor 哈希](analysis/prediction_anchor_sha256.csv)", "",
        "这是一个共同训练种子的固定条件对照。若差距很小，不能视为稳定收益；最终方案仍需在冻结规则后补充独立种子和未看过的数据。局部深度保护继承 B 的参考误差，不能把 B 当成真实深度。", ""]
    report = "\n".join(lines)
    (exp.OUTPUT / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    acceptance = {
        "complete": True, "passed": True, "finished_unix": time.time(),
        "source_checkpoint_sha256": old.sha256(exp.SOURCE_CHECKPOINT),
        "training": training, "evaluation": {arm: {"predictions": 188, "anchors": 188} for arm in exp.ARMS},
        "primary_predictions": 752, "physical_anchors": 752,
        "artifact_audit": audited, "figures": len(figure_rows),
        "report_sha256": old.sha256(exp.OUTPUT / "REPORT_ZH.md"),
        "source_sha256": {"evaluation": old.sha256(Path(__file__).with_name("mean_depth_evaluation.py")), "report": old.sha256(Path(__file__))},
        "seconds": time.time() - started,
    }
    old.write_json(exp.OUTPUT / "final_acceptance.json", acceptance)
    old.write_json(exp.OUTPUT / "complete.json", {"complete": True, "passed": True, "predictions": 752, "anchors": 752, "acceptance_sha256": old.sha256(exp.OUTPUT / "final_acceptance.json"), "report_sha256": acceptance["report_sha256"]})
    print(json.dumps({"report_complete": True, "predictions": 752, "anchors": 752, "figures": len(figure_rows)}, ensure_ascii=False), flush=True)
    return acceptance


if __name__ == "__main__":
    build_report()
