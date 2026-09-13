"""Complete V4 P12 anchor comparison metrics, figures, and Chinese report."""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.matlab_multivolume_dataset import (
    DatasetItemKey, _read_targets, load_dataset_index, load_inference_input,
)
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as evaluation
from tools import v4_p12_anchor_compare_experiment as exp


METHODS = ("mean_rl3", "taylor_rl3_sqrt", *exp.ARMS)
LABELS = {
    "mean_rl3": "Mean-RL3",
    "taylor_rl3_sqrt": "Taylor-RL3-sqrt",
    "taylor_anchor_e3_mean100": "Taylor 主分支网络（含 P12）",
    "mean_anchor_e3_mean100": "Mean 主分支网络（含 P12）",
    "ground_truth": "GT",
}
OLD_ANALYSIS = exp.ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _normalized(volume: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(volume, np.float64), 0)
    return positive / max(float(positive.sum()), 1e-30)


def _case(sample: str, subset: int) -> DatasetItemKey:
    indexed, _ = load_dataset_index(exp.DATA)
    return next(
        item for item in indexed["test"]
        if item.sample_id == sample and item.subset_index == subset
    )


def case_volumes(sample: str, subset: int, role: str = "final") -> tuple[dict[str, np.ndarray], np.ndarray]:
    key = _case(sample, subset)
    raw = load_inference_input(key.sample_dir, subset, var_feature_representation="sqrt")
    target = _read_targets(key, include_ground_truth=True)
    volumes = {
        "mean_rl3": np.asarray(raw["g_mean"][0], np.float32),
        "taylor_rl3_sqrt": np.asarray(raw["f_var"][0], np.float32),
    }
    for arm in exp.ARMS:
        volumes[arm] = np.load(
            exp.OUTPUT / "evaluation" / arm / role / f"{sample}_subset_{subset:02d}"
            / "reconstruction.npy", allow_pickle=False,
        )
    return volumes, np.asarray(target["ground_truth"][0], np.float32)


def augment_anchor_artifacts() -> int:
    cases = evaluation.all_cases()
    count = 0
    for arm in exp.ARMS:
        anchor_kind = exp.ANCHORS[arm]
        for role in ("best", "final"):
            for case in cases:
                folder = exp.OUTPUT / "evaluation" / arm / role / case["id"]
                record_path = folder / "complete.json"
                record = json.loads(record_path.read_text(encoding="utf-8"))
                inputs = load_inference_input(
                    case["path"], case["subset"], var_feature_representation="sqrt"
                )
                volume = inputs["f_var"][0] if anchor_kind == "taylor_sqrt" else inputs["g_mean"][0]
                beta = float(record["metrics"]["beta"])
                pre_gain_anchor = beta * np.asarray(volume, np.float32)
                prediction = np.load(folder / "reconstruction.npy", allow_pickle=False)
                physical_anchor = np.load(folder / "anchor.npy", allow_pickle=False)
                correction = prediction - physical_anchor
                np.save(folder / "pre_gain_anchor.npy", pre_gain_anchor.astype(np.float32))
                np.save(folder / "effective_correction.npy", correction.astype(np.float32))
                record["anchor_artifacts"] = {
                    "anchor_kind": anchor_kind,
                    "correction_to_anchor_l2": float(
                        np.linalg.norm(correction) / max(np.linalg.norm(physical_anchor), 1e-30)
                    ),
                    "signed_correction_mass_ratio": float(
                        correction.sum(dtype=np.float64)
                        / max(physical_anchor.sum(dtype=np.float64), 1e-30)
                    ),
                    "pre_gain_anchor_sha256": old.sha256(folder / "pre_gain_anchor.npy"),
                    "effective_correction_sha256": old.sha256(folder / "effective_correction.npy"),
                }
                old.write_json(record_path, record)
                count += 1
    if count != 376:
        raise ValueError(f"Expected 376 anchor augmentations, found {count}")
    return count


def quality_and_stability() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for sample in ("T02", "T03", "T04"):
        by_method = {method: [] for method in METHODS}
        for subset in range(1, 11):
            volumes, truth = case_volumes(sample, subset)
            for method, volume in volumes.items():
                by_method[method].append(volume)
                quality.append({
                    "sample_id": sample, "subset": subset, "method": method,
                    **evaluation._structure_row(volume, truth),
                })
        for method, values in by_method.items():
            q = np.stack([_normalized(value) for value in values])
            center = q.mean(axis=0)
            profiles = q.sum(axis=(2, 3))
            pairwise = [
                float(np.abs(np.cumsum(profiles[a]) - np.cumsum(profiles[b])).sum() * 10)
                for a, b in itertools.combinations(range(10), 2)
            ]
            masses = np.asarray([
                np.maximum(value, 0).sum(dtype=np.float64) for value in values
            ])
            errors = np.asarray([
                row["gt_scale_aligned_nrmse"] for row in quality
                if row["sample_id"] == sample and row["method"] == method
            ])
            stability.append({
                "sample_id": sample, "method": method,
                "shape_relative_dispersion": float(
                    np.sqrt(np.mean(np.sum((q - center) ** 2, axis=(1, 2, 3))))
                    / max(np.linalg.norm(center), 1e-30)
                ),
                "pairwise_axial_w1_um": float(np.mean(pairwise)),
                "mass_coefficient_of_variation": float(
                    masses.std(ddof=1) / max(masses.mean(), 1e-30)
                ),
                "gt_scale_aligned_nrmse_std": float(errors.std(ddof=1)),
                "subsets": 10, "pair_count": 45,
            })
    quality_summary = []
    quality_metrics = (
        "gt_nrmse", "gt_scale_aligned_nrmse", "gt_axial_w1_um", "gt_axial_mass_l1",
        "gt_support_outside_pm10_mass", "gt_xy_mip_ssim", "background_xy_mass_fraction",
    )
    stability_metrics = (
        "shape_relative_dispersion", "pairwise_axial_w1_um",
        "mass_coefficient_of_variation", "gt_scale_aligned_nrmse_std",
    )
    for method in METHODS:
        for sample in ("T02", "T03", "T04", "object_macro"):
            owners = (sample,) if sample != "object_macro" else ("T02", "T03", "T04")
            owner_means = []
            for owner in owners:
                rows = [r for r in quality if r["method"] == method and r["sample_id"] == owner]
                owner_means.append({
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in quality_metrics
                })
            quality_summary.append({
                "sample_id": sample, "method": method,
                **{metric: float(np.mean([row[metric] for row in owner_means])) for metric in quality_metrics},
            })
        owned = [row for row in stability if row["method"] == method]
        stability.append({
            "sample_id": "object_macro", "method": method,
            **{metric: float(np.mean([float(row[metric]) for row in owned])) for metric in stability_metrics},
            "subsets": 30, "pair_count": 135,
        })
    return quality, quality_summary, stability


def local_audit() -> dict[str, list[dict[str, Any]]]:
    from tools import v3_compare_local_audit as audit
    previous_evaluation = evaluation.exp
    previous_audit = audit.exp
    evaluation.exp = exp
    audit.exp = exp
    try:
        result = audit.run()
    finally:
        evaluation.exp = previous_evaluation
        audit.exp = previous_audit
    return result


def local_summary(tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    methods = METHODS
    for method in methods:
        t02 = [r for r in tables["t02_tubes"] if r["method"] == method and float(r["threshold_fraction"]) == .1]
        tubes = [r for r in t02 if str(r["tube"]).isdigit()]
        gaps = [r for r in t02 if str(r["tube"]).startswith("gap_")]
        t03 = [r for r in tables["t03_lines"] if r["method"] == method and float(r["threshold_fraction"]) == .1]
        t04 = [r for r in tables["t04_axial"] if r["method"] == method and float(r["threshold_fraction"]) == .1]
        pairs = [r for r in t04 if r.get("separation_applicable") in (True, "True")]
        ten = [r for r in t04 if float(r.get("separation_um") or -1) == 10]
        def mean(rows, key):
            values = [float(row[key]) for row in rows if row.get(key) not in (None, "", "nan")]
            return float(np.mean(values)) if values else float("nan")
        def rate(rows, key):
            return float(np.mean([str(row[key]).lower() in ("true", "1") for row in rows])) if rows else float("nan")
        result.append({
            "method": method,
            "t02_centerline_localized_fraction": mean(tubes, "centerline_localized_fraction"),
            "t02_false_break_rate": rate(tubes, "false_break"),
            "t02_gap_bridge_rate": rate(gaps, "bridged"),
            "t03_all_three_localized_rate": rate(t03, "all_three_localized"),
            "t03_separated_rate": rate(t03, "separated"),
            "t03_false_break_rate": float(np.mean([int(float(r["false_break_count"])) > 0 for r in t03])) if t03 else float("nan"),
            "t04_20_40um_separated_rate": rate(pairs, "separated"),
            "t04_20_40um_local_w1_um": mean(pairs, "local_depth_w1_um"),
            "t04_10um_localized_rate": rate(ten, "all_layers_localized"),
            "t04_10um_expected_energy_fraction": mean(ten, "expected_layer_energy_fraction"),
        })
    return result


def _show(ax, image: np.ndarray, scale: float, title: str, aspect: str = "equal") -> None:
    ax.imshow(
        np.clip(image / max(scale, 1e-30), 0, 1), cmap="magma", vmin=0, vmax=1,
        interpolation="nearest", aspect=aspect, origin="lower",
    )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def figures(stability: list[dict[str, Any]]) -> list[str]:
    destination = exp.OUTPUT / "analysis/figures"
    destination.mkdir(parents=True, exist_ok=True)
    created = []
    for sample in ("T02", "T03", "T04"):
        volumes, truth = case_volumes(sample, 1)
        shown = {"ground_truth": truth, **volumes}
        methods = ("ground_truth", *METHODS)
        figure, axes = plt.subplots(3, len(methods), figsize=(3 * len(methods), 8.5), layout="constrained")
        for column, method in enumerate(methods):
            volume = shown[method]
            for row, projection in enumerate((volume.max(0), volume.sum(1), volume.sum(2))):
                _show(
                    axes[row, column], projection, float(np.quantile(projection, .999)),
                    LABELS[method] + (" XY" if row == 0 else " XZ" if row == 1 else " YZ"),
                    "auto" if row else "equal",
                )
        path = destination / f"{sample}_subset01_projections_volume_normalized.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        created.append(str(path.relative_to(exp.OUTPUT)))

        networks = exp.ARMS
        xy_scale = max(float(np.quantile(volumes[name].max(0), .999)) for name in networks)
        xz_scale = max(float(np.quantile(volumes[name].sum(1), .999)) for name in networks)
        figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
        for column, method in enumerate(networks):
            _show(axes[0, column], volumes[method].max(0), xy_scale, LABELS[method] + " XY·共同尺度")
            _show(axes[1, column], volumes[method].sum(1), xz_scale, LABELS[method] + " XZ·共同尺度", "auto")
        path = destination / f"{sample}_subset01_network_shared_scale.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(str(path.relative_to(exp.OUTPUT)))

        figure, axes = plt.subplots(len(methods), 10, figsize=(22, 2.1 * len(methods)), layout="constrained")
        shared_network = max(float(np.quantile(volumes[name], .999)) for name in networks)
        for row, method in enumerate(methods):
            volume = shown[method]
            scale = shared_network if method in networks else float(np.quantile(volume, .999))
            for layer in range(10):
                _show(axes[row, layer], volume[layer], scale, f"{LABELS[method]} · {(layer + 1) * 10} µm")
        path = destination / f"{sample}_subset01_native_layers.png"
        figure.savefig(path, dpi=115)
        plt.close(figure)
        created.append(str(path.relative_to(exp.OUTPUT)))

        for arm in networks:
            folder = exp.OUTPUT / "evaluation" / arm / "final" / f"{sample}_subset_01"
            anchor = np.load(folder / "anchor.npy", allow_pickle=False)
            correction = np.load(folder / "effective_correction.npy", allow_pickle=False)
            prediction = np.load(folder / "reconstruction.npy", allow_pickle=False)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
            _show(axes[0], anchor.max(0), float(np.quantile(anchor, .999)), "校准后基础体")
            signed = correction.sum(0)
            scale = float(np.quantile(np.abs(signed), .999))
            axes[1].imshow(signed, cmap="coolwarm", vmin=-scale, vmax=scale, origin="lower")
            axes[1].set_title("有效修正（正/负）")
            axes[1].set_xticks([]); axes[1].set_yticks([])
            _show(axes[2], prediction.max(0), float(np.quantile(prediction, .999)), "最终重建")
            path = destination / f"{sample}_subset01_{arm}_anchor_correction.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            created.append(str(path.relative_to(exp.OUTPUT)))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, sample in zip(axes, ("T02", "T03", "T04")):
        for arm in exp.ARMS:
            profiles = []
            for subset in range(1, 11):
                volumes, _ = case_volumes(sample, subset)
                profiles.append(_normalized(volumes[arm]).sum((1, 2)))
            values = np.stack(profiles)
            mean = values.mean(0)
            std = values.std(0)
            z = np.arange(10, 101, 10)
            ax.plot(z, mean, label=LABELS[arm])
            ax.fill_between(z, mean - std, mean + std, alpha=.2)
        ax.set_title(sample); ax.set_xlabel("深度 (µm)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("质量占比，10 子集 mean±std")
    axes[-1].legend(fontsize=7)
    path = destination / "network_subset_axial_stability.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(str(path.relative_to(exp.OUTPUT)))
    return created


def _lookup(rows: list[dict[str, Any]], method: str, sample: str = "object_macro") -> dict[str, Any]:
    return next(row for row in rows if row["method"] == method and row["sample_id"] == sample)


def _training_audit() -> dict[str, Any]:
    result = {}
    for arm in exp.ARMS:
        folder = exp.OUTPUT / arm
        complete = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
        training = read_csv(folder / "training_metrics.csv")
        validation = read_csv(folder / "validation_metrics.csv")
        if int(complete["completed_steps"]) != 400 or len(training) != 400 or len(validation) != 600:
            raise ValueError(f"{arm} training inventory is incomplete")
        run_contract = json.loads((folder / "run_contract.json").read_text(encoding="utf-8"))
        if run_contract["train_objects"] != [f"P{i:02d}" for i in range(1, 13)]:
            raise ValueError(f"{arm} run contract does not include P12")
        checkpoints = {}
        for role, filename in (("best", "checkpoint_best.pt"), ("final", "checkpoint_last.pt")):
            path = folder / filename
            payload = torch.load(path, map_location="cpu", weights_only=False)
            checkpoints[role] = {"step": int(payload["completed_steps"]), "sha256": old.sha256(path)}
            del payload
        if checkpoints["final"]["step"] != 400:
            raise ValueError(f"{arm} final checkpoint is not step 400")
        result[arm] = {
            "training_rows": len(training), "validation_rows": len(validation),
            "best_step": int(complete["best_step"]), "checkpoints": checkpoints,
            "world_size": int(json.loads((exp.OUTPUT / "jobs" / f"train_{arm}.json").read_text())["command"].count("--nproc_per_node") > 0) or 1,
            "dataset_fingerprint": run_contract["dataset_fingerprint"],
        }
    return result


def build_report() -> dict[str, Any]:
    for arm in exp.ARMS:
        marker = exp.OUTPUT / "evaluation" / arm / "complete.json"
        if not marker.exists() or not json.loads(marker.read_text(encoding="utf-8")).get("complete"):
            raise ValueError(f"Evaluation incomplete: {arm}")
    previous = evaluation.exp
    evaluation.exp = exp
    try:
        augmented = augment_anchor_artifacts()
        quality, quality_summary, stability = quality_and_stability()
    finally:
        evaluation.exp = previous
    local_tables = local_audit()
    local = local_summary(local_tables)
    analysis = exp.OUTPUT / "analysis"
    write_csv(analysis / "quality_per_subset.csv", quality)
    write_csv(analysis / "quality_summary.csv", quality_summary)
    write_csv(analysis / "stability_per_object.csv", stability)
    write_csv(analysis / "local_summary.csv", local)
    figure_paths = figures(stability)
    training = _training_audit()

    taylor = _lookup(quality_summary, exp.ARMS[0])
    mean = _lookup(quality_summary, exp.ARMS[1])
    taylor_stability = _lookup(stability, exp.ARMS[0])
    mean_stability = _lookup(stability, exp.ARMS[1])
    local_map = {row["method"]: row for row in local}
    old_quality = read_csv(OLD_ANALYSIS / "quality_summary.csv")
    old_stability = read_csv(OLD_ANALYSIS / "stability_per_object.csv")
    old_taylor = _lookup(old_quality, "e3_mean100")
    old_mean = _lookup(old_quality, "mean_anchor_e3_mean100")
    old_taylor_stability = _lookup(old_stability, "e3_mean100")
    old_mean_stability = _lookup(old_stability, "mean_anchor_e3_mean100")

    lower_metrics = ("gt_scale_aligned_nrmse", "gt_axial_w1_um", "background_xy_mass_fraction")
    mean_quality_not_worse = all(float(mean[m]) <= float(taylor[m]) for m in lower_metrics)
    stability_wins = sum(
        _lookup(stability, exp.ARMS[1], sample)["shape_relative_dispersion"]
        < _lookup(stability, exp.ARMS[0], sample)["shape_relative_dispersion"]
        for sample in ("T02", "T03", "T04")
    )
    taylor_local = local_map[exp.ARMS[0]]
    mean_local = local_map[exp.ARMS[1]]
    local_no_regression = (
        float(mean_local["t02_centerline_localized_fraction"]) >= float(taylor_local["t02_centerline_localized_fraction"])
        and float(mean_local["t02_false_break_rate"]) <= float(taylor_local["t02_false_break_rate"])
        and float(mean_local["t03_separated_rate"]) >= float(taylor_local["t03_separated_rate"])
        and float(mean_local["t04_20_40um_separated_rate"]) >= float(taylor_local["t04_20_40um_separated_rate"])
    )
    support_mean = stability_wins >= 2 and mean_quality_not_worse and local_no_regression
    outcome = "支持 Mean 主分支" if support_mean else "不支持自动替换；保留 Taylor 基线或判为混合收益"

    lines = [
        "# V4（加入 P12）Mean/Taylor 主重建分支对照",
        "",
        "两组均从同一份 seed=20260901 初始化重新训练 400 步；训练集 P01–P12，验证集 V01–V03，测试集 T02–T04。两组都保留 Taylor、Mean、Set 编码器以及 E3＋mean≤100% 损失，只切换基础重建体和解析 β。主比较为两组 step 400 final。",
        "",
        "## 最终测试集（对象等权）",
        "",
        "| 方法 | aligned 3D NRMSE↓ | 轴向 W1 (µm)↓ | XY MIP SSIM↑ | 背景质量↓ | 形状离散度↓ | 子集轴向 W1↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in exp.ARMS:
        q = _lookup(quality_summary, method)
        s = _lookup(stability, method)
        lines.append(
            f"| {LABELS[method]} | {float(q['gt_scale_aligned_nrmse']):.4f} | {float(q['gt_axial_w1_um']):.2f} | {float(q['gt_xy_mip_ssim']):.4f} | {float(q['background_xy_mass_fraction']):.4f} | {float(s['shape_relative_dispersion']):.4f} | {float(s['pairwise_axial_w1_um']):.3f} |"
        )
    lines += [
        "",
        "## T02/T03/T04 固定规则",
        "",
        "| 方法 | T02 中心线保留↑ | T02 断裂率↓ | T03 三线分开率↑ | T04 20–40µm 分开率↑ | T04 10µm 定位率↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in exp.ARMS:
        row = local_map[method]
        lines.append(
            f"| {LABELS[method]} | {float(row['t02_centerline_localized_fraction']):.3f} | {float(row['t02_false_break_rate']):.3f} | {float(row['t03_separated_rate']):.3f} | {float(row['t04_20_40um_separated_rate']):.3f} | {float(row['t04_10um_localized_rate']):.3f} |"
        )
    lines += [
        "",
        "T04 的 10 µm 项只评价原生层定位和目标层能量，不通过插值宣称双峰分辨。",
        "",
        "## P12 加入前后",
        "",
        "| 网络 | 加 P12 前 NRMSE | 加 P12 后 NRMSE | 加 P12 前 W1 | 加 P12 后 W1 | 加 P12 前离散度 | 加 P12 后离散度 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Taylor 主分支 | {float(old_taylor['gt_scale_aligned_nrmse']):.4f} | {float(taylor['gt_scale_aligned_nrmse']):.4f} | {float(old_taylor['gt_axial_w1_um']):.2f} | {float(taylor['gt_axial_w1_um']):.2f} | {float(old_taylor_stability['shape_relative_dispersion']):.4f} | {float(taylor_stability['shape_relative_dispersion']):.4f} |",
        f"| Mean 主分支 | {float(old_mean['gt_scale_aligned_nrmse']):.4f} | {float(mean['gt_scale_aligned_nrmse']):.4f} | {float(old_mean['gt_axial_w1_um']):.2f} | {float(mean['gt_axial_w1_um']):.2f} | {float(old_mean_stability['shape_relative_dispersion']):.4f} | {float(mean_stability['shape_relative_dispersion']):.4f} |",
        "",
        "旧实验与本轮只差训练集新增 P12，但仍只有同一个随机种子；前后差值可作为方向性证据，不能当作统计显著。",
        "",
        "## 判定",
        "",
        f"**{outcome}。** Mean 主分支在 3 个测试对象中有 {stability_wins} 个形状离散度更低；整体质量不退步={mean_quality_not_worse}，T02/T03/T04 严格局部规则不退步={local_no_regression}。",
        "",
        "清晰图像位于 [analysis/figures](analysis/figures)：每个完整三维体归一化图用于看结构，两网络共同尺度图用于看真实相对亮度；原生 10 层没有逐层归一化。",
        "",
        f"checkpoint：Taylor best={training[exp.ARMS[0]]['best_step']}，Mean best={training[exp.ARMS[1]]['best_step']}；主结论始终使用 step 400 final。",
    ]
    (exp.OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "complete": True,
        "passed_artifact_acceptance": True,
        "scientific_outcome": outcome,
        "supports_mean_anchor": support_mean,
        "stability_objects_won_by_mean": stability_wins,
        "mean_quality_not_worse": mean_quality_not_worse,
        "local_no_regression": local_no_regression,
        "training": training,
        "predictions": 376,
        "anchor_augmentations": augmented,
        "figures": figure_paths,
        "report_sha256": old.sha256(exp.OUTPUT / "REPORT_ZH.md"),
    }
    old.write_json(exp.OUTPUT / "final_acceptance.json", result)
    old.write_json(exp.OUTPUT / "complete.json", {
        "complete": True, "experiments": list(exp.ARMS), "predictions": 376,
        "report": str(exp.OUTPUT / "REPORT_ZH.md"),
    })
    return result


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, ensure_ascii=False))
