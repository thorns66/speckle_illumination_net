"""Build complete numerical tables and clear normalized figures for V5."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import v3_compare_evaluation as metrics
from tools import v3_compare_local_audit as local
from tools import v5_mixed_real_anchor_experiment as exp
from tools.v5_mixed_dataset import MixedExperimentDataset, RealFieldDataset


OLD_TAYLOR = exp.ROOT / "outputs/v3_mean050_mean100_400_20260908_run01/evaluation/e3_mean100/final"
OLD_MEAN = exp.ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/evaluation/mean_anchor_e3_mean100/final"
REAL_OLD = exp.ROOT / "outputs/spinach_real_reprocess_20260910_run01"
METHODS = (
    "mean_rl3", "taylor_rl3_sqrt", "old_taylor_anchor", "old_mean_anchor",
    "new_taylor_anchor", "new_mean_anchor",
)
LABELS = {
    "ground_truth": "Ground truth", "mean_rl3": "Mean-RL3",
    "taylor_rl3_sqrt": "Taylor-RL3-sqrt", "old_taylor_anchor": "Old Taylor anchor (400)",
    "old_mean_anchor": "Old Mean anchor (400)", "new_taylor_anchor": "New Taylor anchor (600)",
    "new_mean_anchor": "New Mean anchor (600)",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalized(volume: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(volume, np.float64), 0)
    return value / max(value.sum(), 1e-30)


def new_path(arm: str, domain: str, owner: str, subset: int, role: str = "final600") -> Path:
    parent = "simulation" if domain == "simulation" else "real_training_fields"
    case = f"{owner}_subset_{subset:02d}" if domain == "simulation" else f"real_{owner}_subset_{subset:02d}"
    return exp.OUTPUT / "comparison" / parent / arm / role / case / "reconstruction.npy"


def simulation_volumes(dataset, owner: str, subset: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    item = None
    for index in range(len(dataset)):
        candidate = dataset[index]
        if candidate["sample_id"] == owner and int(candidate["subset_index"]) == subset:
            item = candidate
            break
    if item is None:
        raise KeyError(f"Missing simulation case {owner} subset {subset}")
    volumes = {
        "mean_rl3": item["g_mean"][0].numpy(),
        "taylor_rl3_sqrt": item["f_var"][0].numpy(),
        "old_taylor_anchor": np.load(OLD_TAYLOR / f"{owner}_subset_{subset:02d}/reconstruction.npy"),
        "old_mean_anchor": np.load(OLD_MEAN / f"{owner}_subset_{subset:02d}/reconstruction.npy"),
        "new_taylor_anchor": np.load(new_path(exp.ARMS[0], "simulation", owner, subset)),
        "new_mean_anchor": np.load(new_path(exp.ARMS[1], "simulation", owner, subset)),
    }
    return volumes, item["ground_truth"][0].numpy()


def _mat_volume(path: Path, name: str = "reconstruction") -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name], dtype=np.float32).transpose(0, 2, 1).copy()


def real_volumes(dataset: RealFieldDataset, field: str, subset: int) -> dict[str, np.ndarray]:
    item = dataset[(0 if field == "45" else 10) + subset - 1]
    result = {
        "mean_rl3": item["g_mean"][0].numpy(),
        "taylor_rl3_sqrt": item["f_var"][0].numpy(),
        "new_taylor_anchor": np.load(new_path(exp.ARMS[0], "real", field, subset)),
        "new_mean_anchor": np.load(new_path(exp.ARMS[1], "real", field, subset)),
    }
    if subset == 1:
        result.update({
            "old_taylor_anchor": _mat_volume(REAL_OLD / field / "before_p12_taylor/reconstruction.mat"),
            "old_mean_anchor": _mat_volume(REAL_OLD / field / "before_p12_mean/reconstruction.mat"),
        })
    return result


def real_method_volume(field: str, subset: int, method: str) -> np.ndarray:
    folder = exp.REAL_DATA / field / f"subset_{subset:02d}"
    if method == "mean_rl3":
        return _mat_volume(folder / "mean_rl3.mat", "reconstruction_raw")
    if method == "taylor_rl3_sqrt":
        return _mat_volume(folder / "taylor_rl3.mat", "reconstruction_sqrt")
    if method == "new_taylor_anchor":
        return np.load(new_path(exp.ARMS[0], "real", field, subset), allow_pickle=False)
    if method == "new_mean_anchor":
        return np.load(new_path(exp.ARMS[1], "real", field, subset), allow_pickle=False)
    raise KeyError(method)


def quality_and_stability() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = MixedExperimentDataset(
        exp.SIMULATION_DATA, "test", real_root=exp.REAL_DATA,
        cache_dir=exp.OUTPUT / "data_cache", var_feature_representation="sqrt",
    )
    quality: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for owner in ("T02", "T03", "T04"):
        by_method = {name: [] for name in METHODS}
        for subset in range(1, 11):
            volumes, truth = simulation_volumes(dataset, owner, subset)
            for method, volume in volumes.items():
                by_method[method].append(volume)
                quality.append({"sample_id": owner, "subset_index": subset, "method": method,
                                **metrics._structure_row(volume, truth)})
        for method, values in by_method.items():
            q = np.stack([normalized(value) for value in values]); center = q.mean(0)
            profiles = q.sum((2, 3))
            pairwise = [float(np.abs(np.cumsum(profiles[a]) - np.cumsum(profiles[b])).sum() * 10)
                        for a, b in itertools.combinations(range(10), 2)]
            masses = np.asarray([np.maximum(value, 0).sum(dtype=np.float64) for value in values])
            errors = [row["gt_scale_aligned_nrmse"] for row in quality
                      if row["sample_id"] == owner and row["method"] == method]
            stability.append({
                "sample_id": owner, "method": method,
                "shape_relative_dispersion": float(np.sqrt(np.mean(np.sum((q - center) ** 2, axis=(1, 2, 3)))) / max(np.linalg.norm(center), 1e-30)),
                "pairwise_axial_w1_um": float(np.mean(pairwise)),
                "mass_coefficient_of_variation": float(masses.std(ddof=1) / max(masses.mean(), 1e-30)),
                "gt_scale_aligned_nrmse_std": float(np.std(errors, ddof=1)),
                "subsets": 10, "pair_count": 45,
            })
    for method in METHODS:
        selected = [row for row in stability if row["method"] == method]
        stability.append({
            "sample_id": "object_macro", "method": method,
            **{name: float(np.mean([row[name] for row in selected])) for name in (
                "shape_relative_dispersion", "pairwise_axial_w1_um", "mass_coefficient_of_variation",
                "gt_scale_aligned_nrmse_std",
            )}, "subsets": 30, "pair_count": 135,
        })
    return quality, stability


def real_stability_rows() -> list[dict[str, Any]]:
    methods = ("mean_rl3", "taylor_rl3_sqrt", "new_taylor_anchor", "new_mean_anchor")
    result: list[dict[str, Any]] = []
    for field in ("45", "55"):
        for method in methods:
            volumes = [real_method_volume(field, subset, method) for subset in range(1, 11)]
            q = np.stack([normalized(volume) for volume in volumes]); center = q.mean(0)
            profiles = q.sum((2, 3))
            pairwise = [float(np.abs(np.cumsum(profiles[a]) - np.cumsum(profiles[b])).sum() * 10)
                        for a, b in itertools.combinations(range(10), 2)]
            masses = np.asarray([np.maximum(volume, 0).sum(dtype=np.float64) for volume in volumes])
            result.append({
                "field_id": field, "method": method, "result_role": "training_field_diagnostic_no_gt",
                "shape_relative_dispersion": float(np.sqrt(np.mean(np.sum((q - center) ** 2, axis=(1, 2, 3)))) / max(np.linalg.norm(center), 1e-30)),
                "pairwise_axial_w1_um": float(np.mean(pairwise)),
                "mass_coefficient_of_variation": float(masses.std(ddof=1) / max(masses.mean(), 1e-30)),
                "subsets": 10, "pair_count": 45,
            })
    for method in methods:
        selected = [row for row in result if row["method"] == method]
        result.append({
            "field_id": "field_macro", "method": method, "result_role": "training_field_diagnostic_no_gt",
            **{name: float(np.mean([row[name] for row in selected])) for name in (
                "shape_relative_dispersion", "pairwise_axial_w1_um", "mass_coefficient_of_variation",
            )}, "subsets": 20, "pair_count": 90,
        })
    return result


def local_tables() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    dataset = MixedExperimentDataset(exp.SIMULATION_DATA, "test", real_root=exp.REAL_DATA, cache_dir=None)
    tables = {name: [] for name in ("t02", "t03", "t04", "profiles")}
    functions = {"T02": ("t02", local.t02), "T03": ("t03", local.t03), "T04": ("t04", local.t04)}
    for owner, (table, function) in functions.items():
        for subset in range(1, 11):
            volumes, truth = simulation_volumes(dataset, owner, subset)
            for method, volume in volumes.items():
                metadata = {"sample_id": owner, "subset_index": subset, "method": method}
                rows, profiles = function(volume, truth, metadata)
                tables[table].extend(rows); tables["profiles"].extend(profiles)
    summary = []
    for method in METHODS:
        t02 = [row for row in tables["t02"] if row["method"] == method and float(row["threshold_fraction"]) == .1 and isinstance(row["tube"], int)]
        t03 = [row for row in tables["t03"] if row["method"] == method and float(row["threshold_fraction"]) == .1]
        t04 = [row for row in tables["t04"] if row["method"] == method and float(row["threshold_fraction"]) == .1]
        pair = [row for row in t04 if row["separation_applicable"]]
        ten = [row for row in t04 if row["separation_um"] == 10]
        mean = lambda rows, name: float(np.mean([float(row[name]) for row in rows])) if rows else float("nan")
        rate = lambda rows, name: float(np.mean([bool(row[name]) for row in rows])) if rows else float("nan")
        summary.append({
            "method": method,
            "t02_centerline_localized_fraction": mean(t02, "centerline_localized_fraction"),
            "t02_false_break_rate": rate(t02, "false_break"),
            "t03_all_three_localized_rate": rate(t03, "all_three_localized"),
            "t03_separated_rate": rate(t03, "separated"),
            "t03_false_break_rate": float(np.mean([int(row["false_break_count"]) > 0 for row in t03])),
            "t04_20_40um_separated_rate": rate(pair, "separated"),
            "t04_20_40um_local_w1_um": mean(pair, "local_depth_w1_um"),
            "t04_10um_localized_rate": rate(ten, "all_layers_localized"),
            "t04_10um_expected_energy_fraction": mean(ten, "expected_layer_energy_fraction"),
        })
    return tables, summary


def show(ax, image: np.ndarray, scale: float, title: str, aspect: str = "equal") -> None:
    ax.imshow(np.clip(image / max(scale, 1e-30), 0, 1), cmap="magma", vmin=0, vmax=1,
              interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def projection_figure(volumes: dict[str, np.ndarray], path: Path, *, shared_network_scale: bool) -> dict[str, Any]:
    order = list(volumes)
    projections = {name: (value.max(0), value.sum(1), value.sum(2)) for name, value in volumes.items()}
    network_names = [name for name in order if "anchor" in name]
    scales = {}
    for view in range(3):
        common = max(float(np.quantile(projections[name][view], .999)) for name in network_names)
        for name in order:
            scales[f"{name}_{view}"] = common if shared_network_scale and name in network_names else float(np.quantile(projections[name][view], .999))
    fig, axes = plt.subplots(3, len(order), figsize=(3.0 * len(order), 8.5), layout="constrained")
    for column, name in enumerate(order):
        for row, suffix in enumerate(("XY MIP", "XZ sum", "YZ sum")):
            show(axes[row, column], projections[name][row], scales[f"{name}_{row}"], LABELS[name] + " · " + suffix,
                 "equal" if row == 0 else "auto")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=170); plt.close(fig)
    return scales


def anchor_correction_figure(domain: str, owner: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), layout="constrained")
    for row, (arm, label) in enumerate(((exp.ARMS[0], "New Taylor"), (exp.ARMS[1], "New Mean"))):
        parent = "simulation" if domain == "simulation" else "real_training_fields"
        case = f"{owner}_subset_01" if domain == "simulation" else f"real_{owner}_subset_01"
        folder = exp.OUTPUT / "comparison" / parent / arm / "final600" / case
        anchor = np.load(folder / "physical_anchor.npy"); reconstruction = np.load(folder / "reconstruction.npy")
        correction = np.load(folder / "effective_correction.npy")
        common = max(float(np.quantile(anchor.max(0), .999)), float(np.quantile(reconstruction.max(0), .999)))
        show(axes[row, 0], anchor.max(0), common, label + " · calibrated anchor")
        show(axes[row, 1], reconstruction.max(0), common, label + " · output")
        signed = correction.sum(0); scale = float(np.quantile(np.abs(signed), .999))
        axes[row, 2].imshow(np.clip(signed / max(scale, 1e-30), -1, 1), cmap="coolwarm", vmin=-1, vmax=1,
                            interpolation="nearest")
        axes[row, 2].set_title(label + " · signed correction", fontsize=8)
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
    fig.savefig(path, dpi=170); plt.close(fig)


def axial_profile_figure(volumes: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for name, volume in volumes.items():
        profile = normalized(volume).sum((1, 2))
        ax.plot(np.arange(10, 101, 10), profile, marker="o", label=LABELS[name])
    ax.set_xlabel("Depth (um)"); ax.set_ylabel("Normalized axial mass")
    ax.set_xticks(np.arange(10, 101, 10)); ax.legend(fontsize=7, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=170); plt.close(fig)


def subset_fluctuation_figure(domain: str, owner: str, dataset, path: Path) -> None:
    method_names = METHODS if domain == "simulation" else (
        "mean_rl3", "taylor_rl3_sqrt", "new_taylor_anchor", "new_mean_anchor",
    )
    by_method = {name: [] for name in method_names}
    for subset in range(1, 11):
        volumes = simulation_volumes(dataset, owner, subset)[0] if domain == "simulation" else None
        for method in method_names:
            volume = volumes[method] if volumes is not None else real_method_volume(owner, subset, method)
            by_method[method].append(normalized(volume).sum((1, 2)))
    fig, axes = plt.subplots(2, 3 if domain == "simulation" else 2,
                             figsize=(12, 7), layout="constrained")
    for ax, method in zip(np.asarray(axes).reshape(-1), method_names):
        values = np.stack(by_method[method])
        for profile in values:
            ax.plot(np.arange(10, 101, 10), profile, color="#6d79a8", alpha=.28, linewidth=.9)
        ax.plot(np.arange(10, 101, 10), values.mean(0), color="#d62728", marker="o",
                linewidth=2, label="10-subset mean")
        ax.set_title(LABELS[method], fontsize=9); ax.set_xticks(np.arange(10, 101, 20))
        ax.set_xlabel("Depth (um)"); ax.set_ylabel("Normalized axial mass"); ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=170); plt.close(fig)


def figures() -> dict[str, Any]:
    folder = exp.OUTPUT / "comparison/figures"; folder.mkdir(parents=True, exist_ok=True)
    simulation = MixedExperimentDataset(exp.SIMULATION_DATA, "test", real_root=exp.REAL_DATA, cache_dir=None)
    real = RealFieldDataset(exp.REAL_DATA)
    scale_records = {}
    count = 0
    for owner in ("T02", "T03", "T04"):
        values, truth = simulation_volumes(simulation, owner, 1)
        shown = {"ground_truth": truth, **values}
        scale_records[f"{owner}_own"] = projection_figure(shown, folder / f"{owner}_subset01_projections_own.png", shared_network_scale=False)
        scale_records[f"{owner}_shared"] = projection_figure(shown, folder / f"{owner}_subset01_projections_network_shared.png", shared_network_scale=True)
        fig, axes = plt.subplots(10, len(shown), figsize=(2.5 * len(shown), 24), layout="constrained")
        for column, (name, volume) in enumerate(shown.items()):
            scale = float(np.quantile(volume, .999))
            for z in range(10):
                show(axes[z, column], volume[z], scale, f"{LABELS[name]} · {10*(z+1)} um")
        fig.savefig(folder / f"{owner}_subset01_all_native_layers.png", dpi=120); plt.close(fig); count += 3
        axial_profile_figure(shown, folder / f"{owner}_subset01_axial_profiles.png")
        subset_fluctuation_figure("simulation", owner, simulation,
                                  folder / f"{owner}_ten_subset_axial_fluctuation.png")
        anchor_correction_figure("simulation", owner, folder / f"{owner}_subset01_anchor_correction.png")
        count += 3
    for field in ("45", "55"):
        shown = real_volumes(real, field, 1)
        shown = {name: shown[name] for name in METHODS}
        scale_records[f"real_{field}_own"] = projection_figure(shown, folder / f"real_{field}_subset01_projections_own.png", shared_network_scale=False)
        scale_records[f"real_{field}_shared"] = projection_figure(shown, folder / f"real_{field}_subset01_projections_network_shared.png", shared_network_scale=True)
        fig, axes = plt.subplots(10, len(shown), figsize=(2.5 * len(shown), 24), layout="constrained")
        for column, (name, volume) in enumerate(shown.items()):
            scale = float(np.quantile(volume, .999))
            for z in range(10):
                show(axes[z, column], volume[z], scale, f"{LABELS[name]} · {10*(z+1)} um")
        fig.savefig(folder / f"real_{field}_subset01_all_native_layers.png", dpi=100); plt.close(fig)
        yslice, xslice = slice(294, 686), slice(490, 882)
        fig, axes = plt.subplots(10, len(shown), figsize=(2.5 * len(shown), 24), layout="constrained")
        for column, (name, volume) in enumerate(shown.items()):
            scale = float(np.quantile(volume[:, yslice, xslice], .999))
            for z in range(10):
                show(axes[z, column], volume[z, yslice, xslice], scale,
                     f"{LABELS[name]} · ROI · {10*(z+1)} um")
        fig.savefig(folder / f"real_{field}_subset01_fixed_roi_native_layers.png", dpi=120); plt.close(fig)
        axial_profile_figure(shown, folder / f"real_{field}_subset01_axial_profiles.png")
        subset_fluctuation_figure("real", field, real,
                                  folder / f"real_{field}_ten_subset_axial_fluctuation.png")
        anchor_correction_figure("real", field, folder / f"real_{field}_subset01_anchor_correction.png")
        count += 7
    write_json(folder / "display_scales.json", {
        "normalization": "whole-volume/projection 99.9 percentile; linear; no per-layer normalization",
        "network_shared_figures": True, "records": scale_records,
    })
    return {"figures": count, "display_scales": str(folder / "display_scales.json")}


def report(quality: list[dict[str, Any]], stability: list[dict[str, Any]],
           real_stability: list[dict[str, Any]], local_summary: list[dict[str, Any]]) -> None:
    summary = []
    for method in METHODS:
        objects = []
        for owner in ("T02", "T03", "T04"):
            rows = [row for row in quality if row["method"] == method and row["sample_id"] == owner]
            objects.append({name: float(np.mean([row[name] for row in rows])) for name in (
                "gt_scale_aligned_nrmse", "gt_axial_w1_um", "gt_xy_mip_ssim", "background_xy_mass_fraction"
            )})
        summary.append({"method": method, **{name: float(np.mean([row[name] for row in objects])) for name in objects[0]}})
    write_csv(exp.OUTPUT / "comparison/simulation_quality_object_macro.csv", summary)
    lines = [
        "# V5：P12排除、真实45/55混合训练的Taylor／Mean主分支对照", "",
        "主比较为两组第600步；45、55均参与训练，因此只能作为训练视场诊断，不能作为独立真实测试。", "",
        "## 仿真测试集对象等权平均", "",
        "| 方法 | 尺度对齐NRMSE | 轴向W1 (um) | XY MIP SSIM | 背景质量 |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {LABELS[row['method']]} | {row['gt_scale_aligned_nrmse']:.4f} | {row['gt_axial_w1_um']:.2f} | {row['gt_xy_mip_ssim']:.4f} | {row['background_xy_mass_fraction']:.4f} |")
    lines += ["", "## 子集稳定性", "", "| 方法 | 相对离散度 | 45对子集轴向W1 (um) | 质量CV |", "|---|---:|---:|---:|"]
    for method in METHODS:
        row = next(row for row in stability if row["method"] == method and row["sample_id"] == "object_macro")
        lines.append(f"| {LABELS[method]} | {row['shape_relative_dispersion']:.4f} | {row['pairwise_axial_w1_um']:.2f} | {row['mass_coefficient_of_variation']:.4f} |")
    lines += ["", "## 真实训练视场稳定性（无GT）", "",
              "| 方法 | 相对离散度 | 45对子集轴向W1 (um) | 质量CV |", "|---|---:|---:|---:|"]
    for method in ("mean_rl3", "taylor_rl3_sqrt", "new_taylor_anchor", "new_mean_anchor"):
        row = next(row for row in real_stability if row["method"] == method and row["field_id"] == "field_macro")
        lines.append(f"| {LABELS[method]} | {row['shape_relative_dispersion']:.4f} | {row['pairwise_axial_w1_um']:.2f} | {row['mass_coefficient_of_variation']:.4f} |")
    lines += ["", "## 解释边界", "", "- 两组新网络可以归因于重建基础不同，因为初始化、样本顺序、数据和训练预算一致。",
              "- 新网络与旧400步网络的差异同时包含真实数据混合和训练预算变化，不能只归因于真实数据。",
              "- 本轮没有独立真实测试视场，不据45、55结果自动替换正式基线。",
              "- 图像采用整幅体或投影的99.9百分位线性显示，不做逐层归一化。"]
    (exp.OUTPUT / "comparison/REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(exp.OUTPUT / "comparison/local_structure_summary.csv", local_summary)


def main() -> None:
    quality, stability = quality_and_stability()
    real_stability = real_stability_rows()
    write_csv(exp.OUTPUT / "comparison/simulation_quality_per_subset.csv", quality)
    write_csv(exp.OUTPUT / "comparison/simulation_stability.csv", stability)
    write_csv(exp.OUTPUT / "comparison/real_training_field_stability.csv", real_stability)
    tables, local_summary = local_tables()
    for name, rows in tables.items(): write_csv(exp.OUTPUT / f"comparison/local_{name}.csv", rows)
    visual = figures(); report(quality, stability, real_stability, local_summary)
    write_json(exp.OUTPUT / "comparison/report_complete.json", {
        "complete": True, "quality_rows": len(quality), "stability_rows": len(stability),
        "local_rows": {name: len(rows) for name, rows in tables.items()}, **visual,
    })


if __name__ == "__main__":
    main()
