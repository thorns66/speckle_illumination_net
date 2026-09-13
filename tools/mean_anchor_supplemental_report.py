"""Complete the Mean-anchor comparison with fixed V3 local diagnostics."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import mean_anchor_experiment as exp
from tools import three_way_experiment as old
from tools import v3_compare_evaluation as evaluation
from tools import v3_compare_local_audit as local_audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("MEAN_ANCHOR_OUTPUT", exp.OUTPUT)).resolve()
DATA = Path(os.environ.get("MEAN_ANCHOR_DATA", exp.DATA)).resolve()
BASELINE = exp.REFERENCE_OUTPUT
ANALYSIS = OUTPUT / "analysis"
SUPPLEMENT = ANALYSIS / "supplemental"
METHODS = ("mean_rl3", "taylor_rl3", "e3_mean100_final", "mean_anchor_e3_mean100_final")
LABELS = {
    "mean_rl3": "Mean-RL3",
    "taylor_rl3": "Taylor-RL3-sqrt",
    "e3_mean100_final": "Taylor anchor network",
    "mean_anchor_e3_mean100_final": "Mean anchor network",
}
COLORS = {
    "mean_rl3": "#888888",
    "taylor_rl3": "#b07c22",
    "e3_mean100_final": "#3975b7",
    "mean_anchor_e3_mean100_final": "#d1495b",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    return str(value).lower() in ("true", "1")


def finite_mean(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def bool_mean(rows: list[dict], key: str) -> float:
    return float(np.mean([as_bool(row[key]) for row in rows])) if rows else float("nan")


def generate_new_local_tables() -> dict[str, list[dict]]:
    exp.OUTPUT = OUTPUT
    exp.DATA = DATA
    evaluation.exp = exp
    local_audit.exp = exp
    return local_audit.run()


def combine_local(new: dict[str, list[dict]]) -> dict[str, list[dict]]:
    combined = {}
    for name in ("t02_tubes", "t03_lines", "t04_axial", "fixed_profiles"):
        baseline_rows = read_csv(BASELINE / "analysis" / f"{name}.csv")
        reference = [row for row in baseline_rows if row.get("method") == "e3_mean100_final"]
        candidate = [
            row for row in new[name]
            if row.get("method") in ("mean_rl3", "taylor_rl3", "mean_anchor_e3_mean100_final")
        ]
        combined[name] = [*candidate, *reference]
        write_csv(SUPPLEMENT / f"{name}.csv", combined[name])
    return combined


def local_summary(tables: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for method in METHODS:
        t02 = [
            row for row in tables["t02_tubes"]
            if row["method"] == method and abs(float(row["threshold_fraction"]) - .1) < 1e-9
        ]
        tubes = [row for row in t02 if not str(row["tube"]).startswith("gap_")]
        gaps = [row for row in t02 if str(row["tube"]).startswith("gap_")]
        rows.append({
            "method": method,
            "diagnostic": "T02_tube_continuity",
            "items": len(tubes),
            "centerline_localized_fraction": finite_mean(tubes, "centerline_localized_fraction"),
            "false_break_rate": bool_mean(tubes, "false_break"),
            "gap_bridge_rate": bool_mean(gaps, "bridged"),
            "local_depth_w1_um": finite_mean(tubes, "local_depth_w1_um"),
        })

        t03 = [
            row for row in tables["t03_lines"]
            if row["method"] == method and abs(float(row["threshold_fraction"]) - .1) < 1e-9
        ]
        rows.append({
            "method": method,
            "diagnostic": "T03_three_lines",
            "items": len(t03),
            "localized_rate": bool_mean(t03, "all_three_localized"),
            "separated_rate": bool_mean(t03, "separated"),
            "false_break_rate": float(np.mean([int(float(row["false_break_count"])) > 0 for row in t03])),
            "local_depth_w1_um": finite_mean(t03, "local_depth_w1_um"),
            "worst_valley_ratio": float(np.mean([
                max(float(row["valley_ratio_1"]), float(row["valley_ratio_2"])) for row in t03
            ])),
        })

        t04 = [
            row for row in tables["t04_axial"]
            if row["method"] == method and abs(float(row["threshold_fraction"]) - .1) < 1e-9
        ]
        resolvable = [
            row for row in t04
            if row["family"] != "single_control" and 20 <= float(row["separation_um"]) <= 40
        ]
        ten_um = [row for row in t04 if row["family"] != "single_control" and float(row["separation_um"]) == 10]
        rows.append({
            "method": method,
            "diagnostic": "T04_axial_pairs",
            "items": len(t04),
            "localized_rate": bool_mean(t04, "all_layers_localized"),
            "separated_rate_20_40um": bool_mean(resolvable, "separated"),
            "local_depth_w1_um": finite_mean(t04, "local_depth_w1_um"),
            "expected_layer_energy_fraction": finite_mean(t04, "expected_layer_energy_fraction"),
            "ten_um_expected_layer_energy_fraction": finite_mean(ten_um, "expected_layer_energy_fraction"),
            "mean_z_error_layers": finite_mean(t04, "mean_z_error_layers"),
            "false_peaks_mean": finite_mean(t04, "false_peak_count"),
        })
    write_csv(SUPPLEMENT / "local_summary.csv", rows)
    return rows


def object_macro_metrics() -> list[dict]:
    rows = []
    sources = {
        "e3_mean100_final": BASELINE / "evaluation/e3_mean100/metrics.csv",
        "mean_anchor_e3_mean100_final": OUTPUT / "evaluation/mean_anchor_e3_mean100/metrics.csv",
    }
    keys = (
        "native_total_loss", "common_normalized_mean_loss",
        "common_absolute_log_variance_loss", "common_normalized_shape_variance_loss",
        "common_weighted_tv", "common_e3_score",
    )
    for method, path in sources.items():
        data = [row for row in read_csv(path) if row["method"] == method and row["split"] == "test"]
        per_object = []
        for sample in ("T02", "T03", "T04"):
            selected = [row for row in data if row["sample_id"] == sample]
            per_object.append({key: finite_mean(selected, key) for key in keys})
        rows.append({"method": method, **{
            key: float(np.mean([row[key] for row in per_object])) for key in keys
        }})
    write_csv(SUPPLEMENT / "physical_metrics_object_macro.csv", rows)
    return rows


def correction_summary() -> list[dict]:
    source = read_csv(OUTPUT / "evaluation/mean_anchor_e3_mean100/correction_metrics.csv")
    source = [row for row in source if row["checkpoint_role"] == "final" and row["split"] == "test"]
    rows = []
    for sample in ("T02", "T03", "T04"):
        selected = [row for row in source if row["sample_id"] == sample]
        rows.append({
            "sample_id": sample,
            "correction_to_anchor_l2": finite_mean(selected, "correction_to_anchor_l2"),
            "signed_correction_mass_ratio": finite_mean(selected, "signed_correction_mass_ratio"),
        })
    rows.append({
        "sample_id": "object_macro",
        "correction_to_anchor_l2": float(np.mean([row["correction_to_anchor_l2"] for row in rows])),
        "signed_correction_mass_ratio": float(np.mean([row["signed_correction_mass_ratio"] for row in rows])),
    })
    write_csv(SUPPLEMENT / "correction_summary.csv", rows)
    return rows


def validation_curves() -> list[dict]:
    sources = {
        "e3_mean100_final": BASELINE / "e3_mean100/validation_metrics.csv",
        "mean_anchor_e3_mean100_final": OUTPUT / "mean_anchor_e3_mean100/validation_metrics.csv",
    }
    rows = []
    for method, path in sources.items():
        data = read_csv(path)
        for step in range(20, 401, 20):
            selected = [row for row in data if int(row["step"]) == step]
            object_scores = []
            for sample in ("V01", "V02", "V03"):
                object_scores.append(finite_mean([row for row in selected if row["sample_id"] == sample], "selection_score"))
            rows.append({"method": method, "step": step, "object_macro_selection_score": float(np.mean(object_scores))})
    write_csv(SUPPLEMENT / "validation_curve.csv", rows)
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for method in sources:
        selected = [row for row in rows if row["method"] == method]
        axis.plot([row["step"] for row in selected], [row["object_macro_selection_score"] for row in selected],
                  marker="o", ms=3, label=LABELS[method], color=COLORS[method])
    axis.set(xlabel="optimizer step", ylabel="validation selection score (object macro)", title="Fixed validation curve")
    axis.grid(alpha=.25); axis.legend()
    figure.savefig(SUPPLEMENT / "validation_curve.png", dpi=170)
    plt.close(figure)
    return rows


def profile_figures(tables: dict[str, list[dict]]) -> int:
    rows = tables["fixed_profiles"]
    specs = {"T02": (13, 4), "T03": (12, 4), "T04": (16, 4)}
    for sample, (count, columns) in specs.items():
        selected = [row for row in rows if row["sample_id"] == sample and int(row["subset"]) == 1]
        regions = list(dict.fromkeys(row["region"] for row in selected))[:count]
        figure, axes = plt.subplots(math.ceil(count / columns), columns, figsize=(15, 3 * math.ceil(count / columns)),
                                    squeeze=False, layout="constrained")
        for axis, region in zip(axes.ravel(), regions):
            region_rows = [row for row in selected if row["region"] == region]
            for method in METHODS:
                values = sorted([row for row in region_rows if row["method"] == method], key=lambda row: float(row["coordinate"]))
                if not values:
                    continue
                x = np.asarray([float(row["coordinate"]) for row in values])
                y = np.asarray([float(row["value"]) for row in values])
                axis.plot(x, y / max(float(y.max()), 1e-30), label=LABELS[method], color=COLORS[method], lw=1.3)
            if sample != "T03":
                reference = sorted([row for row in region_rows if row["method"] == "e3_mean100_final"], key=lambda row: float(row["coordinate"]))
                if reference and reference[0].get("gt_value") not in (None, ""):
                    gt = np.asarray([float(row["gt_value"]) for row in reference])
                    axis.plot([float(row["coordinate"]) for row in reference], gt / max(float(gt.max()), 1e-30), "k--", lw=1.2, label="GT")
            axis.set_title(f"{sample} region {region}", fontsize=9)
            axis.grid(alpha=.2); axis.set_ylim(-.02, 1.05)
        for axis in axes.ravel()[len(regions):]:
            axis.axis("off")
        axes.ravel()[0].legend(fontsize=6)
        figure.suptitle(f"{sample} subset 01 · fixed ROI profiles · each curve normalized for shape display")
        figure.savefig(SUPPLEMENT / f"{sample}_subset01_local_profiles.png", dpi=160)
        plt.close(figure)
    return len(specs)


def build_report(local: list[dict], physical: list[dict], correction: list[dict]) -> dict:
    lookup = {(row["method"], row["diagnostic"]): row for row in local}
    baseline = "e3_mean100_final"
    candidate = "mean_anchor_e3_mean100_final"
    t02_b, t02_n = lookup[(baseline, "T02_tube_continuity")], lookup[(candidate, "T02_tube_continuity")]
    t03_b, t03_n = lookup[(baseline, "T03_three_lines")], lookup[(candidate, "T03_three_lines")]
    t04_b, t04_n = lookup[(baseline, "T04_axial_pairs")], lookup[(candidate, "T04_axial_pairs")]
    t03_ok = t03_n["separated_rate"] >= t03_b["separated_rate"] and t03_n["false_break_rate"] <= t03_b["false_break_rate"]
    t04_ok = t04_n["separated_rate_20_40um"] >= t04_b["separated_rate_20_40um"] and t04_n["local_depth_w1_um"] <= t04_b["local_depth_w1_um"]
    t02_ok = t02_n["centerline_localized_fraction"] >= t02_b["centerline_localized_fraction"] and t02_n["false_break_rate"] <= t02_b["false_break_rate"]
    base_quality = next(row for row in read_csv(ANALYSIS / "quality_summary.csv") if row["sample_id"] == "object_macro" and row["method"] == "e3_mean100")
    new_quality = next(row for row in read_csv(ANALYSIS / "quality_summary.csv") if row["sample_id"] == "object_macro" and row["method"] == "mean_anchor_e3_mean100")
    base_stability = next(row for row in read_csv(ANALYSIS / "stability_per_object.csv") if row["sample_id"] == "object_macro" and row["method"] == "e3_mean100")
    new_stability = next(row for row in read_csv(ANALYSIS / "stability_per_object.csv") if row["sample_id"] == "object_macro" and row["method"] == "mean_anchor_e3_mean100")
    broad_ok = all(float(new_quality[key]) <= float(base_quality[key]) for key in (
        "gt_scale_aligned_nrmse", "gt_axial_w1_um", "background_xy_mass_fraction"
    ))
    stability_ok = float(new_stability["shape_relative_dispersion"]) < float(base_stability["shape_relative_dispersion"])
    strict_support = stability_ok and broad_ok and t03_ok and t04_ok
    verdict = "supports_mean_anchor" if strict_support else "mixed"
    correction_macro = next(row for row in correction if row["sample_id"] == "object_macro")
    lines = [
        "# Mean主重建分支：完整对照结论", "",
        "主比较固定为两组第400步 final；两组best也都落在step 400，因此best与final结论相同。", "",
        "## 总体质量与稳定性", "",
        "| 网络 | 子集形状离散度↓ | 子集间轴向W1 μm↓ | aligned NRMSE↓ | GT轴向W1 μm↓ | 背景质量↓ |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Taylor主分支 | {float(base_stability['shape_relative_dispersion']):.4f} | {float(base_stability['pairwise_axial_w1_um']):.3f} | {float(base_quality['gt_scale_aligned_nrmse']):.4f} | {float(base_quality['gt_axial_w1_um']):.2f} | {float(base_quality['background_xy_mass_fraction']):.4f} |",
        f"| Mean主分支 | {float(new_stability['shape_relative_dispersion']):.4f} | {float(new_stability['pairwise_axial_w1_um']):.3f} | {float(new_quality['gt_scale_aligned_nrmse']):.4f} | {float(new_quality['gt_axial_w1_um']):.2f} | {float(new_quality['background_xy_mass_fraction']):.4f} |",
        "", "## 固定局部专项", "",
        "| 指标 | Taylor主分支 | Mean主分支 |", "|---|---:|---:|",
        f"| T02中心线恢复率↑ | {t02_b['centerline_localized_fraction']:.3f} | {t02_n['centerline_localized_fraction']:.3f} |",
        f"| T02假断率↓ | {t02_b['false_break_rate']:.3f} | {t02_n['false_break_rate']:.3f} |",
        f"| T03三线分开率↑ | {t03_b['separated_rate']:.3f} | {t03_n['separated_rate']:.3f} |",
        f"| T03假断率↓ | {t03_b['false_break_rate']:.3f} | {t03_n['false_break_rate']:.3f} |",
        f"| T03局部深度W1 μm↓ | {t03_b['local_depth_w1_um']:.2f} | {t03_n['local_depth_w1_um']:.2f} |",
        f"| T04 20–40 μm双层分开率↑ | {t04_b['separated_rate_20_40um']:.3f} | {t04_n['separated_rate_20_40um']:.3f} |",
        f"| T04局部深度W1 μm↓ | {t04_b['local_depth_w1_um']:.2f} | {t04_n['local_depth_w1_um']:.2f} |",
        f"| T04目标层能量占比↑ | {t04_b['expected_layer_energy_fraction']:.3f} | {t04_n['expected_layer_energy_fraction']:.3f} |",
        "",
        f"严格预设判定：`{verdict}`（稳定性={stability_ok}，总体质量={broad_ok}，T03不退步={t03_ok}，T04不退步={t04_ok}；T02连续性补充={t02_ok}）。",
        "",
        "## 重要解释", "",
        f"新网络有效修正的L2范数平均为校准Mean基础体的 {correction_macro['correction_to_anchor_l2']:.2f} 倍，净修正质量为基础体的 {correction_macro['signed_correction_mass_ratio']:.2f} 倍。",
        "这说明模型并不是只在Mean-RL3上做很小修补，而是仍由网络残差主导最终形状；因此本轮只能说明更换anchor带来的整体效果，不能把结果解释成Mean-RL3被原样保留。",
        "T04的10 μm间距是原生相邻层，只报告目标层能量，不用插值宣称双峰分辨。结论仅限当前单种子、10帧RL3和400步预算。",
        "",
        "完整数据见 `analysis/supplemental/`；清晰图见 `analysis/figures/` 与 `analysis/supplemental/*_local_profiles.png`。",
    ]
    report = OUTPUT / "FINAL_REPORT_ZH.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "complete": True,
        "verdict": verdict,
        "strict_support": strict_support,
        "stability_passed": stability_ok,
        "broad_quality_passed": broad_ok,
        "t02_continuity_no_regression": t02_ok,
        "t03_resolution_no_regression": t03_ok,
        "t04_axial_no_regression": t04_ok,
        "report": str(report),
    }
    old.write_json(OUTPUT / "final_verdict.json", result)
    return result


def main() -> None:
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    new = generate_new_local_tables()
    combined = combine_local(new)
    local = local_summary(combined)
    physical = object_macro_metrics()
    correction = correction_summary()
    validation_curves()
    profile_figures(combined)
    result = build_report(local, physical, correction)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
