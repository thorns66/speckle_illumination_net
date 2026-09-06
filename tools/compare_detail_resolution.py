from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _row(label: str, run: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    artifact = run["artifacts"]["best"]
    depth = artifact["depth"]
    resolution = artifact["resolution"]
    with (Path(run["output_dir"]) / "metrics.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    registered = float(resolution["registered_reconstruction_resolution_um"])
    near_mass = 100.0 * float(depth["true_depth_plus_minus_10um_mass_fraction"])
    centroid_error = float(depth["centroid_error_um"])
    peak = float(depth["peak_depth_um"])
    pass_resolution = registered <= baseline["resolution_um"] * 0.95
    pass_peak = abs(peak - 50.0) < 1e-6
    pass_near_mass = near_mass >= baseline["near_mass_percent"] - 2.0
    pass_centroid = centroid_error <= 2.0
    return {
        "label": label,
        "registered_resolution_um": registered,
        "resolution_improvement_vs_e0_percent": 100.0
        * (baseline["resolution_um"] - registered)
        / baseline["resolution_um"],
        "shared_center_resolution_um": float(
            resolution["shared_center_reconstruction_resolution_um"]
        ),
        "peak_depth_um": peak,
        "true_layer_mass_percent": 100.0 * float(depth["true_layer_mass_fraction"]),
        "mass_40_60um_percent": near_mass,
        "centroid_error_um": centroid_error,
        "first_layer_mass_percent": 100.0 * float(depth["first_layer_mass_fraction"]),
        "last_layer_mass_percent": 100.0 * float(depth["last_layer_mass_fraction"]),
        "best_normalized_var_loss": float(saved["best_var_loss"]),
        "best_total_loss_not_cross_comparable": float(saved["best_total_loss"]),
        "best_step": int(saved["best_step"]),
        "median_step_time_s": float(run["loss"]["median_step_time_s"]),
        "peak_gpu_memory_mb": float(run["loss"]["peak_gpu_memory_mb"]),
        "pass_resolution_5pct": pass_resolution,
        "pass_peak_50um": pass_peak,
        "pass_depth_mass_guardrail": pass_near_mass,
        "pass_centroid_guardrail": pass_centroid,
        "passes_all": pass_resolution and pass_peak and pass_near_mass and pass_centroid,
    }


def _plot(rows: list[dict[str, Any]], baseline: dict[str, float], path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(rows)))
    panels = [
        ("registered_resolution_um", "Registered FTC (µm)", baseline["resolution_um"] * 0.95, "max"),
        ("mass_40_60um_percent", "Mass at 40–60 µm (%)", baseline["near_mass_percent"] - 2.0, "min"),
        ("centroid_error_um", "Depth centroid error (µm)", 2.0, "max"),
        ("best_normalized_var_loss", "Best base variance loss", None, None),
        ("median_step_time_s", "Median step time (s)", baseline["step_time_s"], None),
        ("peak_gpu_memory_mb", "Peak GPU memory (MiB)", baseline["gpu_memory_mb"], None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for axis, (key, title, threshold, direction) in zip(axes.flat, panels):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if threshold is not None:
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
            suffix = "≤" if direction == "max" else "≥"
            axis.text(0.01, 0.98, f"gate {suffix} {threshold:.3g}", transform=axis.transAxes, va="top")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    figure.suptitle("50 µm detail-resolution candidates vs historical E0")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detail-resolution candidates with historical E0")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-label", default="E0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    runs = summary["runs"]
    baseline_run = runs[args.baseline_label]
    baseline_artifact = baseline_run["artifacts"]["best"]
    baseline = {
        "resolution_um": float(baseline_artifact["resolution"]["registered_reconstruction_resolution_um"]),
        "near_mass_percent": 100.0 * float(baseline_artifact["depth"]["true_depth_plus_minus_10um_mass_fraction"]),
        "step_time_s": float(baseline_run["loss"]["median_step_time_s"]),
        "gpu_memory_mb": float(baseline_run["loss"]["peak_gpu_memory_mb"]),
    }
    rows = [_row(label, run, baseline) for label, run in runs.items()]
    candidates = [row for row in rows if row["label"] != args.baseline_label]
    ranking = sorted(
        candidates,
        key=lambda row: (
            not row["passes_all"],
            row["registered_resolution_um"],
            -row["mass_40_60um_percent"],
            row["centroid_error_um"],
        ),
    )
    report = {
        "source_summary": str(summary_path),
        "baseline_label": args.baseline_label,
        "baseline": baseline,
        "acceptance": {
            "registered_resolution": "at least 5% finer than E0",
            "peak_depth_um": 50.0,
            "mass_40_60um": "no more than 2 percentage points below E0",
            "centroid_error_um_max": 2.0,
        },
        "warning": "Total losses are not compared across runs because the objectives differ. best_normalized_var_loss is the shared physics term.",
        "rows": rows,
        "candidate_ranking": [row["label"] for row in ranking],
        "recommended_candidate": ranking[0]["label"] if ranking and ranking[0]["passes_all"] else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detail_resolution_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output_dir / "detail_resolution_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, baseline, output_dir / "detail_resolution_comparison.png")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
