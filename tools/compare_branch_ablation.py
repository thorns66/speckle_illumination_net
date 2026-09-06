from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _percent_change(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _best_metrics(run: dict[str, Any]) -> dict[str, float]:
    loss = run["loss"]
    best = run["artifacts"]["best"]
    depth = best["depth"]
    resolution = best["resolution"]
    return {
        "registered_resolution_um": float(
            resolution["registered_reconstruction_resolution_um"]
        ),
        "shared_center_resolution_um": float(
            resolution["shared_center_reconstruction_resolution_um"]
        ),
        "peak_depth_um": float(depth["peak_depth_um"]),
        "true_layer_mass_percent": 100.0 * float(depth["true_layer_mass_fraction"]),
        "true_depth_plus_minus_10um_mass_percent": 100.0
        * float(depth["true_depth_plus_minus_10um_mass_fraction"]),
        "centroid_error_um": float(depth["centroid_error_um"]),
        "best_total_loss": float(loss["best_total_loss"]),
        "last20_median_total_loss": float(loss["last20_median_total_loss"]),
        "median_step_time_s": float(loss["median_step_time_s"]),
        "peak_gpu_memory_mb": float(loss["peak_gpu_memory_mb"]),
    }


def _make_row(
    label: str,
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float | str]:
    return {
        "label": label,
        **metrics,
        "registered_resolution_delta_um_vs_original": metrics[
            "registered_resolution_um"
        ]
        - baseline["registered_resolution_um"],
        "registered_resolution_change_percent_vs_original": _percent_change(
            metrics["registered_resolution_um"], baseline["registered_resolution_um"]
        ),
        "shared_center_resolution_delta_um_vs_original": metrics[
            "shared_center_resolution_um"
        ]
        - baseline["shared_center_resolution_um"],
        "true_layer_mass_delta_percentage_points_vs_original": metrics[
            "true_layer_mass_percent"
        ]
        - baseline["true_layer_mass_percent"],
        "plus_minus_10um_mass_delta_percentage_points_vs_original": metrics[
            "true_depth_plus_minus_10um_mass_percent"
        ]
        - baseline["true_depth_plus_minus_10um_mass_percent"],
        "centroid_error_delta_um_vs_original": metrics["centroid_error_um"]
        - baseline["centroid_error_um"],
        "best_loss_change_percent_vs_original": _percent_change(
            metrics["best_total_loss"], baseline["best_total_loss"]
        ),
        "step_time_change_percent_vs_original": _percent_change(
            metrics["median_step_time_s"], baseline["median_step_time_s"]
        ),
        "gpu_memory_delta_mb_vs_original": metrics["peak_gpu_memory_mb"]
        - baseline["peak_gpu_memory_mb"],
        "gpu_memory_change_percent_vs_original": _percent_change(
            metrics["peak_gpu_memory_mb"], baseline["peak_gpu_memory_mb"]
        ),
    }


def _plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)

    panels = [
        ("registered_resolution_um", "Registered FTC (um)", "lower is better"),
        ("shared_center_resolution_um", "Shared-center FTC (um)", "lower is better"),
        (
            "true_depth_plus_minus_10um_mass_percent",
            "Mass in 40-60 um (%)",
            "higher is better",
        ),
        ("centroid_error_um", "Depth centroid error (um)", "lower is better"),
        ("best_total_loss", "Best training loss", "lower is better"),
    ]
    for axis, (key, title, subtitle) in zip(axes.flat[:5], panels):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(x, values, color=colors[: len(rows)])
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(f"{title}\n{subtitle}")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            precision = 5 if key == "best_total_loss" else 2
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.{precision}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis = axes.flat[5]
    width = 0.36
    time_relative = [
        100.0 * float(row["median_step_time_s"]) / float(rows[0]["median_step_time_s"])
        for row in rows
    ]
    memory_relative = [
        100.0 * float(row["peak_gpu_memory_mb"]) / float(rows[0]["peak_gpu_memory_mb"])
        for row in rows
    ]
    axis.bar(x - width / 2, time_relative, width, label="step time", color="#72B7B2")
    axis.bar(x + width / 2, memory_relative, width, label="GPU memory", color="#B279A2")
    axis.axhline(100.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(x, labels, rotation=15)
    axis.set_title("Cost relative to original (%)\nlower is better")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    fig.suptitle("50 um no_mean branch ablation vs original three-branch SetBranch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare branch ablations against the original three-branch model"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-label", default="E0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    runs = summary["runs"]
    if args.baseline_label not in runs:
        raise KeyError(f"Missing baseline label {args.baseline_label!r}")

    absolute = {label: _best_metrics(run) for label, run in runs.items()}
    baseline = absolute[args.baseline_label]
    rows = [_make_row(label, metrics, baseline) for label, metrics in absolute.items()]
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source_summary": str(summary_path),
        "baseline_label": args.baseline_label,
        "baseline_definition": "original VAR+Mean+mean/std SetBranch+gate, no_mean loss",
        "sign_convention": {
            "resolution_loss_time_memory": "negative delta is better",
            "mass": "positive delta is better",
            "centroid_error": "negative delta is better",
        },
        "rows": rows,
    }
    with (output_dir / "comparison_vs_original.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output_dir / "comparison_vs_original.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output_dir / "comparison_vs_original.png")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
