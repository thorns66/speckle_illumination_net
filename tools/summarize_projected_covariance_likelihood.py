"""Report completed diagnostics, keeping oracle controls visibly separate."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.summarize_nested_frame_count_cs import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um/projected_covariance_likelihood")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows, pending, anchors = [], [], {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.exists():
            pending.append(directory.name)
            continue
        summary = json.loads(path.read_text())
        if summary["history"][-1]["step"] != 199 or summary["arguments"]["steps"] != 200:
            raise RuntimeError(f"Not a completed 200-step experiment: {directory}")
        dataset = Path(summary["arguments"]["dataset"])
        truth = np.load(dataset / "target.npy")
        key = str(dataset)
        if key not in anchors:
            anchors[key] = evaluate(np.load(dataset / "anchor_rl3.npy"), truth)
        for variant in ("best", "final"):
            metrics = evaluate(np.load(directory / f"reconstruction_{variant}.npy"), truth)
            rows.append({"run": directory.name, "checkpoint": variant,
                         "target": summary["arguments"]["target"],
                         "step": summary["best_step"] if variant == "best" else 199,
                         "dimension": summary["arguments"]["dimension"],
                         "best_holdout_score": summary["best_holdout_score"],
                         "optimization_elapsed_s": summary["history"][-1]["elapsed_s"],
                         "peak_gpu_memory_mb": summary["history"][-1]["peak_gpu_memory_mb"],
                         **metrics})
    report = {"complete": not pending, "pending": pending, "anchor_metrics": anchors,
              "scope": "Fixed true 50 um; oracle groups are positive controls, never deployable estimates",
              "selection": "Empirical groups use independent holdout frames; no truth-based checkpoint selection",
              "timing_note": "Optimization times exclude PSF loading, QR, and adjoint precomputation",
              "rows": rows}
    (root / "summary_latest.json").write_text(json.dumps(report, indent=2))
    if rows:
        with (root / "summary_latest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    dataset = root.parent / "frame_count_nested/n800"
    truth = np.load(dataset / "target.npy")[4]
    images = [("RL3 anchor", np.load(dataset / "anchor_rl3.npy")[4])]
    for title, name in (("Empirical MSE / D128", "n800_mse_d128"),
                        ("Empirical likelihood / D128", "n800_nll_d128_r0"),
                        ("Likelihood + weak smoothness", "n800_nll_d128_reg001"),
                        ("Oracle likelihood / D128", "oracle_nll_d128_r0"),
                        ("Oracle likelihood / D1024", "oracle_nll_d1024_r0")):
        if (root / name / "summary.json").exists():
            images.append((title, np.load(root / name / "reconstruction_best.npy")[4]))
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (title, layer) in zip(axes.flat, images):
        layer = layer / layer.sum()
        axis.imshow(layer[:140, :140], cmap="gray", vmin=0, vmax=1.5 * truth.max() / truth.sum())
        axis.set_title(title)
    figure.savefig(root / "upper_left_comparison_latest.png", dpi=150)
    plt.close(figure)
    ridge_names = ("oracle_nll_d1024_r0", "oracle_nll_d1024_j1e4", "oracle_nll_d1024_j1e5")
    if all((root/name/"summary.json").exists() for name in ridge_names):
        ridge_images = [("Anchor", np.load(dataset/"anchor_rl3.npy")[4])]
        ridge_images += [(name.rsplit("_", 1)[-1], np.load(root/name/"reconstruction_best.npy")[4]) for name in ridge_names]
        figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
        for axis, (label, layer) in zip(axes, ridge_images):
            layer = layer/layer.sum()
            axis.imshow(layer[:140, :140], cmap="gray", vmin=0, vmax=1.5*truth.max()/truth.sum())
            axis.set_title(label); axis.axis("off")
        figure.savefig(root/"ridge_sensitivity_upper_left.png", dpi=150)
        plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
