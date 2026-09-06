"""Compare finite-source population Cs with stationary controls, without GT selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.summarize_nested_frame_count_cs import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-fullbatch", action="store_true")
    args = parser.parse_args()
    suffix = "_with_fullbatch" if args.include_fullbatch else ""
    root = Path("outputs/linear_float_oracle_50um").resolve()
    output = root / "exact_population_diagnostics"
    truth = np.load(root / "dataset/target.npy")
    anchor = np.load(root / "dataset/anchor_rl3.npy")
    rows = [{"variant": "anchor", "checkpoint": "initial", "step": -1,
             "holdout_loss": None, **evaluate(anchor, truth)}]
    pictures = {"Truth": truth[4], "RL3 anchor": anchor[4]}
    runs = {
        "stationary_B0.5": root / "oracle_sensor_bandwidth/sigma0_b0p5",
        "exact_B0.5": output / "oracle_b0p5",
        "stationary_B2": root / "oracle_sensor_bandwidth/sigma0_b2",
        "exact_B2": output / "oracle_b2",
    }
    if args.include_fullbatch:
        runs["exact_B0.5_full32"] = root / "exact_population_fullbatch/b0p5"
        runs["exact_B2_full32"] = root / "exact_population_fullbatch/b2"
    for name, directory in runs.items():
        summary = json.loads((directory / "summary.json").read_text())
        if summary["steps"] != 200 or summary["history"][-1]["step"] != 199:
            raise RuntimeError(f"Incomplete: {directory}")
        for checkpoint in ("best", "final"):
            volume = np.load(directory / f"reconstruction_{checkpoint}.npy")
            if volume.shape != truth.shape or not np.isfinite(volume).all() or np.count_nonzero(np.delete(volume, 4, axis=0)):
                raise ValueError(f"Invalid fixed-depth output: {directory}")
            rows.append({"variant": name, "checkpoint": checkpoint,
                         "step": summary["best_step"] if checkpoint == "best" else 199,
                         "holdout_loss": summary["best_holdout_loss"] if checkpoint == "best" else summary["history"][-1]["holdout_loss"],
                         **evaluate(volume, truth)})
            if checkpoint == "best":
                pictures[name] = volume[4]
    audits = []
    for count in (100, 800):
        for window in (0, 16):
            current = json.loads((output / f"n{count}_w{window}.json").read_text())
            previous_path = (root / "oracle_sensor_bandwidth" / f"empirical_transfer_n{count}.json" if window == 0
                             else root / "localized_covariance_audit" / f"n{count}_w{window}.json")
            previous = json.loads(previous_path.read_text())
            if current["sensor_probe_seed"] != previous["sensor_probe_seed"]:
                raise ValueError("Different sensor seeds")
            for model, source in (("stationary", previous), ("exact", current)):
                grad = source["gradient_agreement"]
                scores = source["scores"]
                row = {"frames": count, "window_sigma": window, "model": model,
                       **{key: value for key, value in grad.items() if "cosine" in key}}
                for split in ("train", "holdout"):
                    row[f"{split}_truth_loss"] = scores["truth"][split]["relative_mse"]
                    row[f"{split}_anchor_loss"] = scores["local_anchor"][split]["relative_mse"]
                row["prescreen_pass"] = (all(grad[key] >= .5 for key in (
                    "train_population_cosine", "holdout_population_cosine", "train_holdout_cosine"))
                    and all(row[f"{split}_truth_loss"] < row[f"{split}_anchor_loss"] for split in ("train", "holdout")))
                audits.append(row)
    report = {"complete": True, "rows": rows, "audits": audits,
              "covariance_build": json.loads((root / "exact_population_cs/report.json").read_text()),
              "scope": "Fixed known 50 um, 200-step population-statistic oracle controls; NOT empirical-data reconstruction or unknown-depth validation.",
              "selection": "Independent sensor-probe oracle loss; neither FTC nor truth-image error selects checkpoints.",
              "limits": "Exact Cs applies to the linear finite random-phase simulator only. All existing empirical holdout frames are research validation, not a pristine test set. Gradient prescreen is an engineering heuristic, not an identifiability theorem.",
              "production_changed": False, "mean_backprop_enabled": False}
    (output / f"comparison{suffix}.json").write_text(json.dumps(report, indent=2))
    rows_count = (len(pictures) + 2) // 3
    fig, axes = plt.subplots(rows_count, 3, figsize=(12, 4 * rows_count), constrained_layout=True)
    scale = 1.5 * truth[4].max() / truth[4].sum()
    for ax, (name, value) in zip(axes.flat, pictures.items()):
        value = value.astype(np.float64); value /= value.sum()
        ax.imshow(value[:140, :140], cmap="gray", vmin=0, vmax=scale)
        metric = next((row for row in rows if row["variant"] == ("anchor" if name == "RL3 anchor" else name)
                       and row["checkpoint"] in ("initial", "best")), None)
        title = name if metric is None else f"{name}\nFTC {metric['ftc_um']:.3f} um | L2 {metric['unit_mass_relative_l2']:.3f}"
        ax.set_title(title); ax.axis("off")
    for ax in list(axes.flat)[len(pictures):]:
        ax.axis("off")
    fig.savefig(output / f"upper_left_comparison{suffix}.png", dpi=130)
    plt.close(fig)
    print(json.dumps({"rows": rows, "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
