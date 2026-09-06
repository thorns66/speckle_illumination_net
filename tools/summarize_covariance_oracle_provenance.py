"""Audit oracle provenance without selecting checkpoints from image truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.summarize_nested_frame_count_cs import evaluate


def paired_probe_comparison(candidate, anchor, *, seed=20261051, repeats=4000):
    """Conditional sketch Monte Carlo interval, NOT a frame/generalization CI."""
    candidate, anchor = np.asarray(candidate, dtype=np.float64), np.asarray(anchor, dtype=np.float64)
    if (candidate.ndim != 1 or candidate.shape != anchor.shape or len(candidate) < 2
            or not np.isfinite(candidate).all() or not np.isfinite(anchor).all()
            or np.any(candidate < 0) or np.any(anchor < 0) or anchor.mean() <= 0):
        raise ValueError("Expected paired finite nonnegative per-probe losses")
    indices = np.random.default_rng(seed).integers(len(anchor), size=(repeats, len(anchor)))
    difference = candidate - anchor
    relative = 100 * difference[indices].mean(1) / anchor[indices].mean(1)
    return {"relative_mse": float(candidate.mean()),
            "change_from_anchor_percent": float(100 * difference.mean() / anchor.mean()),
            "probe_only_bootstrap_95_percent": np.quantile(relative, [.025, .975]).tolist(),
            "probe_count": len(anchor)}


def read_empirical_audits(paths, dataset):
    combined, reference = {}, None
    for path in paths:
        report = json.loads(Path(path).read_text())
        args = report["arguments"]
        if (not report.get("complete") or Path(args["dataset"]).resolve() != dataset
                or args["cs_model"] != "finite_phase" or not args["skip_gradients"]):
            raise ValueError(f"Incompatible frozen-candidate audit: {path}")
        signature = tuple(args[key] for key in ("probes", "sensor_seed", "sigma", "window_sigma", "cs_model", "cs_directory"))
        anchor = np.asarray(report["scores"]["local_anchor"]["holdout"]["per_probe_losses"])
        if len(anchor) != args["probes"]:
            raise ValueError("Incomplete probe bank")
        if reference is None:
            reference = (signature, anchor)
        elif signature != reference[0] or not np.allclose(anchor, reference[1], rtol=1e-6, atol=1e-8):
            raise ValueError("Audits must share the exact probe/data protocol")
        for name, scores in report["scores"].items():
            source = report["candidate_sources"][name]
            source_path = Path(source["path"]).resolve()
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"Candidate changed after audit: {source_path}")
            item = {"source": source, "name": name,
                    **paired_probe_comparison(scores["holdout"]["per_probe_losses"], anchor),
                    "correlation": scores["holdout"]["correlation"],
                    "population_oracle_loss": scores["population_oracle_loss"]}
            key = str(source_path)
            if key in combined and not np.isclose(combined[key]["relative_mse"], item["relative_mse"], rtol=1e-6):
                raise ValueError("Repeated candidate scores disagree")
            combined[key] = item
    return {"scope": "New sensor probes, existing acquisition frames reused. Paired bootstrap is only probe Monte Carlo uncertainty conditional on these frames; not a new acquisition, not a generalization confidence interval. No checkpoint reselection.",
            "reports": [str(Path(path).resolve()) for path in paths],
            "candidates_by_path": combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um/oracle_provenance")
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/dataset")
    parser.add_argument("--empirical-audit", action="append", default=[])
    args = parser.parse_args()
    root, dataset = Path(args.root).resolve(), Path(args.dataset).resolve()
    truth = np.load(dataset / "target.npy")
    anchor = np.load(dataset / "anchor_rl3.npy")
    rows = [{"variant": "anchor", "checkpoint": "initial", "step": -1,
             "holdout_loss": None, **evaluate(anchor, truth)}]
    pictures = {"anchor": anchor[4]}
    summaries = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing completed summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary["steps"] != 200 or summary["history"][-1]["step"] != 199:
            raise RuntimeError(f"Incomplete experiment: {directory}")
        if Path(summary["arguments"]["dataset"]).resolve() != dataset:
            raise ValueError(f"Wrong dataset for {directory}")
        expected_steps = [0] + list(range(19, 200, 20))
        if [item["step"] for item in summary["history"]] != expected_steps:
            raise ValueError(f"Unexpected checkpoint steps for {directory}")
        selected_loss, selected_step = summary["initial_holdout_loss"], -1
        for item in summary["history"]:
            if not np.isfinite(item["holdout_loss"]):
                raise ValueError(f"Nonfinite validation loss for {directory}")
            if item["holdout_loss"] < selected_loss:
                selected_loss, selected_step = item["holdout_loss"], item["step"]
        if selected_step != summary["best_step"] or selected_loss != summary["best_holdout_loss"]:
            raise ValueError(f"Best checkpoint is not chosen by held-out loss: {directory}")
        selected_path = directory / ("reconstruction_initial.npy" if selected_step == -1
                                      else f"reconstruction_step{selected_step:03d}.npy")
        if not np.array_equal(np.load(selected_path), np.load(directory/"reconstruction_best.npy")):
            raise ValueError(f"Best volume does not match selected step: {directory}")
        summaries[directory.name] = {key: summary[key] for key in (
            "arguments", "steps", "best_step", "initial_holdout_loss", "true_shape_holdout_loss")}
        for checkpoint in ("best", "final"):
            volume = np.load(directory / f"reconstruction_{checkpoint}.npy")
            if (volume.shape != truth.shape or not np.isfinite(volume).all()
                    or np.any(volume < 0) or not np.isclose(volume.sum(), 1., atol=1e-5)):
                raise ValueError(f"Invalid volume: {directory}/{checkpoint}")
            if np.count_nonzero(np.delete(volume, 4, axis=0)):
                raise ValueError("This report covers only fixed-50-um diagnostics")
            row = {"variant": directory.name, "checkpoint": checkpoint,
                   "step": summary["best_step"] if checkpoint == "best" else 199,
                   "holdout_loss": summary["best_holdout_loss"] if checkpoint == "best"
                       else summary["history"][-1]["holdout_loss"], **evaluate(volume, truth)}
            rows.append(row)
            if checkpoint == "best":
                pictures[directory.name] = volume[4]
    old = dataset.parent / "oracle_cov_b0p5_s4/reconstruction_best.npy"
    replicated = root / "finite1024_matched/reconstruction_best.npy"
    parity = None
    if old.exists() and replicated.exists():
        previous, current = np.load(old), np.load(replicated)
        parity = float(np.linalg.norm(previous-current)/np.linalg.norm(previous))
    report = {
        "complete": True, "pending": [], "dataset": str(dataset),
        "scope": "Fixed known 50 um, unit-mass bounded-log shape, 200 steps; bound and sensor bandwidth recorded per experiment. Oracle target statistics use truth; not deployable results.",
        "selection": "Best by held-out sensor-probe loss, not FTC or truth-image error. The finite matched holdout uses an independent bank, shared by its target and model.",
        "interpretation": "Pattern-bank size is a covariance modeling count, NOT acquired frame count. Stationary models are deterministic idealizations, not exact finite-aperture population covariance.",
        "original_oracle_replication_relative_l2": parity, "experiments": summaries, "rows": rows,
    }
    sources = {item["arguments"].get("target_source", "oracle") for item in summaries.values()}
    if sources == {"frames"}:
        report["scope"] = "Fixed known 50 um, 200 updates, frame-derived covariance targets and fixed population Cs. Known depth and unit mass are diagnostic assumptions, not deployable arbitrary 3D recovery."
        report["selection"] = "Best solely by separate observed holdout-frame covariance-vector loss on 16 independent sensor probes, including the initial candidate. No image truth or FTC for checkpoint choice. Final reported separately."
        report["interpretation"] = "Frames are actual acquisition counts in this linear simulation. Model has no scene/layer gain, mean backprop, or access to acquisition speckle realizations. An independent larger-probe audit is still needed."
    elif sources != {"oracle"}:
        raise ValueError("Do not mix frame-derived and oracle-target experiments in one comparison")
    if args.empirical_audit:
        report["independent_probe_audit"] = read_empirical_audits(args.empirical_audit, dataset)
        report["interpretation"] = report["interpretation"].replace("An independent larger-probe audit is still needed.", "")
        report["interpretation"] += " Larger independent-probe audit attached; uses the same acquired holdout frames, with no checkpoint reselection."
    (root/"summary_comparison.json").write_text(json.dumps(report, indent=2))
    with (root/"summary_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    pictures["truth"] = truth[4]
    labels = ["truth", "anchor"] + sorted(summaries)
    cols = 3 if len(labels) <= 6 else 4
    fig, axes = plt.subplots((len(labels)+cols-1)//cols, cols,
                             figsize=(14, 7.5), constrained_layout=True)
    peak = truth[4].max()/truth[4].sum()
    for ax, name in zip(axes.flat, labels):
        layer = pictures[name].astype(np.float64); layer /= layer.sum()
        ax.imshow(layer[:140, :140], cmap="gray", vmin=0, vmax=1.5*peak)
        if name == "truth":
            ax.set_title("Ground truth (evaluation only)", fontsize=9)
        else:
            metric = next(row for row in rows if row["variant"] == name and row["checkpoint"] in ("initial", "best"))
            ax.set_title(f"{name}\nFTC={metric['ftc_um']:.3f} um | L2={metric['unit_mass_relative_l2']:.3f}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(labels):]:
        ax.axis("off")
    fig.savefig(root/"upper_left_comparison.png", dpi=150)
    plt.close(fig)
    # Best and final must both be visible: a better FTC can accompany spikes.
    labels = ["truth", "anchor"] + sorted(summaries)
    fig, axes = plt.subplots(2, len(labels), figsize=(3 * len(labels), 6.5), constrained_layout=True)
    for row_index, checkpoint in enumerate(("best", "final")):
        for column, name in enumerate(labels):
            if name == "truth":
                layer, title = truth[4], "Truth (evaluation only)"
            elif name == "anchor":
                layer, title = anchor[4], "Initial VAR"
            else:
                layer = np.load(root/name/f"reconstruction_{checkpoint}.npy")[4]
                metric = next(item for item in rows if item["variant"] == name and item["checkpoint"] == checkpoint)
                title = (f"{name.replace('seed202609', 's')} | {checkpoint} {metric['step']}\n"
                         f"FTC {metric['ftc_um']:.2f} | L2 {metric['unit_mass_relative_l2']:.3f}\n"
                         f"peak/truth {metric['peak_ratio']:.2f}")
            layer = np.asarray(layer, dtype=np.float64); layer = layer/layer.sum()
            axes[row_index, column].imshow(layer[:140, :140], cmap="gray", vmin=0, vmax=1.5*peak)
            axes[row_index, column].set_title(title, fontsize=8)
            axes[row_index, column].axis("off")
    fig.suptitle("Fixed 50 um, N800 | common intensity scale | best by 16-probe holdout, no truth selection")
    fig.savefig(root/"best_and_final_comparison.png", dpi=130)
    fig.savefig(root/"best_and_final_comparison.jpg", dpi=100)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
