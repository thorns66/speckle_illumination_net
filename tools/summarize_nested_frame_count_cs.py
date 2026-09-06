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

from utils.artifact_quality import compare_annular_structure, compute_visual_artifact_metrics
from utils.resolution import compute_ftc_curve, prepare_resolution_image


def evaluate(volume: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    shape = np.asarray(volume[4], dtype=np.float64).clip(0)
    target = np.asarray(truth[4], dtype=np.float64).clip(0)
    shape /= shape.sum()
    target /= target.sum()
    image = prepare_resolution_image(shape, 2.0)
    reference = prepare_resolution_image(target, 2.0)
    center = (10.0, 12.0)
    curve = compute_ftc_curve(
        image,
        center_xy_1based=center,
        pixel_size_um=5.2 / 8.93,
        line_pairs=40,
        harmonic=10,
        angular_samples=1000,
        max_radius_px=500,
        threshold=0.1,
        smoothing_window=9,
        consecutive_below=5,
    )
    artifacts = compute_visual_artifact_metrics(
        image,
        center_xy_1based=center,
        inner_radius_px=20.0,
        outer_radius_px=120.0,
        harmonic=10,
        angular_samples=1000,
    )
    annular = compare_annular_structure(
        reference,
        image,
        center_xy_1based=center,
        inner_radius_px=20.0,
        outer_radius_px=120.0,
    )
    return {
        "ftc_um": float(curve.robust_cutoff.resolution_um),
        "unit_mass_relative_l2": float(np.linalg.norm(shape - target) / np.linalg.norm(target)),
        "pearson": float(np.corrcoef(shape.ravel(), target.ravel())[0, 1]),
        "peak_ratio": float(shape.max() / target.max()),
        "support_leak": float(shape[target <= 0].sum()),
        "radial_barb_energy": artifacts.radial_to_tangential_gradient_energy,
        "off_harmonic_energy": artifacts.off_harmonic_angular_energy_fraction,
        "phase_coherence": artifacts.target_harmonic_phase_coherence,
        "normalized_laplacian": artifacts.normalized_laplacian_energy,
        "annular_truth_correlation": annular.intensity_correlation,
        "annular_gradient_cosine": annular.gradient_cosine_similarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "frame_count_results"
    rows = []
    images = {}
    for count in (200, 400, 800):
        dataset = root / "frame_count_nested" / f"n{count}"
        truth = np.load(dataset / "target.npy")
        anchor = np.load(dataset / "anchor_rl3.npy")
        rows.append({"frames": count, "variant": "anchor", "checkpoint": "initial", "step": -1, "holdout_loss": None, **evaluate(anchor, truth)})
        images[(count, "anchor")] = anchor[4]
        for bound in ("0p5", "2p0"):
            directory = output / f"n{count}_b{bound}_s0p5"
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if summary["arguments"]["steps"] != 200 or summary["history"][-1]["step"] != 199:
                raise RuntimeError(f"Incomplete 200-step run: {directory}")
            for checkpoint in ("best", "final"):
                volume = np.load(directory / f"reconstruction_{checkpoint}.npy")
                step = summary["best_step"] if checkpoint == "best" else 199
                loss = summary["best_holdout_loss"] if checkpoint == "best" else summary["history"][-1]["holdout_loss"]
                rows.append({"frames": count, "variant": f"Cs_B{bound}", "checkpoint": checkpoint, "step": step, "holdout_loss": loss, **evaluate(volume, truth)})
                if checkpoint == "best":
                    images[(count, bound)] = volume[4]

    report = {
        "semantics": "Fixed true 50 um, 200 steps, complete Cs in sensor diagonal variance, Gaussian smoothing sigma=0.5; not sensor covariance-vector matching.",
        "dataset_seed": 20260940,
        "selection": "Independent same-N empirical holdout and fixed 512 model probes; truth/FTC not used for selection.",
        "evaluation": {"fixed_center_evaluation_pixels": [10.0, 12.0], "upsampling": 2, "artifact_annulus_evaluation_pixels": [20.0, 120.0]},
        "rows": rows,
    }
    (output / "frame_count_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "frame_count_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    for row_index, count in enumerate((200, 400, 800)):
        truth = np.load(root / "frame_count_nested" / f"n{count}" / "target.npy")[4]
        peak = float(truth.max() / truth.sum())
        for column, label in enumerate(("anchor", "0p5", "2p0")):
            image = images[(count, label)].astype(np.float64)
            image /= image.sum()
            axis = axes[row_index, column]
            axis.imshow(image[:140, :140], cmap="gray", vmin=0, vmax=1.5 * peak)
            variant = "anchor" if label == "anchor" else f"Cs_B{label}"
            row = next(item for item in rows if item["frames"] == count and item["variant"] == variant and item["checkpoint"] in ("initial", "best"))
            title = "RL3 anchor" if label == "anchor" else f"Cs B={label.replace('p', '.')}"
            axis.set_title(f"N={count} | {title}\nFTC={row['ftc_um']:.3f} um, L2={row['unit_mass_relative_l2']:.3f}")
            axis.axis("off")
    figure.savefig(output / "upper_left_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
