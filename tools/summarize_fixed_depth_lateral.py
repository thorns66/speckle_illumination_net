from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_layer(path: Path, depth_index: int = 4) -> np.ndarray:
    value = np.load(path) if path.suffix.lower() == ".npy" else tifffile.imread(path)
    value = np.asarray(value, dtype=np.float64).squeeze()
    if value.ndim != 3:
        raise ValueError(f"Expected [Z,H,W] at {path}, got {value.shape}")
    return np.maximum(value[depth_index], 0.0)


def _display(image: np.ndarray) -> np.ndarray:
    scale = float(np.quantile(image, 0.999))
    return np.clip(image / max(scale, np.finfo(np.float64).tiny), 0.0, 1.0)


def _extract(label: str, reconstruction: Path, metrics_path: Path) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    shared = metrics["native_shared_reference_center"]["no_mean"]["robust"]
    registered = metrics["center_registered_radial_contrast"]["no_mean"]["robust"]
    artifacts = metrics["visual_artifact_quality"]
    relative = artifacts["reconstruction_relative_to_reference"]
    reconstruction_artifacts = artifacts["reconstruction"]
    similarity = artifacts["annular_similarity_to_reference"]
    center = metrics["center_calibration"]
    return {
        "label": label,
        "reconstruction_path": str(reconstruction.resolve()),
        "shared_center_ftc_um": float(shared["resolution_um"]),
        "registered_ftc_um_diagnostic_only": float(registered["resolution_um"]),
        "center_shift_original_px": float(center["center_displacement_px_original_grid"]),
        "radial_barb_ratio_vs_var": float(relative["radial_barb_energy_ratio"]),
        "off_harmonic_ratio_vs_var": float(relative["off_harmonic_energy_ratio"]),
        "laplacian_ratio_vs_var": float(relative["laplacian_energy_ratio"]),
        "target_phase_coherence": float(
            reconstruction_artifacts["target_harmonic_phase_coherence"]
        ),
        "annular_intensity_correlation_vs_var": float(similarity["intensity_correlation"]),
        "annular_gradient_cosine_vs_var": float(similarity["gradient_cosine_similarity"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the fixed-z lateral oracle diagnostic")
    parser.add_argument("--root", default="outputs/fixed_depth_lateral_50um")
    parser.add_argument(
        "--e0-reconstruction",
        default="outputs/depth50_n100_no_mean_loss/reconstruction_best.npy",
    )
    parser.add_argument("--e0-evaluation", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    evaluations: list[tuple[str, Path, Path]] = [
        ("E0 no_mean", Path(args.e0_reconstruction), Path(args.e0_evaluation))
    ]
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        reconstruction = directory / "reconstruction_best.npy"
        metrics = directory / "resolution_best" / "resolution_metrics.json"
        if reconstruction.is_file() and metrics.is_file() and not directory.name.startswith("_"):
            evaluations.append((directory.name, reconstruction, metrics))
    rows = [_extract(label, reconstruction, metrics) for label, reconstruction, metrics in evaluations]
    baseline = rows[0]
    for row in rows:
        row["ftc_change_vs_e0_percent"] = (
            (row["shared_center_ftc_um"] - baseline["shared_center_ftc_um"])
            / baseline["shared_center_ftc_um"]
            * 100.0
        )
        row["laplacian_change_vs_e0_percent"] = (
            (row["laplacian_ratio_vs_var"] - baseline["laplacian_ratio_vs_var"])
            / baseline["laplacian_ratio_vs_var"]
            * 100.0
        )
        row["radial_barb_change_vs_e0_percent"] = (
            (row["radial_barb_ratio_vs_var"] - baseline["radial_barb_ratio_vs_var"])
            / baseline["radial_barb_ratio_vs_var"]
            * 100.0
        )
    candidates = rows[1:]
    best_ftc = min(candidates, key=lambda row: row["shared_center_ftc_um"])
    clean_candidates = [
        row
        for row in candidates
        if row["laplacian_ratio_vs_var"] <= baseline["laplacian_ratio_vs_var"]
        and row["radial_barb_ratio_vs_var"] <= baseline["radial_barb_ratio_vs_var"]
    ]
    best_clean = (
        min(clean_candidates, key=lambda row: row["shared_center_ftc_um"])
        if clean_candidates
        else None
    )
    report = {
        "diagnostic_semantics": (
            "Known-depth oracle upper bound. It measures lateral headroom and cannot be deployed with unknown real-sample depth."
        ),
        "checkpoint_selection": "independent physical holdout loss; resolution metrics were revealed afterward",
        "baseline": baseline,
        "best_ftc_candidate": best_ftc["label"],
        "best_clean_candidate": None if best_clean is None else best_clean["label"],
        "rows": rows,
    }
    with (root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference_layer = _load_layer(Path(args.reference))
    images: list[tuple[str, np.ndarray, dict[str, Any] | None]] = [("VAR", reference_layer, None)]
    for row in rows:
        images.append((row["label"], _load_layer(Path(row["reconstruction_path"])), row))
    figure, axes = plt.subplots(
        2,
        len(images),
        figsize=(3.15 * len(images), 6.6),
        constrained_layout=True,
        squeeze=False,
    )
    for column, (label, image, row) in enumerate(images):
        shown = _display(image)
        title = label if row is None else (
            f"{label}\nFTC {row['shared_center_ftc_um']:.3f} um"
            f" | Lap {row['laplacian_ratio_vs_var']:.2f}x"
            f"\nbarb {row['radial_barb_ratio_vs_var']:.2f}x"
        )
        axes[0, column].imshow(shown, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(title, fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].imshow(shown[:130, :130], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, column].set_title("upper-left 130x130", fontsize=8)
        axes[1, column].axis("off")
    figure.savefig(root / "comparison_full_and_upper_left.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
