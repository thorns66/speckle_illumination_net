#!/usr/bin/env python3
"""Render reproducible per-sample and cross-sample pilot comparisons.

MATLAB arrays written as Y-by-X-by-Z appear through h5py as Z-by-X-by-Y.
The z index is therefore the first Python axis, while every displayed slice
must be transposed back to Y-by-X.  Every image panel is shown at the
configured ground-truth depth.  Display scaling is independent (0--99.7
percentile) and stated in each figure, so these plots are not mistaken for
absolute photometric comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VOLUME_KEYS = (
    ("legacy_mean_raw", "Legacy mean RL"),
    ("legacy_taylor_raw", "Legacy Taylor sqrt"),
    ("physics_mean_raw", "Retained-scale mean RL"),
    ("physics_taylor_raw", "Retained-scale Taylor sqrt"),
)


def _truth_metadata(prepared_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(prepared_path, "r") as handle:
        truth = np.asarray(handle["ground_truth"], dtype=np.float64)
        z_um = np.asarray(handle["cfg/z_um"], dtype=np.float64).reshape(-1)
        truth_index = int(np.asarray(handle["truth_index_one_based"]).reshape(-1)[0]) - 1
    if truth.shape[0] != z_um.size or not (0 <= truth_index < z_um.size):
        raise ValueError(f"Invalid truth metadata in {prepared_path}")
    return truth, z_um, truth_index


def _load_subset(subset_path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with h5py.File(subset_path, "r") as handle:
        z_um = np.asarray(handle["z_um"], dtype=np.float64).reshape(-1)
        result["z_um"] = z_um
        for key, _ in VOLUME_KEYS:
            volume = np.asarray(handle[key], dtype=np.float64)
            if volume.shape[0] != z_um.size or not np.all(np.isfinite(volume)):
                raise ValueError(f"Invalid {key} in {subset_path}")
            # Taylor's solver output estimates g^2; convert it exactly once
            # to the amplitude-domain image used by the legacy workflow.
            if "taylor" in key:
                volume = np.sqrt(np.maximum(volume, 0.0))
            result[key] = volume
    return result


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(finite, 99.7))
    if high <= 0:
        high = float(np.max(finite))
    return 0.0, max(high, np.finfo(np.float64).eps)


def _axial_mass(volume: np.ndarray) -> np.ndarray:
    mass = np.maximum(volume, 0.0).sum(axis=(1, 2), dtype=np.float64)
    total = float(mass.sum())
    if total <= 0:
        raise ValueError("Volume has zero nonnegative mass")
    return mass / total


def _metrics(volume: np.ndarray, z_um: np.ndarray, truth_index: int) -> dict[str, float]:
    mass = _axial_mass(volume)
    return {
        "peak": float(z_um[int(np.argmax(mass))]),
        "centroid": float(np.dot(mass, z_um)),
        "truth_mass": float(mass[truth_index]),
    }


def render_sample(sample_dir: Path, subset_index: int = 1) -> tuple[Path, Path]:
    prepared = sample_dir / "prepared.mat"
    subset = sample_dir / "subsets" / f"subset_{subset_index:02d}.mat"
    if not prepared.is_file() or not subset.is_file():
        raise FileNotFoundError(f"Incomplete sample: {sample_dir}")
    truth, z_um, truth_index = _truth_metadata(prepared)
    data = _load_subset(subset)
    if not np.array_equal(z_um, data["z_um"]):
        raise ValueError(f"Depth mismatch in {sample_dir}")

    sample_id = sample_dir.name
    preview_dir = sample_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = preview_dir / f"{sample_id}_pilot_summary.png"
    axial_path = preview_dir / f"{sample_id}_axial_mass.png"

    panels = [(truth, "Ground truth")]
    for key, label in VOLUME_KEYS:
        panels.append((data[key], label))

    fig, axes = plt.subplots(1, len(panels), figsize=(19, 4.2), constrained_layout=True)
    for axis, (volume, label) in zip(axes, panels):
        image = volume[truth_index].T
        vmin, vmax = _display_limits(image)
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(label, fontsize=10)
        axis.set_axis_off()
        if label != "Ground truth":
            metrics = _metrics(volume, z_um, truth_index)
            axis.text(
                0.02,
                0.02,
                f"peak {metrics['peak']:.0f} um\n"
                f"true-layer mass {100 * metrics['truth_mass']:.1f}%",
                transform=axis.transAxes,
                color="white",
                fontsize=8,
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.60, "pad": 2, "edgecolor": "none"},
            )
    fig.suptitle(
        f"{sample_id} | subset {subset_index:02d} | true depth {z_um[truth_index]:.0f} um\n"
        "Each panel independently scaled to its 99.7th percentile",
        fontsize=12,
    )
    fig.savefig(comparison_path, dpi=180, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    styles = ("o-", "s-", "^-", "D-")
    for (key, label), style in zip(VOLUME_KEYS, styles):
        volume = data[key]
        mass = _axial_mass(volume)
        metrics = _metrics(volume, z_um, truth_index)
        axis.plot(
            z_um,
            mass,
            style,
            linewidth=1.7,
            markersize=4,
            label=(f"{label}: peak {metrics['peak']:.0f}, "
                   f"centroid {metrics['centroid']:.1f} um"),
        )
    axis.axvline(
        z_um[truth_index], color="black", linestyle="--", linewidth=1.2,
        label="True depth",
    )
    axis.set(
        xlabel="Depth (um)", ylabel="Fraction of volume mass",
        title=f"{sample_id} axial mass | subset {subset_index:02d}",
    )
    axis.set_xticks(z_um)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(axial_path, dpi=180, facecolor="white")
    plt.close(fig)
    return comparison_path, axial_path


def render_montage(dataset_root: Path, sample_ids: list[str], subset_index: int = 1) -> Path:
    rows: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]] = []
    complete_ids: list[str] = []
    for sample_id in sample_ids:
        sample_dir = dataset_root / sample_id
        subset_path = sample_dir / "subsets" / f"subset_{subset_index:02d}.mat"
        if not subset_path.is_file():
            continue
        truth, z_um, truth_index = _truth_metadata(sample_dir / "prepared.mat")
        data = _load_subset(subset_path)
        if not np.array_equal(z_um, data["z_um"]):
            raise ValueError(f"Depth mismatch in {sample_dir}")
        rows.append(
            (
                sample_id,
                float(z_um[truth_index]),
                truth[truth_index],
                data["legacy_mean_raw"][truth_index],
                data["legacy_taylor_raw"][truth_index],
            )
        )
        complete_ids.append(sample_id)
    if not rows:
        raise ValueError("No complete samples available for montage")
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(10.2, 3.25 * len(rows)), squeeze=False,
        constrained_layout=True,
    )
    column_titles = ("Ground truth", "10-frame Legacy mean RL", "10-frame Legacy Taylor sqrt")
    for row_index, (sample_id, truth_depth, truth, mean_volume, taylor_volume) in enumerate(rows):
        images = (truth.T, mean_volume.T, taylor_volume.T)
        for column_index, (axis, image) in enumerate(zip(axes[row_index], images)):
            vmin, vmax = _display_limits(image)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if row_index == 0:
                axis.set_title(column_titles[column_index])
            if column_index == 0:
                axis.set_ylabel(f"{sample_id}\n{truth_depth:.0f} um")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"{len(rows)}-sample MATLAB pilot | subset {subset_index:02d} | true-depth slices\n"
        "Each panel independently scaled to its 99.7th percentile",
        fontsize=12,
    )
    default_five = complete_ids == [f"P{i:02d}" for i in range(1, 6)]
    prefix = "pilot5" if default_five else f"pilot{len(rows)}"
    output = dataset_root / f"{prefix}_subset_{subset_index:02d}_comparison.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--samples", nargs="+", default=[f"P{i:02d}" for i in range(1, 6)])
    parser.add_argument("--subset", type=int, default=1)
    args = parser.parse_args()
    completed: list[str] = []
    for sample_id in args.samples:
        sample_dir = args.dataset_root / sample_id
        subset = sample_dir / "subsets" / f"subset_{args.subset:02d}.mat"
        if subset.is_file():
            paths = render_sample(sample_dir, args.subset)
            print(*(str(path) for path in paths), sep="\n")
            completed.append(sample_id)
    if completed:
        print(render_montage(args.dataset_root, completed, args.subset))


if __name__ == "__main__":
    main()
