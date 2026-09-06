from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml


def _load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        volume = np.load(path)
    else:
        volume = tifffile.imread(path)
    volume = np.asarray(volume, dtype=np.float64).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {volume.shape} at {path}")
    return np.clip(volume, 0.0, None)


def _display_image(image: np.ndarray) -> np.ndarray:
    positive = image[image > 0]
    if positive.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(positive, [0.5, 99.8])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw VAR and four branch-ablation reconstructions"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--true-depth-um", type=float, default=50.0)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    reference = _load_volume(Path(summary["reference_path"]))
    runs = summary["runs"]

    first_run = next(iter(runs.values()))
    config_path = Path(first_run["config_path"])
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    z_um = np.asarray(config["psf"]["z_values_um"], dtype=np.float64)
    z_index = int(np.argmin(np.abs(z_um - args.true_depth_um)))

    labels = ["Raw VAR", *runs.keys()]
    volumes = [reference]
    registered_resolution = [
        float(
            next(iter(runs.values()))["artifacts"]["best"]["resolution"]
            ["registered_reference_resolution_um"]
        )
    ]
    near_mass = [
        100.0
        * float(reference.sum(axis=(-2, -1))[np.abs(z_um - args.true_depth_um) <= 10].sum())
        / float(reference.sum())
    ]
    for run in runs.values():
        best = run["artifacts"]["best"]
        volumes.append(_load_volume(Path(best["reconstruction_path"])))
        registered_resolution.append(
            float(best["resolution"]["registered_reconstruction_resolution_um"])
        )
        near_mass.append(
            100.0 * float(best["depth"]["true_depth_plus_minus_10um_mass_fraction"])
        )

    fig = plt.figure(figsize=(16, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, len(labels), height_ratios=[1.0, 0.72])
    for index, (label, volume, resolution, mass) in enumerate(
        zip(labels, volumes, registered_resolution, near_mass)
    ):
        axis = fig.add_subplot(grid[0, index])
        axis.imshow(_display_image(volume[z_index]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"{label}\nFTC {resolution:.3f} um | 40-60 mass {mass:.1f}%")
        axis.axis("off")

    axis = fig.add_subplot(grid[1, :])
    colors = ["black", "#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for label, volume, color in zip(labels, volumes, colors):
        mass = volume.sum(axis=(-2, -1), dtype=np.float64)
        mass /= mass.sum()
        axis.plot(z_um, 100.0 * mass, marker="o", linewidth=2, label=label, color=color)
    axis.axvline(args.true_depth_um, color="gray", linestyle="--", linewidth=1.5)
    axis.set_xlabel("Depth (um)")
    axis.set_ylabel("Layer mass (%)")
    axis.set_title("Axial mass distribution (best checkpoint)")
    axis.set_xticks(z_um)
    axis.grid(alpha=0.25)
    axis.legend(ncol=len(labels), loc="upper center")

    fig.suptitle(
        "50 um no_mean: E1/E2/E3 compared with original three-branch SetBranch (E0)"
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
