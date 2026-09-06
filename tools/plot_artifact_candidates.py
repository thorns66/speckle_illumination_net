from __future__ import annotations

import argparse
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


def _load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        volume = np.asarray(np.load(path), dtype=np.float64)
    else:
        volume = np.asarray(tifffile.imread(path), dtype=np.float64)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,H,W] at {path}, got {volume.shape}")
    return np.clip(volume, 0.0, None)


def _display_normalize(image: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    scale = float(np.quantile(value, 0.999))
    if scale <= 0.0:
        raise ValueError("Cannot display an empty image")
    return np.clip(value / scale, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full and upper-left artifact comparisons")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zoom-size", type=int, default=130)
    parser.add_argument("--artifact-name", default=None)
    args = parser.parse_args()

    with Path(args.summary).expanduser().resolve().open("r", encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    z_values = None
    entries: list[tuple[str, np.ndarray, dict[str, Any] | None]] = []
    reference_path = Path(summary["reference_path"])
    first_run = next(iter(summary["runs"].values()))
    config_z = first_run.get("z_values_um")
    if config_z is not None:
        z_values = np.asarray(config_z, dtype=np.float64)
    else:
        z_values = np.arange(10.0, 101.0, 10.0)
    true_depth_um = float(summary["true_depth_um"])
    depth_index = int(np.argmin(np.abs(z_values - true_depth_um)))
    entries.append(("VAR", _load_volume(reference_path)[depth_index], None))

    for label, run in summary["runs"].items():
        artifacts = run["artifacts"]
        if args.artifact_name is not None:
            artifact = artifacts[args.artifact_name]
        elif len(artifacts) == 1:
            artifact = next(iter(artifacts.values()))
        else:
            artifact = artifacts["best"]
        volume = _load_volume(Path(artifact["reconstruction_path"]))
        entries.append((label, volume[depth_index], artifact))

    columns = len(entries)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(3.2 * columns, 6.6),
        constrained_layout=True,
        squeeze=False,
    )
    for column, (label, image, artifact) in enumerate(entries):
        display = _display_normalize(image)
        if artifact is None:
            title = "VAR reference"
        else:
            resolution = artifact["resolution"]
            depth = artifact["depth"]
            title = (
                f"{label}\nfixed FTC {resolution['shared_center_reconstruction_resolution_um']:.2f} um"
                f" | Lap {resolution['laplacian_energy_ratio_vs_var']:.2f}x"
                f"\nshift {resolution['fitted_center_displacement_px_original']:.2f}px"
                f" | 40-60 mass {100.0 * depth['true_depth_plus_minus_10um_mass_fraction']:.1f}%"
            )
        axes[0, column].imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(title, fontsize=9)
        axes[0, column].axis("off")
        zoom = display[: args.zoom_size, : args.zoom_size]
        axes[1, column].imshow(zoom, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, column].set_title(f"upper-left {args.zoom_size}x{args.zoom_size}", fontsize=9)
        axes[1, column].axis("off")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
