from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np


def _save_image(path: Path, image: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.imshow(image, cmap="gray")
    if title:
        axis.set_title(title)
    axis.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_volume_visualizations(
    output_dir: str | Path,
    volume_zxy: np.ndarray,
    z_values_um: Sequence[float],
) -> None:
    output_dir = Path(output_dir)
    volume = np.asarray(volume_zxy)
    _save_image(output_dir / "MIP_xy.png", volume.max(axis=0), "XY maximum projection")
    _save_image(output_dir / "MIP_xz.png", volume.max(axis=1), "XZ maximum projection")
    _save_image(output_dir / "MIP_yz.png", volume.max(axis=2), "YZ maximum projection")
    depth_dir = output_dir / "depth_layers"
    for index, (layer, z_um) in enumerate(zip(volume, z_values_um)):
        _save_image(depth_dir / f"depth_{index:02d}_{float(z_um):g}um.png", layer, f"z={z_um:g} um")


def save_gate_outputs(output_dir: str | Path, gates: Sequence[np.ndarray]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scale, gate in enumerate(gates):
        value = np.asarray(gate, dtype=np.float32)
        np.save(output_dir / f"gate_scale{scale}.npy", value)
        image = value.mean(axis=tuple(range(value.ndim - 2)))
        _save_image(output_dir / f"gate_scale{scale}_mean.png", image, f"Gate scale {scale}")


def save_loss_curve(path: str | Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        return
    steps = [row["step"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key in ("total_loss", "normalized_mean_loss", "normalized_var_loss"):
        axis.plot(steps, [row[key] for row in rows], label=key)
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Loss")
    axis.set_yscale("log")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
