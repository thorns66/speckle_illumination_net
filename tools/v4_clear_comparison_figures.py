"""Generate font-safe, globally normalized V4 comparison figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import three_way_experiment as old
from tools import v4_p12_anchor_analysis as analysis
from tools import v4_p12_anchor_compare_experiment as exp


LABELS = {
    "ground_truth": "Ground truth",
    "mean_rl3": "Mean-RL3",
    "taylor_rl3_sqrt": "Taylor-RL3-sqrt",
    "taylor_anchor_e3_mean100": "Taylor-anchor net (P12)",
    "mean_anchor_e3_mean100": "Mean-anchor net (P12)",
}
METHODS = ("ground_truth", "mean_rl3", "taylor_rl3_sqrt", *exp.ARMS)
PERCENTILE = 99.9


def scale(volume: np.ndarray) -> float:
    positive = np.maximum(np.asarray(volume, np.float32), 0)
    return max(float(np.percentile(positive, PERCENTILE)), 1e-30)


def show(ax, image: np.ndarray, maximum: float, title: str, *, aspect: str = "equal") -> None:
    ax.imshow(
        np.clip(image / maximum, 0, 1), cmap="magma", vmin=0, vmax=1,
        origin="lower", interpolation="nearest", aspect=aspect,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    output = exp.OUTPUT / "analysis/figures_clear"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in ("T02", "T03", "T04"):
        volumes, truth = analysis.case_volumes(sample, 1)
        values = {"ground_truth": truth, **volumes}
        own_scales = {method: scale(values[method]) for method in METHODS}

        figure, axes = plt.subplots(3, 5, figsize=(17, 9), layout="constrained")
        for column, method in enumerate(METHODS):
            volume = values[method]
            views = (volume.max(0), volume.max(1), volume.max(2))
            for row, view in enumerate(views):
                show(
                    axes[row, column], view, own_scales[method],
                    LABELS[method] + (" | XY MIP" if row == 0 else " | XZ MIP" if row == 1 else " | YZ MIP"),
                    aspect="auto" if row else "equal",
                )
        figure.suptitle(
            f"{sample} subset 01 | one {PERCENTILE:g}th-percentile scale per complete 3D volume"
        )
        path = output / f"{sample}_subset01_all_methods_volume_normalized.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        manifest.append({"path": str(path.relative_to(exp.OUTPUT)), "sample": sample, "display": "per-volume"})

        network_scale = max(scale(volumes[arm]) for arm in exp.ARMS)
        figure, axes = plt.subplots(3, 2, figsize=(9, 10), layout="constrained")
        for column, arm in enumerate(exp.ARMS):
            volume = volumes[arm]
            for row, view in enumerate((volume.max(0), volume.max(1), volume.max(2))):
                show(
                    axes[row, column], view, network_scale,
                    LABELS[arm] + (" | XY" if row == 0 else " | XZ" if row == 1 else " | YZ"),
                    aspect="auto" if row else "equal",
                )
        figure.suptitle(
            f"{sample} subset 01 | both networks share one {PERCENTILE:g}th-percentile scale"
        )
        path = output / f"{sample}_subset01_network_shared_scale.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
        manifest.append({"path": str(path.relative_to(exp.OUTPUT)), "sample": sample, "display": "shared-network"})

        figure, axes = plt.subplots(5, 10, figsize=(23, 11), layout="constrained")
        for row, method in enumerate(METHODS):
            volume = values[method]
            for depth in range(10):
                show(
                    axes[row, depth], volume[depth], own_scales[method],
                    f"{LABELS[method]} | {(depth + 1) * 10} um",
                )
        figure.suptitle(
            f"{sample} subset 01 | native depth planes | no per-plane normalization"
        )
        path = output / f"{sample}_subset01_native_layers_volume_normalized.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        manifest.append({"path": str(path.relative_to(exp.OUTPUT)), "sample": sample, "display": "native-layers"})

    record = {
        "complete": True, "figures": len(manifest), "percentile": PERCENTILE,
        "normalization": "one scale per complete 3D volume; shared-network figures use one scale for both networks; no per-layer normalization",
        "manifest": manifest,
    }
    old.write_json(output / "manifest.json", record)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
