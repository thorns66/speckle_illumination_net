from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def _unit_mass(volume: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(volume, dtype=np.float64), 0.0)
    total = float(value.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("reconstruction must have finite positive mass")
    return value / total


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_flat = left.ravel()
    right_flat = right.ravel()
    epsilon = np.finfo(np.float64).eps
    cosine = float(
        np.dot(left_flat, right_flat)
        / max(float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat)), epsilon)
    )
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    symmetric_relative_l2 = float(
        np.linalg.norm(left_flat - right_flat)
        / max(0.5 * (np.linalg.norm(left_flat) + np.linalg.norm(right_flat)), epsilon)
    )
    return {
        "cosine_similarity": cosine,
        "pearson_correlation": correlation,
        "symmetric_relative_l2": symmetric_relative_l2,
    }


def _save(path: Path, volume: np.ndarray) -> None:
    value = np.asarray(volume, dtype=np.float32)
    np.save(path.with_suffix(".npy"), value)
    tifffile.imwrite(path.with_suffix(".tif"), value, photometric="minisblack")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and ensemble fixed-depth frame-cross-fit reconstructions"
    )
    parser.add_argument(
        "--root", default="outputs/fixed_depth_lateral_50um_h2_crossfit5"
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    fold_dirs = sorted(path for path in root.glob("fold*_seed*") if path.is_dir())
    if len(fold_dirs) < 2:
        raise ValueError(f"Expected at least two fold directories under {root}")

    volumes: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []
    for directory in fold_dirs:
        with (directory / "summary.json").open("r", encoding="utf-8") as handle:
            training = json.load(handle)
        metrics_path = directory / "resolution_best" / "resolution_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        volume = _unit_mass(np.load(directory / "reconstruction_best.npy"))
        volumes.append(volume)
        artifacts = metrics["visual_artifact_quality"]
        folds.append(
            {
                "label": directory.name,
                "best_step": int(training["best_step"]),
                "initial_holdout_score": float(training["initial_holdout_score"]),
                "best_holdout_score": float(training["best_holdout_score"]),
                "holdout_improvement_fraction": float(
                    1.0
                    - training["best_holdout_score"] / training["initial_holdout_score"]
                ),
                "shared_center_ftc_um": float(
                    metrics["native_shared_reference_center"]["no_mean"]["robust"][
                        "resolution_um"
                    ]
                ),
                "laplacian_ratio_vs_var": float(
                    artifacts["reconstruction_relative_to_reference"][
                        "laplacian_energy_ratio"
                    ]
                ),
                "radial_barb_ratio_vs_var": float(
                    artifacts["reconstruction_relative_to_reference"][
                        "radial_barb_energy_ratio"
                    ]
                ),
                "annular_intensity_correlation_vs_var": float(
                    artifacts["annular_similarity_to_reference"]["intensity_correlation"]
                ),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left_index, right_index in itertools.combinations(range(len(volumes)), 2):
        pairwise.append(
            {
                "left": fold_dirs[left_index].name,
                "right": fold_dirs[right_index].name,
                **_pair_metrics(volumes[left_index], volumes[right_index]),
            }
        )

    stack = np.stack(volumes, axis=0)
    ensemble_mean = _unit_mass(stack.mean(axis=0))
    ensemble_median = _unit_mass(np.median(stack, axis=0))
    _save(root / "ensemble_mean", ensemble_mean)
    _save(root / "ensemble_median", ensemble_median)

    report = {
        "semantics": (
            "Frame-cross-fit stability diagnostic. Ensemble construction uses no reference "
            "image or resolution metric, but evaluation against VAR remains diagnostic-only."
        ),
        "fold_count": len(folds),
        "folds": folds,
        "pairwise": pairwise,
        "pairwise_summary": {
            key: {
                "minimum": float(min(row[key] for row in pairwise)),
                "median": float(np.median([row[key] for row in pairwise])),
                "maximum": float(max(row[key] for row in pairwise)),
            }
            for key in (
                "cosine_similarity",
                "pearson_correlation",
                "symmetric_relative_l2",
            )
        },
        "ensemble_mean_path": str((root / "ensemble_mean.npy").resolve()),
        "ensemble_median_path": str((root / "ensemble_median.npy").resolve()),
    }
    with (root / "crossfit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
