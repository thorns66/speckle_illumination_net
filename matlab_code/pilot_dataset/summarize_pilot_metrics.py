#!/usr/bin/env python3
"""Summarize axial metrics for every frozen MATLAB pilot subset.

MATLAB Y-X-Z arrays stored in v7.3 MAT files appear as Z-X-Y through h5py.
Taylor raw volumes estimate g^2 and are square-rooted exactly once before
mass metrics, matching plot_pilot_results.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


METHODS = (
    ("legacy_mean_raw", "legacy_mean"),
    ("legacy_taylor_raw", "legacy_taylor_sqrt"),
    ("physics_mean_raw", "retained_scale_mean"),
    ("physics_taylor_raw", "retained_scale_taylor_sqrt"),
)


def scalar(dataset: h5py.Dataset) -> float:
    return float(np.asarray(dataset).reshape(-1)[0])


def summarize(root: Path) -> tuple[list[dict[str, float | int | str]], dict]:
    rows: list[dict[str, float | int | str]] = []
    for sample_number in range(1, 6):
        sample_id = f"P{sample_number:02d}"
        sample_dir = root / sample_id
        with h5py.File(sample_dir / "prepared.mat", "r") as prepared:
            z_um = np.asarray(prepared["cfg/z_um"], dtype=np.float64).reshape(-1)
            truth_index = int(scalar(prepared["truth_index_one_based"])) - 1
        if z_um.shape != (10,) or not 0 <= truth_index < 10:
            raise ValueError(f"Invalid truth metadata for {sample_id}")
        truth_depth = float(z_um[truth_index])

        for subset_index in range(1, 11):
            path = sample_dir / "subsets" / f"subset_{subset_index:02d}.mat"
            with h5py.File(path, "r") as subset:
                subset_z = np.asarray(subset["z_um"], dtype=np.float64).reshape(-1)
                if not np.array_equal(z_um, subset_z):
                    raise ValueError(f"Depth mismatch in {path}")
                for key, method in METHODS:
                    volume = np.asarray(subset[key], dtype=np.float64)
                    if volume.shape != (10, 260, 260) or not np.all(np.isfinite(volume)):
                        raise ValueError(f"Invalid {key} in {path}")
                    volume = np.maximum(volume, 0.0)
                    if "taylor" in key:
                        volume = np.sqrt(volume)
                    mass = volume.sum(axis=(1, 2), dtype=np.float64)
                    total = float(mass.sum())
                    if not total > 0:
                        raise ValueError(f"Zero mass for {key} in {path}")
                    fraction = mass / total
                    peak_depth = float(z_um[int(np.argmax(fraction))])
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "subset_index": subset_index,
                            "method": method,
                            "truth_depth_um": truth_depth,
                            "peak_depth_um": peak_depth,
                            "centroid_depth_um": float(np.dot(fraction, z_um)),
                            "true_layer_mass_fraction": float(fraction[truth_index]),
                            "peak_hits_truth": int(peak_depth == truth_depth),
                        }
                    )

    if len(rows) != 5 * 10 * len(METHODS):
        raise AssertionError(f"Expected 200 metric rows, got {len(rows)}")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["sample_id"]), str(row["method"]))].append(row)
    aggregate: list[dict[str, float | int | str]] = []
    for (sample_id, method), values in sorted(groups.items()):
        centroid = np.asarray([v["centroid_depth_um"] for v in values], dtype=np.float64)
        truth_mass = np.asarray([v["true_layer_mass_fraction"] for v in values], dtype=np.float64)
        peaks = [float(v["peak_depth_um"]) for v in values]
        peak_counts = Counter(peaks)
        mode_peak, mode_count = sorted(peak_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        aggregate.append(
            {
                "sample_id": sample_id,
                "method": method,
                "truth_depth_um": float(values[0]["truth_depth_um"]),
                "peak_mode_depth_um": mode_peak,
                "peak_mode_count": mode_count,
                "peak_truth_hit_rate": float(np.mean([v["peak_hits_truth"] for v in values])),
                "centroid_mean_um": float(centroid.mean()),
                "centroid_std_um": float(centroid.std(ddof=1)),
                "true_layer_mass_mean": float(truth_mass.mean()),
                "true_layer_mass_std": float(truth_mass.std(ddof=1)),
                "true_layer_mass_min": float(truth_mass.min()),
                "true_layer_mass_max": float(truth_mass.max()),
            }
        )
    return rows, {"row_count": len(rows), "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    rows, summary = summarize(args.dataset_root)
    csv_path = args.dataset_root / "pilot5_all_subset_axial_metrics.csv"
    json_path = args.dataset_root / "pilot5_axial_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(csv_path)
    print(json_path)
    print(f"rows={len(rows)} groups={len(summary['aggregate'])}")


if __name__ == "__main__":
    main()
