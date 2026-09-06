#!/usr/bin/env python3
"""Read-only per-depth mass audit of the selected MATLAB H/Ht tensors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def summarize(array: np.ndarray) -> dict[str, float]:
    phase_mass = array.astype(np.float64).sum(axis=(-2, -1))
    return {
        "phase_mass_mean": float(phase_mass.mean()),
        "phase_mass_min": float(phase_mass.min()),
        "phase_mass_max": float(phase_mass.max()),
        "phase_mass_cv": float(phase_mass.std() / phase_mass.mean()),
        "negative_fraction": float(np.mean(array < 0)),
        "zero_fraction": float(np.mean(array == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    psf, output = args.psf.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with h5py.File(psf) as handle:
        source_z = np.asarray(handle["x3objspace"], dtype=float).reshape(-1)
        # The frozen reconstruction protocol selected source indices 1:10,
        # which map to 10:10:100 um in pilot_load_psf.m.
        for depth_index, z_um in enumerate(range(10, 101, 10)):
            h = np.asarray(handle["H"][depth_index], dtype=np.float32)
            ht = np.asarray(handle["Ht"][depth_index], dtype=np.float32)
            for operator, array in (("H", h), ("Ht", ht), ("H_squared", h**2), ("Ht_squared", ht**2)):
                rows.append(
                    {
                        "depth_index_zero_based": depth_index,
                        "reconstruction_z_um": z_um,
                        "source_x3objspace_value": float(source_z[depth_index]),
                        "operator": operator,
                        **summarize(array),
                    }
                )
    with (output / "psf_depth_mass.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for operator in ("H", "Ht", "H_squared", "Ht_squared"):
        selected = [row for row in rows if row["operator"] == operator]
        means = np.asarray([row["phase_mass_mean"] for row in selected])
        aggregate[operator] = {
            "depth_mass_values": means.tolist(),
            "depth_mass_cv": float(means.std() / means.mean()),
            "depth_mass_max_to_min_ratio": float(means.max() / means.min()),
            "first_to_last_ratio": float(means[0] / means[-1]),
            "strictly_decreasing_with_depth": bool(np.all(np.diff(means) < 0)),
        }
    result = {
        "complete": True,
        "source": str(psf),
        "selected_source_indices_one_based": list(range(1, 11)),
        "reconstruction_z_um": list(range(10, 101, 10)),
        "rows": rows,
        "aggregate": aggregate,
        "interpretation": "A global scalar mass is irrelevant to shape after gain alignment; phase/depth variation is not and can bias unnormalized iterative updates.",
    }
    (output / "psf_depth_mass.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
