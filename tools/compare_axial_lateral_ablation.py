from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_volume(path: Path) -> np.ndarray:
    value = np.load(path) if path.suffix.lower() == ".npy" else tifffile.imread(path)
    value = np.asarray(value, dtype=np.float64)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"Expected [Z,H,W] at {path}, got {value.shape}")
    return np.clip(value, 0.0, None)


def _max_voxel_fraction(path: Path) -> float:
    volume = _load_volume(path)
    return float(volume.max() / max(float(volume.sum()), np.finfo(np.float64).tiny))


def _config_parameters(config_path: Path) -> tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]
    return float(model["lateral_log_residual_bound"]), float(model["axial_logit_scale"])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the pre-registered E0 hard gates to H0-H5"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--baseline-summary",
        default="outputs/artifact_resolution_50um/legacy_summary_step0200/ablation_summary.json",
    )
    parser.add_argument("--baseline-label", default="E0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixed-ftc-max-um", type=float, default=11.123)
    parser.add_argument("--laplacian-ratio-max", type=float, default=1.184)
    parser.add_argument("--radial-barb-ratio-max", type=float, default=1.745)
    parser.add_argument("--phase-coherence-min", type=float, default=0.5262)
    parser.add_argument("--center-shift-max-px", type=float, default=6.2009)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    summary = _load_json(summary_path)
    baseline_summary = _load_json(Path(args.baseline_summary).expanduser().resolve())
    baseline_artifact = baseline_summary["runs"][args.baseline_label]["artifacts"]
    baseline_artifact = next(iter(baseline_artifact.values()))
    baseline_max_voxel = _max_voxel_fraction(
        Path(baseline_artifact["reconstruction_path"])
    )
    max_voxel_limit = 1.1 * baseline_max_voxel

    rows: list[dict[str, Any]] = []
    for label, run in summary["runs"].items():
        bound, scale = _config_parameters(Path(run["config_path"]))
        for artifact_name, artifact in run["artifacts"].items():
            depth = artifact["depth"]
            resolution = artifact["resolution"]
            resolution_valid = bool(
                resolution.get("true_layer_resolution_valid", True)
            )
            max_voxel = float(
                depth.get(
                    "max_voxel_mass_fraction",
                    _max_voxel_fraction(Path(artifact["reconstruction_path"])),
                )
            )
            depth_gates = {
                "pass_peak_50um": float(depth["peak_depth_um"]) == 50.0,
                "pass_mass_40_60": float(
                    depth["true_depth_plus_minus_10um_mass_fraction"]
                )
                >= 0.80,
                "pass_centroid": float(depth["centroid_error_um"]) <= 5.0,
                "pass_first_layer": float(depth["first_layer_mass_fraction"]) < 0.05,
                "pass_last_layer": float(depth["last_layer_mass_fraction"]) < 0.05,
            }
            lateral_gates = {
                "pass_fixed_ftc": resolution_valid and float(
                    resolution["shared_center_reconstruction_resolution_um"]
                )
                <= args.fixed_ftc_max_um,
                "pass_laplacian": resolution_valid and float(
                    resolution["laplacian_energy_ratio_vs_var"]
                )
                <= args.laplacian_ratio_max,
                "pass_radial_barb": resolution_valid and float(
                    resolution["radial_barb_energy_ratio_vs_var"]
                )
                <= args.radial_barb_ratio_max,
                "pass_phase_coherence": resolution_valid and float(
                    resolution["target_phase_coherence_reconstruction"]
                )
                >= args.phase_coherence_min,
                "pass_center_shift": resolution_valid and float(
                    resolution["fitted_center_displacement_px_original"]
                )
                <= args.center_shift_max_px,
                "pass_max_voxel": max_voxel <= max_voxel_limit,
            }
            rows.append(
                {
                    "label": label,
                    "artifact": artifact_name,
                    "lateral_log_residual_bound": bound,
                    "axial_logit_scale": scale,
                    "peak_depth_um": float(depth["peak_depth_um"]),
                    "mass_40_60_fraction": float(
                        depth["true_depth_plus_minus_10um_mass_fraction"]
                    ),
                    "centroid_error_um": float(depth["centroid_error_um"]),
                    "first_layer_mass_fraction": float(
                        depth["first_layer_mass_fraction"]
                    ),
                    "last_layer_mass_fraction": float(depth["last_layer_mass_fraction"]),
                    "fixed_center_ftc_um": _optional_float(
                        resolution["shared_center_reconstruction_resolution_um"]
                    ),
                    "registered_ftc_um_diagnostic_only": _optional_float(
                        resolution["registered_reconstruction_resolution_um"]
                    ),
                    "laplacian_energy_ratio_vs_var": _optional_float(
                        resolution["laplacian_energy_ratio_vs_var"]
                    ),
                    "radial_barb_energy_ratio_vs_var": _optional_float(
                        resolution["radial_barb_energy_ratio_vs_var"]
                    ),
                    "target_phase_coherence": _optional_float(
                        resolution["target_phase_coherence_reconstruction"]
                    ),
                    "center_shift_px_original": _optional_float(
                        resolution["fitted_center_displacement_px_original"]
                    ),
                    "max_voxel_mass_fraction": max_voxel,
                    "true_layer_resolution_valid": resolution_valid,
                    **depth_gates,
                    **lateral_gates,
                    "passes_depth": all(depth_gates.values()),
                    "passes_all": all(depth_gates.values()) and all(lateral_gates.values()),
                }
            )

    best_rows = [row for row in rows if row["artifact"] == "best"]
    depth_passers = sorted(
        (row for row in best_rows if row["passes_depth"]),
        key=lambda row: (row["lateral_log_residual_bound"], row["fixed_center_ftc_um"]),
    )
    full_passers = sorted(
        (row for row in best_rows if row["passes_all"]),
        key=lambda row: (
            row["fixed_center_ftc_um"],
            row["radial_barb_energy_ratio_vs_var"],
            row["laplacian_energy_ratio_vs_var"],
        ),
    )
    thresholds = {
        "peak_depth_um": 50.0,
        "minimum_mass_40_60_fraction": 0.80,
        "maximum_centroid_error_um": 5.0,
        "maximum_first_last_layer_fraction_each": 0.05,
        "maximum_fixed_center_ftc_um": args.fixed_ftc_max_um,
        "maximum_laplacian_energy_ratio_vs_var": args.laplacian_ratio_max,
        "maximum_radial_barb_energy_ratio_vs_var": args.radial_barb_ratio_max,
        "minimum_target_phase_coherence": args.phase_coherence_min,
        "maximum_center_shift_px_original": args.center_shift_max_px,
        "e0_max_voxel_mass_fraction": baseline_max_voxel,
        "maximum_voxel_mass_fraction": max_voxel_limit,
    }
    report = {
        "source_summary": str(summary_path),
        "baseline_summary": str(Path(args.baseline_summary).expanduser().resolve()),
        "baseline_label": args.baseline_label,
        "thresholds": thresholds,
        "checkpoint_selection": "best is selected only by training total loss; final is reported separately",
        "registered_ftc_is_diagnostic_only": True,
        "minimum_bound_depth_passing_candidate": (
            depth_passers[0]["label"] if depth_passers else None
        ),
        "selected_full_gate_candidate": full_passers[0]["label"] if full_passers else None,
        "rows": rows,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "axial_lateral_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output_dir / "axial_lateral_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
