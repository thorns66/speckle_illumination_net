from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _artifact(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return summary["runs"][label]["artifacts"]["best"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply geometry, cleanliness, depth and fixed-center resolution gates"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-label", default="E0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-center-shift-px", type=float, default=1.0)
    parser.add_argument("--minimum-resolution-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-cleanliness-improvement", type=float, default=0.10)
    parser.add_argument("--minimum-phase-delta-improvement", type=float, default=0.05)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if args.baseline_label not in summary["runs"]:
        raise KeyError(f"Missing baseline label {args.baseline_label!r}")
    baseline = _artifact(summary, args.baseline_label)
    baseline_depth = baseline["depth"]
    baseline_resolution = baseline["resolution"]
    true_depth_um = float(summary["true_depth_um"])
    minimum_near_mass = max(
        0.80,
        float(baseline_depth["true_depth_plus_minus_10um_mass_fraction"]) - 0.02,
    )
    maximum_centroid_error = max(
        2.0,
        float(baseline_depth["centroid_error_um"]) + 0.5,
    )
    maximum_shared_center_resolution = (
        float(baseline_resolution["shared_center_reconstruction_resolution_um"])
        * (1.0 - args.minimum_resolution_improvement)
    )
    maximum_laplacian_ratio = min(
        1.25,
        float(baseline_resolution["laplacian_energy_ratio_vs_var"])
        * (1.0 - args.minimum_cleanliness_improvement),
    )
    minimum_phase_delta = (
        float(baseline_resolution["target_phase_coherence_delta_vs_var"])
        + args.minimum_phase_delta_improvement
    )
    maximum_off_harmonic_ratio = float(
        baseline_resolution["off_harmonic_energy_ratio_vs_var"]
    )

    rows: list[dict[str, Any]] = []
    for label, run in summary["runs"].items():
        artifact = run["artifacts"]["best"]
        depth = artifact["depth"]
        resolution = artifact["resolution"]
        gates = {
            "pass_peak_depth": float(depth["peak_depth_um"]) == true_depth_um,
            "pass_near_depth_mass": float(
                depth["true_depth_plus_minus_10um_mass_fraction"]
            )
            >= minimum_near_mass,
            "pass_centroid": float(depth["centroid_error_um"])
            <= maximum_centroid_error,
            "pass_center_geometry": float(
                resolution["fitted_center_displacement_px_original"]
            )
            <= args.max_center_shift_px,
            "pass_fixed_center_resolution": float(
                resolution["shared_center_reconstruction_resolution_um"]
            )
            <= maximum_shared_center_resolution,
            "pass_laplacian_cleanliness": float(
                resolution["laplacian_energy_ratio_vs_var"]
            )
            <= maximum_laplacian_ratio,
            "pass_phase_coherence": float(
                resolution["target_phase_coherence_delta_vs_var"]
            )
            >= minimum_phase_delta,
            "pass_off_harmonic_energy": float(
                resolution["off_harmonic_energy_ratio_vs_var"]
            )
            <= maximum_off_harmonic_ratio,
        }
        passes_all = all(gates.values())
        rows.append(
            {
                "label": label,
                "shared_center_resolution_um": float(
                    resolution["shared_center_reconstruction_resolution_um"]
                ),
                "registered_resolution_um_diagnostic_only": float(
                    resolution["registered_reconstruction_resolution_um"]
                ),
                "center_shift_px_original": float(
                    resolution["fitted_center_displacement_px_original"]
                ),
                "laplacian_energy_ratio_vs_var": float(
                    resolution["laplacian_energy_ratio_vs_var"]
                ),
                "phase_coherence_delta_vs_var": float(
                    resolution["target_phase_coherence_delta_vs_var"]
                ),
                "off_harmonic_energy_ratio_vs_var": float(
                    resolution["off_harmonic_energy_ratio_vs_var"]
                ),
                "annular_gradient_cosine_vs_var": float(
                    resolution["annular_gradient_cosine_vs_var"]
                ),
                "near_depth_mass_fraction": float(
                    depth["true_depth_plus_minus_10um_mass_fraction"]
                ),
                "centroid_error_um": float(depth["centroid_error_um"]),
                **gates,
                "passes_all": passes_all,
            }
        )

    eligible = [row for row in rows if row["passes_all"] and row["label"] != args.baseline_label]
    eligible.sort(
        key=lambda row: (
            row["shared_center_resolution_um"],
            row["laplacian_energy_ratio_vs_var"],
            -row["phase_coherence_delta_vs_var"],
        )
    )
    selected = eligible[0]["label"] if eligible else args.baseline_label
    thresholds = {
        "true_peak_depth_um": true_depth_um,
        "minimum_near_depth_mass_fraction": minimum_near_mass,
        "maximum_centroid_error_um": maximum_centroid_error,
        "maximum_center_shift_px_original": args.max_center_shift_px,
        "maximum_shared_center_resolution_um": maximum_shared_center_resolution,
        "maximum_laplacian_energy_ratio_vs_var": maximum_laplacian_ratio,
        "minimum_phase_coherence_delta_vs_var": minimum_phase_delta,
        "maximum_off_harmonic_energy_ratio_vs_var": maximum_off_harmonic_ratio,
    }
    report = {
        "source_summary": str(summary_path),
        "baseline_label": args.baseline_label,
        "selected_label": selected,
        "selection_result": (
            "candidate_passed_all_hard_gates"
            if eligible
            else "no_candidate_passed_keep_baseline"
        ),
        "registered_ftc_is_diagnostic_only": True,
        "thresholds": thresholds,
        "rows": rows,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "artifact_resolution_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output_dir / "artifact_resolution_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
