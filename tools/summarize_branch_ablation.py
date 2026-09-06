from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent.parent / path).resolve()
    return path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        volume = np.asarray(np.load(path), dtype=np.float64)
    else:
        volume = np.asarray(tifffile.imread(path), dtype=np.float64)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,H,W] volume at {path}, got {volume.shape}")
    return np.clip(volume, 0.0, None)


def _depth_metrics(volume: np.ndarray, z_values_um: np.ndarray, true_depth_um: float) -> dict[str, Any]:
    layer_mass = volume.sum(axis=(-2, -1), dtype=np.float64)
    total = float(layer_mass.sum())
    if total <= 0:
        raise ValueError("Reconstruction has no positive mass")
    fraction = layer_mass / total
    true_index = int(np.argmin(np.abs(z_values_um - true_depth_um)))
    near = np.abs(z_values_um - true_depth_um) <= 10.0 + 1e-6
    return {
        "layer_mass_fraction": fraction.tolist(),
        "peak_depth_um": float(z_values_um[int(np.argmax(fraction))]),
        "centroid_depth_um": float(np.sum(z_values_um * fraction)),
        "centroid_error_um": float(abs(np.sum(z_values_um * fraction) - true_depth_um)),
        "true_layer_mass_fraction": float(fraction[true_index]),
        "true_depth_plus_minus_10um_mass_fraction": float(fraction[near].sum()),
        "first_layer_mass_fraction": float(fraction[0]),
        "last_layer_mass_fraction": float(fraction[-1]),
        "max_voxel_mass_fraction": float(volume.max() / total),
    }


def _loss_metrics(output_dir: Path, through_step: int | None = None) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
    else:
        saved = {}
    with (output_dir / "losses.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if through_step is not None:
        rows = [row for row in rows if int(float(row["step"])) <= through_step]
    if not rows:
        raise ValueError(f"No loss rows found at or before step {through_step}")
    total = np.asarray([float(row["total_loss"]) for row in rows], dtype=np.float64)
    step_time = np.asarray([float(row["step_time_s"]) for row in rows], dtype=np.float64)
    gpu_memory = np.asarray([float(row["gpu_memory_mb"]) for row in rows], dtype=np.float64)
    tail = total[-min(20, len(total)) :]
    best_index = int(np.argmin(total))
    best_step = int(float(rows[best_index]["step"]))
    best_total_loss = float(total[best_index])
    if through_step is None and saved:
        best_step = int(saved["best_step"])
        best_total_loss = float(saved["best_total_loss"])
    return {
        "steps_completed": len(rows),
        "last_included_step": int(float(rows[-1]["step"])),
        "best_step": best_step,
        "best_total_loss": best_total_loss,
        "final_total_loss": float(total[-1]),
        "last20_median_total_loss": float(np.median(tail)),
        "median_step_time_s": float(np.median(step_time)),
        "peak_gpu_memory_mb": float(gpu_memory.max()),
        "registered_parameter_count": saved.get("registered_parameter_count"),
        "active_parameter_count": saved.get("active_parameter_count"),
    }


def _run_resolution(
    config_path: Path,
    reconstruction_path: Path,
    reference_path: Path,
    true_depth_um: float,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "resolution_metrics.json"
    latest_input_mtime = max(
        path.stat().st_mtime for path in (config_path, reconstruction_path, reference_path)
    )
    if report_path.is_file() and report_path.stat().st_mtime >= latest_input_mtime:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.evaluate_resolution",
        "--config",
        str(config_path),
        "--reconstruction",
        str(reconstruction_path),
        "--reference",
        str(reference_path),
        "--true-depth-um",
        str(true_depth_um),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    (output_dir / "evaluation_stdout.log").write_text(completed.stdout, encoding="utf-8")
    with (output_dir / "resolution_metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolution_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    registered = metrics["center_registered_radial_contrast"]
    shared = metrics["native_shared_reference_center"]
    sensitivity = metrics["center_sensitivity"]
    visual = metrics["visual_artifact_quality"]
    relative_artifacts = visual["reconstruction_relative_to_reference"]
    similarity = visual["annular_similarity_to_reference"]
    reference_key = next(key for key in registered if key not in {"no_mean", "comparison", "meaning"})
    return {
        "true_layer_resolution_valid": True,
        "true_layer_resolution_invalid_reason": None,
        "registered_reference_resolution_um": float(
            registered[reference_key]["robust"]["resolution_um"]
        ),
        "registered_reconstruction_resolution_um": float(
            registered["no_mean"]["robust"]["resolution_um"]
        ),
        "registered_change_vs_var_percent": float(
            registered["comparison"]["reconstruction_change_vs_reference_percent"]
        ),
        "shared_center_reconstruction_resolution_um": float(
            shared["no_mean"]["robust"]["resolution_um"]
        ),
        "fitted_center_displacement_um": float(
            metrics["center_calibration"]["center_displacement_um"]
        ),
        "fitted_center_displacement_px_original": float(
            metrics["center_calibration"]["center_displacement_px_original_grid"]
        ),
        "geometry_warning": bool(metrics["center_calibration"]["geometry_warning"]),
        "registered_center_sensitivity_min_um": float(
            sensitivity["no_mean_registered"]["minimum_resolution_um"]
        ),
        "registered_center_sensitivity_max_um": float(
            sensitivity["no_mean_registered"]["maximum_resolution_um"]
        ),
        "radial_barb_energy_ratio_vs_var": float(
            relative_artifacts["radial_barb_energy_ratio"]
        ),
        "off_harmonic_energy_ratio_vs_var": float(
            relative_artifacts["off_harmonic_energy_ratio"]
        ),
        "laplacian_energy_ratio_vs_var": float(
            relative_artifacts["laplacian_energy_ratio"]
        ),
        "target_phase_coherence_delta_vs_var": float(
            relative_artifacts["target_phase_coherence_delta"]
        ),
        "target_phase_coherence_reconstruction": float(
            visual["reconstruction"]["target_harmonic_phase_coherence"]
        ),
        "annular_intensity_correlation_vs_var": float(
            similarity["intensity_correlation"]
        ),
        "annular_gradient_cosine_vs_var": float(
            similarity["gradient_cosine_similarity"]
        ),
        "annular_affine_nrmse_vs_var": float(similarity["affine_normalized_rmse"]),
    }


def _invalid_resolution_fields(reason: str) -> dict[str, Any]:
    """Return a schema-compatible result when the requested layer has no signal."""
    names = (
        "registered_reference_resolution_um",
        "registered_reconstruction_resolution_um",
        "registered_change_vs_var_percent",
        "shared_center_reconstruction_resolution_um",
        "fitted_center_displacement_um",
        "fitted_center_displacement_px_original",
        "registered_center_sensitivity_min_um",
        "registered_center_sensitivity_max_um",
        "radial_barb_energy_ratio_vs_var",
        "off_harmonic_energy_ratio_vs_var",
        "laplacian_energy_ratio_vs_var",
        "target_phase_coherence_delta_vs_var",
        "target_phase_coherence_reconstruction",
        "annular_intensity_correlation_vs_var",
        "annular_gradient_cosine_vs_var",
        "annular_affine_nrmse_vs_var",
    )
    result: dict[str, Any] = {name: None for name in names}
    result.update(
        {
            "true_layer_resolution_valid": False,
            "true_layer_resolution_invalid_reason": reason,
            "geometry_warning": True,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and summarize branch ablations")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=CONFIG_PATH; repeat once per run",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--true-depth-um", type=float, default=50.0)
    parser.add_argument("--output-dir", default="outputs/ablation_50um/summary_seed20260901")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument(
        "--artifact-step",
        type=int,
        default=None,
        help="Evaluate only reconstruction_step_NNNN.tif and truncate loss statistics there",
    )
    args = parser.parse_args()
    if args.artifact_step is not None and args.include_final:
        parser.error("--artifact-step and --include-final are mutually exclusive")

    root = Path.cwd().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    summary_dir = Path(args.output_dir).expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []

    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be LABEL=CONFIG_PATH, got {item!r}")
        label, raw_config_path = item.split("=", 1)
        config_path = Path(raw_config_path).expanduser().resolve()
        config = _load_config(config_path)
        output_dir = _resolve(config_path, config["experiment"]["output_dir"])
        z_values = np.asarray(config["psf"]["z_values_um"], dtype=np.float64)
        if args.artifact_step is not None:
            artifact_name = f"step_{args.artifact_step:04d}"
            artifacts = {
                artifact_name: output_dir / f"reconstruction_step_{args.artifact_step:04d}.tif"
            }
        else:
            artifacts = {"best": output_dir / "reconstruction_best.npy"}
            if args.include_final:
                final_step = int(config["optimization"]["max_steps"]) - 1
                artifacts["final"] = output_dir / f"reconstruction_step_{final_step:04d}.tif"
        run_result: dict[str, Any] = {
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "loss": _loss_metrics(output_dir, through_step=args.artifact_step),
            "artifacts": {},
        }
        for artifact_name, reconstruction_path in artifacts.items():
            if not reconstruction_path.is_file():
                raise FileNotFoundError(reconstruction_path)
            evaluation_dir = output_dir / (
                f"resolution_evaluation_{args.true_depth_um:g}um_raw_var_{artifact_name}"
            )
            volume = _load_volume(reconstruction_path)
            depth = _depth_metrics(volume, z_values, args.true_depth_um)
            true_index = int(np.argmin(np.abs(z_values - args.true_depth_um)))
            true_layer_peak = float(volume[true_index].max())
            if true_layer_peak > 0.0:
                resolution = _resolution_fields(
                    _run_resolution(
                        config_path,
                        reconstruction_path,
                        reference_path,
                        args.true_depth_um,
                        evaluation_dir,
                    )
                )
                resolution_report: str | None = str(
                    evaluation_dir / "resolution_metrics.json"
                )
            else:
                resolution = _invalid_resolution_fields(
                    "The reconstruction has zero signal at the requested true-depth layer."
                )
                resolution_report = None
            artifact_result = {
                "reconstruction_path": str(reconstruction_path),
                "depth": depth,
                "resolution": resolution,
                "resolution_report": resolution_report,
            }
            run_result["artifacts"][artifact_name] = artifact_result
            flat_rows.append(
                {
                    "label": label,
                    "artifact": artifact_name,
                    **run_result["loss"],
                    **artifact_result["depth"],
                    **artifact_result["resolution"],
                }
            )
        results[label] = run_result

    report = {
        "true_depth_um": args.true_depth_um,
        "reference_path": str(reference_path),
        "selection_rule": (
            f"fixed reconstruction checkpoint at step {args.artifact_step}"
            if args.artifact_step is not None
            else "best artifact is selected only by training total loss"
        ),
        "runs": results,
    }
    with (summary_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    scalar_rows = [
        {key: value for key, value in row.items() if key != "layer_mass_fraction"}
        for row in flat_rows
    ]
    with (summary_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
