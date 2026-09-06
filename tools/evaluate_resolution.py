from __future__ import annotations

import argparse
import csv
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
import yaml

from utils.artifact_quality import (
    compare_annular_structure,
    compute_visual_artifact_metrics,
)
from utils.io import load_volume_tiff
from utils.resolution import FTCCurve, compute_ftc_curve, prepare_resolution_image, sample_quarter_circle
from utils.star_center import CenterCalibration, calibrate_star_center


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent.parent / path).resolve()
    return path


def _load_volume(path: Path, depths: int, intensity_mode: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        volume = np.asarray(np.load(path), dtype=np.float32)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        volume = load_volume_tiff(path, depths, intensity_mode=intensity_mode)
    else:
        raise ValueError(f"Unsupported volume format: {path.suffix}")
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3 or volume.shape[0] != depths:
        raise ValueError(f"Expected volume [Z,H,W] with Z={depths}, got {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Volume contains NaN or Inf: {path}")
    return volume


def _select_depth(z_values: np.ndarray, target_um: float) -> int:
    matches = np.flatnonzero(np.isclose(z_values, target_um, rtol=0.0, atol=1e-5))
    if len(matches) != 1:
        raise ValueError(
            f"true depth {target_um:g} um must exactly match one configured layer; "
            f"available={z_values.tolist()}"
        )
    return int(matches[0])


def _cutoff_dict(curve: FTCCurve) -> dict[str, Any]:
    return {
        "matlab": {
            "radius_px": curve.matlab_cutoff.radius_px,
            "resolution_um": curve.matlab_cutoff.resolution_um,
            "status": curve.matlab_cutoff.status,
        },
        "robust": {
            "radius_px": curve.robust_cutoff.radius_px,
            "resolution_um": curve.robust_cutoff.resolution_um,
            "status": curve.robust_cutoff.status,
        },
        "tested_radius_px": [float(curve.radii_px[0]), float(curve.radii_px[-1])],
        "tested_resolution_um": [
            float(curve.resolution_um[0]),
            float(curve.resolution_um[-1]),
        ],
        "minimum_valid_arc_fraction": float(curve.valid_fraction.min()),
    }


def _calibration_dict(calibration: CenterCalibration) -> dict[str, Any]:
    return {
        "center_xy_evaluation_1based": list(calibration.center_xy_1based),
        "score": calibration.score,
        "median_target_ftc": calibration.median_ftc,
        "target_phase_coherence": calibration.phase_coherence,
        "target_harmonic_purity": calibration.harmonic_purity,
        "calibration_radii_px": calibration.calibration_radii_px.tolist(),
        "search_bounds_xy_evaluation_1based": list(calibration.search_bounds_xy_1based),
    }


def _comparison_dict(reference: FTCCurve, reconstruction: FTCCurve) -> dict[str, Any]:
    reference_um = reference.robust_cutoff.resolution_um
    reconstruction_um = reconstruction.robust_cutoff.resolution_um
    return {
        "reference_resolution_um": reference_um,
        "reconstruction_resolution_um": reconstruction_um,
        "reconstruction_change_vs_reference_percent": (
            (reconstruction_um - reference_um) / reference_um * 100.0
        ),
        "reference_finer_than_reconstruction_percent": (
            (reconstruction_um - reference_um) / reconstruction_um * 100.0
        ),
        "interpretation": (
            "Smaller is better; a positive reconstruction_change_vs_reference_percent "
            "means the reconstruction is coarser than the reference."
        ),
    }


def _center_sensitivity(
    image: np.ndarray,
    center_xy_1based: tuple[float, float],
    curve_options: dict[str, Any],
    perturbation_px: int,
) -> dict[str, Any]:
    values: list[float] = []
    for dy in range(-perturbation_px, perturbation_px + 1):
        for dx in range(-perturbation_px, perturbation_px + 1):
            options = dict(curve_options)
            options["center_xy_1based"] = (
                center_xy_1based[0] + dx,
                center_xy_1based[1] + dy,
            )
            values.append(compute_ftc_curve(image, **options).robust_cutoff.resolution_um)
    array = np.asarray(values, dtype=np.float64)
    return {
        "perturbation_px_evaluation_grid": int(perturbation_px),
        "samples": int(len(array)),
        "median_resolution_um": float(np.median(array)),
        "minimum_resolution_um": float(array.min()),
        "maximum_resolution_um": float(array.max()),
        "range_resolution_um": float(array.max() - array.min()),
    }


def _save_curve_csv(path: Path, reference: FTCCurve, reconstruction: FTCCurve) -> None:
    count = min(len(reference.radii_px), len(reconstruction.radii_px))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "radius_px",
                "resolution_um",
                "reference_ftc",
                "reference_ftc_robust",
                "reconstruction_ftc",
                "reconstruction_ftc_robust",
                "valid_arc_fraction",
            ]
        )
        for index in range(count):
            writer.writerow(
                [
                    float(reference.radii_px[index]),
                    float(reference.resolution_um[index]),
                    float(reference.ftc[index]),
                    float(reference.robust_ftc[index]),
                    float(reconstruction.ftc[index]),
                    float(reconstruction.robust_ftc[index]),
                    float(min(reference.valid_fraction[index], reconstruction.valid_fraction[index])),
                ]
            )


def _plot_images(path: Path, images: dict[str, np.ndarray], true_depth_um: float) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, (label, image) in zip(axes, images.items()):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"{label}, z={true_depth_um:g} um")
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_curves(
    path: Path,
    curves: dict[str, FTCCurve],
    threshold: float,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = {next(iter(curves)): "tab:blue", "no_mean": "tab:orange"}
    for label, curve in curves.items():
        color = colors[label]
        axis.plot(curve.resolution_um, curve.ftc, color=color, alpha=0.25, linewidth=1)
        axis.plot(
            curve.resolution_um,
            curve.robust_ftc,
            color=color,
            linewidth=2,
            label=f"{label} ({curve.robust_cutoff.resolution_um:.3f} um)",
        )
        axis.axvline(curve.robust_cutoff.resolution_um, color=color, linestyle=":", alpha=0.8)
    axis.axhline(threshold, color="black", linestyle="--", label=f"FTC={threshold:g}")
    axis.set_title(title)
    axis.set_xlabel("Resolution / line-pair period (um)")
    axis.set_ylabel("Fourier contrast")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_overlays(
    path: Path,
    images: dict[str, np.ndarray],
    curves: dict[str, FTCCurve],
    centers: dict[str, tuple[float, float]],
    angular_samples: int,
    title_suffix: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, (label, image) in zip(axes, images.items()):
        curve = curves[label]
        center = centers[label]
        _, x, y, _ = sample_quarter_circle(
            image,
            curve.robust_cutoff.radius_px,
            center_xy_1based=center,
            angular_samples=angular_samples,
        )
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.plot(x - 1, y - 1, color="red", linewidth=1.2)
        axis.scatter([center[0] - 1], [center[1] - 1], color="cyan", s=15)
        axis.set_title(
            f"{label}: r={curve.robust_cutoff.radius_px:g}px, "
            f"{curve.robust_cutoff.resolution_um:.3f} um"
        )
        axis.axis("off")
    figure.suptitle(title_suffix)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Siemens-star resolution at one known physical depth"
    )
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--reconstruction", default=None)
    parser.add_argument("--reference", default=None, help="Defaults to data.f_var_tiff_path")
    parser.add_argument("--true-depth-um", type=float, default=50.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--upsample-factor", type=float, default=2.0)
    parser.add_argument("--center-mode", choices=("reference_calibrated", "fixed"), default="reference_calibrated")
    parser.add_argument("--center-x-original", type=float, default=11.0)
    parser.add_argument("--center-y-original", type=float, default=11.0)
    parser.add_argument("--center-search-radius-px", type=int, default=16)
    parser.add_argument("--center-sensitivity-px", type=int, default=2)
    parser.add_argument("--pixel-size-um", type=float, default=5.2 / 8.93)
    parser.add_argument("--line-pairs", type=int, default=40)
    parser.add_argument("--harmonic", type=int, default=10)
    parser.add_argument("--angular-samples", type=int, default=1000)
    parser.add_argument("--max-radius-px", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--consecutive-below", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    z_values = np.asarray(config["psf"]["z_values_um"], dtype=np.float64)
    depth_index = _select_depth(z_values, args.true_depth_um)
    intensity_mode = config["data"].get("tiff_intensity_mode", "matlab_im2double")
    reference_path = _resolve(
        config_path,
        args.reference if args.reference is not None else config["data"]["f_var_tiff_path"],
    )
    configured_output = _resolve(config_path, config["experiment"]["output_dir"])
    reconstruction_path = (
        _resolve(config_path, args.reconstruction)
        if args.reconstruction is not None
        else configured_output / "reconstruction_best.npy"
    )
    output_dir = (
        _resolve(config_path, args.output_dir)
        if args.output_dir is not None
        else configured_output / f"resolution_evaluation_{args.true_depth_um:g}um"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_volume = _load_volume(reference_path, len(z_values), intensity_mode)
    reconstruction_volume = _load_volume(reconstruction_path, len(z_values), intensity_mode)
    if reference_volume.shape != reconstruction_volume.shape:
        raise ValueError(
            f"Reference/reconstruction shape mismatch: {reference_volume.shape} "
            f"vs {reconstruction_volume.shape}"
        )
    reference_label = "F_var" if args.reference is None else "VAR"
    images = {
        reference_label: prepare_resolution_image(reference_volume[depth_index], args.upsample_factor),
        "no_mean": prepare_resolution_image(reconstruction_volume[depth_index], args.upsample_factor),
    }
    initial_center = (
        args.center_x_original * args.upsample_factor,
        args.center_y_original * args.upsample_factor,
    )
    calibration_options = {
        "initial_center_xy_1based": initial_center,
        "search_radius_px": args.center_search_radius_px,
        "harmonic": args.harmonic,
        "angular_samples": args.angular_samples,
        "max_radius_px": args.max_radius_px,
    }
    if args.center_mode == "reference_calibrated":
        reference_calibration = calibrate_star_center(images[reference_label], **calibration_options)
        reconstruction_calibration = calibrate_star_center(images["no_mean"], **calibration_options)
    else:
        fixed_options = dict(calibration_options)
        fixed_options["search_radius_px"] = 0
        reference_calibration = calibrate_star_center(images[reference_label], **fixed_options)
        reconstruction_calibration = calibrate_star_center(images["no_mean"], **fixed_options)

    reference_center = reference_calibration.center_xy_1based
    reconstruction_center = reconstruction_calibration.center_xy_1based
    curve_options = {
        "pixel_size_um": args.pixel_size_um,
        "line_pairs": args.line_pairs,
        "harmonic": args.harmonic,
        "angular_samples": args.angular_samples,
        "max_radius_px": args.max_radius_px,
        "threshold": args.threshold,
        "smoothing_window": args.smoothing_window,
        "consecutive_below": args.consecutive_below,
    }
    shared_curves = {
        label: compute_ftc_curve(image, center_xy_1based=reference_center, **curve_options)
        for label, image in images.items()
    }
    registered_centers = {reference_label: reference_center, "no_mean": reconstruction_center}
    registered_curves = {
        label: compute_ftc_curve(
            image,
            center_xy_1based=registered_centers[label],
            **curve_options,
        )
        for label, image in images.items()
    }
    displacement = float(np.linalg.norm(np.subtract(reconstruction_center, reference_center)))
    reference_cutoff_radius = float(
        shared_curves[reference_label].robust_cutoff.radius_px
    )
    complete_radius = float(
        np.floor(
            min(
                images[reference_label].shape[1] - reference_center[0],
                images[reference_label].shape[0] - reference_center[1],
            )
        )
    )
    artifact_inner_radius = max(4.0, 0.25 * reference_cutoff_radius)
    artifact_outer_radius = min(
        complete_radius,
        max(artifact_inner_radius + 4.0, 1.5 * reference_cutoff_radius),
    )
    artifact_options = {
        "center_xy_1based": reference_center,
        "inner_radius_px": artifact_inner_radius,
        "outer_radius_px": artifact_outer_radius,
        "harmonic": args.harmonic,
        "angular_samples": args.angular_samples,
    }
    artifact_metrics = {
        label: compute_visual_artifact_metrics(image, **artifact_options)
        for label, image in images.items()
    }
    artifact_similarity = compare_annular_structure(
        images[reference_label],
        images["no_mean"],
        center_xy_1based=reference_center,
        inner_radius_px=artifact_inner_radius,
        outer_radius_px=artifact_outer_radius,
    )
    geometry_warning = displacement > 2.0
    summary: dict[str, Any] = {
        "true_depth_um": float(args.true_depth_um),
        "depth_index_zero_based": depth_index,
        "z_values_um": z_values.tolist(),
        "reference_path": str(reference_path),
        "reconstruction_path": str(reconstruction_path),
        "reference_semantics": (
            "Configured F_var anchor (sqrt-variance Taylor reconstruction)."
            if args.reference is None
            else "Explicit reference supplied by --reference."
        ),
        "parameters": {
            "upsample_factor": args.upsample_factor,
            "initial_center_xy_original_1based": [args.center_x_original, args.center_y_original],
            "initial_center_xy_evaluation_1based": list(initial_center),
            "center_mode": args.center_mode,
            "center_search_radius_px_evaluation_grid": args.center_search_radius_px,
            "pixel_size_um_on_evaluation_grid": args.pixel_size_um,
            "line_pairs": args.line_pairs,
            "fft_harmonic_zero_based": args.harmonic,
            "matlab_fft_index_one_based": args.harmonic + 1,
            "angular_samples": args.angular_samples,
            "requested_max_radius_px": args.max_radius_px,
            "threshold": args.threshold,
            "smoothing_window": args.smoothing_window,
            "consecutive_below": args.consecutive_below,
        },
        "center_calibration": {
            "reference": _calibration_dict(reference_calibration),
            "reconstruction_diagnostic_only": _calibration_dict(reconstruction_calibration),
            "center_displacement_px_evaluation_grid": displacement,
            "center_displacement_px_original_grid": displacement / args.upsample_factor,
            "center_displacement_um": displacement * args.pixel_size_um,
            "geometry_warning": geometry_warning,
            "note": (
                "The reference center is frozen for native-coordinate comparison. "
                "The reconstruction center is fitted only to separate radial contrast "
                "from center displacement; it is never used to tune the reference."
            ),
        },
        "native_shared_reference_center": {
            reference_label: _cutoff_dict(shared_curves[reference_label]),
            "no_mean": _cutoff_dict(shared_curves["no_mean"]),
            "comparison": _comparison_dict(shared_curves[reference_label], shared_curves["no_mean"]),
            "meaning": "Includes center displacement/geometric fidelity and radial contrast.",
        },
        "center_registered_radial_contrast": {
            reference_label: _cutoff_dict(registered_curves[reference_label]),
            "no_mean": _cutoff_dict(registered_curves["no_mean"]),
            "comparison": _comparison_dict(
                registered_curves[reference_label], registered_curves["no_mean"]
            ),
            "meaning": (
                "Compensates only the fitted star-center displacement. This is the closer "
                "estimate of pure radial resolution, but it is still relative to VAR."
            ),
        },
        "visual_artifact_quality": {
            "evaluation_center_xy_evaluation_1based": list(reference_center),
            "reference_cutoff_radius_px": reference_cutoff_radius,
            "reference": artifact_metrics[reference_label].to_dict(),
            "reconstruction": artifact_metrics["no_mean"].to_dict(),
            "reconstruction_relative_to_reference": {
                "radial_barb_energy_ratio": (
                    artifact_metrics["no_mean"].radial_to_tangential_gradient_energy
                    / max(
                        artifact_metrics[reference_label].radial_to_tangential_gradient_energy,
                        np.finfo(np.float64).eps,
                    )
                ),
                "off_harmonic_energy_ratio": (
                    artifact_metrics["no_mean"].off_harmonic_angular_energy_fraction
                    / max(
                        artifact_metrics[reference_label].off_harmonic_angular_energy_fraction,
                        np.finfo(np.float64).eps,
                    )
                ),
                "laplacian_energy_ratio": (
                    artifact_metrics["no_mean"].normalized_laplacian_energy
                    / max(
                        artifact_metrics[reference_label].normalized_laplacian_energy,
                        np.finfo(np.float64).eps,
                    )
                ),
                "target_phase_coherence_delta": (
                    artifact_metrics["no_mean"].target_harmonic_phase_coherence
                    - artifact_metrics[reference_label].target_harmonic_phase_coherence
                ),
            },
            "annular_similarity_to_reference": artifact_similarity.to_dict(),
            "meaning": (
                "Uses the fixed VAR center. Lower radial-barb, off-harmonic and "
                "Laplacian energies are cleaner; higher phase coherence and "
                "similarity are better. Registered FTC cannot compensate these metrics."
            ),
        },
        "center_sensitivity": {
            reference_label: _center_sensitivity(
                images[reference_label], reference_center, curve_options, args.center_sensitivity_px
            ),
            "no_mean_registered": _center_sensitivity(
                images["no_mean"], reconstruction_center, curve_options, args.center_sensitivity_px
            ),
        },
        "validity": {
            "old_fixed_22px_result_valid": False,
            "absolute_optical_resolution_claim_supported": False,
            "relative_var_comparison_supported": True,
            "warning": (
                "No ideal/ground-truth star is available. Treat these as relative FTC and "
                "geometric-fidelity measurements, not absolute optical resolution."
            ),
        },
    }
    with (output_dir / "resolution_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _save_curve_csv(
        output_dir / "ftc_curve_shared_center.csv",
        shared_curves[reference_label],
        shared_curves["no_mean"],
    )
    _save_curve_csv(
        output_dir / "ftc_curve_center_registered.csv",
        registered_curves[reference_label],
        registered_curves["no_mean"],
    )
    tifffile.imwrite(
        output_dir / "reference_true_depth_normalized.tif",
        images[reference_label].astype(np.float32),
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / "no_mean_true_depth_normalized.tif",
        images["no_mean"].astype(np.float32),
        photometric="minisblack",
    )
    _plot_images(output_dir / "true_depth_comparison.png", images, args.true_depth_um)
    _plot_curves(
        output_dir / "ftc_curves_shared_center.png",
        shared_curves,
        args.threshold,
        "Shared VAR-calibrated center (includes displacement)",
    )
    _plot_curves(
        output_dir / "ftc_curves_center_registered.png",
        registered_curves,
        args.threshold,
        "Individually centered radial contrast",
    )
    _plot_overlays(
        output_dir / "cutoff_overlays_shared_center.png",
        images,
        shared_curves,
        {reference_label: reference_center, "no_mean": reference_center},
        args.angular_samples,
        "Shared VAR-calibrated center",
    )
    _plot_overlays(
        output_dir / "cutoff_overlays_center_registered.png",
        images,
        registered_curves,
        registered_centers,
        args.angular_samples,
        "Individually centered radial contrast",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
