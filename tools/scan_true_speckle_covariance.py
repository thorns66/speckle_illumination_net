from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml

from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from physics.speckle_oracle import (
    SpeckleGeneratorConfig,
    analytic_intensity_covariance,
    ensemble_covariance_action,
    generate_speckle_ensemble,
    stationary_covariance_action,
    stationary_covariance_from_ensemble,
)
from tools.scan_covariance_sketch_depth import (
    _common_template,
    _empirical_action,
    _fit_nonnegative_scale,
    _load_f_var,
    _load_raw_depth,
    _lowpass_probes,
    _resolve,
    _sketch_loss,
)
from utils.io import load_volume_tiff


LOGGER = logging.getLogger("scan_true_speckle_covariance")


def _load_g_mean(config_path: Path, config: dict, depth_um: int) -> np.ndarray:
    z_count = len(config["psf"]["z_values_um"])
    configured = _resolve(config_path, config["data"]["g_mean_tiff_path"])
    path = configured if depth_um == 50 else configured.with_name(
        configured.name.replace("detph50", f"detph{depth_um}")
    )
    return load_volume_tiff(
        path,
        z_count,
        intensity_mode=config["data"]["tiff_intensity_mode"],
    )


def _normalized_projection(volume: np.ndarray) -> np.ndarray:
    projection = np.maximum(volume, 0.0).sum(axis=0, dtype=np.float64)
    return (projection / projection.sum()).astype(np.float32)


def _attachment_comparison(
    attachment_path: str | Path,
    analytic: np.ndarray,
    monte_carlo: np.ndarray,
) -> dict[str, float]:
    attachment = np.asarray(tifffile.imread(attachment_path), dtype=np.float64)
    center = np.unravel_index(int(np.argmax(attachment)), attachment.shape)
    attachment /= attachment[center]
    target_radius = min(
        center[0], center[1], attachment.shape[0] - 1 - center[0], attachment.shape[1] - 1 - center[1]
    )
    radius = min(target_radius, analytic.shape[0] // 2)
    attachment_crop = attachment[
        center[0] - radius : center[0] + radius + 1,
        center[1] - radius : center[1] + radius + 1,
    ]
    source_center = analytic.shape[0] // 2
    analytic_crop = analytic[
        source_center - radius : source_center + radius + 1,
        source_center - radius : source_center + radius + 1,
    ]
    monte_carlo_crop = monte_carlo[
        source_center - radius : source_center + radius + 1,
        source_center - radius : source_center + radius + 1,
    ]

    def relative(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second) / np.linalg.norm(second))

    return {
        "attachment_vs_analytic_relative_error": relative(attachment_crop, analytic_crop),
        "attachment_vs_monte_carlo_relative_error": relative(attachment_crop, monte_carlo_crop),
        "analytic_vs_monte_carlo_relative_error": relative(analytic_crop, monte_carlo_crop),
        "comparison_radius_px": float(radius),
    }


@torch.inference_mode()
def _precompute_depth_operators(
    h_all: torch.Tensor,
    probes: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    phase_chunk_size: int,
) -> tuple[list[LFMOperator], list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    typed_probes = torch.from_numpy(probes)[:, None].to(
        device=h_all.device, dtype=h_all.dtype
    )
    operators: list[LFMOperator] = []
    backprojections: list[torch.Tensor] = []
    mean_projections = {name: [] for name in templates}
    for index in range(h_all.shape[0]):
        operator = LFMOperator(
            h_all[index : index + 1],
            mode="optimized",
            phase_chunk_size=phase_chunk_size,
        )
        operators.append(operator)
        backprojections.append(operator.adjoint(typed_probes)[:, 0, 0])
        for name, template in templates.items():
            typed = torch.from_numpy(template)[None, None, None].to(
                device=h_all.device, dtype=h_all.dtype
            )
            mean_projections[name].append(operator(typed)[0, 0])
        LOGGER.info("Precomputed H/Ht products for depth index %d", index)
    return operators, backprojections, mean_projections


@torch.inference_mode()
def _theory_action(
    variant: str,
    operator: LFMOperator,
    backprojection: torch.Tensor,
    unit_mean_prediction: torch.Tensor,
    template: np.ndarray,
    measured_mean: np.ndarray,
    *,
    analytic_kernel: torch.Tensor,
    monte_carlo_kernel: torch.Tensor,
    patterns: torch.Tensor,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    device, dtype = backprojection.device, backprojection.dtype
    typed_template = torch.from_numpy(template).to(device=device, dtype=dtype)
    target_mean = torch.from_numpy(measured_mean).to(device=device, dtype=dtype)
    if variant == "generator_nonstationary":
        illumination_mean = patterns.mean(dim=0)
        mean_input = typed_template * illumination_mean
        unit_mean_prediction = operator(mean_input[None, None, None])[0, 0]
    gain = (
        (unit_mean_prediction * target_mean).sum()
        / unit_mean_prediction.square().sum().clamp_min(torch.finfo(dtype).eps)
    ).clamp_min(torch.finfo(dtype).eps)
    mean_error = (
        torch.linalg.vector_norm(gain * unit_mean_prediction - target_mean)
        / torch.linalg.vector_norm(target_mean).clamp_min(torch.finfo(dtype).eps)
    )
    reconstruction = gain * typed_template
    right = reconstruction[None] * backprojection
    if variant == "analytic_stationary":
        correlated, report = stationary_covariance_action(right, analytic_kernel)
    elif variant == "generator_stationary":
        correlated, report = stationary_covariance_action(right, monte_carlo_kernel)
    elif variant == "generator_nonstationary":
        correlated = ensemble_covariance_action(right, patterns)
        report = {"negative_spectral_mass_fraction": 0.0, "minimum_raw_eigenvalue": 0.0}
    else:
        raise ValueError(f"Unknown oracle variant {variant}")
    left = reconstruction[None] * correlated
    prediction = operator(left[:, None, None])[:, 0]
    return prediction.cpu().numpy(), float(gain), float(mean_error), report


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle Cs value-of-information depth scan")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ensemble-size", type=int, default=4096)
    parser.add_argument("--convergence-size", type=int, default=1024)
    parser.add_argument("--speckle-batch-size", type=int, default=32)
    parser.add_argument("--speckle-seed", type=int, default=20260904)
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-sigma", type=float, default=16.0)
    parser.add_argument("--probe-seed", type=int, default=20260901)
    parser.add_argument("--phase-chunk-size", type=int, default=16)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument(
        "--attachment-cs",
        default="psf/illumination_cs_NA0.04479.tif",
    )
    parser.add_argument(
        "--output-dir", default="outputs/covariance_sketch_validation/oracle_true_cs"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.convergence_size >= args.ensemble_size:
        raise ValueError("convergence-size must be smaller than ensemble-size")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full oracle experiment")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[int, dict[str, np.ndarray]] = {}
    for depth_um in (20, 50, 80):
        frames = _load_raw_depth(config_path, config, depth_um)
        datasets[depth_um] = {
            "frames": frames,
            "mean": frames.mean(axis=0, dtype=np.float64).astype(np.float32),
            "fvar_projection": _common_template(_load_f_var(config_path, config, depth_um)),
            "gmean_projection": _normalized_projection(_load_g_mean(config_path, config, depth_um)),
        }
    e0 = np.load(config_path.parent.parent / "outputs/depth50_n100_no_mean_loss/reconstruction_best.npy")
    if e0.ndim == 5:
        e0 = e0[0, 0]
    datasets[50]["e0_true_layer"] = (
        np.maximum(e0[4], 0.0) / np.maximum(np.maximum(e0[4], 0.0).sum(), np.finfo(np.float32).eps)
    ).astype(np.float32)
    probes = _lowpass_probes(
        args.num_probes,
        tuple(int(value) for value in datasets[50]["mean"].shape),
        sigma=args.probe_sigma,
        seed=args.probe_seed,
    )
    empirical_full = {
        depth: _empirical_action(data["frames"], probes)
        for depth, data in datasets.items()
    }
    permutation = np.random.default_rng(args.probe_seed + 2000).permutation(100)
    empirical_splits = [
        _empirical_action(datasets[50]["frames"][indices], probes)
        for indices in (permutation[:50], permutation[50:])
    ]

    generator_config = SpeckleGeneratorConfig(seed=args.speckle_seed)
    LOGGER.info("Generating %d independent oracle speckles on %s", args.ensemble_size, device)
    patterns = generate_speckle_ensemble(
        args.ensemble_size,
        generator_config,
        device=device,
        batch_size=args.speckle_batch_size,
    )
    LOGGER.info("Estimating analytic and Monte Carlo stationary Cs")
    analytic_kernel = analytic_intensity_covariance(
        generator_config, device=device, dtype=torch.float32
    )
    monte_carlo_kernel = stationary_covariance_from_ensemble(patterns)
    convergence_kernel = stationary_covariance_from_ensemble(
        patterns[: args.convergence_size]
    )
    convergence_relative_error = float(
        (
            torch.linalg.vector_norm(convergence_kernel - monte_carlo_kernel)
            / torch.linalg.vector_norm(monte_carlo_kernel).clamp_min(
                torch.finfo(monte_carlo_kernel.dtype).eps
            )
        ).item()
    )
    analytic_relative_error = float(
        (
            torch.linalg.vector_norm(analytic_kernel - monte_carlo_kernel)
            / torch.linalg.vector_norm(monte_carlo_kernel).clamp_min(
                torch.finfo(monte_carlo_kernel.dtype).eps
            )
        ).item()
    )
    np.save(output_dir / "cs_analytic.npy", analytic_kernel.cpu().numpy())
    np.save(output_dir / "cs_generator_monte_carlo.npy", monte_carlo_kernel.cpu().numpy())
    attachment_comparison = _attachment_comparison(
        _resolve(config_path, args.attachment_cs),
        analytic_kernel.cpu().numpy(),
        monte_carlo_kernel.cpu().numpy(),
    )
    diagnostics = {
        "generator": generator_config.to_dict(),
        "ensemble_size": args.ensemble_size,
        "convergence_size": args.convergence_size,
        "convergence_relative_error": convergence_relative_error,
        "analytic_vs_monte_carlo_full_relative_error": analytic_relative_error,
        **attachment_comparison,
        "pattern_mean": float(patterns.mean().item()),
        "pattern_mean_spatial_cv": float(
            (patterns.mean(dim=0).std() / patterns.mean(dim=0).mean()).item()
        ),
    }
    with (output_dir / "truth_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    LOGGER.info("Oracle diagnostics: %s", diagnostics)

    psf_config = config["psf"]
    LOGGER.info("Loading full selected PSF")
    psf_data = load_psf(
        _resolve(config_path, psf_config["H_path"]),
        psf_config["z_values_um"],
        h_variable_name=psf_config["H_variable_name"],
        ht_variable_name=psf_config["Ht_variable_name"],
        psf_z_all_um=psf_config.get("psf_z_all_um"),
        depth_unit=psf_config.get("depth_unit", "auto"),
        load_h=True,
        load_ht=False,
    )
    h_all = torch.from_numpy(psf_data.H).to(device=device, dtype=torch.float32)
    z_values = [float(value) for value in psf_data.metadata.selected_z_um]
    del psf_data
    gc.collect()
    templates_50 = {
        name: datasets[50][name]
        for name in ("fvar_projection", "gmean_projection", "e0_true_layer")
    }
    operators, backprojections, mean_projections = _precompute_depth_operators(
        h_all,
        probes,
        templates_50,
        phase_chunk_size=args.phase_chunk_size,
    )
    variants = (
        "analytic_stationary",
        "generator_stationary",
        "generator_nonstationary",
    )
    depth_to_index = {int(round(value)): index for index, value in enumerate(z_values)}
    scan_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for variant in variants:
        LOGGER.info("Calibrating variant %s on 20/50/80 um", variant)
        calibration_scales: list[float] = []
        calibration_rows: list[dict[str, float]] = []
        for depth_um in (20, 50, 80):
            index = depth_to_index[depth_um]
            template = datasets[depth_um]["fvar_projection"]
            operator = operators[index]
            typed = torch.from_numpy(template)[None, None, None].to(
                device=device, dtype=h_all.dtype
            )
            unit_mean = operator(typed)[0, 0]
            action, gain, mean_error, _ = _theory_action(
                variant,
                operator,
                backprojections[index],
                unit_mean,
                template,
                datasets[depth_um]["mean"],
                analytic_kernel=analytic_kernel,
                monte_carlo_kernel=monte_carlo_kernel,
                patterns=patterns,
            )
            scale = _fit_nonnegative_scale(action, empirical_full[depth_um])
            calibration_scales.append(scale)
            calibration_rows.append(
                {
                    "depth_um": float(depth_um),
                    "scale": scale,
                    "gain": gain,
                    "mean_relative_error": mean_error,
                }
            )
        scales = np.asarray(calibration_scales)
        system_scale = float(np.exp(np.median(np.log(scales))))
        scale_cv = float(scales.std(ddof=1) / scales.mean())
        LOGGER.info("%s system scale %.7g CV %.4f", variant, system_scale, scale_cv)

        for template_name, template in templates_50.items():
            variant_rows: list[dict[str, object]] = []
            psd_reports: list[dict[str, float]] = []
            for index, z_um in enumerate(z_values):
                action, gain, mean_error, psd_report = _theory_action(
                    variant,
                    operators[index],
                    backprojections[index],
                    mean_projections[template_name][index],
                    template,
                    datasets[50]["mean"],
                    analytic_kernel=analytic_kernel,
                    monte_carlo_kernel=monte_carlo_kernel,
                    patterns=patterns,
                )
                psd_reports.append(psd_report)
                prediction = system_scale * action
                losses = {
                    "full": _sketch_loss(
                        prediction, empirical_full[50], correlation_weight=args.correlation_weight
                    ),
                    "split_a": _sketch_loss(
                        prediction, empirical_splits[0], correlation_weight=args.correlation_weight
                    ),
                    "split_b": _sketch_loss(
                        prediction, empirical_splits[1], correlation_weight=args.correlation_weight
                    ),
                }
                row: dict[str, object] = {
                    "variant": variant,
                    "template": template_name,
                    "candidate_depth_um": z_um,
                    "photometric_gain": gain,
                    "mean_relative_error": mean_error,
                }
                for subset, values in losses.items():
                    row.update({f"{subset}_{key}": value for key, value in values.items()})
                scan_rows.append(row)
                variant_rows.append(row)
                LOGGER.info(
                    "%s/%s %.0f um: %.6f / %.6f / %.6f",
                    variant,
                    template_name,
                    z_um,
                    row["full_combined_loss"],
                    row["split_a_combined_loss"],
                    row["split_b_combined_loss"],
                )
            minima = {
                subset: float(
                    min(variant_rows, key=lambda item: float(item[f"{subset}_combined_loss"]))[
                        "candidate_depth_um"
                    ]
                )
                for subset in ("full", "split_a", "split_b")
            }
            true_row = next(
                row for row in variant_rows if round(float(row["candidate_depth_um"])) == 50
            )
            wrong = [
                row for row in variant_rows if round(float(row["candidate_depth_um"])) != 50
            ]
            margins = {
                subset: float(
                    min(float(row[f"{subset}_combined_loss"]) for row in wrong)
                    / max(float(true_row[f"{subset}_combined_loss"]), np.finfo(np.float64).eps)
                    - 1.0
                )
                for subset in ("full", "split_a", "split_b")
            }
            passed = (
                all(round(value) == 50 for value in minima.values())
                and all(value >= 0.10 for value in margins.values())
                and scale_cv <= 0.10
            )
            summaries.append(
                {
                    "variant": variant,
                    "template": template_name,
                    "system_scale": system_scale,
                    "calibration_scale_cv": scale_cv,
                    "calibration": calibration_rows,
                    "minima_um": minima,
                    "wrong_depth_relative_margin": margins,
                    "maximum_negative_spectral_mass_fraction": max(
                        report["negative_spectral_mass_fraction"] for report in psd_reports
                    ),
                    "passed": passed,
                }
            )

    with (output_dir / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)
    result = {
        "truth_diagnostics": diagnostics,
        "summaries": summaries,
        "primary_fvar_passed": any(
            row["template"] == "fvar_projection" and bool(row["passed"])
            for row in summaries
        ),
        "any_template_passed": any(bool(row["passed"]) for row in summaries),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["primary_fvar_passed"]:
        print(
            "ORACLE_LINEAR_STAGE_BLOCKED: test exact sensor-normalized generator before training",
            flush=True,
        )
        raise SystemExit(2)
    print("ORACLE_LINEAR_STAGE_PASSED: 200-step ablation is eligible", flush=True)


if __name__ == "__main__":
    main()
