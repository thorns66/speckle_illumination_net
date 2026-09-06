from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from physics.speckle_oracle import SpeckleGeneratorConfig, generate_speckle_ensemble
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


LOGGER = logging.getLogger("scan_exact_normalized_speckle_forward")


@torch.inference_mode()
def _normalized_sensor_statistics(
    operator: LFMOperator,
    template: np.ndarray,
    patterns: torch.Tensor,
    probes: np.ndarray,
    *,
    batch_size: int,
    convergence_size: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute mean and C_y q after per-frame sensor max normalization."""

    count, height, width = patterns.shape
    if not 2 <= convergence_size < count:
        raise ValueError("convergence_size must lie in [2, ensemble_size)")
    device, dtype = patterns.device, patterns.dtype
    typed_template = torch.from_numpy(template).to(device=device, dtype=dtype)
    typed_probes = torch.from_numpy(probes.reshape(probes.shape[0], -1)).to(
        device=device, dtype=dtype
    )
    checkpoints = {convergence_size, count}
    sum_y = torch.zeros((height * width,), device=device, dtype=torch.float64)
    sum_yytq = torch.zeros(
        (probes.shape[0], height * width), device=device, dtype=torch.float64
    )
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    processed = 0
    while processed < count:
        next_checkpoint = min(value for value in checkpoints if value > processed)
        current = min(batch_size, next_checkpoint - processed)
        speckles = patterns[processed : processed + current]
        # [B,H,W] -> [B,1,1,H,W]
        volumes = (speckles * typed_template)[:, None, None]
        sensor = operator(volumes)[:, 0]
        sensor = sensor / sensor.amax(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(sensor.dtype).eps
        )
        flat = sensor.reshape(current, -1).to(torch.float64)
        coefficients = flat @ typed_probes.T.to(torch.float64)
        sum_y += flat.sum(dim=0)
        sum_yytq += coefficients.T @ flat
        processed += current
        if processed in checkpoints:
            mean = sum_y / float(processed)
            mean_probe = mean.to(dtype) @ typed_probes.T
            covariance_action = (
                sum_yytq - float(processed) * mean_probe.to(torch.float64)[:, None] * mean[None]
            ) / float(processed - 1)
            results[processed] = (
                mean.reshape(height, width).cpu().numpy().astype(np.float32),
                covariance_action.reshape(probes.shape).cpu().numpy().astype(np.float32),
            )
            LOGGER.info("Exact normalized forward accumulated %d patterns", processed)
    return results


def _mean_gain(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    denominator = float(np.vdot(prediction.reshape(-1), prediction.reshape(-1)).real)
    gain = max(
        float(np.vdot(prediction.reshape(-1), target.reshape(-1)).real)
        / max(denominator, np.finfo(np.float64).eps),
        0.0,
    )
    relative_error = float(
        np.linalg.norm(gain * prediction - target)
        / max(np.linalg.norm(target), np.finfo(np.float64).eps)
    )
    return gain, relative_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle depth scan with exact per-frame sensor max normalization"
    )
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ensemble-size", type=int, default=1024)
    parser.add_argument("--convergence-size", type=int, default=512)
    parser.add_argument("--speckle-seed", type=int, default=20260904)
    parser.add_argument("--speckle-generation-batch-size", type=int, default=32)
    parser.add_argument("--forward-batch-size", type=int, default=64)
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-sigma", type=float, default=16.0)
    parser.add_argument("--probe-seed", type=int, default=20260901)
    parser.add_argument("--phase-chunk-size", type=int, default=16)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/covariance_sketch_validation/exact_sensor_normalization",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[int, dict[str, np.ndarray]] = {}
    for depth_um in (20, 50, 80):
        frames = _load_raw_depth(config_path, config, depth_um)
        datasets[depth_um] = {
            "frames": frames,
            "mean": frames.mean(axis=0, dtype=np.float64).astype(np.float32),
            "fvar_projection": _common_template(_load_f_var(config_path, config, depth_um)),
        }
    e0 = np.load(config_path.parent.parent / "outputs/depth50_n100_no_mean_loss/reconstruction_best.npy")
    if e0.ndim == 5:
        e0 = e0[0, 0]
    e0_layer = np.maximum(e0[4], 0.0)
    datasets[50]["e0_true_layer"] = (e0_layer / e0_layer.sum()).astype(np.float32)
    sensor_shape = tuple(int(value) for value in datasets[50]["mean"].shape)
    probes = _lowpass_probes(
        args.num_probes,
        sensor_shape,
        sigma=args.probe_sigma,
        seed=args.probe_seed,
    )
    empirical = {
        depth: _empirical_action(data["frames"], probes)
        for depth, data in datasets.items()
    }
    permutation = np.random.default_rng(args.probe_seed + 2000).permutation(100)
    empirical_splits = [
        _empirical_action(datasets[50]["frames"][indices], probes)
        for indices in (permutation[:50], permutation[50:])
    ]

    generator_config = SpeckleGeneratorConfig(seed=args.speckle_seed)
    LOGGER.info("Generating %d independent speckles", args.ensemble_size)
    patterns = generate_speckle_ensemble(
        args.ensemble_size,
        generator_config,
        device=device,
        batch_size=args.speckle_generation_batch_size,
    )
    psf_config = config["psf"]
    LOGGER.info("Loading selected PSF")
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
    operators = [
        LFMOperator(
            h_all[index : index + 1],
            mode="optimized",
            phase_chunk_size=args.phase_chunk_size,
        )
        for index in range(len(z_values))
    ]
    depth_to_index = {int(round(value)): index for index, value in enumerate(z_values)}

    cache: dict[tuple[int, str, int], dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def statistics(
        dataset_depth: int, template_name: str, candidate_index: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        key = (dataset_depth, template_name, candidate_index)
        if key not in cache:
            cache[key] = _normalized_sensor_statistics(
                operators[candidate_index],
                datasets[dataset_depth][template_name],
                patterns,
                probes,
                batch_size=args.forward_batch_size,
                convergence_size=args.convergence_size,
            )
        return cache[key]

    calibration: list[dict[str, float]] = []
    for depth_um in (20, 50, 80):
        result = statistics(depth_um, "fvar_projection", depth_to_index[depth_um])
        predicted_mean, action = result[args.ensemble_size]
        gain, mean_error = _mean_gain(predicted_mean, datasets[depth_um]["mean"])
        scaled_by_mean = gain**2 * action
        covariance_scale = _fit_nonnegative_scale(scaled_by_mean, empirical[depth_um])
        convergence_action = result[args.convergence_size][1]
        convergence_error = float(
            np.linalg.norm(convergence_action - action)
            / max(np.linalg.norm(action), np.finfo(np.float64).eps)
        )
        calibration.append(
            {
                "depth_um": float(depth_um),
                "photometric_gain": gain,
                "mean_relative_error": mean_error,
                "residual_covariance_scale": covariance_scale,
                "512_vs_1024_action_relative_error": convergence_error,
            }
        )
        LOGGER.info("Calibration %d um: %s", depth_um, calibration[-1])
    scales = np.asarray([row["residual_covariance_scale"] for row in calibration])
    system_scale = float(np.exp(np.median(np.log(scales))))
    scale_cv = float(scales.std(ddof=1) / scales.mean())
    LOGGER.info("Exact normalization system scale %.7g CV %.4f", system_scale, scale_cv)

    scan_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for template_name in ("fvar_projection", "e0_true_layer"):
        template_rows: list[dict[str, object]] = []
        convergence_errors: list[float] = []
        for index, z_um in enumerate(z_values):
            result = statistics(50, template_name, index)
            predicted_mean, action = result[args.ensemble_size]
            gain, mean_error = _mean_gain(predicted_mean, datasets[50]["mean"])
            prediction = system_scale * gain**2 * action
            convergence_error = float(
                np.linalg.norm(result[args.convergence_size][1] - action)
                / max(np.linalg.norm(action), np.finfo(np.float64).eps)
            )
            convergence_errors.append(convergence_error)
            losses = {
                "full": _sketch_loss(
                    prediction, empirical[50], correlation_weight=args.correlation_weight
                ),
                "split_a": _sketch_loss(
                    prediction, empirical_splits[0], correlation_weight=args.correlation_weight
                ),
                "split_b": _sketch_loss(
                    prediction, empirical_splits[1], correlation_weight=args.correlation_weight
                ),
            }
            row: dict[str, object] = {
                "template": template_name,
                "candidate_depth_um": z_um,
                "photometric_gain": gain,
                "mean_relative_error": mean_error,
                "action_convergence_relative_error": convergence_error,
            }
            for subset, values in losses.items():
                row.update({f"{subset}_{key}": value for key, value in values.items()})
            scan_rows.append(row)
            template_rows.append(row)
            LOGGER.info(
                "%s %.0f um: %.6f / %.6f / %.6f conv=%.4f",
                template_name,
                z_um,
                row["full_combined_loss"],
                row["split_a_combined_loss"],
                row["split_b_combined_loss"],
                convergence_error,
            )
        minima = {
            subset: float(
                min(template_rows, key=lambda row: float(row[f"{subset}_combined_loss"]))[
                    "candidate_depth_um"
                ]
            )
            for subset in ("full", "split_a", "split_b")
        }
        true_row = next(
            row for row in template_rows if round(float(row["candidate_depth_um"])) == 50
        )
        wrong = [
            row for row in template_rows if round(float(row["candidate_depth_um"])) != 50
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
            and max(convergence_errors) <= 0.05
        )
        summaries.append(
            {
                "template": template_name,
                "minima_um": minima,
                "wrong_depth_relative_margin": margins,
                "maximum_action_convergence_relative_error": max(convergence_errors),
                "passed": passed,
            }
        )

    with (output_dir / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)
    result = {
        "model": "exact_generator_with_sensor_max_normalization",
        "generator": generator_config.to_dict(),
        "ensemble_size": args.ensemble_size,
        "convergence_size": args.convergence_size,
        "system_residual_covariance_scale": system_scale,
        "calibration_scale_cv": scale_cv,
        "calibration": calibration,
        "summaries": summaries,
        "primary_fvar_passed": any(
            row["template"] == "fvar_projection" and bool(row["passed"])
            for row in summaries
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["primary_fvar_passed"]:
        print("EXACT_NORMALIZED_STAGE_BLOCKED: do not start covariance training", flush=True)
        raise SystemExit(2)
    print("EXACT_NORMALIZED_STAGE_PASSED: 200-step training is authorized", flush=True)


if __name__ == "__main__":
    main()
