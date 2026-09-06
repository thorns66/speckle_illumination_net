from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from physics.speckle_oracle import (
    SpeckleGeneratorConfig,
    generate_speckle_ensemble,
)
from tools.scan_covariance_sketch_depth import (
    _empirical_action,
    _load_f_var,
    _load_raw_depth,
    _lowpass_probes,
    _resolve,
)


LOGGER = logging.getLogger("diagnose_fixed_depth_lateral")


def bounded_anchor_shape(anchor: Tensor, raw: Tensor, bound: float) -> tuple[Tensor, Tensor]:
    """Positive unit-mass image with a bounded log correction around an anchor."""

    if anchor.shape != raw.shape or anchor.ndim != 2:
        raise ValueError("anchor and raw must be matching 2D tensors")
    if bound <= 0.0:
        raise ValueError("bound must be positive")
    correction = float(bound) * torch.tanh(raw / float(bound))
    unnormalized = anchor * torch.exp(correction)
    shape = unnormalized / unnormalized.sum().clamp_min(torch.finfo(raw.dtype).eps)
    return shape, correction


def hessian_schatten2(correction: Tensor, epsilon: float = 1e-6) -> Tensor:
    if correction.ndim != 2 or min(correction.shape) < 3:
        raise ValueError("correction must be a 2D image at least 3x3")
    center = correction[1:-1, 1:-1]
    dxx = correction[1:-1, 2:] - 2.0 * center + correction[1:-1, :-2]
    dyy = correction[2:, 1:-1] - 2.0 * center + correction[:-2, 1:-1]
    dxy = (
        correction[2:, 2:]
        - correction[2:, :-2]
        - correction[:-2, 2:]
        + correction[:-2, :-2]
    ) * 0.25
    return torch.sqrt(dxx.square() + dyy.square() + 2.0 * dxy.square() + epsilon**2).mean()


def _positive_log_fit(prediction: Tensor, target: Tensor, epsilon: Tensor) -> Tensor:
    valid = target > epsilon
    if not bool(valid.any()):
        raise ValueError("measured sensor variance has no positive pixels")
    offset = (torch.log(target[valid] + epsilon) - torch.log(prediction[valid] + epsilon)).median()
    return torch.exp(offset).detach()


def diagonal_variance_loss(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    epsilon = target.detach().mean().clamp_min(torch.finfo(target.dtype).eps) * 1e-6
    scale = _positive_log_fit(prediction, target, epsilon)
    residual = torch.log(scale * prediction.clamp_min(0.0) + epsilon) - torch.log(target + epsilon)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual)), scale


def covariance_vector_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    correlation_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    epsilon = torch.finfo(pred.dtype).eps
    scale = (
        (pred * truth).sum() / pred.square().sum().clamp_min(epsilon)
    ).clamp_min(0.0).detach()
    scaled = scale * pred
    relative_mse_per_probe = (scaled - truth).square().sum(dim=1) / truth.square().sum(
        dim=1
    ).clamp_min(epsilon)
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    truth_centered = truth - truth.mean(dim=1, keepdim=True)
    correlation = (pred_centered * truth_centered).sum(dim=1) / (
        pred_centered.square().sum(dim=1).sqrt()
        * truth_centered.square().sum(dim=1).sqrt()
    ).clamp_min(epsilon)
    rel_mse = relative_mse_per_probe.mean()
    corr = correlation.mean()
    combined = rel_mse + float(correlation_weight) * (1.0 - corr)
    return combined, {
        "scale": scale,
        "relative_mse": rel_mse,
        "correlation": corr,
        "correlation_loss": 1.0 - corr,
    }


def _operator_forward(operator: LFMOperator, volume: Tensor, *, squared: bool) -> Tensor:
    if squared:
        return checkpoint(operator.forward_squared, volume, use_reentrant=False)
    return checkpoint(operator, volume, use_reentrant=False)


def _oracle_prediction(
    shape: Tensor,
    backprojected: Tensor,
    centered_patterns: Tensor,
    operator: LFMOperator,
) -> Tensor:
    batch_shape = shape[None, None, None].expand(backprojected.shape[0], 1, 1, -1, -1)
    right = (batch_shape * backprojected).squeeze(1).squeeze(1)
    flat_right = right.flatten(1)
    flat_patterns = centered_patterns.flatten(1)
    weights = flat_patterns @ flat_right.T
    correlated = (weights.T @ flat_patterns).reshape_as(right) / float(
        centered_patterns.shape[0] - 1
    )
    forward_input = batch_shape * correlated[:, None, None]
    return _operator_forward(operator, forward_input, squared=False)[:, 0]


@torch.no_grad()
def _precompute_adjoint(operator: LFMOperator, probes: Tensor, chunk_size: int) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, probes.shape[0], chunk_size):
        chunks.append(operator.adjoint(probes[start : start + chunk_size, None]))
    return torch.cat(chunks, dim=0)


def _gradient_norm(loss: Tensor, parameter: Tensor, *, retain_graph: bool = True) -> float:
    gradient = torch.autograd.grad(loss, parameter, retain_graph=retain_graph)[0]
    return float(torch.linalg.vector_norm(gradient).detach().item())


def _save_volume(path: Path, layer: np.ndarray, depth_index: int, z_count: int) -> None:
    volume = np.zeros((z_count, *layer.shape), dtype=np.float32)
    volume[depth_index] = layer.astype(np.float32, copy=False)
    np.save(path.with_suffix(".npy"), volume)
    tifffile.imwrite(path.with_suffix(".tif"), volume, photometric="minisblack")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix z=50 um and directly optimize only the lateral nonnegative structure"
    )
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mode", choices=("h2", "covariance", "joint"), required=True)
    parser.add_argument("--anchor", choices=("e0", "f_var"), default="e0")
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--covariance-gradient-ratio", type=float, default=0.3)
    parser.add_argument("--curvature-gradient-ratio", type=float, default=0.0)
    parser.add_argument("--curvature-start-step", type=int, default=20)
    parser.add_argument("--train-probes", type=int, default=16)
    parser.add_argument("--probes-per-step", type=int, default=4)
    parser.add_argument("--holdout-probes", type=int, default=16)
    parser.add_argument("--probe-sigma", type=float, default=16.0)
    parser.add_argument("--probe-seed", type=int, default=20260901)
    parser.add_argument("--holdout-probe-seed", type=int, default=20260902)
    parser.add_argument("--train-patterns", type=int, default=1024)
    parser.add_argument("--holdout-patterns", type=int, default=2048)
    parser.add_argument("--train-pattern-seed", type=int, default=20260904)
    parser.add_argument("--holdout-pattern-seed", type=int, default=20260905)
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--adjoint-chunk-size", type=int, default=1)
    parser.add_argument("--phase-chunk-size", type=int, default=16)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--train-frame-fraction", type=float, default=1.0)
    parser.add_argument("--frame-split-seed", type=int, default=20260906)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.steps < 1 or args.train_probes < args.probes_per_step:
        raise ValueError("invalid step/probe counts")
    if args.train_probes % args.probes_per_step != 0:
        raise ValueError("train-probes must be divisible by probes-per-step")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This full-size diagnostic requires CUDA")
    torch.cuda.set_device(device)

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    z_values = np.asarray(config["psf"]["z_values_um"], dtype=np.float64)
    matches = np.flatnonzero(np.isclose(z_values, 50.0, atol=1e-6, rtol=0.0))
    if len(matches) != 1:
        raise ValueError("The configured grid must contain exactly one 50 um layer")
    depth_index = int(matches[0])
    frames_np = _load_raw_depth(config_path, config, 50).astype(np.float32, copy=False)
    if not 0.0 < args.train_frame_fraction <= 1.0:
        raise ValueError("train-frame-fraction must lie in (0,1]")
    if args.train_frame_fraction < 1.0:
        permutation = np.random.default_rng(args.frame_split_seed).permutation(frames_np.shape[0])
        train_count = int(round(args.train_frame_fraction * frames_np.shape[0]))
        train_count = min(max(train_count, 2), frames_np.shape[0] - 2)
        train_indices = permutation[:train_count]
        holdout_indices = permutation[train_count:]
        train_frames_np = frames_np[train_indices]
        holdout_frames_np = frames_np[holdout_indices]
    else:
        train_indices = np.arange(frames_np.shape[0])
        holdout_indices = train_indices.copy()
        train_frames_np = frames_np
        holdout_frames_np = frames_np
    LOGGER.info("Frame split: train=%d holdout=%d seed=%d", len(train_indices), len(holdout_indices), args.frame_split_seed)
    f_var = _load_f_var(config_path, config, 50).astype(np.float32, copy=False)
    if args.anchor == "e0":
        e0_path = config_path.parent.parent / "outputs/depth50_n100_no_mean_loss/reconstruction_best.npy"
        anchor_np = np.load(e0_path).squeeze()[depth_index]
        anchor_source = str(e0_path)
    else:
        anchor_np = f_var[depth_index]
        anchor_source = str(_resolve(config_path, config["data"]["f_var_tiff_path"]))
    anchor_np = np.maximum(anchor_np, 0.0).astype(np.float32, copy=False)
    floor = max(float(anchor_np.max()) * 1e-6, np.finfo(np.float32).tiny)
    anchor_np = np.maximum(anchor_np, floor)
    anchor_np /= float(anchor_np.sum(dtype=np.float64))

    LOGGER.info("Loading only the true 50 um PSF layer")
    psf_config = config["psf"]
    psf_data = load_psf(
        _resolve(config_path, psf_config["H_path"]),
        [50.0],
        h_variable_name=psf_config["H_variable_name"],
        ht_variable_name=psf_config["Ht_variable_name"],
        psf_z_all_um=psf_config.get("psf_z_all_um"),
        depth_unit=psf_config.get("depth_unit", "auto"),
        load_h=True,
        load_ht=False,
    )
    operator = LFMOperator(
        torch.from_numpy(psf_data.H).to(device=device, dtype=torch.float32),
        mode="optimized",
        phase_chunk_size=args.phase_chunk_size,
    )
    del psf_data
    anchor = torch.from_numpy(anchor_np).to(device)
    raw = torch.nn.Parameter(torch.zeros_like(anchor))
    measured_variance_train = torch.from_numpy(train_frames_np.var(axis=0, ddof=1)).to(device)
    measured_variance_holdout = torch.from_numpy(holdout_frames_np.var(axis=0, ddof=1)).to(device)

    use_covariance = args.mode in {"covariance", "joint"}
    if use_covariance:
        sensor_shape = tuple(int(value) for value in frames_np.shape[-2:])
        train_probes_np = _lowpass_probes(
            args.train_probes, sensor_shape, sigma=args.probe_sigma, seed=args.probe_seed
        )
        holdout_probes_np = _lowpass_probes(
            args.holdout_probes,
            sensor_shape,
            sigma=args.probe_sigma,
            seed=args.holdout_probe_seed,
        )
        train_target = torch.from_numpy(_empirical_action(train_frames_np, train_probes_np)).to(device)
        holdout_target = torch.from_numpy(_empirical_action(holdout_frames_np, holdout_probes_np)).to(device)
        train_probes = torch.from_numpy(train_probes_np).to(device)
        holdout_probes = torch.from_numpy(holdout_probes_np).to(device)
        LOGGER.info("Precomputing exact H^T q for independent train/holdout probes")
        train_back = _precompute_adjoint(operator, train_probes, args.adjoint_chunk_size)
        holdout_back = _precompute_adjoint(operator, holdout_probes, args.adjoint_chunk_size)
        LOGGER.info("Generating independent oracle Cs factors")
        train_patterns = generate_speckle_ensemble(
            args.train_patterns,
            SpeckleGeneratorConfig(seed=args.train_pattern_seed),
            device=device,
            batch_size=args.pattern_batch_size,
        )
        holdout_patterns = generate_speckle_ensemble(
            args.holdout_patterns,
            SpeckleGeneratorConfig(seed=args.holdout_pattern_seed),
            device=device,
            batch_size=args.pattern_batch_size,
        )
        train_patterns = train_patterns - train_patterns.mean(dim=0, keepdim=True)
        holdout_patterns = holdout_patterns - holdout_patterns.mean(dim=0, keepdim=True)
    else:
        train_target = holdout_target = train_back = holdout_back = None
        train_patterns = holdout_patterns = None

    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
    covariance_weight = 0.0 if args.mode == "h2" else 1.0
    curvature_weight = 0.0
    history: list[dict[str, float | int]] = []
    best_score = math.inf
    best_layer: np.ndarray | None = None
    best_step = -1
    best_primary_scale = 1.0
    started = time.perf_counter()

    def loss_components(step: int, *, holdout: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
        values: dict[str, Tensor] = {"curvature": hessian_schatten2(correction)}
        if args.mode != "covariance" or holdout:
            volume = shape[None, None, None]
            h2_prediction = _operator_forward(operator, volume.square(), squared=True)[:, 0]
            variance_target = measured_variance_holdout if holdout else measured_variance_train
            h2_loss, h2_scale = diagonal_variance_loss(h2_prediction, variance_target[None])
            values["h2_loss"] = h2_loss
            values["h2_scale"] = h2_scale
        else:
            values["h2_loss"] = raw.new_zeros(())
            values["h2_scale"] = raw.new_ones(())
        if use_covariance:
            if holdout:
                assert holdout_back is not None and holdout_patterns is not None
                assert holdout_target is not None
                cov_prediction = _oracle_prediction(shape, holdout_back, holdout_patterns, operator)
                cov_target = holdout_target
            else:
                assert train_back is not None and train_patterns is not None
                assert train_target is not None
                bank_count = args.train_probes // args.probes_per_step
                bank = step % bank_count
                selection = slice(bank * args.probes_per_step, (bank + 1) * args.probes_per_step)
                cov_prediction = _oracle_prediction(
                    shape, train_back[selection], train_patterns, operator
                )
                cov_target = train_target[selection]
            cov_loss, cov_values = covariance_vector_loss(
                cov_prediction,
                cov_target,
                correlation_weight=args.correlation_weight,
            )
            values["cov_loss"] = cov_loss
            values.update({f"cov_{key}": value for key, value in cov_values.items()})
        else:
            values["cov_loss"] = raw.new_zeros(())
            values["cov_scale"] = raw.new_ones(())
            values["cov_relative_mse"] = raw.new_zeros(())
            values["cov_correlation"] = raw.new_zeros(())
            values["cov_correlation_loss"] = raw.new_zeros(())
        if args.mode == "h2":
            data = values["h2_loss"]
        elif args.mode == "covariance":
            data = values["cov_loss"]
        else:
            data = values["h2_loss"] + float(covariance_weight) * values["cov_loss"]
        total = data + float(curvature_weight) * values["curvature"]
        values["data_loss"] = data
        values["total_loss"] = total
        values["correction_rms"] = correction.square().mean().sqrt()
        values["correction_abs_max"] = correction.abs().max()
        values["shape_max_fraction"] = shape.max()
        return total, values

    # Fix the joint data-term balance once at the common initialization.
    if args.mode == "joint":
        _, initial = loss_components(0)
        h2_norm = _gradient_norm(initial["h2_loss"], raw)
        cov_norm = _gradient_norm(initial["cov_loss"], raw)
        covariance_weight = args.covariance_gradient_ratio * h2_norm / max(cov_norm, 1e-20)
        LOGGER.info(
            "Fixed covariance weight %.7g (requested gradient ratio %.4g; h2 %.4g, cov %.4g)",
            covariance_weight,
            args.covariance_gradient_ratio,
            h2_norm,
            cov_norm,
        )

    # The untouched anchor is an admissible checkpoint. This is essential for
    # frame-cross-fit early stopping: optimization must prove that it generalizes.
    _, initial_validation = loss_components(0, holdout=True)
    if args.mode == "covariance":
        initial_holdout_score = float(initial_validation["cov_loss"].detach().item())
        best_primary_scale = float(initial_validation["cov_scale"].detach().item())
    elif args.mode == "joint":
        initial_holdout_score = float(
            (initial_validation["h2_loss"] + covariance_weight * initial_validation["cov_loss"])
            .detach()
            .item()
        )
        best_primary_scale = float(initial_validation["h2_scale"].detach().item())
    else:
        initial_holdout_score = float(initial_validation["h2_loss"].detach().item())
        best_primary_scale = float(initial_validation["h2_scale"].detach().item())
    best_score = initial_holdout_score
    best_step = -1
    best_layer = anchor.detach().cpu().numpy()
    _save_volume(output_dir / "reconstruction_best", best_layer, depth_index, len(z_values))
    LOGGER.info("step=-01 untouched-anchor holdout=%.6g", initial_holdout_score)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total, values = loss_components(step)
        if (
            args.curvature_gradient_ratio > 0.0
            and curvature_weight == 0.0
            and step == args.curvature_start_step
        ):
            data_norm = _gradient_norm(values["data_loss"], raw)
            curvature_norm = _gradient_norm(values["curvature"], raw)
            curvature_weight = (
                args.curvature_gradient_ratio * data_norm / max(curvature_norm, 1e-20)
            )
            total = values["data_loss"] + curvature_weight * values["curvature"]
            values["total_loss"] = total
            LOGGER.info(
                "Step %d fixed curvature weight %.7g (data %.4g, curvature %.4g)",
                step,
                curvature_weight,
                data_norm,
                curvature_norm,
            )
        total.backward()
        if not bool(torch.isfinite(raw.grad).all()):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()

        should_validate = step == 0 or (step + 1) % args.save_every == 0 or step == args.steps - 1
        if should_validate:
            with torch.enable_grad():
                validation_total, validation = loss_components(step, holdout=True)
            if args.mode == "covariance":
                score = float(validation["cov_loss"].detach().item())
                primary_scale = float(validation["cov_scale"].detach().item())
            elif args.mode == "joint":
                score = float(
                    (validation["h2_loss"] + covariance_weight * validation["cov_loss"])
                    .detach()
                    .item()
                )
                primary_scale = float(validation["h2_scale"].detach().item())
            else:
                score = float(validation["h2_loss"].detach().item())
                primary_scale = float(validation["h2_scale"].detach().item())
            shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
            row: dict[str, float | int] = {
                "step": step,
                "train_total_loss": float(values["total_loss"].detach().item()),
                "train_data_loss": float(values["data_loss"].detach().item()),
                "train_h2_loss": float(values["h2_loss"].detach().item()),
                "train_cov_loss": float(values["cov_loss"].detach().item()),
                "train_cov_relative_mse": float(values["cov_relative_mse"].detach().item()),
                "train_cov_correlation": float(values["cov_correlation"].detach().item()),
                "holdout_score": score,
                "holdout_h2_loss": float(validation["h2_loss"].detach().item()),
                "holdout_cov_loss": float(validation["cov_loss"].detach().item()),
                "holdout_cov_relative_mse": float(
                    validation["cov_relative_mse"].detach().item()
                ),
                "holdout_cov_correlation": float(validation["cov_correlation"].detach().item()),
                "h2_scale": float(validation["h2_scale"].detach().item()),
                "cov_scale": float(validation["cov_scale"].detach().item()),
                "curvature": float(validation["curvature"].detach().item()),
                "curvature_weight": float(curvature_weight),
                "covariance_weight": float(covariance_weight),
                "correction_rms": float(validation["correction_rms"].detach().item()),
                "correction_abs_max": float(validation["correction_abs_max"].detach().item()),
                "shape_max_fraction": float(validation["shape_max_fraction"].detach().item()),
                "elapsed_s": float(time.perf_counter() - started),
                "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
            }
            history.append(row)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if score < best_score:
                best_score = score
                best_step = step
                best_primary_scale = primary_scale
                best_layer = shape.detach().cpu().numpy()
                _save_volume(output_dir / "reconstruction_best", best_layer, depth_index, len(z_values))
            LOGGER.info(
                "step=%03d train=%.6g holdout=%.6g h2=%.6g cov=%.6g corr=%.4f rms=%.4f max=%.4f mem=%.0fMB",
                step,
                row["train_total_loss"],
                score,
                row["holdout_h2_loss"],
                row["holdout_cov_loss"],
                row["holdout_cov_correlation"],
                row["correction_rms"],
                row["correction_abs_max"],
                row["peak_gpu_memory_mb"],
            )

    final_shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
    final_layer = final_shape.detach().cpu().numpy()
    _save_volume(output_dir / "reconstruction_final", final_layer, depth_index, len(z_values))
    if best_layer is None:
        raise RuntimeError("no checkpoint was saved")
    summary = {
        "diagnostic_semantics": (
            "Oracle upper-bound diagnostic: z is fixed to the known 50 um layer and is not a deployable reconstruction prior."
        ),
        "selection": "minimum independent holdout physical loss; FTC was not used for checkpoint selection",
        "mode": args.mode,
        "anchor": args.anchor,
        "anchor_source": anchor_source,
        "bound": args.bound,
        "steps": args.steps,
        "best_step": best_step,
        "best_holdout_score": best_score,
        "initial_holdout_score": initial_holdout_score,
        "frame_split": {
            "seed": args.frame_split_seed,
            "train_indices_zero_based": train_indices.tolist(),
            "holdout_indices_zero_based": holdout_indices.tolist(),
        },
        "best_primary_quadratic_scale": best_primary_scale,
        "covariance_weight": covariance_weight,
        "curvature_weight": curvature_weight,
        "arguments": vars(args),
        "history": history,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Completed: best step %d, score %.7g", best_step, best_score)


if __name__ == "__main__":
    main()
