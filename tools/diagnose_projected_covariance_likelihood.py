"""Fixed-depth diagnostic for projected covariance likelihood.

This is deliberately not part of the production trainer.  It fixes z=50 um and
optimizes only a nonnegative, unit-mass lateral image.  The sensor covariance is
projected to a small, fixed subspace so that the exact model covariance can be
formed without ever constructing the full sensor covariance matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter

from tools.diagnose_cs_information import CachedCs, load_operator
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape


def fixed_sensor_basis(count: int, shape: tuple[int, int], sigma: float, seed: int) -> np.ndarray:
    """Make a data-independent, zero-mean, orthonormal sensor subspace."""
    rng = np.random.default_rng(seed)
    raw = rng.choice((-1.0, 1.0), size=(count, *shape)).astype(np.float32)
    smooth = np.stack([gaussian_filter(item, sigma=sigma, mode="reflect") for item in raw])
    smooth -= smooth.mean(axis=(1, 2), keepdims=True)
    # QR in float64 prevents a poorly conditioned projected covariance from
    # being confused with a non-orthogonal sketch.
    basis, _ = np.linalg.qr(smooth.reshape(count, -1).T.astype(np.float64), mode="reduced")
    return basis.T.reshape(count, *shape).astype(np.float32)


def empirical_projected_covariance(frames: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flat = frames.reshape(frames.shape[0], -1).astype(np.float64, copy=False)
    centered = flat - flat.mean(axis=0, keepdims=True)
    scores = centered @ basis.reshape(basis.shape[0], -1).T.astype(np.float64)
    return (scores.T @ scores / float(frames.shape[0] - 1)).astype(np.float32)


def projected_model_covariance(shape: torch.Tensor, backprojected: torch.Tensor, cs: CachedCs) -> torch.Tensor:
    """Q' H D_g Cs D_g H' Q using one Cs action and a small Gram matrix."""
    back = backprojected[:, 0, 0]
    left = shape[None] * back
    right = cs.action(left)
    covariance = left.flatten(1) @ right.flatten(1).T
    return 0.5 * (covariance + covariance.T)


def gaussian_covariance_divergence(prediction: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    """Per-dimension KL(target Gaussian || prediction Gaussian).

    Both covariances receive the same fixed, dimensionless ridge after a fixed
    target-trace normalization.  No scene-wise/model-wise scale is fitted.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching square matrices")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dimension = prediction.shape[0]
    scale = (torch.trace(target.detach()) / dimension).clamp_min(torch.finfo(target.dtype).tiny)
    eye = torch.eye(dimension, device=prediction.device, dtype=prediction.dtype)
    model = prediction / scale + float(ridge) * eye
    data = target / scale + float(ridge) * eye
    chol_model = torch.linalg.cholesky(model)
    chol_data = torch.linalg.cholesky(data)
    trace_term = torch.trace(torch.cholesky_solve(data, chol_model))
    logdet_model = 2.0 * torch.log(torch.diagonal(chol_model)).sum()
    logdet_data = 2.0 * torch.log(torch.diagonal(chol_data)).sum()
    return 0.5 * (trace_term + logdet_model - logdet_data - dimension) / dimension


def relative_covariance_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(torch.finfo(target.dtype).tiny)


def quadratic_smoothness(shape: torch.Tensor) -> torch.Tensor:
    scaled = shape * shape.numel()
    return (scaled[1:] - scaled[:-1]).square().mean() + (scaled[:, 1:] - scaled[:, :-1]).square().mean()


def gradient_norm(loss: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, parameter, retain_graph=True)[0].norm().detach()


def save_volume(path: Path, layer: torch.Tensor) -> None:
    volume = np.zeros((10, *layer.shape), np.float32)
    volume[4] = layer.detach().cpu().numpy()
    np.save(path.with_suffix(".npy"), volume)
    tifffile.imwrite(path.with_suffix(".tif"), volume, photometric="minisblack")


@torch.no_grad()
def precompute_adjoint(operator, basis: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    return torch.cat([operator.adjoint(basis[start:start + chunk, None])
                      for start in range(0, len(basis), chunk)]).detach()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed output already exists: {output}")

    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset / "metadata.json").read_text())
    cs = CachedCs(torch.from_numpy(np.load(dataset / "cs_kernel.npy")).to(device)
                  * metadata["model_pattern_variance_mean"], (260, 260))
    basis_np = fixed_sensor_basis(args.dimension, (260, 260), args.basis_sigma, args.basis_seed)
    basis = torch.from_numpy(basis_np).to(device)
    back = precompute_adjoint(operator, basis)

    anchor = torch.from_numpy(np.load(dataset / "anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max() * 1e-8)
    anchor /= anchor.sum()
    truth = torch.from_numpy(np.load(dataset / "target.npy")[4]).to(device)
    truth = truth.clamp_min(0); truth /= truth.sum()
    if args.target == "empirical":
        train_target = torch.from_numpy(empirical_projected_covariance(
            np.load(dataset / "train_frames.npy", mmap_mode="r"), basis_np)).to(device)
        holdout_target = torch.from_numpy(empirical_projected_covariance(
            np.load(dataset / "holdout_frames.npy", mmap_mode="r"), basis_np)).to(device)
    else:
        # Positive-control upper bound only: this must never select a deployable model.
        train_target = projected_model_covariance(truth, back, cs).detach()
        holdout_target = train_target

    raw = torch.nn.Parameter(torch.zeros_like(anchor))
    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)

    def scores(shape: torch.Tensor, target: torch.Tensor):
        covariance = projected_model_covariance(shape, back, cs)
        nll = gaussian_covariance_divergence(covariance.double(), target.double(), args.ridge).float()
        mse = relative_covariance_mse(covariance, target)
        return nll, mse, covariance

    shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
    initial_nll, initial_mse, initial_cov = scores(shape, holdout_target)
    train_data, _, _ = scores(shape, train_target)
    smooth = quadratic_smoothness(shape)
    regularizer_weight = 0.0
    if args.regularizer_gradient_ratio > 0:
        regularizer_weight = float(args.regularizer_gradient_ratio * gradient_norm(train_data, raw)
                                   / gradient_norm(smooth, raw).clamp_min(1e-30))
    with torch.no_grad():
        normalized = initial_cov / (torch.trace(holdout_target) / args.dimension)
        eig = torch.linalg.eigvalsh(normalized.double())
        condition = float(eig.max() / eig.clamp_min(args.ridge).min())

    def objective(shape: torch.Tensor, target: torch.Tensor):
        nll, mse, _ = scores(shape, target)
        data = nll if args.loss == "nll" else mse
        return data + regularizer_weight * quadratic_smoothness(shape), nll, mse

    initial_score = float(initial_nll if args.loss == "nll" else initial_mse)
    best_score, best_step = initial_score, -1
    save_volume(output / "reconstruction_initial", anchor)
    save_volume(output / "reconstruction_best", anchor)
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
        loss, train_nll, train_mse = objective(shape, train_target)
        loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"Invalid loss or gradient at step {step}")
        grad = float(raw.grad.norm())
        optimizer.step()
        if step == 0 or (step + 1) % args.save_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
                val_loss, val_nll, val_mse = objective(shape, holdout_target)
                score = float(val_nll if args.loss == "nll" else val_mse)
            row = {"step": step, "train_objective": float(loss.detach()),
                   "train_nll": float(train_nll.detach()), "train_mse": float(train_mse.detach()),
                   "holdout_selection_score": score, "holdout_nll": float(val_nll),
                   "holdout_mse": float(val_mse), "gradient_norm": grad,
                   "correction_rms": float(correction.square().mean().sqrt()),
                   "elapsed_s": time.perf_counter() - started,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}
            history.append(row); save_volume(output / f"reconstruction_step{step:03d}", shape)
            if score < best_score:
                best_score, best_step = score, step
                save_volume(output / "reconstruction_best", shape)
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
    save_volume(output / "reconstruction_final", shape)
    summary = {"arguments": vars(args), "scope": "fixed true 50 um lateral diagnostic only",
               "selection": "independent holdout frames; truth metrics never used for empirical selection",
               "target_warning": ("oracle is a solver/information upper bound and not an estimator"
                                  if args.target == "oracle" else None),
               "physics": "exact projected stationary Cs covariance; no fitted covariance amplitude",
               "regularizer_weight": regularizer_weight, "initial_holdout_nll": float(initial_nll),
               "initial_holdout_mse": float(initial_mse), "initial_model_condition": condition,
               "best_step": best_step, "best_holdout_score": best_score, "history": history}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--output", required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--loss", choices=("nll", "mse"), required=True)
    parser.add_argument("--target", choices=("empirical", "oracle"), default="empirical")
    parser.add_argument("--dimension", type=int, default=128); parser.add_argument("--basis-sigma", type=float, default=4.0)
    parser.add_argument("--basis-seed", type=int, default=20260911); parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bound", type=float, default=0.5); parser.add_argument("--regularizer-gradient-ratio", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200); parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01); parser.add_argument("--seed", type=int, default=20260901)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
