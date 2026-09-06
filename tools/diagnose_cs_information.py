"""Isolate finite-probe bias from missing off-diagonal covariance information.

Research diagnostic only: the depth is fixed at 50 um, never inferred.
No imports from ephemeral /tmp prototypes and no scene-wise covariance scaling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint

from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.scan_covariance_sketch_depth import _empirical_action, _lowpass_probes


def probe_seed(master: int, domain: str, step: int, chunk: int) -> int:
    value = f"cs-information-v1:{master}:{domain}:{step}:{chunk}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little") & ((1 << 63) - 1)


class CachedCs:
    def __init__(self, kernel: torch.Tensor, shape: tuple[int, int]):
        self.shape = shape
        kh, kw = kernel.shape
        self.fft_shape = (shape[0] + kh - 1, shape[1] + kw - 1)
        grid = F.pad(kernel, (0, self.fft_shape[1] - kw, 0, self.fft_shape[0] - kh))
        grid = torch.roll(grid, (-(kh // 2), -(kw // 2)), (-2, -1))
        self.spectrum = torch.fft.rfft2(grid).real.clamp_min(0)
        self.sqrt_spectrum = self.spectrum.sqrt()

    def action(self, value: torch.Tensor) -> torch.Tensor:
        result = torch.fft.irfft2(
            torch.fft.rfft2(value, s=self.fft_shape) * self.spectrum, s=self.fft_shape
        )
        return result[..., : self.shape[0], : self.shape[1]]

    @torch.no_grad()
    def draw(self, count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.spectrum.device).manual_seed(seed)
        white = torch.randn((count, *self.fft_shape), generator=generator,
                            device=self.spectrum.device, dtype=self.spectrum.dtype)
        sample = torch.fft.irfft2(torch.fft.rfft2(white) * self.sqrt_spectrum, s=self.fft_shape)
        return sample[..., : self.shape[0], : self.shape[1]]


def blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return value
    radius = max(1, math.ceil(4 * sigma))
    x = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    weights = torch.exp(-0.5 * (x / sigma).square())
    weights /= weights.sum()
    shape = value.shape
    batch = value.reshape(-1, 1, *shape[-2:])
    batch = F.pad(batch, (radius, radius, radius, radius), mode="reflect")
    batch = F.conv2d(batch, weights[None, None, None, :])
    batch = F.conv2d(batch, weights[None, None, :, None])
    return batch.reshape(shape)


def observation_loss(observations, target, mask, unbiased: bool):
    """All-pairs U statistic equals MSE(sample mean) - sample variance / K."""
    count = observations.shape[0]
    if count < 2:
        raise ValueError("At least two independent observations required")
    scale = target[mask].square().mean().sqrt().clamp_min(torch.finfo(target.dtype).tiny)
    values = observations[:, mask] / scale
    measured = target[mask] / scale
    mean = values.mean(dim=0)
    self_loss = (mean - measured).square().mean()
    bias = values.var(dim=0, unbiased=True).mean() / count
    return (self_loss - bias if unbiased else self_loss), self_loss, bias


def vector_loss(prediction, target):
    pred, data = prediction.flatten(1), target.flatten(1)
    # No detached scene/bank gain. The known system scale remains fixed.
    tiny = torch.finfo(pred.dtype).tiny
    mse = ((pred - data).square().sum(1) / data.square().sum(1).clamp_min(tiny)).mean()
    px, tx = pred - pred.mean(1, keepdim=True), data - data.mean(1, keepdim=True)
    corr = ((px * tx).sum(1) / (px.norm(dim=1) * tx.norm(dim=1)).clamp_min(tiny)).mean()
    return mse, corr


def model_action(shape, backprojected, cs, operator, checkpoint_enabled=True):
    right = shape[None] * backprojected[:, 0, 0]
    volume = (shape[None] * cs.action(right))[:, None, None]
    return (checkpoint(operator, volume, use_reentrant=False)
            if checkpoint_enabled and torch.is_grad_enabled() else operator(volume))[:, 0]


def observations(shape, cs, operator, *, count, batch, master, domain, step, sigma,
                 checkpoint_enabled=True):
    outputs = []
    for index, start in enumerate(range(0, count, batch)):
        probes = cs.draw(min(batch, count - start), probe_seed(master, domain, step, index))
        volume = (shape * probes)[:, None, None]
        sensor = (checkpoint(operator, volume, use_reentrant=False)
                  if checkpoint_enabled and torch.is_grad_enabled() else operator(volume))[:, 0]
        outputs.append(blur(sensor.square(), sigma))
    return torch.cat(outputs)


def load_operator(config_path, device):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    psf = config["psf"]
    h_path = Path(psf["H_path"])
    if not h_path.is_absolute():
        h_path = config_path.parent.parent / h_path
    loaded = load_psf(h_path, [50.0], h_variable_name=psf["H_variable_name"],
                      ht_variable_name=psf["Ht_variable_name"], psf_z_all_um=psf.get("psf_z_all_um"),
                      depth_unit=psf.get("depth_unit", "auto"), load_h=True, load_ht=False)
    return LFMOperator(torch.from_numpy(loaded.H).to(device=device, dtype=torch.float32),
                       mode="optimized", phase_chunk_size=16)


def save_volume(path, shape):
    volume = np.zeros((10, *shape.shape), np.float32)
    volume[4] = shape.detach().cpu().numpy()
    np.save(path.with_suffix(".npy"), volume)
    tifffile.imwrite(path.with_suffix(".tif"), volume, photometric="minisblack")


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("A completed result already exists; choose a new output")
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset / "metadata.json").read_text())
    cs = CachedCs(torch.from_numpy(np.load(dataset / "cs_kernel.npy")).to(device)
                  * metadata["model_pattern_variance_mean"], (260, 260))
    anchor = torch.from_numpy(np.load(dataset / "anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max() * 1e-8)
    anchor /= anchor.sum()
    raw = torch.nn.Parameter(torch.zeros_like(anchor))
    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)

    if args.mode == "vector":
        banks = {}
        for domain, count, filename in (("train_sensor", args.sensor_probes, "train_frames.npy"),
                                        ("holdout_sensor", args.eval_sensor_probes, "holdout_frames.npy")):
            # scipy/numpy's RandomState accepts a 32-bit seed.
            seeds = probe_seed(args.seed, domain, 0, 0) % (2**32)
            probes_np = _lowpass_probes(count, (260, 260), sigma=args.probe_sigma, seed=seeds)
            frames = np.load(dataset / filename, mmap_mode="r")
            target = torch.from_numpy(_empirical_action(frames, probes_np)).to(device)
            with torch.no_grad():
                back = torch.cat([operator.adjoint(torch.from_numpy(probes_np[i:i+4]).to(device)[:, None])
                                  for i in range(0, count, 4)]).detach()
            banks[domain] = (back, target)

        def data_loss(shape, step):
            back, target = banks["train_sensor"]
            start = (step * args.probes_per_step) % len(back)
            indices = (torch.arange(args.probes_per_step, device=device) + start) % len(back)
            predicted = model_action(shape, back[indices], cs, operator)
            loss, corr = vector_loss(predicted, target[indices])
            return loss, {"train_mse": float(loss.detach()), "train_correlation": float(corr.detach())}

        @torch.no_grad()
        def validate(shape):
            back, target = banks["holdout_sensor"]
            scores, correlations = [], []
            for start in range(0, len(back), 4):
                prediction = model_action(shape, back[start:start+4], cs, operator)
                loss, corr = vector_loss(prediction, target[start:start+4])
                scores.append(loss)
                correlations.append(corr)
            return float(torch.stack(scores).mean()), float(torch.stack(correlations).mean())
    else:
        targets, masks = {}, {}
        for name, filename in (("train", "train_frames.npy"), ("holdout", "holdout_frames.npy")):
            frames = np.load(dataset / filename, mmap_mode="r")
            targets[name] = blur(torch.from_numpy(frames.var(0, ddof=1)).to(device), args.sigma)
            mean = torch.from_numpy(frames.mean(0)).to(device)
            masks[name] = mean > 0.005 * mean.max()

        def data_loss(shape, step):
            samples = observations(shape, cs, operator, count=args.object_probes, batch=args.batch,
                                   master=args.seed, domain="train_object", step=step, sigma=args.sigma)
            loss, self_loss, bias = observation_loss(samples, targets["train"], masks["train"],
                                                     unbiased=args.mode == "ustat")
            return loss, {"train_self_mse": float(self_loss.detach()), "estimated_self_noise_bias": float(bias.detach())}

        @torch.no_grad()
        def validate(shape):
            samples = observations(shape, cs, operator, count=args.eval_object_probes, batch=args.batch,
                                   master=args.seed, domain="holdout_object", step=0, sigma=args.sigma)
            prediction = samples.mean(0)[masks["holdout"]][None]
            target = targets["holdout"][masks["holdout"]][None]
            loss, corr = vector_loss(prediction, target)
            return float(loss), float(corr)

    initial_loss, initial_corr = validate(anchor)
    best_loss, best_step = initial_loss, -1
    save_volume(output / "reconstruction_best", anchor)
    save_volume(output / "reconstruction_initial", anchor)
    print(json.dumps({"initial_loss": initial_loss, "initial_correlation": initial_corr,
                      "mode": args.mode, "frames": metadata["train_frames"]}), flush=True)
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
        loss, extra = data_loss(shape, step)
        loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"Invalid objective/gradient at step {step}")
        grad_norm = float(raw.grad.norm())
        optimizer.step()
        if step == 0 or (step + 1) % args.save_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
                val, corr = validate(shape)
            row = {"step": step, "train_objective": float(loss.detach()), "gradient_norm": grad_norm,
                   "holdout_loss": val, "holdout_correlation": corr,
                   "correction_rms": float(correction.square().mean().sqrt()),
                   "elapsed_s": time.perf_counter() - started,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, **extra}
            history.append(row)
            save_volume(output / f"reconstruction_step{step:03d}", shape)
            if val < best_loss:
                best_loss, best_step = val, step
                save_volume(output / "reconstruction_best", shape)
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
    save_volume(output / "reconstruction_final", shape)
    summary = {"arguments": vars(args), "frames": metadata["train_frames"],
               "selection": "same-N independent frames and disjoint holdout probes; no truth-based selection",
               "physics": "fixed calibrated stationary Cs, no free covariance amplitude",
               "initial_holdout_loss": initial_loss, "best_step": best_step, "best_holdout_loss": best_loss,
               "seed_domains": ["train_object", "holdout_object", "final_object", "train_sensor", "holdout_sensor", "final_sensor"],
               "history": history}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mode", choices=["self", "ustat", "vector"], required=True)
    parser.add_argument("--object-probes", type=int, default=16)
    parser.add_argument("--eval-object-probes", type=int, default=512)
    parser.add_argument("--sensor-probes", type=int, default=32)
    parser.add_argument("--eval-sensor-probes", type=int, default=32)
    parser.add_argument("--probes-per-step", type=int, default=4)
    parser.add_argument("--probe-sigma", type=float, default=8)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260901)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
