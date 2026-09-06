"""Refresh sensor probes and test fixed-global versus per-probe loss weights."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.diagnose_cs_information import (
    CachedCs, bounded_anchor_shape, load_operator, model_action,
    probe_seed, save_volume, vector_loss, _lowpass_probes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization", choices=["per_probe", "global"], required=True)
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed result exists")
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((root / "metadata.json").read_text())
    cs = CachedCs(torch.from_numpy(np.load(root / "cs_kernel.npy")).to(device) * metadata["model_pattern_variance_mean"], (260, 260))
    anchor = torch.from_numpy(np.load(root / "anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max() * 1e-8)
    anchor /= anchor.sum()
    raw = torch.nn.Parameter(torch.zeros_like(anchor))
    optimizer = torch.optim.Adam([raw], lr=0.01)

    frames = {}
    for name in ("train", "holdout"):
        data = torch.from_numpy(np.load(root / f"{name}_frames.npy")).to(device).flatten(1)
        frames[name] = data - data.mean(0, keepdim=True)

    @torch.no_grad()
    def probes(count, domain, step=0):
        q = torch.from_numpy(_lowpass_probes(count, (260, 260), sigma=8,
                            seed=probe_seed(args.seed, domain, step, 0) % (2**32))).to(device)
        return q

    @torch.no_grad()
    def empirical(q, name):
        data = frames[name]
        return ((q.flatten(1) @ data.T) @ data / (len(data)-1)).reshape_as(q)

    qh = probes(32, "holdout_sensor")
    target_h = empirical(qh, "holdout")
    with torch.no_grad():
        back_h = torch.cat([operator.adjoint(qh[i:i+4, None]) for i in range(0, 32, 4)])
        train_energy = empirical(probes(256, "train_normalizer"), "train").flatten(1).square().sum(1).mean().detach()
        holdout_energy = target_h.flatten(1).square().sum(1).mean().detach()

    def score(prediction, target, energy):
        per_probe, corr = vector_loss(prediction, target)
        global_score = (prediction - target).flatten(1).square().sum(1).mean() / energy.clamp_min(torch.finfo(energy.dtype).tiny)
        selected = per_probe if args.normalization == "per_probe" else global_score
        return selected, per_probe, global_score, corr

    @torch.no_grad()
    def validate(shape):
        values = [score(model_action(shape, back_h[i:i+4], cs, operator), target_h[i:i+4], holdout_energy)
                  for i in range(0, 32, 4)]
        return [float(torch.stack([item[j] for item in values]).mean()) for j in range(4)]

    initial = validate(anchor)
    best, best_step = initial[0], -1
    history = []
    save_volume(output / "reconstruction_best", anchor)
    save_volume(output / "reconstruction_initial", anchor)
    print(json.dumps({"initial_holdout": initial, "normalization": args.normalization}), flush=True)
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
        q = probes(4, "train_sensor_refresh", step)
        target = empirical(q, "train")
        with torch.no_grad():
            back = operator.adjoint(q[:, None]).detach()
        loss, _, _, _ = score(model_action(shape, back, cs, operator), target, train_energy)
        loss.backward()
        if not torch.isfinite(raw.grad).all():
            raise FloatingPointError(f"Invalid gradient at {step}")
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
                val, per, global_score, corr = validate(shape)
            row = {"step": step, "train_loss": float(loss.detach()), "holdout_loss": val,
                   "holdout_per_probe": per, "holdout_global": global_score,
                   "holdout_correlation": corr, "elapsed_s": time.perf_counter() - started,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}
            history.append(row)
            save_volume(output / f"reconstruction_step{step:03d}", shape)
            if val < best:
                best, best_step = val, step
                save_volume(output / "reconstruction_best", shape)
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        shape, _ = bounded_anchor_shape(anchor, raw, args.bound)
    save_volume(output / "reconstruction_final", shape)
    summary = {"arguments": vars(args), "frames": metadata["train_frames"], "initial_holdout_loss": initial[0],
               "best_step": best_step, "best_holdout_loss": best, "history": history,
               "semantics": "fresh sensor probes per step, fixed calibrated Cs, no free photometric/covariance scale",
               "train_fixed_loss_denominator": float(train_energy)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
