from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml


def _prototype():
    spec = importlib.util.spec_from_file_location(
        "linear_float_oracle", "/tmp/linear_float_oracle.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the validated linear-float prototype")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nested train/holdout frame-count datasets for fixed-depth Cs diagnostics."
    )
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--counts", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--model-patterns", type=int, default=2048)
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--rl-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260940)
    args = parser.parse_args()

    counts = sorted(set(args.counts))
    if any(count < 2 or count > args.max_frames for count in counts):
        raise ValueError("Every count must be between 2 and max-frames")

    prototype = _prototype()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    operator = prototype._operator(config_path, config, device)

    generator_config = prototype.SpeckleGeneratorConfig(seed=args.seed)
    total = 2 * args.max_frames + args.model_patterns
    patterns = prototype._raw_speckles(
        total, generator_config, device, args.pattern_batch_size
    )
    model_start = 2 * args.max_frames
    system_scale = patterns[model_start:].mean()
    patterns /= system_scale

    target_np = prototype._target(patterns.shape[-1])
    target = torch.from_numpy(target_np).to(device)
    train = prototype._forward_frames(
        operator, target, patterns[: args.max_frames], args.forward_batch_size
    )
    holdout = prototype._forward_frames(
        operator,
        target,
        patterns[args.max_frames : model_start],
        args.forward_batch_size,
    )
    model_patterns = patterns[model_start:].to(device)
    cs_variance = float(model_patterns.var(dim=0, unbiased=True).mean().item())
    cs_kernel = prototype.stationary_covariance_from_ensemble(model_patterns).cpu().numpy()

    target_volume = np.zeros((10, *target_np.shape), dtype=np.float32)
    target_volume[4] = target_np

    for count in counts:
        output = output_root / f"n{count}"
        output.mkdir(parents=True, exist_ok=True)
        measured_variance = torch.from_numpy(train[:count].var(axis=0, ddof=1)).to(device)
        anchor = prototype._rl_anchor(
            operator, measured_variance, args.rl_iterations
        ).cpu().numpy()
        anchor_volume = np.zeros_like(target_volume)
        anchor_volume[4] = anchor

        np.save(output / "train_frames.npy", train[:count])
        np.save(output / "holdout_frames.npy", holdout[:count])
        np.save(output / "cs_kernel.npy", cs_kernel.astype(np.float32))
        np.save(output / "target.npy", target_volume)
        np.save(output / "anchor_rl3.npy", anchor_volume)
        tifffile.imwrite(output / "target.tif", target_volume, photometric="minisblack")
        tifffile.imwrite(output / "anchor_rl3.tif", anchor_volume, photometric="minisblack")

        metadata = {
            "semantics": "nested linear floating-point sensor data; one system-wide illumination scale",
            "train_frames": count,
            "holdout_frames": count,
            "maximum_generated_frames_per_split": args.max_frames,
            "nested_counts": counts,
            "model_patterns": args.model_patterns,
            "speckle_seed": args.seed,
            "speckle_system_mean_before_scaling": float(system_scale.item()),
            "model_pattern_mean": float(model_patterns.mean().item()),
            "model_pattern_variance_mean": cs_variance,
            "per_frame_sensor_normalization": False,
            "quantization": False,
            "noise": "none",
            "rl_anchor_iterations": args.rl_iterations,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps({"dataset": str(output), **metadata}), flush=True)


if __name__ == "__main__":
    main()
