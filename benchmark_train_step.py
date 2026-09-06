from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.checkpoint import checkpoint

from losses.self_supervised_losses import TaylorH2VarianceModel
from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from train_volume import _model_from_config


def timed(function, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one 260x260x10, N=100 optimization step")
    parser.add_argument("--config", default="configs/v1_100_simulation.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=260)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", default="train_step_benchmark.json")
    parser.add_argument("--toy", action="store_true", help="Validate timing code with a small synthetic PSF")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device or config["runtime"]["device"])
    psf_config = config["psf"]
    if args.toy:
        h = torch.rand((10, 3, 3, 9, 9), device=device)
    else:
        psf_path = Path(psf_config["H_path"])
        if not psf_path.is_absolute():
            psf_path = config_path.parent.parent / psf_path
        psf = load_psf(
            psf_path,
            psf_config["z_values_um"],
            h_variable_name=psf_config["H_variable_name"],
            ht_variable_name=psf_config["Ht_variable_name"],
        )
        h = torch.from_numpy(psf.H).to(device=device, dtype=torch.float32)
        del psf
    gc.collect()
    operator = LFMOperator(
        h,
        mode=config["runtime"]["operator_mode"],
        phase_chunk_size=config["runtime"]["operator_phase_chunk_size"],
    )
    model = _model_from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimization"]["lr_network"])
    z_count = operator.num_depths
    shape = (1, 1, z_count, args.height, args.width)
    f_var = torch.rand(shape, device=device)
    g_mean = torch.rand(shape, device=device)
    residual = torch.randn((1, args.frames, 1, args.height, args.width), device=device)
    residual -= residual.mean(dim=1, keepdim=True)
    z_values = torch.tensor(psf_config["z_values_um"], device=device)
    measured_mean = torch.rand((1, 1, args.height, args.width), device=device)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])
    physics_use_checkpoint = bool(config["runtime"].get("physics_use_checkpoint", False))
    model.initialize_beta(f_var, measured_mean, operator)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    output, network_time = timed(lambda: model(f_var, g_mean, residual, z_values), device)
    if physics_use_checkpoint:
        mean_function = lambda: checkpoint(
            operator,
            output.reconstruction,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        variance_function = lambda: checkpoint(
            variance_model,
            output.reconstruction,
            measured_mean,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        mean_function = lambda: operator(output.reconstruction)
        variance_function = lambda: variance_model(output.reconstruction, measured_mean)
    pred_mean, h_time = timed(mean_function, device)
    pred_var, h2_time = timed(variance_function, device)
    loss = pred_mean.mean() + pred_var.mean() + output.reconstruction.mean() * 1e-5
    _, backward_time = timed(lambda: loss.backward(), device)
    _, optimizer_time = timed(lambda: optimizer.step(), device)
    total_time = time.perf_counter() - total_start
    report = {
        "device": str(device),
        "volume_shape": list(shape),
        "num_frames": args.frames,
        "toy_psf": args.toy,
        "physics_use_checkpoint": physics_use_checkpoint,
        "network_forward_s": network_time,
        "H_forward_s": h_time,
        "H2_forward_s": h2_time,
        "backward_s": backward_time,
        "optimizer_step_s": optimizer_time,
        "total_step_s": total_time,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
