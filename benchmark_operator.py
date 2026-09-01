from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf


def measure(function, device: torch.device, repeats: int) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark H and H**2 LFM operators")
    parser.add_argument("--config", default="configs/v1_100_simulation.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=260)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="operator_benchmark.json")
    parser.add_argument("--toy", action="store_true", help="Validate timing code with a small synthetic PSF")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device or config["runtime"]["device"])
    root = config_path.parent.parent
    psf_config = config["psf"]
    if args.toy:
        h = torch.rand((10, 3, 3, 9, 9), device=device)
    else:
        psf_path = Path(psf_config["H_path"])
        if not psf_path.is_absolute():
            psf_path = root / psf_path
        psf = load_psf(
            psf_path,
            psf_config["z_values_um"],
            h_variable_name=psf_config["H_variable_name"],
            ht_variable_name=psf_config["Ht_variable_name"],
            psf_z_all_um=psf_config.get("psf_z_all_um"),
            depth_unit=psf_config.get("depth_unit", "auto"),
        )
        h = torch.from_numpy(psf.H).to(device=device, dtype=torch.float32)
        del psf
    gc.collect()
    operator = LFMOperator(
        h,
        mode=config["runtime"]["operator_mode"],
        phase_chunk_size=config["runtime"]["operator_phase_chunk_size"],
    )
    volume = torch.rand((1, 1, operator.num_depths, args.height, args.width), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        h_mean, h_min = measure(lambda: operator(volume), device, args.repeats)
        h2_mean, h2_min = measure(lambda: operator.forward_squared(volume.square()), device, args.repeats)
    report = {
        "device": str(device),
        "volume_shape": list(volume.shape),
        "psf_shape_canonical": list(operator.H.shape),
        "operator_mode": operator.mode,
        "phase_chunk_size": operator.phase_chunk_size,
        "toy_psf": args.toy,
        "T_H_mean_s": h_mean,
        "T_H_min_s": h_min,
        "T_H2_mean_s": h2_mean,
        "T_H2_min_s": h2_min,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
