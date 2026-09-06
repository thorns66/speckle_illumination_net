"""Exact second/fourth phase moments of the finite random-phase speckle generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from physics.speckle_oracle import SpeckleGeneratorConfig, circular_pupil


def exact_moments(config, lag=(0, 0)):
    """For independent unit-modulus phases: Cov(Ia,Ib)=|Gamma_ab|^2-sum_j |Pa_j Pb_j|^2."""
    pupil = circular_pupil(config, device="cpu", dtype=torch.float64)
    impulse = torch.fft.ifft2(torch.fft.ifftshift(pupil))
    side, pad = config.sampling, config.sampling // 2
    mask = F.pad(torch.ones((side, side), dtype=torch.float64), (pad, pad, pad, pad))
    mask_fft = torch.fft.fft2(mask)

    def convolve(kernel):
        return torch.fft.ifft2(mask_fft * torch.fft.fft2(kernel))

    power = impulse.abs().square()
    shifted = torch.roll(impulse, shifts=(-lag[0], -lag[1]), dims=(-2, -1))
    mean = convolve(power).real
    gamma = convolve(impulse * shifted.conj())
    fourth_correction = convolve(power * shifted.abs().square()).real
    covariance = gamma.abs().square() - fourth_correction
    crop = (slice(pad, pad+side), slice(pad, pad+side))
    return mean[crop].contiguous(), covariance[crop].contiguous(), fourth_correction[crop].contiguous()


def validate_small_generator(count=8192):
    config = SpeckleGeneratorConfig(sampling=16, seed=79)
    pupil = circular_pupil(config, device="cpu", dtype=torch.float64)
    rng = torch.Generator().manual_seed(config.seed)
    samples = []
    for start in range(0, count, 128):
        phase = torch.rand((min(128, count-start), 16, 16), generator=rng, dtype=torch.float64) * (2*torch.pi)
        source = F.pad(torch.polar(torch.ones_like(phase), phase), (8, 8, 8, 8))
        field = torch.fft.ifft2(torch.fft.fft2(source) * torch.fft.ifftshift(pupil))
        samples.append(field.abs().square()[:, 8:24, 8:24])
    samples = torch.cat(samples)
    mean, var, _ = exact_moments(config)
    centered = samples - samples.mean(0)
    empirical_variance = centered.square().sum(0)/(count-1)
    _, lag1, _ = exact_moments(config, (1, 0))
    empirical_lag = (centered[:, :-1] * centered[:, 1:]).sum(0)/(count-1)
    errors = {
        "mean_relative_l2": float((samples.mean(0)-mean).norm()/mean.norm()),
        "variance_relative_l2": float((empirical_variance-var).norm()/var.norm()),
        "lag1_relative_l2": float((empirical_lag-lag1[:-1]).norm()/lag1[:-1].norm()),
    }
    if errors["mean_relative_l2"] > 0.04 or max(errors["variance_relative_l2"], errors["lag1_relative_l2"]) > 0.08:
        raise AssertionError(errors)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--output", default="outputs/linear_float_oracle_50um/cs_information/stationarity")
    args = parser.parse_args()
    torch.set_num_threads(4)
    validation = validate_small_generator()
    metadata = json.loads((Path(args.dataset)/"metadata.json").read_text())
    scale = metadata["speckle_system_mean_before_scaling"]
    mean, variance, correction = exact_moments(SpeckleGeneratorConfig())
    mean = mean / scale
    variance = variance / scale**2
    correction = correction / scale**2
    assumed = metadata["model_pattern_variance_mean"]
    yy, xx = np.mgrid[:260, :260]
    radius = np.hypot(xx-4.5, yy-5.5)
    masks = {
        "all": np.ones((260, 260), bool),
        "center_100_square": (yy >=80)&(yy<180)&(xx>=80)&(xx<180),
        "corner_10_square": (yy<10)&(xx<10),
        "star_annulus_r10_60": (xx>=4.5)&(yy>=5.5)&(radius>=10)&(radius<=60),
        "edge_first_2_pixels": (xx<2)|(yy<2)|(xx>=258)|(yy>=258),
    }
    regions = {}
    for name, mask in masks.items():
        values = variance.numpy()[mask]/assumed
        regions[name] = {"pixels": int(mask.sum()), "mean_illumination": float(mean.numpy()[mask].mean()),
                         "true_to_stationary_variance_mean": float(values.mean()),
                         "ratio_min": float(values.min()), "ratio_max": float(values.max()),
                         "variance_relative_l2_mismatch": float(np.linalg.norm(values-1)/np.linalg.norm(values))}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"exact_illumination_moments.npz", mean=mean.numpy(), variance=variance.numpy(),
                        phase_fourth_cumulant_correction=correction.numpy())
    report = {"small_generator_validation": validation, "fixed_system_scale": scale,
              "assumed_stationary_variance": assumed, "regions": regions,
              "formula": "Gamma_ab=sum_j M_j P_(a-j) conj(P_(b-j)); Cov(Ia,Ib)=abs(Gamma_ab)^2-sum_j M_j abs(P_(a-j)P_(b-j))^2",
              "scope": "Exact object illumination moments; not a sensor loss or reconstruction result."}
    (output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
