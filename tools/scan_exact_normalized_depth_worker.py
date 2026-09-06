from __future__ import annotations

import argparse
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
    _load_f_var,
    _load_raw_depth,
    _lowpass_probes,
    _resolve,
    _sketch_loss,
)
from tools.scan_exact_normalized_speckle_forward import (
    _mean_gain,
    _normalized_sensor_statistics,
)


LOGGER = logging.getLogger("scan_exact_normalized_depth_worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-depth 4096-speckle oracle worker")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--candidate-depth", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ensemble-size", type=int, default=4096)
    parser.add_argument("--convergence-size", type=int, default=2048)
    parser.add_argument("--speckle-seed", type=int, default=20260904)
    parser.add_argument("--speckle-generation-batch-size", type=int, default=32)
    parser.add_argument("--forward-batch-size", type=int, default=64)
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-sigma", type=float, default=16.0)
    parser.add_argument("--probe-seed", type=int, default=20260901)
    parser.add_argument("--phase-chunk-size", type=int, default=16)
    parser.add_argument("--system-scale", type=float, default=0.9184511051397161)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    frames = _load_raw_depth(config_path, config, 50)
    measured_mean = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    fvar_template = _common_template(_load_f_var(config_path, config, 50))
    e0 = np.load(config_path.parent.parent / "outputs/depth50_n100_no_mean_loss/reconstruction_best.npy")
    if e0.ndim == 5:
        e0 = e0[0, 0]
    e0_layer = np.maximum(e0[4], 0.0)
    templates = {
        "fvar_projection": fvar_template,
        "e0_true_layer": (e0_layer / e0_layer.sum()).astype(np.float32),
    }
    probes = _lowpass_probes(
        args.num_probes,
        tuple(int(value) for value in frames.shape[-2:]),
        sigma=args.probe_sigma,
        seed=args.probe_seed,
    )
    empirical_full = _empirical_action(frames, probes)
    permutation = np.random.default_rng(args.probe_seed + 2000).permutation(100)
    empirical_splits = [
        _empirical_action(frames[indices], probes)
        for indices in (permutation[:50], permutation[50:])
    ]

    generator_config = SpeckleGeneratorConfig(seed=args.speckle_seed)
    LOGGER.info("Depth %d: generating %d speckles", args.candidate_depth, args.ensemble_size)
    patterns = generate_speckle_ensemble(
        args.ensemble_size,
        generator_config,
        device=device,
        batch_size=args.speckle_generation_batch_size,
    )
    psf_config = config["psf"]
    psf_data = load_psf(
        _resolve(config_path, psf_config["H_path"]),
        [args.candidate_depth],
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
    results: list[dict[str, object]] = []
    for template_name, template in templates.items():
        statistics = _normalized_sensor_statistics(
            operator,
            template,
            patterns,
            probes,
            batch_size=args.forward_batch_size,
            convergence_size=args.convergence_size,
        )
        predicted_mean, action = statistics[args.ensemble_size]
        gain, mean_error = _mean_gain(predicted_mean, measured_mean)
        prediction = args.system_scale * gain**2 * action
        convergence_error = float(
            np.linalg.norm(statistics[args.convergence_size][1] - action)
            / max(np.linalg.norm(action), np.finfo(np.float64).eps)
        )
        losses = {
            "full": _sketch_loss(
                prediction, empirical_full, correlation_weight=args.correlation_weight
            ),
            "split_a": _sketch_loss(
                prediction, empirical_splits[0], correlation_weight=args.correlation_weight
            ),
            "split_b": _sketch_loss(
                prediction, empirical_splits[1], correlation_weight=args.correlation_weight
            ),
        }
        row: dict[str, object] = {
            "candidate_depth_um": args.candidate_depth,
            "template": template_name,
            "ensemble_size": args.ensemble_size,
            "convergence_size": args.convergence_size,
            "action_convergence_relative_error": convergence_error,
            "photometric_gain": gain,
            "mean_relative_error": mean_error,
        }
        for subset, values in losses.items():
            row.update({f"{subset}_{key}": value for key, value in values.items()})
        results.append(row)
        LOGGER.info("Depth %d/%s result: %s", args.candidate_depth, template_name, row)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generator": generator_config.to_dict(),
                "system_scale": args.system_scale,
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
