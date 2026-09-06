"""Matrix-free covariance sampling risk, not a bound on image reconstruction.

For iid centered x with covariance C, unbiased sample covariance S_N obeys
E||S_N-C||_F^2 = (E||x||^4-||C||_F^2)/N
                 + ((tr C)^2+||C||_F^2)/(N*(N-1)).
We estimate the radial fourth moment from existing frames using the exact
population mean. Trace/Frobenius moments use independent Rademacher probes.
No image fitting, mean backprop, scene scale fitting, or hidden illumination.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.diagnose_cs_information import load_operator, model_action


def sampling_risk(count, trace, frobenius_squared, radial_fourth):
    if count < 2 or frobenius_squared <= 0 or radial_fourth < 0:
        raise ValueError("Invalid covariance moment or sample count")
    return ((radial_fourth - frobenius_squared) / count
            + (trace**2 + frobenius_squared) / (count * (count - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probes", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    root = Path("outputs/linear_float_oracle_50um")
    cs = DenseFinitePhaseCs.load(root / "exact_population_cs", device=device, system_mean=1.)
    operator = load_operator(Path("configs/depth50_n100_no_mean_loss.yaml").resolve(), device)
    truth_np = np.load(root / "dataset/target.npy")
    truth = torch.from_numpy(truth_np[4]).to(device); truth /= truth.sum()
    q = torch.from_numpy(np.random.default_rng(20260987).choice((-1., 1.), (args.probes, 260, 260)).astype(np.float32) / 260).to(device)
    with torch.no_grad():
        raw_actions = []
        for start in range(0, len(q), 4):
            back = operator.adjoint(q[start:start+4, None])
            raw_actions.append(model_action(truth, back, cs, operator))
        raw_actions = torch.cat(raw_actions)
        mean_illumination = torch.from_numpy(np.load(root / "exact_population_cs/mean_raw.npy")).to(device)
        mean_image_raw = operator((truth * mean_illumination)[None, None, None])[0, 0]
    pixels = 260 * 260
    probe_trace = (q.double() * raw_actions.double()).sum((1, 2)).cpu().numpy() * pixels
    probe_frobenius = raw_actions.double().square().sum((1, 2)).cpu().numpy() * pixels
    trace_raw, frobenius_raw = float(probe_trace.mean()), float(probe_frobenius.mean())
    rng = np.random.default_rng(20260988)
    boot = rng.integers(len(q), size=(2000, len(q)))
    effective_ranks = probe_trace[boot].mean(1)**2 / probe_frobenius[boot].mean(1)
    reports = []
    with torch.no_grad():
        for count in (100, 200, 400, 800):
            dataset = root / ("dataset" if count == 100 else f"frame_count_nested/n{count}")
            metadata = json.loads((dataset / "metadata.json").read_text())
            if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
                raise ValueError("Linear float data required")
            np.testing.assert_allclose(np.load(dataset / "target.npy"), truth_np, atol=1e-9)
            scale = metadata["speckle_system_mean_before_scaling"]
            trace, frobenius = trace_raw / scale**2, frobenius_raw / scale**4
            theory = raw_actions / scale**2
            fourths, observed = [], {}
            for split in ("train", "holdout"):
                frames = torch.from_numpy(np.load(dataset / f"{split}_frames.npy")).to(device).flatten(1)
                if len(frames) != count:
                    raise ValueError("Unexpected acquired frame count")
                centered = frames - frames.mean(0, keepdim=True)
                empirical = ((q.flatten(1) @ centered.T) @ centered / (count - 1)).reshape_as(q)
                observed[split] = float((empirical.double() - theory.double()).square().sum() / theory.double().square().sum())
                residual = frames.double() - mean_image_raw.double().flatten()[None] / scale
                fourths.append(residual.square().sum(1).square().cpu().numpy())
                del frames, centered, residual, empirical
            fourth_samples = np.concatenate(fourths)
            fourth = float(fourth_samples.mean())
            risk = sampling_risk(count, trace, frobenius, fourth) / frobenius
            row = {"frames": count, "trace": trace, "frobenius_squared": frobenius,
                   "radial_fourth_moment": fourth,
                   "radial_fourth_relative_standard_error": float(fourth_samples.std(ddof=1) / np.sqrt(len(fourth_samples)) / fourth),
                   "predicted_relative_frobenius_mse": risk,
                   "gaussian_reference_relative_mse": (trace**2 / frobenius + 1) / (count - 1),
                   "observed_relative_frobenius_mse": observed,
                   "caution": "Fourth moment estimated using both existing splits for diagnosis only; not a training objective or unseen test. Trace/Frobenius are Monte Carlo estimates."}
            reports.append(row); print(json.dumps(row), flush=True)
    report = {"complete": True, "arguments": vars(args), "probe_seed": 20260987,
              "scope": "Exact known-50-um linear simulator statistics; estimates accuracy of full sample covariance, NOT minimum frames needed for image recovery.",
              "effective_rank_trace_squared_over_frobenius": trace_raw**2 / frobenius_raw,
              "effective_rank_probe_bootstrap_95_interval": np.quantile(effective_ranks, [.025, .975]).tolist(),
              "relative_probe_standard_errors": {"trace": float(probe_trace.std(ddof=1) / np.sqrt(len(q)) / trace_raw),
                                                  "frobenius_squared": float(probe_frobenius.std(ddof=1) / np.sqrt(len(q)) / frobenius_raw)},
              "gaussian_reference_frames_for_full_covariance_rmse": {
                  str(tol): int(np.ceil(1 + (trace_raw**2 / frobenius_raw + 1) / tol**2)) for tol in (.5, .2, .1)},
              "rows": reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
