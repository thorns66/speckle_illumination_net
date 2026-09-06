"""Separate sensor-sketch noise from object-space tangent instability.

Uses nested sensor-probe prefixes and fixed, scene-independent DCT tangent
subspaces. Does not fit an image or select a checkpoint. Population truth is
only an oracle gradient reference, never a deployable measurement.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn

from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.audit_oracle_candidate_empirical_transfer import cosine
from tools.diagnose_cs_information import load_operator, model_action
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.scan_covariance_sketch_depth import _lowpass_probes


def tangent_report(gradients, cuts=(4, 8, 16, 32, 64, 128, 260)):
    values = {key: np.asarray(value, dtype=np.float64) for key, value in gradients.items()}
    coeffs = {key: dctn(value, type=2, norm="ortho") for key, value in values.items()}
    # Unit-mass log parameterization has a constant-shift null direction at
    # initialization; exclude DC instead of interpreting its rounding noise.
    for value in coeffs.values():
        value[0, 0] = 0.
    def cos(a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.vdot(a, b) / norm) if norm > 0 else None
    results = []
    for cut in cuts:
        projected = {key: value[:cut, :cut] for key, value in coeffs.items()}
        row = {"dct_side": cut, "parameters_excluding_dc": projected["train"].size - 1,
               "train_population_cosine": cos(projected["train"], projected["population"]),
               "holdout_population_cosine": cos(projected["holdout"], projected["population"]),
               "train_holdout_cosine": cos(projected["train"], projected["holdout"]),
               "retained_gradient_energy": {key: float(np.sum(value**2) / max(np.sum(coeffs[key]**2), 1e-300))
                                            for key, value in projected.items()}}
        row["direction_gate_pass"] = all(row[key] is not None and row[key] >= .5 for key in (
            "train_population_cosine", "holdout_population_cosine", "train_holdout_cosine"))
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probes", type=int, choices=(256, 512, 1024), default=512)
    args = parser.parse_args()
    output, dataset = Path(args.output), Path(args.dataset)
    if (output / "report.json").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    metadata = json.loads((dataset / "metadata.json").read_text())
    if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
        raise ValueError("Linear unquantized data required")
    operator = load_operator(Path("configs/depth50_n100_no_mean_loss.yaml").resolve(), device)
    cs = DenseFinitePhaseCs.load("outputs/linear_float_oracle_50um/exact_population_cs", device=device,
                                system_mean=metadata["speckle_system_mean_before_scaling"])
    q = torch.from_numpy(_lowpass_probes(args.probes, (260, 260), sigma=0, seed=20260983)).to(device)
    anchor = torch.from_numpy(np.load(dataset / "anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max()*1e-8); anchor /= anchor.sum()
    truth = torch.from_numpy(np.load(dataset / "target.npy")[4]).to(device)
    truth = truth.clamp_min(0); truth /= truth.sum()
    targets = {}
    with torch.no_grad():
        back = torch.cat([operator.adjoint(q[i:i+4, None]) for i in range(0, len(q), 4)])
        for split in ("train", "holdout"):
            frames = torch.from_numpy(np.load(dataset / f"{split}_frames.npy")).to(device).flatten(1)
            centered = frames - frames.mean(0, keepdim=True)
            targets[split] = ((q.flatten(1) @ centered.T) @ centered / (len(frames)-1)).reshape_as(q)
            del centered, frames
        targets["population"] = torch.cat([model_action(truth, back[i:i+4], cs, operator)
                                            for i in range(0, len(q), 4)])
    normalizers = {key: target.square().flatten(1).sum(1).mean().detach() for key, target in targets.items()}
    raw = torch.zeros_like(anchor, requires_grad=True)
    gradients = {mode: {key: torch.zeros_like(raw) for key in targets} for mode in ("per_probe", "global_fixed")}
    score_sums = {mode: {key: {name: 0. for name in ("anchor", "truth")} for key in targets}
                  for mode in gradients}
    reports = []; snapshots = {}; started = time.perf_counter()
    milestones = sorted({16, 64, 256, args.probes})
    for start in range(0, len(q), 4):
        stop = min(start + 4, len(q)); count = stop - start
        shape, _ = bounded_anchor_shape(anchor, raw, .5)
        prediction = model_action(shape, back[start:stop], cs, operator)
        for mode in gradients:
            for key, target in targets.items():
                denominator = (target[start:stop].square().flatten(1).sum(1) if mode == "per_probe" else normalizers[key])
                denominator = denominator.clamp_min(torch.finfo(target.dtype).tiny)
                loss = ((prediction - target[start:stop]).square().flatten(1).sum(1) / denominator).mean()
                gradient = torch.autograd.grad(loss, raw, retain_graph=not (mode == "global_fixed" and key == "population"))[0]
                gradients[mode][key] += gradient.detach() * count
                score_sums[mode][key]["anchor"] += float(loss.detach()) * count
                oracle_loss = ((targets["population"][start:stop] - target[start:stop]).square().flatten(1).sum(1) / denominator).mean()
                score_sums[mode][key]["truth"] += float(oracle_loss) * count
        if stop in milestones:
            snapshot = {mode: {key: (grad / stop).cpu().numpy() for key, grad in values.items()}
                        for mode, values in gradients.items()}
            snapshots[stop] = snapshot
            for mode, values in snapshot.items():
                full = {"train_population_cosine": cosine(torch.from_numpy(values["train"]), torch.from_numpy(values["population"])),
                        "holdout_population_cosine": cosine(torch.from_numpy(values["holdout"]), torch.from_numpy(values["population"])),
                        "train_holdout_cosine": cosine(torch.from_numpy(values["train"]), torch.from_numpy(values["holdout"]))}
                scores = {key: {name: value / stop for name, value in names.items()} for key, names in score_sums[mode].items()}
                row = {"probes": stop, "loss": mode, "full_gradient_agreement": full,
                       "scores": scores, "tangent_subspaces": tangent_report(values),
                       "elapsed_s": time.perf_counter()-started}
                reports.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "tangent_subspaces"}), flush=True)
                np.savez(output / f"gradients_k{stop}_{mode}.npz", **values)
    for row in reports:
        previous = snapshots[row["probes"]][row["loss"]]
        final = snapshots[len(q)][row["loss"]]
        row["cosine_to_largest_probe_prefix"] = {key: float(np.vdot(previous[key], final[key]) /
             (np.linalg.norm(previous[key])*np.linalg.norm(final[key]))) for key in targets}
    report = {"complete": True, "arguments": vars(args), "frames": metadata["train_frames"], "results": reports,
              "scope": "Initial gradient audit only, same acquired frames with increasing nested sensor probes. No image fitting, no mean backprop. Largest prefix is not a proof of exact gradient convergence.",
              "normalization": "global_fixed is one fixed scalar denominator per target set, not a scene gain. It changes gradient magnitude but not unnormalized squared-error gradient direction. per_probe is the existing objective.",
              "tangent": "Orthonormal object-space DCT log-residual coefficients; DC excluded. Smoothing/parameter restriction has not been adopted for training. Low-frequency direction agreement alone cannot prove resolution improvement.",
              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)/2**30}
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
