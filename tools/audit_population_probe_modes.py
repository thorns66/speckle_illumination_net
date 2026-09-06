"""No-training audit of DC and deterministic DCT covariance-vector probes.

Data are centered over frames, never normalized per frame. A constant SENSOR
probe retains Cov(y_i, total intensity); this is not an image-mean loss and
does not require the illumination realization. This is not ghost imaging:
both sides are measured after interaction with the object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.audit_oracle_candidate_empirical_transfer import cosine
from tools.diagnose_cs_information import load_operator, model_action, vector_loss
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.scan_covariance_sketch_depth import _empirical_action


def dct_modes(shape, count=65):
    height, width = shape
    if count < 1 or count > height * width:
        raise ValueError("Invalid mode count")
    extent = min(max(height, width), count)
    frequencies = sorted(((y, x) for y in range(min(height, extent)) for x in range(min(width, extent))),
                         key=lambda pair: (pair[0]**2 + pair[1]**2, pair))[:count]
    y, x = np.arange(height) + .5, np.arange(width) + .5
    values = np.stack([np.cos(np.pi * fy * y[:, None] / height) * np.cos(np.pi * fx * x[None] / width)
                       for fy, fx in frequencies])
    values /= np.linalg.norm(values.reshape(count, -1), axis=1)[:, None, None]
    return values.astype(np.float32), frequencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    args = parser.parse_args()
    output, dataset = Path(args.output), Path(args.dataset)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    metadata = json.loads((dataset / "metadata.json").read_text())
    if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
        raise ValueError("Requires linear unquantized data")
    operator = load_operator(Path(args.config).resolve(), device)
    cs = DenseFinitePhaseCs.load("outputs/linear_float_oracle_50um/exact_population_cs", device=device,
                                system_mean=metadata["speckle_system_mean_before_scaling"])
    q_np, frequencies = dct_modes((260, 260))
    q = torch.from_numpy(q_np).to(device)
    with torch.no_grad():
        back = torch.cat([operator.adjoint(q[i:i+4, None]) for i in range(0, len(q), 4)])
    targets = {split: torch.from_numpy(_empirical_action(np.load(dataset / f"{split}_frames.npy"), q_np)).to(device)
               for split in ("train", "holdout")}
    anchor = torch.from_numpy(np.load(dataset / "anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max()*1e-8); anchor /= anchor.sum()
    truth = torch.from_numpy(np.load(dataset / "target.npy")[4]).to(device)
    truth = truth.clamp_min(0); truth /= truth.sum()
    predictions = {}
    with torch.no_grad():
        for name, shape in (("truth", truth), ("anchor", anchor)):
            predictions[name] = torch.cat([model_action(shape, back[i:i+4], cs, operator)
                                           for i in range(0, len(back), 4)])
    groups = {"dc_only": [0], "low16_with_dc": list(range(16)),
              "low16_without_dc": list(range(1, 17)), "low64_with_dc": list(range(64))}
    results = []
    for name, indices in groups.items():
        idx = torch.tensor(indices, device=device)
        # Same per-probe relative MSE as earlier, without a fitted scene scale.
        scores = {candidate: {split: float(vector_loss(value[idx], target[idx])[0])
                              for split, target in targets.items()} for candidate, value in predictions.items()}
        raw = torch.zeros_like(anchor, requires_grad=True)
        gradients = {key: torch.zeros_like(anchor) for key in ("train", "holdout", "population")}
        for start in range(0, len(idx), 4):
            selected = idx[start:start+4]
            shape, _ = bounded_anchor_shape(anchor, raw, .5)
            prediction = model_action(shape, back[selected], cs, operator)
            for key, target in (*targets.items(), ("population", predictions["truth"])):
                loss = vector_loss(prediction, target[selected])[0]
                gradient = torch.autograd.grad(loss, raw, retain_graph=key != "population")[0]
                gradients[key] += gradient.detach() * len(selected) / len(idx)
        agreement = {"train_population_cosine": cosine(gradients["train"], gradients["population"]),
                     "holdout_population_cosine": cosine(gradients["holdout"], gradients["population"]),
                     "train_holdout_cosine": cosine(gradients["train"], gradients["holdout"])}
        result = {"family": name, "mode_indices": indices, "scores": scores, "gradient_agreement": agreement,
                  "gradient_norms": {k: float(v.norm()) for k, v in gradients.items()},
                  "empirical_action_cosine": cosine(targets["train"][idx], targets["holdout"][idx]),
                  "prescreen_pass": all(v >= .5 for v in agreement.values())
                       and all(scores["truth"][split] < scores["anchor"][split] for split in targets)}
        results.append(result); print(json.dumps(result), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"complete": True, "arguments": vars(args), "frames": metadata["train_frames"],
              "dct_frequencies": frequencies, "results": results,
              "scope": "Fixed 50 um, exact finite-source population covariance; no training, no mean backprop, no ground-truth choice of probes. Existing holdout is reused for research, not a final test.",
              "prescreen": "All three initial gradient cosines >= 0.5 and truth ranked ahead of anchor on both independent frame sets. Engineering filter, not a proof of identifiability."}
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
