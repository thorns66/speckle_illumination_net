"""No-training audit of ideal covariance candidates on existing independent frames.

Reports objective rankings and gradient agreement, not a new reconstruction.
All candidate objects are normalized to unit mass. No fitted scene scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from physics.speckle_oracle import SpeckleGeneratorConfig, analytic_intensity_covariance
from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.diagnose_cs_information import CachedCs, load_operator, model_action, vector_loss
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.scan_covariance_sketch_depth import _empirical_action
from tools.localized_covariance import localized_probe_bank


def cosine(left, right):
    return float((left.flatten()@right.flatten()) / (left.norm()*right.norm()).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.)
    parser.add_argument("--sensor-seed", type=int, default=20260983)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH",
                        help="Additional frozen reconstruction to evaluate; never optimize it")
    parser.add_argument("--only-custom-candidates", action="store_true",
                        help="Keep truth/local anchor and explicit candidates, omit legacy runs")
    parser.add_argument("--skip-gradients", action="store_true",
                        help="Evaluate frozen candidates only, no initialization gradient audit")
    parser.add_argument("--window-sigma", type=float, default=0.)
    parser.add_argument("--cs-model", choices=("stationary", "finite_phase"), default="stationary")
    parser.add_argument("--cs-directory", default="outputs/linear_float_oracle_50um/exact_population_cs")
    args = parser.parse_args()
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    root = Path("outputs/linear_float_oracle_50um").resolve()
    metadata = json.loads((dataset/"metadata.json").read_text())
    truth_np = np.load(dataset/"target.npy")
    np.testing.assert_allclose(truth_np, np.load(root/"dataset/target.npy"), rtol=1e-5, atol=1e-9)
    operator = load_operator(Path(args.config).resolve(), device)
    if args.cs_model == "finite_phase":
        if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
            raise ValueError("Exact phase moments require linear unquantized simulation data")
        cs = DenseFinitePhaseCs.load(args.cs_directory, device=device,
                                     system_mean=metadata["speckle_system_mean_before_scaling"])
    else:
        cs = CachedCs(analytic_intensity_covariance(SpeckleGeneratorConfig(), device=device, dtype=torch.float32)
                      * metadata["model_pattern_variance_mean"], (260, 260))
    q_np, windows_np = localized_probe_bank(args.probes, (260,260), seed=args.sensor_seed,
                                           window_sigma=args.window_sigma, probe_sigma=args.sigma)
    windows = torch.from_numpy(windows_np).to(device)
    q = torch.from_numpy(q_np).to(device)
    with torch.no_grad():
        back = torch.cat([operator.adjoint(q[i:i+4,None]) for i in range(0, len(q), 4)])
    targets = {}
    for split in ("train", "holdout"):
        frames = np.load(dataset/f"{split}_frames.npy", mmap_mode="r")
        targets[split] = torch.from_numpy(_empirical_action(frames, q_np)).to(device)*windows
    candidates = {
        "truth": dataset/"target.npy",
        "local_anchor": dataset/"anchor_rl3.npy",
        "n100_anchor": root/"dataset/anchor_rl3.npy",
        "finite1024_matched": root/"oracle_provenance/finite1024_matched/reconstruction_best.npy",
        "stationary_sigma4_b0p5": root/"oracle_provenance/stationary_analytic/reconstruction_best.npy",
        "stationary_sigma0_b0p5": root/"oracle_sensor_bandwidth/sigma0_b0p5/reconstruction_best.npy",
        "stationary_sigma0_b2": root/"oracle_sensor_bandwidth/sigma0_b2/reconstruction_best.npy",
    }
    if args.only_custom_candidates:
        candidates = {name:candidates[name] for name in ("truth", "local_anchor")}
    for specification in args.candidate:
        name, separator, filename = specification.partition("=")
        if not separator or not name or not filename or name in candidates:
            raise ValueError("Each custom candidate requires a unique NAME=PATH")
        candidates[name] = Path(filename).resolve()
    for path in candidates.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    scores = {}; truth_action = None
    with torch.no_grad():
        for name, path in candidates.items():
            volume = np.load(path)
            if (volume.shape != truth_np.shape or not np.isfinite(volume).all()
                    or np.any(volume < 0) or volume[4].sum() <= 0):
                raise ValueError(f"Invalid nonnegative volume: {path}")
            shape = torch.from_numpy(volume[4]).to(device).clamp_min(0)
            shape /= shape.sum()
            prediction = torch.cat([model_action(shape, back[i:i+4], cs, operator)
                                    for i in range(0, len(back), 4)])*windows
            if name == "truth":
                truth_action = prediction
            scores[name] = {}
            for split, target in targets.items():
                loss, corr = vector_loss(prediction, target)
                per_probe = (prediction-target).flatten(1).square().sum(1)/target.flatten(1).square().sum(1)
                scores[name][split] = {"relative_mse": float(loss), "correlation": float(corr),
                                      "per_probe_losses": per_probe.cpu().tolist()}
            scores[name]["population_oracle_loss"] = float(vector_loss(prediction, truth_action)[0])
            population_per_probe = ((prediction-truth_action).flatten(1).square().sum(1)
                                    / truth_action.flatten(1).square().sum(1))
            scores[name]["population_per_probe_losses"] = population_per_probe.cpu().tolist()
    gradient_report = None
    if not args.skip_gradients:
        # Use each dataset's own anchor. Gradient wrt bounded log residual includes
        # the unit-mass normalization and is the actual optimizer's tangent space.
        anchor = torch.from_numpy(np.load(dataset/"anchor_rl3.npy")[4]).to(device)
        anchor = anchor.clamp_min(anchor.max()*1e-8); anchor /= anchor.sum()
        raw = torch.zeros_like(anchor, requires_grad=True)
        gradients = {key: torch.zeros_like(raw) for key in ("train", "holdout", "population")}
        for start in range(0, len(back), 4):
            shape, _ = bounded_anchor_shape(anchor, raw, .5)
            prediction = model_action(shape, back[start:start+4], cs, operator)*windows[start:start+4]
            for key, target in (("train",targets["train"]), ("holdout",targets["holdout"]), ("population",truth_action)):
                loss = vector_loss(prediction, target[start:start+4])[0]
                gradient = torch.autograd.grad(loss, raw, retain_graph=key != "population")[0]
                gradients[key] += gradient.detach() * len(prediction)/len(back)
        gradient_report = {
            "train_population_cosine": cosine(gradients["train"], gradients["population"]),
            "holdout_population_cosine": cosine(gradients["holdout"], gradients["population"]),
            "train_holdout_cosine": cosine(gradients["train"], gradients["holdout"]),
            "norms": {key:float(value.norm()) for key,value in gradients.items()},
            "train_relative_gradient_error": float((gradients["train"]-gradients["population"]).norm()/gradients["population"].norm()),
            "holdout_relative_gradient_error": float((gradients["holdout"]-gradients["population"]).norm()/gradients["population"].norm()),
        }
    report = {"complete": True, "arguments":vars(args), "frames":metadata["train_frames"],
              "scope":"Frozen-candidate forward/gradient audit only; existing train/holdout frames reused, not a pristine final test set. New sensor probes do not constitute a new independent acquisition. Candidate source paths and hashes are recorded; no candidate is trained or selected by this audit.",
              "candidate_sources": {name: {"path": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in candidates.items()},
              "model":("Exact finite-source population phase moments; no sampled pattern banks." if args.cs_model == "finite_phase"
                       else "Deterministic stationary analytic Cs; includes mismatch to finite-source population moments."),
              "sensor_probe_seed":args.sensor_seed, "covariance_measurement":"P_w C_y P_w r with fixed, scene-independent windows; width zero means global",
              "scores":scores, "gradient_agreement":gradient_report}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames":report["frames"], "gradient_agreement":gradient_report,
                      "scores":{name:{split:{key:value for key,value in score.items() if key!="per_probe_losses"}
                                         for split,score in value.items() if isinstance(score,dict)}
                                for name,value in scores.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
