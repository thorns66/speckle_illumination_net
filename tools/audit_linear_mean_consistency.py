"""Forward-only mean consistency check; no gradients and no reconstruction.

Compares constant illumination with the exact finite random-phase source mean.
The fitted scalar is reported diagnostically, never applied to saved objects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics.speckle_oracle import SpeckleGeneratorConfig
from tools.audit_finite_speckle_stationarity import exact_moments
from tools.diagnose_cs_information import load_operator


def score(predicted, measured):
    p, t = predicted.flatten().double(), measured.flatten().double()
    gain = (p@t)/(p@p)
    return {"relative_l2": float((p-t).norm()/t.norm()),
            "correlation": float(torch.corrcoef(torch.stack((p,t)))[0,1]),
            "diagnostic_gain": float(gain),
            "gain_adjusted_relative_l2": float((gain*p-t).norm()/t.norm())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--include-fullbatch", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device); torch.cuda.set_device(device)
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    root = Path("outputs/linear_float_oracle_50um").resolve()
    operator = load_operator(Path(args.config).resolve(), device)
    finite_mean, _, _ = exact_moments(SpeckleGeneratorConfig())
    finite_mean = finite_mean.to(device=device, dtype=torch.float32)
    rows = []
    for count, dataset in ((100, root/"dataset"), (800, root/"frame_count_nested/n800")):
        metadata = json.loads((dataset/"metadata.json").read_text())
        targets = {name:torch.from_numpy(np.load(dataset/f"{name}_frames.npy", mmap_mode="r").mean(0, dtype=np.float64)).to(device)
                   for name in ("train", "holdout")}
        shapes = {"truth":dataset/"target.npy", "local_anchor":dataset/"anchor_rl3.npy",
                  "n100_anchor":root/"dataset/anchor_rl3.npy",
                  "stationary_sigma0_b2":root/"oracle_sensor_bandwidth/sigma0_b2/reconstruction_best.npy",
                  "finite1024_matched":root/"oracle_provenance/finite1024_matched/reconstruction_best.npy"}
        if args.include_fullbatch:
            shapes.update({
                "exact_b0p5_single": root/"exact_population_diagnostics/oracle_b0p5/reconstruction_best.npy",
                "exact_b2_single": root/"exact_population_diagnostics/oracle_b2/reconstruction_best.npy",
                "exact_b0p5_full32": root/"exact_population_fullbatch/b0p5/reconstruction_best.npy",
                "exact_b2_full32": root/"exact_population_fullbatch/b2/reconstruction_best.npy",
            })
        if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
            raise ValueError("Linear unquantized frames required")
        truth = torch.from_numpy(np.load(dataset/"target.npy")[4]).to(device).clamp_min(0)
        truth /= truth.sum()
        population_predictions = {
            "constant": operator(truth[None,None,None])[0,0],
            "finite_source_exact": operator((truth*finite_mean/metadata["speckle_system_mean_before_scaling"])[None,None,None])[0,0],
        }
        for name, path in shapes.items():
            g = torch.from_numpy(np.load(path)[4]).to(device).clamp_min(0); g /= g.sum()
            for model, illumination in (("constant",torch.ones_like(g)),
                                        ("finite_source_exact",finite_mean/metadata["speckle_system_mean_before_scaling"])):
                prediction = operator((g*illumination)[None,None,None])[0,0]
                rows.append({"frames":count,"candidate":name,"mean_model":model,
                             **{split:score(prediction,target) for split,target in targets.items()},
                             "population_truth_mean":score(prediction,population_predictions[model])})
        rows.append({"frames":count,"candidate":"observed_train_vs_holdout","mean_model":"data",
                     "holdout":score(targets["train"],targets["holdout"])})
    report = {"arguments":vars(args), "scope":"Forward only, torch.no_grad; no model updates, no mean backpropagation, no scale applied to saved reconstruction.",
              "population_mean":"Exact finite random-phase generator moments normalized by each dataset's fixed system illumination mean.",
              "warning":"Only linear float simulated frames. Historical per-frame-max-normalized uint8 data are excluded. No new real-data calibration is claimed.",
              "rows":rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
