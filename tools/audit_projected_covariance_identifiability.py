"""Truth-only audit: distinguish representation capacity from objective fit.

The box/simplex projection uses truth, so its image is NEVER a reconstruction
candidate. It only certifies the best possible L2 error in the bounded family.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from tools.diagnose_cs_information import CachedCs, load_operator
from tools.diagnose_projected_covariance_likelihood import (
    fixed_sensor_basis, precompute_adjoint, projected_model_covariance,
    gaussian_covariance_divergence, relative_covariance_mse,
)


def closest_bounded_shape(anchor, truth, bound):
    a = np.asarray(anchor, np.float64).ravel(); a = np.maximum(a, a.max()*1e-8); a /= a.sum()
    t = np.asarray(truth, np.float64).ravel(); t /= t.sum()
    factor = np.exp(2 * bound)
    def project(c):
        lower, upper = c*a, c*factor*a
        lo, hi = float((lower-t).min()), float((upper-t).max())
        for _ in range(60):
            tau = (lo+hi)/2
            if np.clip(t+tau, lower, upper).sum() > 1: hi = tau
            else: lo = tau
        return np.clip(t+(lo+hi)/2, lower, upper)
    result = minimize_scalar(lambda c: np.square(project(c)-t).sum(),
                             bounds=(1/factor, 1), method="bounded", options={"xatol": 1e-10})
    return project(result.x).reshape(anchor.shape), float(result.x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um/projected_covariance_likelihood")
    args = parser.parse_args()
    device = torch.device(args.device); torch.cuda.set_device(device)
    dataset, root = Path(args.dataset), Path(args.root)
    anchor = np.load(dataset/"anchor_rl3.npy")[4].astype(np.float64)
    anchor = np.maximum(anchor, anchor.max()*1e-8); anchor /= anchor.sum()
    truth = np.load(dataset/"target.npy")[4].astype(np.float64); truth /= truth.sum()
    shapes = {"truth": truth, "anchor": anchor}
    capacity = {}
    for bound in (0.5, 1.0, 2.0):
        candidate, scale = closest_bounded_shape(anchor, truth, bound)
        capacity[str(bound)] = {"relative_l2_infimum": float(np.linalg.norm(candidate-truth)/np.linalg.norm(truth)),
                                "mass": float(candidate.sum()), "box_scale": scale}
        if bound == 0.5: shapes["closest_feasible_b0p5"] = candidate
    for name in ("oracle_nll_d1024_r0", "oracle_mse_d1024_r0"):
        volume = np.load(root/name/"reconstruction_best.npy")[4].astype(np.float64)
        shapes[name] = volume/volume.sum()
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset/"metadata.json").read_text())
    cs = CachedCs(torch.from_numpy(np.load(dataset/"cs_kernel.npy")).to(device)
                  * metadata["model_pattern_variance_mean"], (260, 260))
    q = torch.from_numpy(fixed_sensor_basis(1024, (260, 260), 4, 20260911)).to(device)
    back = precompute_adjoint(operator, q)
    with torch.no_grad():
        target = projected_model_covariance(torch.tensor(truth, dtype=torch.float32, device=device), back, cs)
        scale = torch.trace(target)/len(target)
        eigenvalues = torch.linalg.eigvalsh((target/scale).double())
        scores = {}
        for name, value in shapes.items():
            prediction = projected_model_covariance(torch.tensor(value, dtype=torch.float32, device=device), back, cs)
            scores[name] = {"oracle_nll": float(gaussian_covariance_divergence(prediction.double(),target.double(),1e-3)),
                            "oracle_mse": float(relative_covariance_mse(prediction,target)),
                            "unit_mass_relative_l2": float(np.linalg.norm(value-truth)/np.linalg.norm(truth))}
    report = {"scope": "Truth-only feasibility audit, never a deployable estimate", "capacity": capacity,
              "scores": scores, "true_covariance_eigenvalues_below_ridge": int((eigenvalues<1e-3).sum()),
              "dimension": 1024, "ridge": 1e-3}
    (root/"capacity_and_objective_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
