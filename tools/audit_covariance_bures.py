"""Frozen-candidate/initial-gradient Bures audit; never trains or selects a volume."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.diagnose_cs_information import load_operator
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.diagnose_projected_covariance_likelihood import (
    empirical_projected_covariance, fixed_sensor_basis, precompute_adjoint,
    projected_model_covariance, gaussian_covariance_divergence, relative_covariance_mse,
)


def bures_covariance_distance(prediction, target, ridge=1e-6):
    """Squared BW / target trace, common numerical ridge; NO fitted gain.

    Target is fixed. The eigenvectors of its square root are not differentiated.
    Eigenvalues of the symmetric sandwich suffice for trace(sqrt(.)).
    Bhatia/Jain/Lim (2019), arXiv:1712.01504, Eq. (1).
    """
    if (prediction.ndim != 2 or prediction.shape != target.shape
            or prediction.shape[0] != prediction.shape[1] or ridge < 0):
        raise ValueError("Matching square covariance matrices and nonnegative ridge required")
    target = target.detach().double()
    prediction = prediction.double()
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite covariance")
    dimension = len(target)
    scale = target.trace()/dimension
    if float(scale) <= 0:
        raise ValueError("Positive target trace required")
    eye = torch.eye(dimension, dtype=target.dtype, device=target.device)
    data = (target+target.T)/(2*scale) + ridge*eye
    model = (prediction+prediction.T)/(2*scale) + ridge*eye
    if float(torch.linalg.eigvalsh(model.detach()).min()) < -1e-12:
        raise ValueError("Model covariance is not PSD after declared ridge")
    eig, vec = torch.linalg.eigh(data)
    if float(eig.min()) < -1e-12:
        raise ValueError("Target covariance is not PSD after declared ridge")
    root = (vec*eig.clamp_min(0).sqrt()[None]) @ vec.T
    sandwich = root @ model @ root
    values = torch.linalg.eigvalsh((sandwich+sandwich.T)/2)
    if float(values.detach().min()) < -1e-10:
        raise ValueError("Indefinite Bures sandwich")
    # PSD boundary supported for value-only tests; actual gradient audits use ridge.
    fidelity = values.clamp_min(0).sqrt().sum()
    return (model.trace()+data.trace()-2*fidelity)/dimension


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    denominator = a.norm()*b.norm()
    return float(a@b/denominator) if denominator > 0 else None


def metric(name, prediction, target, ridge):
    if name == "bures":
        return bures_covariance_distance(prediction, target, ridge)
    if name == "mse":
        return relative_covariance_mse(prediction.double(), target.double())
    if name == "nll":
        return gaussian_covariance_divergence(prediction.double(), target.double(), ridge)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--basis-seed", type=int, default=20261060)
    parser.add_argument("--dimension", type=int, choices=(128,256), default=256)
    args = parser.parse_args()
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = json.loads((dataset/"metadata.json").read_text())
    if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
        raise ValueError("This audit requires linear floating-point frames")
    root = Path("outputs/linear_float_oracle_50um/n800_cnn_empirical").resolve()
    paths = {"truth": dataset/"target.npy", "anchor": dataset/"anchor_rl3.npy"}
    for variant in ("pixels_b2", "cnn_seed20260901", "cnn_seed20260902",
                    "cnn_seed20260903", "cnn_matched_step_seed20260901"):
        summary = json.loads((root/variant/"summary.json").read_text())
        if summary["steps"] != 200 or summary["history"][-1]["step"] != 199:
            raise ValueError("Candidate run is not complete")
        if Path(summary["arguments"]["dataset"]).resolve() != dataset:
            raise ValueError("Frozen candidate and audit must use the same dataset")
        for state in ("best", "final"):
            paths[f"{variant}_{state}"] = root/variant/f"reconstruction_{state}.npy"
    arrays = {}
    sources = {}
    for name, path in paths.items():
        array = np.load(path)
        if (array.shape != (10,260,260) or not np.isfinite(array).all() or (array<0).any()
                or np.count_nonzero(np.delete(array,4,axis=0)) or array[4].sum() <= 0):
            raise ValueError(f"Invalid fixed-layer candidate {path}")
        arrays[name] = array[4].astype(np.float32)/array[4].sum()
        sources[name] = {"path":str(path), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    operator = load_operator(Path("configs/depth50_n100_no_mean_loss.yaml").resolve(), device)
    cs = DenseFinitePhaseCs.load("outputs/linear_float_oracle_50um/exact_population_cs", device=device,
                                system_mean=metadata["speckle_system_mean_before_scaling"])
    basis_np = fixed_sensor_basis(args.dimension, (260,260), 0., args.basis_seed)
    basis = torch.from_numpy(basis_np).to(device)
    back = precompute_adjoint(operator, basis, chunk=4)
    flat_basis = basis.flatten(1)
    orthogonality = float((flat_basis@flat_basis.T-torch.eye(len(basis),device=device)).abs().max())
    if orthogonality > 2e-5:
        raise ValueError("Nonorthogonal sensor basis")
    targets = {}
    for split in ("train","holdout"):
        frames = np.load(dataset/f"{split}_frames.npy", mmap_mode="r")
        targets[split] = torch.from_numpy(empirical_projected_covariance(frames,basis_np)).to(device)
    candidate_covariances = {}
    started = time.perf_counter()
    with torch.no_grad():
        for name, value in arrays.items():
            # Identical arrays share a covariance, not extra evidence.
            duplicate = next((old for old in candidate_covariances if np.array_equal(arrays[old],value)),None)
            cov = (candidate_covariances[duplicate] if duplicate is not None else
                   projected_model_covariance(torch.from_numpy(value).to(device),back,cs))
            candidate_covariances[name] = cov.detach()
            print(json.dumps({"phase":"forward","candidate":name,"elapsed_s":time.perf_counter()-started}),flush=True)
    targets["population"] = candidate_covariances["truth"]
    records = []
    for dimension in sorted({min(128,args.dimension),args.dimension}):
        for ridge in (1e-5,1e-6):
            scores = {}
            with torch.no_grad():
                for name,cov in candidate_covariances.items():
                    scores[name] = {split:{loss:float(metric(loss,cov[:dimension,:dimension],
                        target[:dimension,:dimension],ridge)) for loss in ("mse","nll","bures")}
                        for split,target in targets.items()}
            # Only at the unchanged anchor. No optimizer, no finite step, no output image.
            anchor = torch.from_numpy(arrays["anchor"]).to(device)
            anchor = anchor.clamp_min(anchor.max()*1e-8); anchor = anchor/anchor.sum()
            raw = torch.zeros_like(anchor,requires_grad=True)
            shape,_ = bounded_anchor_shape(anchor,raw,2.)
            prediction = projected_model_covariance(shape,back[:dimension],cs)
            gradients = {}
            for loss in ("mse","nll","bures"):
                grads = {}
                for split,target in targets.items():
                    value = metric(loss,prediction,target[:dimension,:dimension],ridge)
                    grad = torch.autograd.grad(value,raw,retain_graph=True)[0].detach()
                    if not torch.isfinite(grad).all() or grad.norm() == 0:
                        raise FloatingPointError("Nonfinite or zero initial gradient")
                    grads[split] = grad
                direction = -anchor*(grads["train"]-(anchor*grads["train"]).sum())
                truth_delta = torch.from_numpy(arrays["truth"]).to(device)-anchor
                gradients[loss] = {
                    "train_holdout_cosine":cosine(grads["train"],grads["holdout"]),
                    "train_population_cosine":cosine(grads["train"],grads["population"]),
                    "holdout_population_cosine":cosine(grads["holdout"],grads["population"]),
                    "gd_image_truth_correction_cosine":cosine(direction,truth_delta),
                    "norms":{key:float(g.norm()) for key,g in grads.items()},
                }
                gradients[loss]["preflight_pass"] = (
                    all(gradients[loss][key] >= .5 for key in (
                        "train_holdout_cosine","train_population_cosine","holdout_population_cosine"))
                    and gradients[loss]["gd_image_truth_correction_cosine"] > 0)
            records.append({"dimension":dimension,"ridge":ridge,"scores":scores,"gradients":gradients})
            print(json.dumps({"phase":"gradient","dimension":dimension,"ridge":ridge,
                              "gradients":gradients}),flush=True)
            del prediction,shape,raw,grads
    output.parent.mkdir(parents=True,exist_ok=True)
    report = {"complete":True,"arguments":vars(args),"candidate_sources":sources,
        "scope":"No training or checkpoint selection. Fixed known 50 um and unit mass. Existing N800 train/holdout acquisition reused; new orthonormal sensor bases. Population truth only an oracle diagnostic.",
        "metric":"Squared Bures covariance distance normalized by fixed target trace; both arguments receive the same declared numerical ridge. No fitted intensity/covariance gain. Not a Gaussian likelihood or a RIM-STD reproduction.",
        "reference":"https://arxiv.org/abs/1712.01504",
        "gate":"Conservative initial preflight, not a necessary/sufficient reconstruction guarantee. Require all gradient pair cosines >=0.5 and positive image-truth correction cosine; do not train based on favorable candidate ranking alone.",
        "orthogonality_max_error":orthogonality,
        "elapsed_s":time.perf_counter()-started,
        "peak_gpu_memory_gib":torch.cuda.max_memory_allocated(device)/2**30,
        "records":records}
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps({"complete":True,"output":str(output)}),flush=True)


if __name__ == "__main__":
    main()
