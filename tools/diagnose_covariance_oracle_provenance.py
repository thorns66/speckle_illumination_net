"""Separate finite-ensemble oracle information from population covariance.

The complete sensor vector Cy q is retained. This is a fixed-50-um diagnostic.
Default oracle targets are generated from truth. The explicit frames target
mode uses only acquired training/validation frames to optimize/select images.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from physics.speckle_oracle import SpeckleGeneratorConfig, analytic_intensity_covariance
from physics.finite_phase_cs import DenseFinitePhaseCs
from physics.covariance_sketch import empirical_covariance_action
from tools.audit_covariance_convolutional_tangent import make_network_input
from tools.audit_cs_covariance_model import raw_patterns
from tools.diagnose_cs_information import CachedCs, load_operator, model_action, save_volume, vector_loss
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape, hessian_schatten2
from tools.scan_covariance_sketch_depth import _lowpass_probes


class FixedCurvaturePenalty:
    """Calibrate once against accumulated data gradients, then keep lambda fixed.

    The penalty acts on the pre-tanh log residual, not the anchor or image.
    Pixel finite differences preserve 90-degree rotations; arbitrary rotations
    are only approximated on the discrete square lattice.
    """
    def __init__(self, ratio: float, start_step: int = 20):
        if not math.isfinite(ratio) or ratio < 0 or not 0 <= start_step < 200:
            raise ValueError("Require finite nonnegative ratio and start step in 0..199")
        self.ratio, self.start_step = float(ratio), int(start_step)
        self.weight = None
        self.calibration = None

    def apply(self, raw: torch.Tensor, step: int) -> dict:
        stats = {"curvature_loss": 0.0, "curvature_weight": 0.0,
                 "curvature_gradient_ratio": 0.0, "weighted_curvature_loss": 0.0}
        if self.ratio == 0 or step < self.start_step:
            return stats
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise FloatingPointError("Finite accumulated covariance gradient required")
        data_norm = float(raw.grad.norm())
        penalty = hessian_schatten2(raw)
        regularizer_gradient = torch.autograd.grad(penalty, raw)[0]
        regularizer_norm = float(regularizer_gradient.norm())
        if not torch.isfinite(penalty) or not torch.isfinite(regularizer_gradient).all():
            raise FloatingPointError("Nonfinite curvature penalty or gradient")
        if self.weight is None:
            if step != self.start_step:
                raise RuntimeError("Curvature calibration step was skipped")
            if data_norm <= 1e-20 or regularizer_norm <= 1e-20:
                raise FloatingPointError("Cannot calibrate curvature against a zero gradient")
            self.weight = self.ratio * data_norm / regularizer_norm
            self.calibration = {"step": step, "data_gradient_norm": data_norm,
                                "curvature_gradient_norm": regularizer_norm,
                                "weight": self.weight, "requested_ratio": self.ratio}
        raw.grad.add_(regularizer_gradient, alpha=self.weight)
        if not torch.isfinite(raw.grad).all():
            raise FloatingPointError("Nonfinite combined covariance/curvature gradient")
        return {"curvature_loss": float(penalty.detach()),
                "curvature_weight": self.weight,
                "curvature_gradient_ratio": self.weight*regularizer_norm/max(data_norm, 1e-20),
                "weighted_curvature_loss": self.weight*float(penalty.detach())}


class FiniteEnsembleCs:
    def __init__(self, patterns: torch.Tensor):
        if patterns.ndim != 3 or len(patterns) < 2:
            raise ValueError("Expected at least two [H,W] illumination patterns")
        self.shape = tuple(patterns.shape[-2:])
        self.centered = (patterns - patterns.mean(0, keepdim=True)).flatten(1).detach()

    def action(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(-1, self.centered.shape[1])
        result = (flat @ self.centered.T) @ self.centered / (len(self.centered) - 1)
        return result.reshape_as(value)


class FrozenOffsetResidual(torch.nn.Module):
    """Activate CNN parameters while starting at exactly the same anchor.

    The initial output is persisted as a buffer, never recomputed as theta
    changes. This is an isolated fixed-depth diagnostic, not production code.
    """
    def __init__(self, anchor, seed, *, context=40, channels=(16, 32, 64)):
        super().__init__()
        self.context = context
        model, value = make_network_input(anchor.detach().cpu().numpy(), "anchor", seed,
                                          context=context, channels=channels,
                                          device=anchor.device, zero_readout=False)
        self.model = model
        self.register_buffer("fixed_input", value.detach())
        with torch.no_grad():
            initial = self._uncentered_output().detach().clone()
        self.register_buffer("initial_output", initial)

    def _uncentered_output(self):
        result = self.model(self.fixed_input)[0, 0]
        if self.context:
            result = result[self.context:-self.context, self.context:-self.context]
        return result

    def forward(self):
        return self._uncentered_output() - self.initial_output


def checked_gradient_norm(parameters):
    norms = []
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Missing or nonfinite parameter gradient")
        norms.append(parameter.grad.square().sum())
    return float(torch.stack(norms).sum().sqrt())


@torch.no_grad()
def compute_actions(shape, back, cs, operator, chunk=4):
    return torch.cat([model_action(shape, back[start:start+chunk], cs, operator)
                      for start in range(0, len(back), chunk)])


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists() or (output/"reconstruction_initial.npy").exists():
        raise FileExistsError("Experiment output already contains a run; use a new directory")
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset/"metadata.json").read_text())
    scale = metadata["speckle_system_mean_before_scaling"]

    def finite(seed):
        patterns = raw_patterns(args.patterns, device, seed).to(device)/scale
        return FiniteEnsembleCs(patterns)

    if args.cs_type == "finite":
        train_cs, holdout_cs = finite(20261012), finite(20261013)
        if args.target_link == "independent":
            train_target_cs, holdout_target_cs = finite(20261022), finite(20261023)
        else:
            train_target_cs, holdout_target_cs = train_cs, holdout_cs
    elif args.cs_type == "finite_phase":
        if args.target_link != "matched":
            raise ValueError("A population moment model has no independent realization bank")
        if metadata["per_frame_sensor_normalization"] or metadata["quantization"]:
            raise ValueError("Population phase moments require the linear floating-point simulator")
        train_cs = DenseFinitePhaseCs.load(args.cs_directory, device=device, system_mean=scale)
        holdout_cs = train_target_cs = holdout_target_cs = train_cs
    else:
        if args.target_link != "matched":
            raise ValueError("A deterministic stationary model has no independent realization bank")
        if args.cs_type == "stationary_empirical":
            kernel = torch.from_numpy(np.load(dataset/"cs_kernel.npy")).to(device)
        else:
            kernel = analytic_intensity_covariance(SpeckleGeneratorConfig(), device=device, dtype=torch.float32)
        train_cs = CachedCs(kernel*metadata["model_pattern_variance_mean"], (260,260))
        holdout_cs = train_target_cs = holdout_target_cs = train_cs

    anchor = torch.from_numpy(np.load(dataset/"anchor_rl3.npy")[4]).to(device)
    anchor = anchor.clamp_min(anchor.max()*1e-8); anchor /= anchor.sum()
    truth = torch.from_numpy(np.load(dataset/"target.npy")[4]).to(device)
    truth = truth.clamp_min(0); truth /= truth.sum()
    with torch.no_grad():
        train_q = torch.from_numpy(_lowpass_probes(32, (260,260), sigma=args.probe_sigma, seed=20261010)).to(device)
        holdout_q = torch.from_numpy(_lowpass_probes(16, (260,260), sigma=args.probe_sigma, seed=20261011)).to(device)
        train_back = torch.cat([operator.adjoint(train_q[start:start+4,None]) for start in range(0,32,4)])
        holdout_back = torch.cat([operator.adjoint(holdout_q[start:start+4,None]) for start in range(0,16,4)])
        if args.target_source == "frames":
            train_frames = torch.from_numpy(np.load(dataset/"train_frames.npy")).to(device)
            holdout_frames = torch.from_numpy(np.load(dataset/"holdout_frames.npy")).to(device)
            if not torch.isfinite(train_frames).all() or not torch.isfinite(holdout_frames).all():
                raise FloatingPointError("Nonfinite observed frames")
            train_target = empirical_covariance_action(train_frames, train_q)
            holdout_target = empirical_covariance_action(holdout_frames, holdout_q)
            del train_frames, holdout_frames
        else:
            train_target = compute_actions(truth, train_back, train_target_cs, operator)
            holdout_target = compute_actions(truth, holdout_back, holdout_target_cs, operator)
        true_model_prediction = compute_actions(truth, holdout_back, holdout_cs, operator)
        true_shape_holdout_loss = float(vector_loss(true_model_prediction, holdout_target)[0])
    # Independent target banks are no longer needed once their sensor actions
    # have been computed, reducing memory without changing target statistics.
    del train_target_cs, holdout_target_cs

    @torch.no_grad()
    def validate(shape):
        prediction = compute_actions(shape, holdout_back, holdout_cs, operator)
        loss, corr = vector_loss(prediction, holdout_target)
        return float(loss), float(corr)

    initial_loss, initial_corr = validate(anchor)
    best_loss, best_step = initial_loss, -1
    save_volume(output/"reconstruction_initial", anchor)
    save_volume(output/"reconstruction_best", anchor)
    if args.shape_model == "cnn":
        residual_model = FrozenOffsetResidual(anchor, args.seed)
        parameters = list(residual_model.parameters())
        get_raw = residual_model
        optimizer = torch.optim.Adam(parameters, lr=args.network_lr)
        with torch.no_grad():
            initial_raw_error = float(get_raw().abs().max())
        if initial_raw_error > 1e-6:
            raise AssertionError(f"CNN does not preserve initial shape: raw error {initial_raw_error}")
    else:
        residual_model = None
        raw_parameter = torch.nn.Parameter(torch.zeros_like(anchor))
        parameters = [raw_parameter]
        get_raw = lambda: raw_parameter
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        initial_raw_error = 0.
    curvature = FixedCurvaturePenalty(args.curvature_gradient_ratio, args.curvature_start_step)
    def save_state(step):
        state = {"step": step, "arguments": vars(args), "optimizer": optimizer.state_dict(),
                 "curvature_calibration": curvature.calibration}
        if residual_model is None:
            state["raw"] = raw_parameter.detach().cpu()
        else:
            state["residual_model"] = {key: value.detach().cpu() for key, value in residual_model.state_dict().items()}
        label = "initial" if step == -1 else f"step{step:03d}"
        torch.save(state, output/f"state_{label}.pt")
    if args.save_states or residual_model is not None:
        save_state(-1)
    history = []; started = time.perf_counter()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        corr = torch.zeros((), device=device)
        indices = (torch.arange(args.train_probe_batch, device=device)
                   + step * args.train_probe_batch) % len(train_back)
        for start in range(0, len(indices), 4):
            selected = indices[start:start+4]
            raw = get_raw()
            shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
            prediction = model_action(shape, train_back[selected], train_cs, operator)
            chunk_loss, chunk_corr = vector_loss(prediction, train_target[selected])
            weight = len(selected) / len(indices)
            (chunk_loss * weight).backward()
            loss += chunk_loss.detach() * weight
            corr += chunk_corr.detach() * weight
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Invalid loss/gradient at step {step}")
        data_gradient = checked_gradient_norm(parameters)
        curvature_stats = curvature.apply(raw, step)
        gradient = checked_gradient_norm(parameters)
        if curvature.calibration is not None and step == curvature.start_step:
            print(json.dumps({"curvature_calibration": curvature.calibration}), flush=True)
        optimizer.step()
        if step == 0 or (step+1)%20 == 0:
            with torch.no_grad():
                raw = get_raw()
                shape, correction = bounded_anchor_shape(anchor, raw, args.bound)
                val_loss, val_corr = validate(shape)
            row = {"step": step, "train_loss": float(loss.detach()), "train_correlation": float(corr.detach()),
                   "holdout_loss": val_loss, "holdout_correlation": val_corr, "gradient_norm": gradient,
                   "data_gradient_norm": data_gradient, **curvature_stats,
                   "total_train_loss": float(loss.detach()) + curvature_stats["weighted_curvature_loss"],
                   "correction_rms": float(correction.square().mean().sqrt()),
                   "elapsed_s": time.perf_counter()-started,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device)/2**20}
            history.append(row); save_volume(output/f"reconstruction_step{step:03d}", shape)
            if args.curvature_gradient_ratio > 0 or args.save_states or residual_model is not None:
                save_state(step)
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step; save_volume(output/"reconstruction_best", shape)
            print(json.dumps(row), flush=True)
    with torch.no_grad(): shape, _ = bounded_anchor_shape(anchor,get_raw(),args.bound)
    save_volume(output/"reconstruction_final",shape)
    summary = {"arguments":vars(args), "steps":200,
               "warning":("Fixed true 50 um, so not a deployable depth reconstruction. Targets use observed train/holdout frames; image truth only logs a diagnostic noise floor, not gradients or selection." if args.target_source == "frames" else "Fixed true 50 um; targets computed from image truth, not deployable. Finite ensembles share sampled covariance; finite_phase uses exact population moment formulas."),
               "cs_pattern_count_applies":args.cs_type=="finite",
               "selection":"Independent sensor probes; independent model ensembles for finite Cs; no truth-image metrics for checkpoint selection",
               "target_link":args.target_link, "initial_holdout_loss":initial_loss,
               "target_source": args.target_source, "shape_model": args.shape_model,
               "initial_raw_max_error": initial_raw_error,
               "optimized_parameter_count": sum(p.numel() for p in parameters),
               "frames": {"train": metadata["train_frames"], "holdout": metadata["holdout_frames"]},
               "initial_holdout_correlation":initial_corr, "true_shape_holdout_loss":true_shape_holdout_loss,
               "best_step":best_step,"best_holdout_loss":best_loss,"history":history,
               "curvature_calibration":curvature.calibration,
               "curvature_domain":"pre-tanh raw log residual; not reconstructed image or anchor",
               "loss_timing":"training losses and gradients before update; holdout and image metrics after update"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--config",default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--dataset",default="outputs/linear_float_oracle_50um/dataset")
    parser.add_argument("--cs-type",choices=("finite","stationary_empirical","stationary_analytic","finite_phase"),required=True)
    parser.add_argument("--cs-directory",default="outputs/linear_float_oracle_50um/exact_population_cs")
    parser.add_argument("--patterns",type=int,default=1024)
    parser.add_argument("--probe-sigma",type=float,default=4.0)
    parser.add_argument("--bound",type=float,default=0.5)
    parser.add_argument("--target-link",choices=("matched","independent"),default="matched")
    parser.add_argument("--seed",type=int,default=20260901)
    parser.add_argument("--train-probe-batch", type=int, choices=range(1,33), default=1)
    parser.add_argument("--curvature-gradient-ratio", type=float, default=0.0)
    parser.add_argument("--curvature-start-step", type=int, default=20)
    parser.add_argument("--target-source", choices=("oracle", "frames"), default="oracle")
    parser.add_argument("--shape-model", choices=("pixels", "cnn"), default="pixels")
    parser.add_argument("--network-lr", type=float, default=1e-3)
    parser.add_argument("--save-states", action="store_true")
    args = parser.parse_args()
    # Validate before any expensive physics loading.
    FixedCurvaturePenalty(args.curvature_gradient_ratio, args.curvature_start_step)
    if args.target_source == "frames" and args.cs_type != "finite_phase":
        parser.error("Frame-target diagnostic requires the fixed finite_phase population model")
    if args.shape_model == "cnn" and args.curvature_gradient_ratio != 0:
        parser.error("CNN diagnostic does not combine a curvature regularizer")
    if not math.isfinite(args.network_lr) or args.network_lr <= 0:
        parser.error("network-lr must be finite and positive")
    train(args)


if __name__=="__main__":
    main()
