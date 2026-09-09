"""Isolated second-round mean/variance refinement; original sources stay frozen.

The shape-gradient budget is measured at q, not at optimizer parameters.  All
validation calls use the same original E3 mean + normalized variance + TV score.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import LossBreakdown, TaylorH2VarianceModel, total_variation_3d
from tools import three_way_experiment as old

ROOT = old.ROOT
OUTPUT = ROOT / "outputs/mean_refinement_round2_20260908"
DATA = old.DATA
SOURCE_CHECKPOINT = old.OUTPUT / "e3/checkpoint_last.pt"
KINDS = ("r0_continue", "r1_no_mean", "r2_input_scale", "r3_mean_shape_005", "r4_mean_shape_010")
INPUT_FIELDS = ("f_var", "f_var_feature", "g_mean", "input_mean", "residual_frames")
_ORIGINAL_PREFLIGHT = trainer._preflight
_ORIGINAL_CONFIGURE = trainer._configure_trainable_parameters
_ORIGINAL_VALIDATE = trainer._validate_config
_STATE = {"completed_steps": 0, "micro_call": 0, "phase": "standalone", "evaluation": False}


def _validate_config(config):
    spec = config["round2"]
    if spec["kind"] not in KINDS or config["three_way"]["kind"] != "e3":
        raise ValueError("Round 2 requires a registered E3 model and a known arm")
    if int(spec["start_step"]) != 200:
        raise ValueError("Every arm must start from the original E3 final step 200")
    if Path(spec["source_checkpoint"]).resolve() != SOURCE_CHECKPOINT.resolve():
        raise ValueError("Unexpected round-2 initialization checkpoint")
    expected_budget = {"r3_mean_shape_005": .05, "r4_mean_shape_010": .10}.get(spec["kind"], 0.)
    if float(spec["shape_gradient_budget"]) != expected_budget:
        raise ValueError("Shape-gradient budget does not match arm")
    if bool(spec["normalize_input"]) != (spec["kind"] == "r2_input_scale"):
        raise ValueError("Only r2 normalizes network inputs")
    if bool(spec["mean_gain_trainable"]) != (spec["kind"] != "r1_no_mean"):
        raise ValueError("Only r1 freezes the mean gain")
    if int(spec["ramp_steps"]) < 1 or not 0 < float(spec["shape_coefficient_cap"]) <= 1:
        raise ValueError("Invalid gradient ramp or coefficient cap")


def validate_trainer_config(config):
    _validate_config(config)
    if float(config["loss"]["lambda_mean"]) != 1. or float(config["loss"]["lambda_var"]) != 1.:
        raise ValueError("Round 2 requires the common E3 mean and variance scoring weights")
    # Keep every original data/model/runtime constraint. Its one no-mean-only
    # assertion belongs to the first experiment, whose loss is replaced here.
    compatibility = copy.deepcopy(config)
    compatibility["loss"]["lambda_mean"] = 0.
    _ORIGINAL_VALIDATE(compatibility)


def configure_precision():
    # The round-2 homogeneity audit localized a 1e-4 discrepancy to cuDNN
    # TF32 input quantization. Apply full FP32 equally to every arm.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def build_model(config, initial=False):
    configure_precision()
    _validate_config(config)
    model = old.build_model(config, initial=False)
    if initial:
        source = Path(config["round2"]["source_checkpoint"])
        if old.sha256(source) != config["round2"]["source_checkpoint_sha256"]:
            raise ValueError("E3 source checkpoint changed")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if int(checkpoint["completed_steps"]) != 200:
            raise ValueError("Initialization is not the E3 final checkpoint")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        with torch.no_grad():
            model.mean_gain_gamma.zero_()
    model.mean_gain_gamma.requires_grad_(bool(config["round2"]["mean_gain_trainable"]))
    return model


def load_operator(config, device, *, sparse=True):
    configure_precision()
    return old.load_operator(config, device, sparse=sparse)


def forward(model, item, operator, beta0=None, *, config):
    spec = config["round2"]
    working = item
    input_scale = item["input_mean"].new_ones(item["input_mean"].shape[0])
    if spec["normalize_input"]:
        reference = float(config["three_way"]["brightness_reference"])
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("Brightness reference must be positive and fixed")
        input_scale = item["input_mean"].mean(dim=(1, 2, 3)) / reference
        divisor = torch.where(input_scale > 0, input_scale, torch.ones_like(input_scale))
        working = dict(item)
        for name in INPUT_FIELDS:
            working[name] = item[name] / divisor.reshape((-1,) + (1,) * (item[name].ndim - 1))
        beta0 = trainer._analytic_beta0(operator, working["f_var"], working["input_mean"])
    output, beta0 = old.ORIGINAL_FORWARD(model, working, operator, beta0)
    g0 = output.reconstruction
    total = g0.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-30)
    q = g0 / total
    with torch.no_grad():
        hq = operator(q.detach())
        a0 = (hq * item["input_mean"]).sum(dim=(1, 2, 3)) / hq.square().sum(dim=(1, 2, 3)).clamp_min(1e-30)
        a0 = a0.clamp_min(0)
    base = model.module if hasattr(model, "module") else model
    gain = a0 * (1 + float(config["three_way"]["gain_bound"]) * torch.tanh(base.mean_gain_gamma))
    physical_factor = gain[:, None, None, None, None] / total
    output.reconstruction = q * gain[:, None, None, None, None]
    output._shape = q
    output._gain = gain
    output._mean_prediction = hq * gain[:, None, None, None]
    output._input_scale = input_scale
    output._network_anchor = working["f_var"] * output.beta[:, None, None, None, None]
    output._physical_anchor = output._network_anchor * physical_factor
    output._round2_training = bool(model.training and torch.is_grad_enabled())
    output._round2_diagnostics = {}
    return output, beta0


class Round2LossBreakdown(LossBreakdown):
    """Keep the old public fields; diagnostic names retain their true meaning."""

    def scalar_metrics(self):
        result = super().scalar_metrics()
        result.update(getattr(self, "round2_metrics", {}))
        return result


def _norm(tensor):
    return torch.linalg.vector_norm(tensor.detach())


def _append_diagnostic(config, item, values):
    _STATE["micro_call"] += 1
    record = {
        "rank": int(os.environ.get("RANK", 0)),
        "phase": _STATE["phase"],
        "optimizer_completed_steps": int(_STATE["completed_steps"]),
        "update_step": int(_STATE["completed_steps"]) + 1,
        "source_completed_steps": 200,
        "micro_call": int(_STATE["micro_call"]),
        "sample_id": str(item.get("sample_id", "")),
        "subset_index": int(item.get("subset_index", 0)),
        **values,
    }
    directory = Path(config["experiment"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gradient_diagnostics_rank{record['rank']}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def loss(output, item, operator, variance_model, config):
    spec = config["round2"]
    q = output._shape
    mu = item["measured_mean"]
    target = item["measured_variance"]
    pred_mu = output._mean_prediction
    scale = mu.detach().abs().mean().clamp_min(1e-8)
    raw_mean = F.smooth_l1_loss(pred_mu, mu)
    mean = F.smooth_l1_loss(pred_mu / scale, mu / scale)
    pred = variance_model(q, mu)
    eps = float(config["loss"]["var_log_eps"])
    normalized_pred = pred / pred.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    normalized_target = target / target.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    var = F.smooth_l1_loss(torch.log(normalized_pred.clamp_min(0) + eps), torch.log(normalized_target + eps))
    physical_pred = pred * output._gain.detach()[:, None, None, None].square()
    raw_var = F.smooth_l1_loss(physical_pred, target)
    tv = total_variation_3d(q * output._gain.detach()[:, None, None, None, None], z_weight=config["loss"]["lambda_tv_z"])
    weighted_tv = tv * float(config["loss"]["lambda_tv"])
    zero = tv.new_zeros(())
    training = bool(output._round2_training and torch.is_grad_enabled() and not _STATE["evaluation"])
    weighted_mean = mean if (not training or spec["mean_gain_trainable"]) else mean * 0.
    shape_mean = zero
    coefficient = zero
    diagnostics = {}
    # Even the controls measure these derivatives: this is a diagnostic and has
    # zero coefficient in their optimization objective.
    if training:
        hq_shape = operator(q)
        pred_shape = hq_shape / hq_shape.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        target_shape = mu / mu.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        shape_mean = F.smooth_l1_loss(pred_shape, target_shape)
        grad_var = torch.autograd.grad(var, q, retain_graph=True, create_graph=False)[0]
        grad_mean = torch.autograd.grad(shape_mean, q, retain_graph=True, create_graph=False)[0]
        grad_tv = torch.autograd.grad(weighted_tv, q, retain_graph=True, create_graph=False)[0]
        finite = torch.stack([torch.isfinite(g).all() for g in (grad_var, grad_mean, grad_tv)]).all()
        norm_var, norm_mean, norm_tv = [_norm(g) for g in (grad_var, grad_mean, grad_tv)]
        denominator_floor = float(spec.get("shape_gradient_eps", 1e-12))
        ramp = min((int(_STATE["completed_steps"]) + 1) / int(spec["ramp_steps"]), 1.)
        budget = float(spec["shape_gradient_budget"]) * ramp
        coefficient = (budget * norm_var / (norm_mean + denominator_floor)).clamp(max=float(spec["shape_coefficient_cap"])).detach()
        ratio = coefficient * norm_mean / (norm_var + denominator_floor)
        cosine = (grad_var.detach() * grad_mean.detach()).sum() / (norm_var * norm_mean).clamp_min(denominator_floor)
        numerical = torch.stack([norm_var, norm_mean, norm_tv, coefficient, ratio, cosine]).cpu().tolist()
        diagnostics = dict(zip(("var_q_gradient_norm", "mean_shape_q_gradient_norm", "weighted_tv_q_gradient_norm", "shape_coefficient", "shape_gradient_ratio", "mean_var_gradient_cosine"), numerical))
        diagnostics.update({"shape_gradient_budget": float(spec["shape_gradient_budget"]), "ramp": ramp,
                            "effective_gradient_budget": budget, "gradients_finite": bool(finite.item()),
                            "shape_mean_loss": float(shape_mean.detach()), "weighted_shape_mean_loss": float((coefficient * shape_mean).detach())})
        diagnostics["budget_passed"] = bool(diagnostics["shape_gradient_ratio"] <= budget + max(1e-7, budget * 1e-5))
        output._round2_diagnostics = diagnostics
        if not diagnostics["gradients_finite"] or not diagnostics["budget_passed"] or not all(math.isfinite(v) for v in numerical):
            raise FloatingPointError(f"Round-2 gradient acceptance failed: {diagnostics}")
        _append_diagnostic(config, item, diagnostics)
        del grad_var, grad_mean, grad_tv
    total_loss = weighted_mean + var + weighted_tv + coefficient * shape_mean
    result = Round2LossBreakdown(total_loss, raw_mean, mean, weighted_mean, raw_var, var, var,
                                zero, zero, tv, weighted_tv, pred_mu, physical_pred)
    result.round2_metrics = {
        "mean_shape_loss": float(shape_mean.detach()),
        "weighted_mean_shape_loss": float((coefficient * shape_mean).detach()),
        "mean_shape_coefficient": float(coefficient.detach()),
        "mean_shape_gradient_ratio": diagnostics.get("shape_gradient_ratio", 0.),
        "mean_var_q_gradient_cosine": diagnostics.get("mean_var_gradient_cosine", 0.),
        "gradient_diagnostics_computed": float(training),
    }
    return result


def install_training(config):
    _validate_config(config)
    old.install_data()
    _STATE.update(completed_steps=0, micro_call=0, phase="training", evaluation=False)
    trainer._validate_config = validate_trainer_config
    trainer._model_from_config = lambda c: build_model(c, initial=True)
    trainer._load_operator = lambda c, p, d, **kwargs: load_operator(c, d)
    trainer._forward = lambda model, item, operator, beta0=None: forward(model, item, operator, beta0, config=config)
    trainer._loss = loss
    trainer.TaylorH2VarianceModel = TaylorH2VarianceModel

    def configure(model):
        _ORIGINAL_CONFIGURE(model)
        model.mean_gain_gamma.requires_grad_(bool(config["round2"]["mean_gain_trainable"]))
        return _ORIGINAL_CONFIGURE(model)

    trainer._configure_trainable_parameters = configure

    def optimizer(model, current):
        model.mean_gain_gamma.requires_grad_(bool(current["round2"]["mean_gain_trainable"]))
        groups = [{"params": [p for n, p in model.named_parameters() if n not in ("raw_beta", "mean_gain_gamma") and p.requires_grad],
                   "lr": float(current["optimization"]["lr_network"])},
                  {"params": [model.raw_beta], "lr": float(current["optimization"]["lr_beta"])}]
        if model.mean_gain_gamma.requires_grad:
            groups.append({"params": [model.mean_gain_gamma], "lr": float(current["three_way"]["lr_gain"])})
        instance = torch.optim.Adam(groups)
        def after_step(*args, **kwargs):
            _STATE["completed_steps"] += 1
        instance.register_step_post_hook(after_step)
        return instance

    trainer._optimizer = optimizer

    def preflight(*args, **kwargs):
        previous = _STATE["phase"]
        _STATE["phase"] = "preflight"
        try:
            return _ORIGINAL_PREFLIGHT(*args, **kwargs)
        finally:
            _STATE["phase"] = previous

    trainer._preflight = preflight

    def evaluate(**kwargs):
        _STATE["evaluation"] = True
        try:
            summary, rows = old.ORIGINAL_EVALUATE(**kwargs)
        finally:
            _STATE["evaluation"] = False
        if kwargs["rank"] == 0:
            objects = {}
            for row in rows:
                row["selection_score"] = row["total_loss"]
                objects.setdefault(row["sample_id"], []).append(row["selection_score"])
            summary["selection_score"] = float(np.mean([np.mean(values) for values in objects.values()]))
        summary = trainer._broadcast_object(summary if kwargs["rank"] == 0 else None, kwargs["rank"], kwargs["world_size"])
        return summary, rows

    trainer._evaluate = evaluate


def run_train(config_path, resume=False):
    config = yaml.safe_load(Path(config_path).read_text())
    _validate_config(config)
    torch.set_num_threads(4)
    contract = json.loads((OUTPUT / "preflight.json").read_text())
    config_absolute = str(Path(config_path).resolve())
    if old.sha256(config_absolute) != contract["config_hashes"].get(config_absolute):
        raise ValueError("Configuration changed after round-2 preflight")
    _, fingerprint = old.relocated_index(DATA)
    if fingerprint != contract["dataset_fingerprint"]:
        raise ValueError("Dataset changed after round-2 preflight")
    sources = contract["source_hashes"]
    if not sources:
        raise ValueError("Round-2 preflight contains no frozen source hashes")
    for name, digest in sources.items():
        path = Path(name)
        if not path.is_absolute() or old.sha256(path) != digest:
            raise ValueError(f"Source changed after round-2 preflight: {name}")
    original_contract = json.loads((old.OUTPUT / "preflight.json").read_text())
    for name, digest in original_contract["source_sha256"].items():
        if old.sha256(ROOT / name) != digest:
            raise ValueError(f"Original experiment source changed: {name}")
    folder = Path(config["experiment"]["output_dir"]).resolve()
    if folder.parent != OUTPUT.resolve():
        raise ValueError("Round-2 training output must stay inside the independent output directory")
    install_training(config)
    if resume:
        checkpoint = torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        _STATE["completed_steps"] = int(checkpoint["completed_steps"])
        del checkpoint
    arguments = SimpleNamespace(config=str(config_path), max_steps=None, global_batch_size=None, validate_every=None,
                                phase_chunk_size=None, resume=str(folder / "checkpoint_last.pt") if resume else None,
                                output_dir=str(folder), limit_validation_items=None, limit_test_items=None)
    trainer.run_training(arguments)
