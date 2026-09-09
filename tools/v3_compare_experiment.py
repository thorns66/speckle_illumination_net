"""Frozen adapters for the V3 baseline/E3/bounded-mean comparison.

This module deliberately patches the established trainer at process scope instead
of changing the historical trainer, model, or loss sources.  The three arms thus
share the same data path and network implementation while differing only in the
declared objective.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import LossBreakdown, TaylorH2VarianceModel, total_variation_3d
from train_volume import _model_from_config as native_model_from_config
from tools import three_way_experiment as old


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01"
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
ARMS = ("baseline", "e3", "e3_mean005")

_ORIGINAL_FORWARD = trainer._forward
_ORIGINAL_LOSS = trainer._loss
_ORIGINAL_EVALUATE = trainer._evaluate
_ORIGINAL_PREFLIGHT = trainer._preflight
_ORIGINAL_CONFIGURE = trainer._configure_trainable_parameters
_ORIGINAL_ATOMIC_SAVE = trainer._atomic_torch_save
_ORIGINAL_WRITE_JSON = trainer._write_json

_STATE: dict[str, Any] = {
    "completed_steps": 0,
    "micro_call": 0,
    "phase": "standalone",
    "evaluation": False,
}


def configure_precision() -> None:
    """Use the same strict FP32 policy in all arms."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _spec(config: dict[str, Any]) -> dict[str, Any]:
    return config["v3_compare"]


def validate_config(config: dict[str, Any]) -> None:
    spec = _spec(config)
    kind = str(spec["kind"])
    if kind not in ARMS:
        raise ValueError(f"Unknown V3 comparison arm: {kind}")
    expected_budget = 0.05 if kind == "e3_mean005" else 0.0
    if float(spec["shape_gradient_budget"]) != expected_budget:
        raise ValueError("The bounded mean gradient budget does not match the arm")
    if int(spec["ramp_steps"]) != 50 or int(spec["lr_drop_after_steps"]) != 200:
        raise ValueError("The V3 comparison requires the fixed 50-step ramp and 200-step LR boundary")
    if int(config["optimization"]["max_steps"]) != 400:
        raise ValueError("Every V3 comparison arm must train for exactly 400 steps")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("The fixed global batch size is 8")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("Validation must run every 20 steps")
    if bool(config["runtime"]["amp"]):
        raise ValueError("AMP must be disabled")
    if config["data"].get("var_feature_representation", "sqrt") != "sqrt":
        raise ValueError("Taylor must use the saved square-root representation exactly once")
    if float(config["loss"]["lambda_mean"]) != 0.0 or float(config["loss"]["lambda_var"]) != 1.0:
        raise ValueError("The compatibility loss fields must retain the baseline values")
    compatibility = copy.deepcopy(config)
    trainer_validation = getattr(validate_config, "native_validation", None)
    if trainer_validation is not None:
        trainer_validation(compatibility)


validate_config.native_validation = trainer._validate_config


def build_model(config: dict[str, Any], *, initial: bool = False) -> torch.nn.Module:
    configure_precision()
    kind = str(_spec(config)["kind"])
    model = native_model_from_config(config)
    if initial:
        source = Path(_spec(config)["shared_initial_state"])
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        if old.state_hash(model.state_dict()) != payload["common_state_sha256"]:
            raise ValueError("The common network initialization changed")
    if kind != "baseline":
        model.register_parameter("mean_gain_gamma", torch.nn.Parameter(torch.zeros(())))
    return model


def load_operator(config: dict[str, Any], device: torch.device):
    configure_precision()
    # Reuse the already numerically checked sparse implementation and selected PSF.
    return old.load_operator(config, device, sparse=True)


def forward(
    model: torch.nn.Module,
    item: dict[str, Any],
    operator: Any,
    beta0: torch.Tensor | None = None,
    *,
    config: dict[str, Any],
):
    kind = str(_spec(config)["kind"])
    output, beta0 = _ORIGINAL_FORWARD(model, item, operator, beta0)
    output._network_anchor = item["f_var"] * output.beta[:, None, None, None, None]
    if kind == "baseline":
        output._physical_anchor = output._network_anchor
        output._shape = output.reconstruction / output.reconstruction.sum(
            dim=(1, 2, 3, 4), keepdim=True
        ).clamp_min(1e-30)
        output._gain = output.reconstruction.sum(dim=(1, 2, 3, 4))
        output._mean_prediction = None
        output._v3_training = bool(model.training and torch.is_grad_enabled())
        return output, beta0

    g0 = output.reconstruction
    total = g0.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-30)
    q = g0 / total
    # Detaching q here is the defining E3 split: the ordinary mean term may tune
    # only gamma, while structure is controlled by normalized variance (and, in C,
    # by the explicitly bounded shape term below).
    with torch.no_grad():
        hq = operator(q.detach())
        a0 = (hq * item["input_mean"]).sum(dim=(1, 2, 3)) / hq.square().sum(
            dim=(1, 2, 3)
        ).clamp_min(1e-30)
        a0 = a0.clamp_min(0)
    base = model.module if hasattr(model, "module") else model
    gain = a0 * (
        1.0 + float(config["three_way"]["gain_bound"]) * torch.tanh(base.mean_gain_gamma)
    )
    output.reconstruction = q * gain[:, None, None, None, None]
    output._shape = q
    output._gain = gain
    output._mean_prediction = hq * gain[:, None, None, None]
    output._physical_anchor = output._network_anchor * (
        gain[:, None, None, None, None] / total
    )
    output._v3_training = bool(model.training and torch.is_grad_enabled())
    return output, beta0


class V3LossBreakdown(LossBreakdown):
    def scalar_metrics(self) -> dict[str, float]:
        result = super().scalar_metrics()
        result.update(getattr(self, "v3_metrics", {}))
        return result


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _gradient_diagnostic(
    config: dict[str, Any], item: dict[str, Any], values: dict[str, Any]
) -> None:
    _STATE["micro_call"] += 1
    rank = int(os.environ.get("RANK", "0"))
    record = {
        "rank": rank,
        "phase": _STATE["phase"],
        "optimizer_completed_steps": int(_STATE["completed_steps"]),
        "update_step": int(_STATE["completed_steps"]) + 1,
        "micro_call": int(_STATE["micro_call"]),
        "sample_id": str(item.get("sample_id", "")),
        "subset_index": int(item.get("subset_index", 0)),
        **values,
    }
    _append_jsonl(
        Path(config["experiment"]["output_dir"])
        / f"gradient_diagnostics_rank{rank}.jsonl",
        record,
    )


def loss(output, item, operator, variance_model, config):
    kind = str(_spec(config)["kind"])
    if kind == "baseline":
        return _ORIGINAL_LOSS(output, item, operator, variance_model, config)

    q = output._shape
    mu = item["measured_mean"]
    target = item["measured_variance"]
    pred_mu = output._mean_prediction
    scale = mu.detach().abs().mean().clamp_min(1e-8)
    raw_mean = F.smooth_l1_loss(pred_mu, mu)
    normalized_mean = F.smooth_l1_loss(pred_mu / scale, mu / scale)

    pred_var_shape = variance_model(q, mu)
    eps = float(config["loss"]["var_log_eps"])
    normalized_prediction = pred_var_shape / pred_var_shape.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-30)
    normalized_target = target / target.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    normalized_var = F.smooth_l1_loss(
        torch.log(normalized_prediction.clamp_min(0) + eps),
        torch.log(normalized_target + eps),
    )
    physical_var = pred_var_shape * output._gain.detach()[:, None, None, None].square()
    raw_var = F.smooth_l1_loss(physical_var, target)
    tv = total_variation_3d(
        q * output._gain.detach()[:, None, None, None, None],
        z_weight=float(config["loss"]["lambda_tv_z"]),
    )
    weighted_tv = tv * float(config["loss"]["lambda_tv"])
    zero = tv.new_zeros(())

    shape_mean = zero
    coefficient = zero
    gradient_ratio = 0.0
    cosine = 0.0
    ramp = min((int(_STATE["completed_steps"]) + 1) / int(_spec(config)["ramp_steps"]), 1.0)
    effective_budget = float(_spec(config)["shape_gradient_budget"]) * ramp
    training = bool(output._v3_training and torch.is_grad_enabled() and not _STATE["evaluation"])
    if training:
        hq_shape = operator(q)
        predicted_shape = hq_shape / hq_shape.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        target_shape = mu / mu.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        shape_mean = F.smooth_l1_loss(predicted_shape, target_shape)
        grad_var = torch.autograd.grad(normalized_var, q, retain_graph=True, create_graph=False)[0]
        grad_mean = torch.autograd.grad(shape_mean, q, retain_graph=True, create_graph=False)[0]
        grad_tv = torch.autograd.grad(weighted_tv, q, retain_graph=True, create_graph=False)[0]
        norm_var = torch.linalg.vector_norm(grad_var.detach())
        norm_mean = torch.linalg.vector_norm(grad_mean.detach())
        norm_tv = torch.linalg.vector_norm(grad_tv.detach())
        floor = float(_spec(config).get("shape_gradient_eps", 1e-12))
        coefficient = torch.minimum(
            q.new_tensor(1.0),
            q.new_tensor(effective_budget) * norm_var / (norm_mean + floor),
        ).detach()
        ratio_tensor = coefficient * norm_mean / (norm_var + floor)
        cosine_tensor = (grad_var.detach() * grad_mean.detach()).sum() / (
            norm_var * norm_mean
        ).clamp_min(floor)
        finite = bool(
            torch.stack([torch.isfinite(x).all() for x in (grad_var, grad_mean, grad_tv)]).all().item()
        )
        gradient_ratio = float(ratio_tensor.cpu())
        cosine = float(cosine_tensor.cpu())
        passed = finite and gradient_ratio <= effective_budget + max(1e-7, effective_budget * 1e-5)
        values = {
            "shape_gradient_budget": float(_spec(config)["shape_gradient_budget"]),
            "ramp": ramp,
            "effective_gradient_budget": effective_budget,
            "var_q_gradient_norm": float(norm_var.cpu()),
            "mean_shape_q_gradient_norm": float(norm_mean.cpu()),
            "weighted_tv_q_gradient_norm": float(norm_tv.cpu()),
            "shape_coefficient": float(coefficient.cpu()),
            "shape_gradient_ratio": gradient_ratio,
            "mean_var_gradient_cosine": cosine,
            "shape_mean_loss": float(shape_mean.detach().cpu()),
            "weighted_shape_mean_loss": float((coefficient * shape_mean).detach().cpu()),
            "gradients_finite": finite,
            "budget_passed": passed,
        }
        _gradient_diagnostic(config, item, values)
        if not passed or not all(
            math.isfinite(float(values[name]))
            for name in (
                "var_q_gradient_norm",
                "mean_shape_q_gradient_norm",
                "weighted_tv_q_gradient_norm",
                "shape_coefficient",
                "shape_gradient_ratio",
                "mean_var_gradient_cosine",
            )
        ):
            raise FloatingPointError(f"Bounded mean-gradient check failed: {values}")
        del grad_var, grad_mean, grad_tv

    total = normalized_mean + normalized_var + weighted_tv + coefficient * shape_mean
    result = V3LossBreakdown(
        total,
        raw_mean,
        normalized_mean,
        normalized_mean,
        raw_var,
        normalized_var,
        normalized_var,
        zero,
        zero,
        tv,
        weighted_tv,
        pred_mu,
        physical_var,
    )
    result.v3_metrics = {
        "mean_shape_loss": float(shape_mean.detach()),
        "weighted_mean_shape_loss": float((coefficient * shape_mean).detach()),
        "mean_shape_coefficient": float(coefficient.detach()),
        "mean_shape_gradient_ratio": gradient_ratio,
        "mean_var_q_gradient_cosine": cosine,
        "effective_shape_gradient_budget": effective_budget if training else 0.0,
        "gradient_diagnostics_computed": float(training),
    }
    return result


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    spec = _spec(config)
    kind = str(spec["kind"])
    network = [
        parameter
        for name, parameter in model.named_parameters()
        if name not in ("raw_beta", "mean_gain_gamma") and parameter.requires_grad
    ]
    groups: list[dict[str, Any]] = [
        {"params": network, "lr": float(config["optimization"]["lr_network"]), "group_name": "network"},
        {"params": [dict(model.named_parameters())["raw_beta"]], "lr": float(config["optimization"]["lr_beta"]), "group_name": "beta"},
    ]
    if kind != "baseline":
        groups.append(
            {
                "params": [dict(model.named_parameters())["mean_gain_gamma"]],
                "lr": float(config["three_way"]["lr_gain"]),
                "group_name": "mean_gain_gamma",
            }
        )
    instance = torch.optim.Adam(groups)

    def after_step(_optimizer, *_args, **_kwargs):
        used_step = int(_STATE["completed_steps"]) + 1
        used = {group["group_name"]: float(group["lr"]) for group in instance.param_groups}
        _STATE["completed_steps"] = used_step
        if used_step == int(spec["lr_drop_after_steps"]):
            for group in instance.param_groups:
                if group["group_name"] == "network":
                    group["lr"] = float(spec["lr_network_after_drop"])
                else:
                    group["lr"] = float(spec["lr_scalar_after_drop"])
        if _STATE["phase"] == "training":
            rank = int(os.environ.get("RANK", "0"))
            _append_jsonl(
                Path(config["experiment"]["output_dir"]) / f"learning_rate_rank{rank}.jsonl",
                {
                    "rank": rank,
                    "step": used_step,
                    "used_lrs": used,
                    "next_lrs": {
                        group["group_name"]: float(group["lr"]) for group in instance.param_groups
                    },
                },
            )

    instance.register_step_post_hook(after_step)
    return instance


def install_training(config: dict[str, Any]) -> None:
    configure_precision()
    validate_config(config)
    _STATE.update(completed_steps=0, micro_call=0, phase="training", evaluation=False)
    trainer._validate_config = validate_config
    trainer._model_from_config = lambda _config: build_model(_config, initial=True)
    trainer._load_operator = lambda current, _path, device, **_kwargs: load_operator(current, device)
    trainer._forward = lambda model, item, operator, beta0=None: forward(
        model, item, operator, beta0, config=config
    )
    trainer._loss = loss
    trainer.TaylorH2VarianceModel = TaylorH2VarianceModel
    trainer._optimizer = _optimizer

    def configure(model):
        return _ORIGINAL_CONFIGURE(model)

    trainer._configure_trainable_parameters = configure

    def evaluate(**kwargs):
        _STATE["evaluation"] = True
        try:
            summary, rows = _ORIGINAL_EVALUATE(**kwargs)
        finally:
            _STATE["evaluation"] = False
        if kwargs["rank"] == 0 and str(_spec(config)["kind"]) != "baseline":
            by_object: dict[str, list[float]] = {}
            for row in rows:
                row["selection_score"] = float(row["total_loss"])
                by_object.setdefault(str(row["sample_id"]), []).append(row["selection_score"])
            summary["selection_score"] = float(
                np.mean([np.mean(values) for values in by_object.values()])
            )
        summary = trainer._broadcast_object(
            summary if kwargs["rank"] == 0 else None,
            kwargs["rank"],
            kwargs["world_size"],
        )
        return summary, rows

    trainer._evaluate = evaluate

    def preflight(*args, **kwargs):
        previous = _STATE["phase"]
        _STATE["phase"] = "trainer_preflight"
        try:
            return _ORIGINAL_PREFLIGHT(*args, **kwargs)
        finally:
            _STATE["phase"] = previous

    trainer._preflight = preflight

    def atomic_save(payload, path):
        _ORIGINAL_ATOMIC_SAVE(payload, path)
        if path.name == "checkpoint_last.pt" and int(payload["completed_steps"]) == 200:
            _ORIGINAL_ATOMIC_SAVE(payload, path.with_name("checkpoint_step_000200.pt"))

    trainer._atomic_torch_save = atomic_save

    def write_json(path, value):
        if path.name == "run_contract.json":
            value = dict(value)
            value.update(
                train_objects=[f"P{i:02d}" for i in range(1, 12)],
                validation_objects=["V01", "V02", "V03"],
                test_objects=["T02", "T03", "T04"],
                numerical_precision="fp32_tf32_disabled",
                comparison_arm=str(_spec(config)["kind"]),
            )
        _ORIGINAL_WRITE_JSON(path, value)

    trainer._write_json = write_json


def run_train(config_path: str | Path, resume: bool = False) -> None:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    torch.set_num_threads(4)
    contract = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    if old.sha256(config_path) != contract["config_hashes"][str(config_path)]:
        raise ValueError("Configuration changed after preflight")
    for name, digest in contract["training_source_hashes"].items():
        if old.sha256(name) != digest:
            raise ValueError(f"Frozen training source changed after preflight: {name}")
    from datasets.matlab_multivolume_dataset import load_dataset_index

    _indexed, fingerprint = load_dataset_index(DATA)
    if fingerprint != contract["dataset_fingerprint"]:
        raise ValueError("V3 dataset fingerprint changed after preflight")
    folder = Path(config["experiment"]["output_dir"]).resolve()
    if folder.parent != OUTPUT.resolve():
        raise ValueError("Training output escaped the isolated V3 comparison directory")
    install_training(config)
    checkpoint_path = folder / "checkpoint_last.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _STATE["completed_steps"] = int(checkpoint["completed_steps"])
        del checkpoint
    arguments = SimpleNamespace(
        config=str(config_path),
        max_steps=None,
        global_batch_size=None,
        validate_every=None,
        phase_chunk_size=None,
        resume=str(checkpoint_path) if resume else None,
        output_dir=str(folder),
        limit_validation_items=None,
        limit_test_items=None,
    )
    trainer.run_training(arguments)

