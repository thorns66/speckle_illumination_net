"""Isolated from-scratch E3+mean<=100% experiment with a Mean-RL3 anchor."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

import training.multivolume_trainer as trainer
from models.configurable_anchor_lfm_net import model_from_config
from tools import three_way_experiment as old
from tools import v3_compare_experiment as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "MEAN_ANCHOR_OUTPUT",
        ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01",
    )
)
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
REFERENCE_OUTPUT = ROOT / "outputs/v3_mean050_mean100_400_20260908_run01"
SHARED_INITIAL_STATE = (
    ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01/shared_initial_state.pt"
)
ARM = "mean_anchor_e3_mean100"
ARMS = (ARM,)
BUDGETS = {ARM: 1.0}
ANCHOR = "mean_rl3"

_NATIVE_VALIDATION = base.validate_config.native_validation
_BASE_LOAD_OPERATOR = base.load_operator
_BASE_LOSS = base.loss
_BASE_INSTALL = base.install_training
_BASE_RUN_TRAIN = base.run_train
_STATE = base._STATE
_ORIGINAL_FORWARD = base._ORIGINAL_FORWARD


def validate_config(config: dict[str, Any]) -> None:
    current = config["v3_compare"]
    if str(current["kind"]) != ARM:
        raise ValueError(f"Expected comparison arm {ARM!r}")
    if config["model"].get("reconstruction_anchor", "taylor_sqrt") != ANCHOR:
        raise ValueError("The Mean-anchor experiment must use reconstruction_anchor=mean_rl3")
    if current.get("beta_anchor") != ANCHOR or current.get("beta_cache_key") != ANCHOR:
        raise ValueError("Analytic beta and its cache must both identify the Mean-RL3 anchor")
    if float(current["shape_gradient_budget"]) != 1.0:
        raise ValueError("Mean-structure gradient budget must remain 100%")
    if Path(current["shared_initial_state"]).resolve() != SHARED_INITIAL_STATE.resolve():
        raise ValueError("Experiment does not use the original shared initialization")
    if int(current["ramp_steps"]) != 50 or int(current["lr_drop_after_steps"]) != 200:
        raise ValueError("Expected the fixed 50-step ramp and step-200 LR drop")
    if float(current.get("shape_coefficient_cap", 1.0)) != 1.0:
        raise ValueError("The shape coefficient cap must remain one")
    if int(config["optimization"]["max_steps"]) != 400:
        raise ValueError("Training must contain exactly 400 optimizer updates")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("Global batch size must remain eight")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("Validation interval must remain 20 updates")
    if bool(config["runtime"]["amp"]):
        raise ValueError("AMP must remain disabled")
    if config["data"].get("var_feature_representation", "sqrt") != "sqrt":
        raise ValueError("The Taylor feature branch must use the saved square root once")
    if float(config["loss"]["lambda_mean"]) != 0.0 or float(config["loss"]["lambda_var"]) != 1.0:
        raise ValueError("Compatibility loss fields changed")
    compatibility = copy.deepcopy(config)
    _NATIVE_VALIDATION(compatibility)


validate_config.native_validation = _NATIVE_VALIDATION


def configure_precision() -> None:
    base.configure_precision()


def build_model(config: dict[str, Any], *, initial: bool = False) -> torch.nn.Module:
    configure_precision()
    model = model_from_config(config)
    if initial:
        payload = torch.load(SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        if old.state_hash(model.state_dict()) != payload["common_state_sha256"]:
            raise ValueError("Common network initialization changed")
    model.register_parameter("mean_gain_gamma", torch.nn.Parameter(torch.zeros(())))
    return model


def load_operator(config: dict[str, Any], device: torch.device):
    configure_precision()
    return _BASE_LOAD_OPERATOR(config, device)


def analytic_beta0(operator, anchor: torch.Tensor, input_mean: torch.Tensor) -> torch.Tensor:
    prediction = operator(anchor)
    numerator = (prediction * input_mean).flatten(1).sum(dim=1)
    denominator = prediction.square().flatten(1).sum(dim=1).add(1e-8)
    value = (numerator / denominator).clamp_min(1e-8)
    if not torch.isfinite(value).all():
        raise FloatingPointError("Mean-anchor analytic beta0 is non-finite")
    return value


def cached_beta0(cache: dict, item: dict[str, Any], operator) -> torch.Tensor:
    key = (str(item["sample_id"]), int(item["subset_index"]), ANCHOR)
    if key not in cache:
        cache[key] = float(analytic_beta0(operator, item["g_mean"], item["input_mean"]).item())
    return item["g_mean"].new_tensor([cache[key]])


def forward(model, item, operator, beta0=None, *, config):
    if beta0 is None:
        beta0 = analytic_beta0(operator, item["g_mean"], item["input_mean"])
    output, beta0 = _ORIGINAL_FORWARD(model, item, operator, beta0)
    network_anchor = item["g_mean"] * output.beta[:, None, None, None, None]
    g0 = output.reconstruction
    total = g0.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-30)
    q = g0 / total
    with torch.no_grad():
        hq = operator(q.detach())
        a0 = (hq * item["input_mean"]).sum(dim=(1, 2, 3)) / hq.square().sum(
            dim=(1, 2, 3)
        ).clamp_min(1e-30)
        a0 = a0.clamp_min(0)
    instance = model.module if hasattr(model, "module") else model
    gain = a0 * (
        1.0 + float(config["three_way"]["gain_bound"]) * torch.tanh(instance.mean_gain_gamma)
    )
    reconstruction = q * gain[:, None, None, None, None]
    physical_anchor = network_anchor * (gain[:, None, None, None, None] / total)
    output.reconstruction = reconstruction
    output._shape = q
    output._gain = gain
    output._mean_prediction = hq * gain[:, None, None, None]
    output._network_anchor = network_anchor
    output._physical_anchor = physical_anchor
    output._pre_gain_reconstruction = g0
    output._effective_correction = reconstruction - physical_anchor
    output._anchor_kind = ANCHOR
    output._v3_training = bool(model.training and torch.is_grad_enabled())
    return output, beta0


def loss(output, item, operator, variance_model, config):
    return _BASE_LOSS(output, item, operator, variance_model, config)


def optimizer(model: torch.nn.Module, config: dict[str, Any]):
    return base._optimizer(model, config)


def activate() -> None:
    base.OUTPUT = OUTPUT
    base.DATA = DATA
    base.ARMS = ARMS
    base.validate_config = validate_config
    base.build_model = build_model
    base.load_operator = load_operator
    base.forward = forward
    base.loss = loss
    base.install_training = install_training


def install_training(config: dict[str, Any]) -> None:
    activate()
    _BASE_INSTALL(config)
    trainer._cached_beta0 = cached_beta0
    original_write = trainer._write_json

    def write_json(path, value):
        if path.name == "run_contract.json":
            value = dict(value)
            value.update(
                reconstruction_anchor=ANCHOR,
                analytic_beta_anchor=ANCHOR,
                beta_cache_key_includes_anchor=True,
            )
        original_write(path, value)

    trainer._write_json = write_json


def run_train(config_path: str | Path, resume: bool = False) -> None:
    activate()
    _BASE_RUN_TRAIN(config_path, resume=resume)

