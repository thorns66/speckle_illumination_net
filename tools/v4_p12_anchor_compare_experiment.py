"""Frozen two-arm V4 experiment: Taylor versus Mean reconstruction anchors.

Both arms retain the Taylor, Mean, and Set feature encoders and differ only in
the volume used as the positive reconstruction anchor and for analytic beta.
"""
from __future__ import annotations

import copy
import json
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
        "V4_ANCHOR_OUTPUT",
        ROOT / "outputs/v4_p12_anchor_compare_e3_mean100_400_20260909_run01",
    )
)
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
SHARED_INITIAL_STATE = (
    ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01/shared_initial_state.pt"
)
ARMS = ("taylor_anchor_e3_mean100", "mean_anchor_e3_mean100")
ANCHORS = {
    "taylor_anchor_e3_mean100": "taylor_sqrt",
    "mean_anchor_e3_mean100": "mean_rl3",
}

_NATIVE_VALIDATION = base.validate_config.native_validation
_BASE_LOAD_OPERATOR = base.load_operator
_BASE_LOSS = base.loss
_BASE_INSTALL = base.install_training
_BASE_RUN_TRAIN = base.run_train
_ORIGINAL_FORWARD = base._ORIGINAL_FORWARD
_STATE = base._STATE


def _arm(config: dict[str, Any]) -> str:
    return str(config["v3_compare"]["kind"])


def _anchor(config: dict[str, Any]) -> str:
    return ANCHORS[_arm(config)]


def validate_config(config: dict[str, Any]) -> None:
    arm = _arm(config)
    if arm not in ARMS:
        raise ValueError(f"Unknown V4 anchor comparison arm: {arm}")
    anchor = ANCHORS[arm]
    if config["model"].get("reconstruction_anchor", "taylor_sqrt") != anchor:
        raise ValueError("Model reconstruction anchor does not match the arm")
    spec = config["v3_compare"]
    if spec.get("beta_anchor") != anchor or spec.get("beta_cache_key") != anchor:
        raise ValueError("Analytic beta and its cache must match the reconstruction anchor")
    if float(spec["shape_gradient_budget"]) != 1.0:
        raise ValueError("Mean-structure gradient budget must remain 100%")
    if Path(spec["shared_initial_state"]).resolve() != SHARED_INITIAL_STATE.resolve():
        raise ValueError("Both arms must use the frozen common initialization")
    if int(spec["ramp_steps"]) != 50 or int(spec["lr_drop_after_steps"]) != 200:
        raise ValueError("Expected the fixed 50-step ramp and step-200 LR drop")
    if float(spec.get("shape_coefficient_cap", 1.0)) != 1.0:
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
        raise ValueError("The Taylor feature encoder must keep the saved sqrt input")
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


def anchor_volume(item: dict[str, Any], anchor: str) -> torch.Tensor:
    if anchor == "taylor_sqrt":
        return item["f_var"]
    if anchor == "mean_rl3":
        return item["g_mean"]
    raise ValueError(f"Unknown anchor: {anchor}")


def analytic_beta0(operator, anchor: torch.Tensor, input_mean: torch.Tensor) -> torch.Tensor:
    prediction = operator(anchor)
    numerator = (prediction * input_mean).flatten(1).sum(dim=1)
    denominator = prediction.square().flatten(1).sum(dim=1).add(1e-8)
    value = (numerator / denominator).clamp_min(1e-8)
    if not torch.isfinite(value).all():
        raise FloatingPointError("Analytic beta0 is non-finite")
    return value


def cached_beta0(cache: dict, item: dict[str, Any], operator, *, config: dict[str, Any]) -> torch.Tensor:
    anchor = _anchor(config)
    key = (str(item["sample_id"]), int(item["subset_index"]), anchor)
    volume = anchor_volume(item, anchor)
    if key not in cache:
        cache[key] = float(analytic_beta0(operator, volume, item["input_mean"]).item())
    return volume.new_tensor([cache[key]])


def forward(model, item, operator, beta0=None, *, config: dict[str, Any]):
    anchor = _anchor(config)
    volume = anchor_volume(item, anchor)
    if beta0 is None:
        beta0 = analytic_beta0(operator, volume, item["input_mean"])
    output, beta0 = _ORIGINAL_FORWARD(model, item, operator, beta0)
    network_anchor = volume * output.beta[:, None, None, None, None]
    pre_gain = output.reconstruction
    total = pre_gain.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-30)
    q = pre_gain / total
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
    output._pre_gain_reconstruction = pre_gain
    output._effective_correction = reconstruction - physical_anchor
    output._anchor_kind = anchor
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
    trainer._cached_beta0 = lambda cache, item, operator: cached_beta0(
        cache, item, operator, config=config
    )
    previous_write = trainer._write_json

    def write_json(path, value):
        if path.name == "run_contract.json":
            value = dict(value)
            value.update(
                train_objects=[f"P{i:02d}" for i in range(1, 13)],
                validation_objects=["V01", "V02", "V03"],
                test_objects=["T02", "T03", "T04"],
                reconstruction_anchor=_anchor(config),
                analytic_beta_anchor=_anchor(config),
                beta_cache_key_includes_anchor=True,
                p12_in_training=True,
            )
        previous_write(path, value)

    trainer._write_json = write_json


def run_train(config_path: str | Path, resume: bool = False) -> None:
    activate()
    _BASE_RUN_TRAIN(config_path, resume=resume)
