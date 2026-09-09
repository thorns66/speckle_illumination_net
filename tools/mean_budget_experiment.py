"""Isolated 50%/100% bounded mean-structure experiments on the V3 dataset."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from tools import v3_compare_experiment as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v3_mean050_mean100_400_20260908_run01"
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
REFERENCE_OUTPUT = ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01"
SHARED_INITIAL_STATE = REFERENCE_OUTPUT / "shared_initial_state.pt"
ARMS = ("e3_mean050", "e3_mean100")
BUDGETS = {"e3_mean050": 0.5, "e3_mean100": 1.0}

_NATIVE_VALIDATION = base.validate_config.native_validation
_STATE = base._STATE
_ORIGINAL_FORWARD = base._ORIGINAL_FORWARD
_ORIGINAL_LOSS = base._ORIGINAL_LOSS


def validate_config(config: dict[str, Any]) -> None:
    current = config["v3_compare"]
    kind = str(current["kind"])
    if kind not in ARMS:
        raise ValueError(f"Unknown mean-budget arm: {kind}")
    if float(current["shape_gradient_budget"]) != BUDGETS[kind]:
        raise ValueError("Mean-structure gradient budget does not match the arm")
    if Path(current["shared_initial_state"]).resolve() != SHARED_INITIAL_STATE.resolve():
        raise ValueError("Experiment does not use the original shared initialization")
    if int(current["ramp_steps"]) != 50 or int(current["lr_drop_after_steps"]) != 200:
        raise ValueError("Expected a 50-step ramp and the fixed step-200 learning-rate drop")
    if float(current.get("shape_coefficient_cap", 1.0)) != 1.0:
        raise ValueError("The loss coefficient cap must remain 1")
    if int(config["optimization"]["max_steps"]) != 400:
        raise ValueError("Every arm must train for exactly 400 optimizer updates")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("Global batch size must remain 8")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("Validation interval must remain 20 updates")
    if bool(config["runtime"]["amp"]):
        raise ValueError("AMP must remain disabled")
    if config["data"].get("var_feature_representation", "sqrt") != "sqrt":
        raise ValueError("Taylor input must use the saved square root exactly once")
    if float(config["loss"]["lambda_mean"]) != 0.0 or float(config["loss"]["lambda_var"]) != 1.0:
        raise ValueError("Compatibility loss fields changed")
    compatibility = copy.deepcopy(config)
    _NATIVE_VALIDATION(compatibility)


validate_config.native_validation = _NATIVE_VALIDATION


def activate() -> None:
    """Point the proven V3 adapter at this process-local experiment."""
    base.OUTPUT = OUTPUT
    base.DATA = DATA
    base.ARMS = ARMS
    base.validate_config = validate_config


def configure_precision() -> None:
    activate()
    base.configure_precision()


def build_model(config: dict[str, Any], *, initial: bool = False) -> torch.nn.Module:
    activate()
    return base.build_model(config, initial=initial)


def load_operator(config: dict[str, Any], device: torch.device):
    activate()
    return base.load_operator(config, device)


def forward(model, item, operator, beta0=None, *, config):
    activate()
    return base.forward(model, item, operator, beta0, config=config)


def loss(output, item, operator, variance_model, config):
    activate()
    return base.loss(output, item, operator, variance_model, config)


def optimizer(model: torch.nn.Module, config: dict[str, Any]):
    activate()
    return base._optimizer(model, config)


def run_train(config_path: str | Path, resume: bool = False) -> None:
    activate()
    base.run_train(config_path, resume=resume)

