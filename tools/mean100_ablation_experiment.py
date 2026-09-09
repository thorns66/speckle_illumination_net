"""Process-local adapter for E3+100% continuation and no-Set experiments."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

import training.multivolume_trainer as trainer
from tools import v3_compare_experiment as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v3_mean100_extend600_noset600_20260908_run01"
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
BASELINE_OUTPUT = ROOT / "outputs/v3_mean050_mean100_400_20260908_run01"
BASELINE_ARM = BASELINE_OUTPUT / "e3_mean100"
REFERENCE_OUTPUT = BASELINE_OUTPUT
SHARED_INITIAL_STATE = (
    ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01/shared_initial_state.pt"
)
ARMS = ("extend600", "noset600")

_NATIVE_VALIDATION = base.validate_config.native_validation
_BASE_INSTALL_TRAINING = base.install_training


def arm_from_config(config: dict[str, Any]) -> str:
    arm = Path(config["experiment"]["output_dir"]).name
    if arm not in ARMS:
        raise ValueError(f"Unknown E3+100% 600-step arm: {arm}")
    return arm


def validate_config(config: dict[str, Any]) -> None:
    arm = arm_from_config(config)
    spec = config["v3_compare"]
    if str(spec["kind"]) != "e3_mean100":
        raise ValueError("Both arms must retain the E3+100% objective")
    if float(spec["shape_gradient_budget"]) != 1.0:
        raise ValueError("The mean-structure gradient budget must remain 100%")
    if float(spec.get("shape_coefficient_cap", 1.0)) != 1.0:
        raise ValueError("The mean-structure coefficient cap must remain 1")
    if int(spec["ramp_steps"]) != 50 or int(spec["lr_drop_after_steps"]) != 200:
        raise ValueError("Expected the fixed 50-step ramp and step-200 LR drop")
    if int(config["optimization"]["max_steps"]) != 600:
        raise ValueError("Both arms must end at optimizer step 600")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("Global batch size must remain 8")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("Validation must run every 20 steps")
    expected_set = arm == "extend600"
    if bool(config["ablation"]["use_set_branch"]) != expected_set:
        raise ValueError(f"Unexpected SetBranch state for {arm}")
    if not bool(config["ablation"]["use_gate"]):
        raise ValueError("The legacy trainer requires the gate compatibility flag")
    if arm == "noset600" and not bool(spec.get("from_scratch", False)):
        raise ValueError("noset600 must be declared as a from-scratch ablation")
    if bool(config["runtime"]["amp"]):
        raise ValueError("AMP must remain disabled")
    if config["data"].get("var_feature_representation") != "sqrt":
        raise ValueError("Taylor must read the saved square root exactly once")
    compatibility = copy.deepcopy(config)
    _NATIVE_VALIDATION(compatibility)


validate_config.native_validation = _NATIVE_VALIDATION


def activate() -> None:
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


def install_training(config: dict[str, Any]) -> None:
    activate()
    _BASE_INSTALL_TRAINING(config)
    arm = arm_from_config(config)
    installed_save = trainer._atomic_torch_save

    def save_milestones(payload, path):
        installed_save(payload, path)
        if (
            arm == "noset600"
            and path.name == "checkpoint_last.pt"
            and int(payload["completed_steps"]) == 400
        ):
            base._ORIGINAL_ATOMIC_SAVE(payload, path.with_name("checkpoint_step_000400.pt"))

    trainer._atomic_torch_save = save_milestones


def run_train(config_path: str | Path, resume: bool = False) -> None:
    activate()
    original_install = base.install_training
    try:
        base.install_training = install_training
        base.run_train(config_path, resume=resume)
    finally:
        base.install_training = original_install
