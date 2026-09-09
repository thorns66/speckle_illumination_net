"""Pre-training contracts for the 50%/100% mean-gradient comparison."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import time

import torch
import yaml

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean_budget_experiment as exp
from tools import three_way_experiment as old
from tools import v3_compare_checks as original_checks
from tools import v3_compare_experiment as original_exp


def _config(arm: str) -> dict:
    return yaml.safe_load((exp.OUTPUT / f"{arm}.yaml").read_text(encoding="utf-8"))


def _item(config: dict, device: torch.device):
    previous = original_checks.exp
    original_checks.exp = exp
    try:
        return original_checks._gpu_item(config, device)
    finally:
        original_checks.exp = previous


def _dataset() -> dict:
    previous = original_checks.exp
    original_checks.exp = exp
    try:
        return original_checks._dataset_check()
    finally:
        original_checks.exp = previous


def _initialization() -> dict:
    expected = json.loads((exp.REFERENCE_OUTPUT / "preflight.json").read_text())[
        "shared_initial_common_state_sha256"
    ]
    rows = {}
    for arm in exp.ARMS:
        model = exp.build_model(_config(arm), initial=True)
        common = {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
        digest = old.state_hash(common)
        if digest != expected or float(model.mean_gain_gamma) != 0.0:
            raise ValueError(f"{arm} does not reproduce the original common initialization")
        rows[arm] = digest
    return {"expected_common_state": expected, "arm_common_states": rows}


def _bounded_loss(config: dict, model, item, operator, device, completed_steps: int):
    temporary = Path("/tmp/mean_budget_gpu_check") / config["v3_compare"]["kind"]
    shutil.rmtree(temporary, ignore_errors=True)
    current = copy.deepcopy(config)
    current["experiment"]["output_dir"] = str(temporary)
    exp._STATE.update(completed_steps=completed_steps, phase="gpu_check", evaluation=False, micro_call=0)
    beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
    output, _ = exp.forward(model, item, operator, beta0, config=current)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])
    result = exp.loss(output, item, operator, variance_model, current)
    return output, result


def _compatibility_and_budgets(device: torch.device) -> dict:
    config50 = _config("e3_mean050")
    operator = exp.load_operator(config50, device)
    item = _item(config50, device)

    # The current core must still be the exact source used by the completed 5% run.
    source_contract = json.loads((exp.REFERENCE_OUTPUT / "preflight.json").read_text())
    source_path = str((exp.ROOT / "tools/v3_compare_experiment.py").resolve())
    current_hash = old.sha256(source_path)
    if source_contract["training_source_hashes"].get(source_path) != current_hash:
        raise ValueError("The proven 5% training core changed after the reference experiment")

    old_config = yaml.safe_load((exp.REFERENCE_OUTPUT / "e3_mean005.yaml").read_text())
    compatibility = copy.deepcopy(config50)
    compatibility["v3_compare"].update(kind="e3_mean005", shape_gradient_budget=0.05)
    compatibility["experiment"]["output_dir"] = "/tmp/mean_budget_compatibility"
    old_model = original_exp.build_model(old_config, initial=True).to(device).train()
    new_model = original_exp.build_model(compatibility, initial=True).to(device).train()
    if old.state_hash(old_model.state_dict()) != old.state_hash(new_model.state_dict()):
        raise ValueError("5% compatibility model state differs")
    beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
    original_exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, micro_call=0)
    old_out, _ = original_exp.forward(old_model, item, operator, beta0, config=old_config)
    old_loss = original_exp.loss(
        old_out, item, operator, TaylorH2VarianceModel(operator, **old_config["noise"]), old_config
    )
    original_exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, micro_call=0)
    new_out, _ = original_exp.forward(new_model, item, operator, beta0, config=compatibility)
    new_loss = original_exp.loss(
        new_out, item, operator, TaylorH2VarianceModel(operator, **compatibility["noise"]), compatibility
    )
    forward_error = float(
        (old_out.reconstruction - new_out.reconstruction).float().norm()
        / old_out.reconstruction.float().norm().clamp_min(1e-30)
    )
    loss_error = abs(float(old_loss.total) - float(new_loss.total))
    if forward_error > 1e-4:
        raise ValueError(f"5% compatibility forward differs: relative L2={forward_error}")
    if loss_error > 2e-7:
        raise ValueError(f"5% compatibility loss differs: abs error={loss_error}")

    budget_rows = {}
    for arm in exp.ARMS:
        config = _config(arm)
        model = exp.build_model(config, initial=True).to(device).train()
        output, result = _bounded_loss(config, model, item, operator, device, completed_steps=48)
        metrics = result.scalar_metrics()
        bound = exp.BUDGETS[arm] * 49 / 50
        ratio = float(metrics["mean_shape_gradient_ratio"])
        if not math.isfinite(float(result.total)) or ratio > bound + max(1e-7, bound * 1e-5):
            raise ValueError(f"{arm} exceeded the step-49 gradient bound")
        # Ordinary E3 mean is detached from q and therefore cannot alter structure directly.
        ordinary_q_grad = torch.autograd.grad(
            result.normalized_mean, output._shape, allow_unused=True, retain_graph=True
        )[0]
        if ordinary_q_grad is not None:
            raise ValueError("Ordinary E3 mean unexpectedly has a q gradient")
        result.total.backward()
        network_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name != "mean_gain_gamma" and parameter.grad is not None
        )
        gamma_grad = float(model.mean_gain_gamma.grad.abs())
        if network_grad <= 0 or gamma_grad <= 0:
            raise ValueError(f"{arm} must update structure and brightness")
        budget_rows[arm] = {
            "step49_bound": bound,
            "step49_actual_ratio": ratio,
            "coefficient": float(metrics["mean_shape_coefficient"]),
            "network_gradient_l1": network_grad,
            "gamma_gradient_abs": gamma_grad,
        }

    # Explicitly test the production formula's limiting cases without inventing a loss weight.
    formula = {}
    for name, var_norm, mean_norm in (("zero_mean", 2.0, 0.0), ("near_zero_mean", 2.0, 1e-20), ("zero_var", 0.0, 2.0)):
        coefficient = min(1.0, 1.0 * var_norm / (mean_norm + 1e-12))
        ratio = coefficient * mean_norm / (var_norm + 1e-12)
        if not math.isfinite(coefficient + ratio) or ratio > 1.0 + 1e-7:
            raise ValueError(f"Unstable coefficient edge case: {name}")
        formula[name] = {"coefficient": coefficient, "ratio": ratio}

    # Inference must not read GT or either 90-frame target statistic.
    config = config50
    model = exp.build_model(config, initial=True).to(device).eval()
    clean = {k: v for k, v in item.items() if k not in ("ground_truth", "measured_mean", "measured_variance")}
    beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
    with torch.inference_mode():
        full, _ = exp.forward(model, item, operator, beta0, config=config)
        stripped, _ = exp.forward(model, clean, operator, beta0, config=config)
    denominator = full.reconstruction.norm().clamp_min(1e-30)
    isolation = float((full.reconstruction - stripped.reconstruction).norm() / denominator)
    if isolation != 0.0:
        raise ValueError("Prediction changed after GT and target statistics were removed")
    return {
        "reference_5percent_core_sha256": current_hash,
        "compatibility_forward_relative_l2": forward_error,
        "compatibility_loss_absolute_error": loss_error,
        "budgets": budget_rows,
        "coefficient_edge_cases": formula,
        "target_contamination_relative_l2": isolation,
    }


def _lr_resume() -> dict:
    config = _config("e3_mean050")
    model = exp.build_model(config, initial=True)
    exp._STATE.update(completed_steps=0, phase="gpu_check")
    optimizer = exp.optimizer(model, config)
    boundaries = {}
    for step in range(1, 202):
        optimizer.step()
        if step in (49, 50, 199, 200, 201):
            boundaries[str(step)] = {g["group_name"]: float(g["lr"]) for g in optimizer.param_groups}
    high = {"network": 1e-3, "beta": 1e-4, "mean_gain_gamma": 1e-4}
    low = {"network": 1e-4, "beta": 1e-5, "mean_gain_gamma": 1e-5}
    if boundaries["199"] != high or boundaries["200"] != low or boundaries["201"] != low:
        raise ValueError(f"Learning-rate boundary failed: {boundaries}")
    state = optimizer.state_dict()
    resumed_model = exp.build_model(config, initial=True)
    resumed = exp.optimizer(resumed_model, config)
    resumed.load_state_dict(state)
    before = {g["group_name"]: float(g["lr"]) for g in resumed.param_groups}
    resumed.step()
    after = {g["group_name"]: float(g["lr"]) for g in resumed.param_groups}
    if before != low or after != low:
        raise ValueError("Optimizer state did not preserve the low learning rate")
    return {"rates_after_step": boundaries, "resume_before": before, "resume_after": after}


def gpu_checks() -> dict:
    started = time.time()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    exp.configure_precision()
    result = {"passed": False, "device": torch.cuda.get_device_name(device)}
    try:
        result.update(
            dataset=_dataset(),
            initialization=_initialization(),
            forward_loss_gradient=_compatibility_and_budgets(device),
            lr_and_resume=_lr_resume(),
            numerical_precision="full FP32; TF32 disabled",
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            passed=True,
            seconds=time.time() - started,
        )
        old.write_json(exp.OUTPUT / "gpu_checks.json", result)
        return result
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), seconds=time.time() - started)
        old.write_json(exp.OUTPUT / "gpu_checks.json", result)
        raise
