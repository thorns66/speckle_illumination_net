"""Numerical and data-contract checks run before V3 comparison training."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch
import yaml

import training.multivolume_trainer as trainer
from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset, load_dataset_index
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import three_way_experiment as old
from tools import v3_compare_experiment as exp


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm((left - right).float()).item()
        / max(torch.linalg.vector_norm(right.float()).item(), 1e-30)
    )


def _matlab_yx(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[()]).T.astype(np.float32)


def _matlab_yxz(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[()]).transpose(0, 2, 1).astype(np.float32)


def _raw_frame(sample_dir: Path, index: int) -> np.ndarray:
    with h5py.File(sample_dir / "sensor_frames" / f"frame_{index:03d}.mat", "r") as handle:
        return _matlab_yx(handle["sensor_pre_detector"])


def _dataset_check() -> dict:
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(items) for split, items in indexed.items()}
    if counts != {"train": 110, "validation": 30, "test": 30}:
        raise ValueError(f"Wrong V3 split counts: {counts}")
    key = indexed["validation"][0]
    subset_path = key.sample_dir / "subsets/subset_01.mat"
    with h5py.File(subset_path, "r") as handle:
        input_indices = np.asarray(handle["input_indices"][()]).reshape(-1).astype(int)
        holdout_indices = np.asarray(handle["holdout_indices"][()]).reshape(-1).astype(int)
        saved_input_mean = _matlab_yx(handle["input_physics_mean_float"])
        saved_holdout_mean = _matlab_yx(handle["holdout_physics_mean_float"])
        saved_holdout_var = _matlab_yx(handle["holdout_physics_variance_nminus1_float"])
        taylor_raw = _matlab_yxz(handle["physics_taylor_raw"])
        taylor_sqrt = _matlab_yxz(handle["physics_taylor_sqrt_float"])
    if set(input_indices) & set(holdout_indices) or set(input_indices) | set(holdout_indices) != set(range(1, 101)):
        raise ValueError("The sampled 10/90 frame partition is invalid")
    input_frames = np.stack([_raw_frame(key.sample_dir, int(i)) for i in input_indices])
    holdout_frames = np.stack([_raw_frame(key.sample_dir, int(i)) for i in holdout_indices])
    input_mean = input_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    holdout_mean = holdout_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    holdout_var = holdout_frames.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    checks = {
        "input_mean_exact": bool(np.array_equal(input_mean, saved_input_mean)),
        "holdout_mean_exact": bool(np.array_equal(holdout_mean, saved_holdout_mean)),
        "holdout_variance_exact": bool(np.array_equal(holdout_var, saved_holdout_var)),
        "taylor_sqrt_once_max_abs": float(np.max(np.abs(np.square(taylor_sqrt) - taylor_raw))),
        "taylor_sqrt_once_relative_l2": float(
            np.linalg.norm((np.square(taylor_sqrt) - taylor_raw).ravel())
            / max(np.linalg.norm(taylor_raw.ravel()), 1e-30)
        ),
    }
    if not all(checks[name] for name in ("input_mean_exact", "holdout_mean_exact", "holdout_variance_exact")):
        raise ValueError(f"Raw frame recomputation failed: {checks}")
    # Float square roots do not invert bitwise, so use an explicit numerical line.
    if checks["taylor_sqrt_once_relative_l2"] > 2e-7:
        raise ValueError(f"Taylor saved sqrt is inconsistent with raw Taylor: {checks}")
    return {"fingerprint": fingerprint, "counts": counts, "sample": f"{key.sample_id}/subset_01", **checks}


def _load_config(arm: str) -> dict:
    return yaml.safe_load((exp.OUTPUT / f"{arm}.yaml").read_text(encoding="utf-8"))


def _gpu_item(config: dict, device: torch.device):
    dataset = MatlabMultiVolumeDataset(
        exp.DATA,
        "validation",
        cache_dir=config["data"]["cache_dir"],
        include_ground_truth=True,
        var_feature_representation="sqrt",
    )
    return trainer._to_device(dataset[0], device)


def _same_common_initial() -> dict:
    hashes = {}
    common_hashes = {}
    for arm in exp.ARMS:
        config = _load_config(arm)
        model = exp.build_model(config, initial=True)
        hashes[arm] = old.state_hash(model.state_dict())
        common = {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
        common_hashes[arm] = old.state_hash(common)
        if hasattr(model, "mean_gain_gamma") and float(model.mean_gain_gamma) != 0.0:
            raise ValueError(f"{arm} gamma is not initialized to zero")
    if len(set(common_hashes.values())) != 1:
        raise ValueError(f"Arms do not share exactly equal network parameters: {common_hashes}")
    return {"full_state_hashes": hashes, "common_state_hashes": common_hashes}


def _forward_loss_checks(device: torch.device) -> dict:
    result = {}
    operator = None
    for arm in ("baseline", "e3"):
        config = _load_config(arm)
        model = exp.build_model(config, initial=True).to(device).eval()
        if operator is None:
            operator = exp.load_operator(config, device)
        item = _gpu_item(config, device)
        beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
        with torch.inference_mode():
            if arm == "baseline":
                reference, _ = exp._ORIGINAL_FORWARD(model, item, operator, beta0)
            else:
                reference_config = copy.deepcopy(config)
                reference_config["three_way"]["kind"] = "e3"
                reference, _ = old.experiment_forward(
                    model, item, operator, beta0, config=reference_config
                )
            actual, _ = exp.forward(model, item, operator, beta0, config=config)
        error = _relative_l2(actual.reconstruction, reference.reconstruction)
        # The experiment protocol fixes 1e-4 as the FP32 investigation line.
        # Sparse/cuDNN repeated forwards can differ in their last few bits.
        if error > 1e-4:
            raise ValueError(f"{arm} forward reproduction failed: relative L2={error}")
        variance_model = TaylorH2VarianceModel(operator, **config["noise"])
        if arm == "baseline":
            actual_loss = exp.loss(actual, item, operator, variance_model, config)
            reference_loss = exp._ORIGINAL_LOSS(reference, item, operator, variance_model, config)
        else:
            with torch.inference_mode():
                actual_loss = exp.loss(actual, item, operator, variance_model, config)
                reference_loss = old.experiment_loss(reference, item, operator, variance_model, reference_config)
        loss_error = abs(float(actual_loss.total) - float(reference_loss.total))
        if loss_error > 1e-7:
            raise ValueError(f"{arm} loss reproduction failed: abs error={loss_error}")
        result[arm] = {"forward_relative_l2": error, "loss_absolute_error": loss_error}

    config = _load_config("e3_mean005")
    temporary = Path("/tmp/v3_compare_gpu_check")
    shutil.rmtree(temporary, ignore_errors=True)
    check_config = copy.deepcopy(config)
    check_config["experiment"]["output_dir"] = str(temporary)
    model = exp.build_model(config, initial=True).to(device).train()
    item = _gpu_item(config, device)
    beta0 = trainer._analytic_beta0(operator, item["f_var"], item["input_mean"])
    exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, micro_call=0)
    output, _ = exp.forward(model, item, operator, beta0, config=check_config)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])
    bounded = exp.loss(output, item, operator, variance_model, check_config)
    bounded.total.backward()
    metrics = bounded.scalar_metrics()
    expected_budget = 0.05 * 49 / 50
    if metrics["mean_shape_gradient_ratio"] > expected_budget + 1e-7:
        raise ValueError(f"Step-49 bounded gradient exceeded its ramp: {metrics}")
    network_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name != "mean_gain_gamma" and parameter.grad is not None
    )
    gamma_grad = float(model.mean_gain_gamma.grad.abs())
    if network_grad <= 0 or gamma_grad <= 0:
        raise ValueError("C must update both network structure and the E3 gain parameter")
    result["e3_mean005"] = {
        "step49_expected_budget": expected_budget,
        "step49_actual_ratio": metrics["mean_shape_gradient_ratio"],
        "network_gradient_l1": network_grad,
        "gamma_gradient_abs": gamma_grad,
    }

    # The forward path must not read GT or the 90-frame statistics.
    model.eval()
    clean = {name: value for name, value in item.items() if name not in ("ground_truth", "measured_mean", "measured_variance")}
    with torch.inference_mode():
        full, _ = exp.forward(model, item, operator, beta0, config=config)
        stripped, _ = exp.forward(model, clean, operator, beta0, config=config)
    contamination_error = _relative_l2(full.reconstruction, stripped.reconstruction)
    if contamination_error != 0.0:
        raise ValueError("Network prediction changed after GT/target statistics were removed")
    result["target_contamination_relative_l2"] = contamination_error
    return result


def _lr_and_resume_check() -> dict:
    config = _load_config("e3_mean005")
    model = exp.build_model(config, initial=True)
    exp._STATE.update(completed_steps=0, phase="gpu_check")
    optimizer = exp._optimizer(model, config)
    boundaries = {}
    for step in range(1, 202):
        optimizer.step()
        if step in (49, 50, 199, 200, 201):
            boundaries[str(step)] = {
                group["group_name"]: float(group["lr"]) for group in optimizer.param_groups
            }
    if boundaries["199"] != {"network": 1e-3, "beta": 1e-4, "mean_gain_gamma": 1e-4}:
        raise ValueError(f"Unexpected high learning rates: {boundaries}")
    if boundaries["200"] != {"network": 1e-4, "beta": 1e-5, "mean_gain_gamma": 1e-5}:
        raise ValueError(f"Step 200 did not prepare the low LR for step 201: {boundaries}")
    if boundaries["201"] != boundaries["200"]:
        raise ValueError(f"Step 201 did not use/retain low rates: {boundaries}")
    state = optimizer.state_dict()
    resumed_model = exp.build_model(config, initial=True)
    exp._STATE.update(completed_steps=201, phase="gpu_check")
    resumed = exp._optimizer(resumed_model, config)
    resumed.load_state_dict(state)
    before = {g["group_name"]: float(g["lr"]) for g in resumed.param_groups}
    resumed.step()
    after = {g["group_name"]: float(g["lr"]) for g in resumed.param_groups}
    if before != boundaries["201"] or after != before:
        raise ValueError("Optimizer state did not preserve the low LR on resume")
    return {"rates_after_step": boundaries, "resume_before": before, "resume_after": after}


def gpu_checks() -> dict:
    started = time.time()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    exp.configure_precision()
    result = {
        "passed": False,
        "device": str(torch.cuda.get_device_name(device)),
        "numerical_precision": "fp32_tf32_disabled",
        "dataset": _dataset_check(),
        "initialization": _same_common_initial(),
    }
    try:
        result["forward_and_loss"] = _forward_loss_checks(device)
        result["lr_and_resume"] = _lr_and_resume_check()
        result.update(
            passed=True,
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            finished_unix=time.time(),
            seconds=time.time() - started,
        )
        old.write_json(exp.OUTPUT / "gpu_checks.json", result)
        return result
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), finished_unix=time.time())
        old.write_json(exp.OUTPUT / "gpu_checks.json", result)
        raise


if __name__ == "__main__":
    print(json.dumps(gpu_checks(), indent=2, ensure_ascii=False))
