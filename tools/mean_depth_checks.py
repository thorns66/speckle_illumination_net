"""Pre-training numerical and contract checks for the mean/depth comparison."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
import yaml

import training.multivolume_trainer as trainer
from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset, load_dataset_index
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean_depth_experiment as exp
from tools import three_way_experiment as old
from tools.v3_compare_checks import _dataset_check


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).norm() / right.float().norm().clamp_min(1e-30))


def _config(arm: str) -> dict:
    return yaml.safe_load((exp.OUTPUT / f"{arm}.yaml").read_text(encoding="utf-8"))


def _item(config: dict, device: torch.device):
    dataset = MatlabMultiVolumeDataset(
        exp.DATA, "validation", cache_dir=config["data"]["cache_dir"],
        include_ground_truth=True, var_feature_representation="sqrt",
    )
    return trainer._to_device(dataset[0], device)


def _synthetic_depth_checks(device: torch.device) -> dict:
    torch.manual_seed(20260908)
    reference = torch.rand(1, 1, 10, 37, 35, device=device)
    reference = reference / reference.sum()
    same, same_stats = exp.local_depth_protection(reference, reference)
    scaled, _ = exp.local_depth_protection(reference * 7.3, reference * 2.1)
    shifted = torch.zeros_like(reference)
    shifted[:, :, 1:] = reference[:, :, :-1]
    shifted = shifted / shifted.sum().clamp_min(1e-30)
    wrong, wrong_stats = exp.local_depth_protection(shifted, reference)

    first = torch.zeros(1, 1, 10, 25, 25, device=device)
    second = torch.zeros_like(first)
    # Both have the same axial centroid (45 um) but different two-layer distributions.
    first[:, :, 2, 8:17, 8:17] = 0.5
    first[:, :, 5, 8:17, 8:17] = 0.5
    second[:, :, 1, 8:17, 8:17] = 0.5
    second[:, :, 6, 8:17, 8:17] = 0.5
    first /= first.sum(); second /= second.sum()
    same_centroid, centroid_stats = exp.local_depth_protection(first, second)
    result = {
        "identical_loss": float(same),
        "scaled_identical_loss": float(scaled),
        "one_layer_shift_loss": float(wrong),
        "one_layer_shift_max_distance_um": wrong_stats["depth_distance_max_um"],
        "same_centroid_different_distribution_loss": float(same_centroid),
        "same_centroid_max_distance_um": centroid_stats["depth_distance_max_um"],
    }
    if result["identical_loss"] != 0.0 or result["scaled_identical_loss"] > 1e-10:
        raise ValueError(f"Depth protection is not brightness-scale invariant at equality: {result}")
    if result["one_layer_shift_loss"] <= 0 or result["same_centroid_different_distribution_loss"] <= 0:
        raise ValueError(f"Depth protection missed an axial perturbation: {result}")
    return result


def _initial_and_loss_checks(device: torch.device) -> dict:
    states = {}
    predictions = {}
    operator = None
    base_item = None
    result = {}
    for arm in exp.ARMS:
        config = _config(arm)
        model = exp.build_model(config, initial=True).to(device)
        model.train()
        states[arm] = old.state_hash({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if operator is None:
            operator = exp.load_operator(config, device)
            base_item = _item(config, device)
        assert base_item is not None
        beta0 = trainer._analytic_beta0(operator, base_item["f_var"], base_item["input_mean"])
        exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, teacher=None, teacher_device=None)
        output, _ = exp.forward(model, base_item, operator, beta0, config=config)
        predictions[arm] = output.reconstruction.detach().clone()
        variance = TaylorH2VarianceModel(operator, **config["noise"])
        breakdown = exp.loss(output, base_item, operator, variance, config)
        mean_to_q = torch.autograd.grad(
            breakdown.normalized_mean, output._shape, retain_graph=True, allow_unused=True
        )[0]
        breakdown.total.backward()
        metrics = breakdown.scalar_metrics()
        if mean_to_q is not None:
            raise ValueError("Ordinary E3 mean loss unexpectedly changed normalized structure q")
        if arm in ("r1_mean005", "r3_mean005_depth"):
            expected = 0.05 * 49 / 50
            if metrics["mean_shape_gradient_ratio"] > expected + 1e-7:
                raise ValueError(f"Step-49 mean q-gradient exceeded its ramp: {metrics}")
        if arm in ("r2_depth", "r3_mean005_depth") and metrics["depth_protection_loss"] > 1e-8:
            raise ValueError(f"Depth protection is not zero at the frozen teacher starting point: {metrics}")
        result[arm] = {
            "state_sha256": states[arm],
            "initial_depth_loss": metrics["depth_protection_loss"],
            "step49_mean_gradient_ratio": metrics["mean_shape_gradient_ratio"],
            "ordinary_mean_to_q_is_none": True,
            "teacher_present": output._reference_q is not None,
        }
        if arm in ("r2_depth", "r3_mean005_depth"):
            # A controlled, mass-preserving axial perturbation on a real source
            # output must cross the 1 um hinge and remain finite.  Run01 failed
            # here after its first update
            # because near-empty local profiles generated unbounded gradients.
            perturbed_q = output._shape.detach().clone()
            delta = 0.2 * perturbed_q[:, :, :-1]
            perturbed_q[:, :, :-1] -= delta
            perturbed_q[:, :, 1:] += delta
            perturbed_q.requires_grad_(True)
            raw_depth, depth_stats = exp.local_depth_protection(
                perturbed_q,
                output._reference_q,
                windows=tuple(config["mean_depth"]["depth_window_pixels"]),
                stride=int(config["mean_depth"]["depth_window_stride"]),
                z_spacing_um=float(config["mean_depth"]["z_spacing_um"]),
                tolerance_um=float(config["mean_depth"]["depth_tolerance_um"]),
                reference_floor_fraction=float(config["mean_depth"]["reference_floor_fraction"]),
            )
            raw_gradient = torch.autograd.grad(raw_depth, perturbed_q)[0]
            if not torch.isfinite(raw_depth) or not torch.isfinite(raw_gradient).all():
                raise ValueError("Real-volume depth perturbation is numerically unstable")
            if float(raw_depth) <= 0 or depth_stats["depth_distance_max_um"] <= 1.0:
                raise ValueError("Real-volume depth perturbation did not cross the frozen hinge")
            result[arm].update(
                real_perturbation_depth_loss=float(raw_depth.detach()),
                real_perturbation_depth_gradient_norm=float(raw_gradient.norm()),
                real_perturbation_depth_max_um=depth_stats["depth_distance_max_um"],
                depth_gradient_budget=float(config["mean_depth"]["depth_gradient_budget"]),
            )
        del model, output, breakdown
        torch.cuda.empty_cache()
    if len(set(states.values())) != 1:
        raise ValueError(f"Initial model states differ: {states}")
    reference = predictions[exp.ARMS[0]]
    forward_errors = {arm: _relative_l2(value, reference) for arm, value in predictions.items()}
    if max(forward_errors.values()) > 1e-4:
        raise ValueError(f"Initial forwards differ: {forward_errors}")
    result["initial_forward_relative_l2"] = forward_errors

    # Forward prediction must depend only on the ten-frame inputs and RL3 products.
    config = _config("r3_mean005_depth")
    model = exp.build_model(config, initial=True).to(device).eval()
    clean = {k: v for k, v in base_item.items() if k not in ("ground_truth", "measured_mean", "measured_variance")}
    poisoned = dict(clean)
    poisoned["ground_truth"] = torch.full((1, 1, 10, 260, 260), 999.0, device=device)
    poisoned["measured_mean"] = torch.full_like(base_item["measured_mean"], 999.0)
    poisoned["measured_variance"] = torch.full_like(base_item["measured_variance"], 999.0)
    beta0 = trainer._analytic_beta0(operator, clean["f_var"], clean["input_mean"])
    exp._STATE.update(phase="gpu_check", evaluation=False, teacher=None, teacher_device=None)
    with torch.inference_mode():
        reference, _ = exp.forward(model, clean, operator, beta0, config=config)
        repeated, _ = exp.forward(model, clean, operator, beta0, config=config)
        poisoned_result, _ = exp.forward(model, poisoned, operator, beta0, config=config)
    repeat_error = _relative_l2(repeated.reconstruction, reference.reconstruction)
    contamination = _relative_l2(poisoned_result.reconstruction, reference.reconstruction)
    # Sparse/cuDNN FP32 repeats can differ in their final bits.  The frozen V3
    # protocol uses 1e-4 as its investigation line and records both errors.
    if max(repeat_error, contamination) > 1e-4:
        raise ValueError(
            f"Same-input repeatability or GT/holdout isolation failed: "
            f"repeat={repeat_error}, contamination={contamination}"
        )
    result["same_input_repeat_relative_l2"] = repeat_error
    result["target_or_gt_contamination_relative_l2"] = contamination
    return result


def _optimizer_check() -> dict:
    config = _config("r3_mean005_depth")
    model = exp.build_model(config, initial=True)
    exp._STATE.update(completed_steps=0, phase="optimizer_check")
    instance = exp.optimizer(model, config)
    if any(state for state in instance.state.values()):
        raise ValueError("Fresh Adam already contains inherited state")
    expected = {"network": 1e-4, "beta": 1e-5, "mean_gain_gamma": 1e-5}
    observed = {group["group_name"]: float(group["lr"]) for group in instance.param_groups}
    if observed != expected:
        raise ValueError(f"Wrong continuation learning rates: {observed}")
    return {"fresh_adam_state_entries": 0, "learning_rates": observed}


def gpu_checks() -> dict:
    started = time.time()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    exp.configure_precision()
    indexed, fingerprint = load_dataset_index(exp.DATA)
    record = {
        "passed": False,
        "device": torch.cuda.get_device_name(device),
        "dataset": _dataset_check(),
        "split_counts": {k: len(v) for k, v in indexed.items()},
        "dataset_fingerprint": fingerprint,
        "synthetic_depth": _synthetic_depth_checks(device),
        "initial_and_loss": _initial_and_loss_checks(device),
        "optimizer": _optimizer_check(),
        "numerical_precision": "full FP32; TF32 disabled",
        "seconds": time.time() - started,
    }
    record["passed"] = True
    old.write_json(exp.OUTPUT / "gpu_checks.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == "__main__":
    gpu_checks()
