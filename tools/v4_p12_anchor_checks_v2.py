"""Corrected GPU preflight that includes P12 holdout physics targets."""
from __future__ import annotations

import json
import time

import torch
import yaml

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import three_way_experiment as old
from tools import v4_p12_anchor_compare_experiment as exp
from tools.three_way_checks import input_item


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def _training_item(device: torch.device) -> dict:
    item = input_item(exp.DATA / "P12", 1, device)
    key = DatasetItemKey("P12", 1, "train", exp.DATA / "P12")
    target = _read_targets(key, include_ground_truth=True)
    for name in ("measured_mean", "measured_variance", "ground_truth"):
        item[name] = torch.from_numpy(target[name]).float().unsqueeze(0).to(device)
    return item


def gpu_checks() -> dict:
    started = time.time()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    configs = {
        arm: yaml.safe_load((exp.OUTPUT / f"{arm}.yaml").read_text(encoding="utf-8"))
        for arm in exp.ARMS
    }
    for config in configs.values():
        exp.validate_config(config)
    operator = exp.load_operator(configs[exp.ARMS[0]], device)
    item = _training_item(device)
    common_states = {}
    records = {}
    for arm, config in configs.items():
        anchor = exp.ANCHORS[arm]
        model = exp.build_model(config, initial=True).to(device)
        common_states[arm] = old.state_hash(
            {name: value for name, value in model.state_dict().items() if name != "mean_gain_gamma"}
        )
        volume = exp.anchor_volume(item, anchor)
        beta0 = exp.analytic_beta0(operator, volume, item["input_mean"])
        with torch.inference_mode():
            native = model(
                item["f_var"], item["g_mean"], item["residual_frames"],
                item["z_values_um"], var_feature_volume=item["f_var_feature"], beta0=beta0,
            )
        anchor_error = _relative_l2(
            native.reconstruction, beta0[:, None, None, None, None] * volume
        )
        if anchor_error > 1e-6:
            raise ValueError(f"{arm} zero-head output does not preserve its anchor")
        cache = {}
        cached = exp.cached_beta0(cache, item, operator, config=config)
        if set(cache) != {(str(item["sample_id"]), int(item["subset_index"]), anchor)}:
            raise ValueError(f"{arm} beta cache is not anchor-qualified")
        torch.testing.assert_close(cached, beta0)

        model.train()
        exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, micro_call=0)
        output, _ = exp.forward(model, item, operator, beta0, config=config)
        losses = exp.loss(
            output, item, operator,
            TaylorH2VarianceModel(operator, **config["noise"]), config,
        )
        losses.total.backward()
        metrics = losses.scalar_metrics()
        expected_budget = 49 / 50
        if metrics["mean_shape_gradient_ratio"] > expected_budget + 1e-6:
            raise ValueError(f"{arm} step-49 gradient budget was exceeded")
        network_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name not in ("raw_beta", "mean_gain_gamma") and parameter.grad is not None
        )
        if network_gradient <= 0 or model.mean_gain_gamma.grad is None:
            raise ValueError(f"{arm} loss did not reach structure and gain parameters")

        model.eval()
        clean = {
            name: value for name, value in item.items()
            if name not in ("ground_truth", "measured_mean", "measured_variance")
        }
        with torch.inference_mode():
            full, _ = exp.forward(model, item, operator, beta0, config=config)
            stripped, _ = exp.forward(model, clean, operator, beta0, config=config)
        isolation_error = _relative_l2(full.reconstruction, stripped.reconstruction)
        if isolation_error != 0.0:
            raise ValueError(f"{arm} inference depends on holdout targets or GT")
        records[arm] = {
            "anchor": anchor,
            "zero_head_anchor_relative_l2": anchor_error,
            "analytic_beta0": float(beta0),
            "step49_gradient_ratio": metrics["mean_shape_gradient_ratio"],
            "network_gradient_l1": network_gradient,
            "target_gt_isolation_relative_l2": isolation_error,
        }
        del model
        torch.cuda.empty_cache()
    if len(set(common_states.values())) != 1:
        raise ValueError("The two arms do not share identical common initialization")
    result = {
        "passed": True,
        "check_version": 2,
        "device": torch.cuda.get_device_name(device),
        "sample": "P12_subset_01",
        "holdout_physics_targets_loaded": True,
        "common_initial_state_sha256": next(iter(common_states.values())),
        "arms": records,
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "seconds": time.time() - started,
    }
    old.write_json(exp.OUTPUT / "gpu_checks.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(gpu_checks(), indent=2, ensure_ascii=False))
