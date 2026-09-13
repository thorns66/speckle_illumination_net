"""GPU preflight checks for the isolated Mean-RL3 anchor experiment."""
from __future__ import annotations

import json
import time

import torch
import yaml

from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean_anchor_experiment as exp
from tools import three_way_experiment as old
from tools.three_way_checks import input_item


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def gpu_checks() -> dict:
    started = time.time()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = yaml.safe_load((exp.OUTPUT / f"{exp.ARM}.yaml").read_text(encoding="utf-8"))
    exp.validate_config(config)
    model = exp.build_model(config, initial=True).to(device)
    operator = exp.load_operator(config, device)
    item = input_item(exp.DATA / "T03", 1, device)
    beta_mean = exp.analytic_beta0(operator, item["g_mean"], item["input_mean"])
    beta_taylor = exp.analytic_beta0(operator, item["f_var"], item["input_mean"])
    with torch.inference_mode():
        native = model(
            item["f_var"], item["g_mean"], item["residual_frames"], item["z_values_um"],
            var_feature_volume=item["f_var_feature"], beta0=beta_mean,
        )
    anchor_error = relative_l2(
        native.reconstruction,
        beta_mean[:, None, None, None, None] * item["g_mean"],
    )
    if anchor_error > 1e-6:
        raise ValueError(f"Zero-head output is not the Mean-RL3 anchor: {anchor_error}")
    cache = {}
    cached = exp.cached_beta0(cache, item, operator)
    if set(cache) != {(str(item["sample_id"]), int(item["subset_index"]), "mean_rl3")}:
        raise ValueError(f"Beta cache is not anchor-qualified: {cache.keys()}")
    torch.testing.assert_close(cached, beta_mean)

    model.train()
    exp._STATE.update(completed_steps=48, phase="gpu_check", evaluation=False, micro_call=0)
    output, _ = exp.forward(model, item, operator, beta_mean, config=config)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])
    losses = exp.loss(output, item, operator, variance_model, config)
    losses.total.backward()
    metrics = losses.scalar_metrics()
    expected_budget = 49 / 50
    if metrics["mean_shape_gradient_ratio"] > expected_budget + 1e-6:
        raise ValueError("Step-49 mean-shape gradient exceeded the frozen ramp")
    network_gradient = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name not in ("raw_beta", "mean_gain_gamma") and parameter.grad is not None
    )
    if network_gradient <= 0 or model.mean_gain_gamma.grad is None:
        raise ValueError("Mean-anchor loss did not reach both structure and gain parameters")

    model.eval()
    clean = {
        name: value for name, value in item.items()
        if name not in ("ground_truth", "measured_mean", "measured_variance")
    }
    with torch.inference_mode():
        full, _ = exp.forward(model, item, operator, beta_mean, config=config)
        stripped, _ = exp.forward(model, clean, operator, beta_mean, config=config)
    isolation_error = relative_l2(full.reconstruction, stripped.reconstruction)
    if isolation_error != 0.0:
        raise ValueError("Inference changed after holdout targets and GT were removed")

    result = {
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "reconstruction_anchor": "mean_rl3",
        "zero_head_anchor_relative_l2": anchor_error,
        "mean_beta0": float(beta_mean),
        "taylor_beta0_reference": float(beta_taylor),
        "beta_cache_key_includes_anchor": True,
        "step49_expected_gradient_budget": expected_budget,
        "step49_actual_gradient_ratio": metrics["mean_shape_gradient_ratio"],
        "network_gradient_l1": network_gradient,
        "target_gt_isolation_relative_l2": isolation_error,
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "seconds": time.time() - started,
    }
    old.write_json(exp.OUTPUT / "gpu_checks.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(gpu_checks(), indent=2, ensure_ascii=False))

