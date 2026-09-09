"""Numerical and gradient acceptance for the isolated second-round adapters."""
from __future__ import annotations

import copy
import json
import time

import torch
import yaml

from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import mean_round2_experiment as experiment
from tools import three_way_experiment as old
from training.multivolume_trainer import _to_device


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def gpu_checks():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    old.install_data()
    started = time.monotonic()
    configs = {kind: yaml.safe_load((experiment.OUTPUT / f"{kind}.yaml").read_text()) for kind in experiment.KINDS}
    operator = experiment.load_operator(configs["r0_continue"], device)
    variance = TaylorH2VarianceModel(operator)
    dataset = MatlabMultiVolumeDataset(old.DATA, "validation", cache_dir=old.OUTPUT / "data_cache")
    index = next(i for i, key in enumerate(dataset.keys) if key.sample_id == "P09" and key.subset_index == 1)
    item = _to_device(dataset[index], device)
    torch.manual_seed(2026090809)
    contamination = dict(item)
    contamination["ground_truth"] = torch.rand_like(item["ground_truth"]) * 1e5
    contamination["measured_mean"] = torch.rand_like(item["measured_mean"]) * 1e5
    contamination["measured_variance"] = torch.rand_like(item["measured_variance"]) * 1e5
    source = torch.load(experiment.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    original_config = source["config"]
    rows = []
    initialized_hashes = {}
    original_state = dict(experiment._STATE)
    experiment._STATE.update(completed_steps=49, micro_call=0, phase="gpu_checks", evaluation=False)
    try:
        for kind, config in configs.items():
            experiment.validate_trainer_config(config)
            model = experiment.build_model(config, initial=True).to(device)
            initialized_hashes[kind] = old.state_hash(model.state_dict())
            if float(model.mean_gain_gamma.detach()) != 0.:
                raise AssertionError("Initial gamma must be zero in every arm")
            model.eval()
            row = {"kind": kind, "gamma_trainable": bool(model.mean_gain_gamma.requires_grad)}
            with torch.no_grad():
                output, _ = experiment.forward(model, item, operator, config=config)
                reference = output.reconstruction.clone()
                repeated, _ = experiment.forward(model, item, operator, config=config)
                repeated_value = repeated.reconstruction.clone()
                changed, _ = experiment.forward(model, contamination, operator, config=config)
                row["same_input_repeat_relative_l2"] = relative(repeated_value, reference)
                row["target_gt_contamination_relative_l2"] = relative(changed.reconstruction, reference)
                common = experiment.loss(output, item, operator, variance, config)
                original_common = old.experiment_loss(output, item, operator, variance, original_config)
                row["common_evaluation_loss_absolute_error"] = abs(float(common.total - original_common.total))
                row["common_evaluation_mean_is_included"] = bool(torch.equal(common.weighted_mean, common.normalized_mean))
                row["prediction_finite_nonnegative"] = bool(torch.isfinite(reference).all() and (reference >= 0).all())
                if kind == "r0_continue":
                    original, _ = old.experiment_forward(model, item, operator, config=original_config)
                    row["E3_gamma0_reproduction_relative_l2"] = relative(reference, original.reconstruction)
                if kind == "r2_input_scale":
                    scaling = []
                    for factor in (0., .1, .5, 2.):
                        scaled = dict(item)
                        for key in experiment.INPUT_FIELDS:
                            scaled[key] = item[key] * factor
                        prediction, _ = experiment.forward(model, scaled, operator, config=config)
                        error = relative(prediction.reconstruction, reference * factor) if factor else float(prediction.reconstruction.abs().max())
                        scaling.append({"gain": factor, "relative_l2_or_zero_max": error})
                    row["brightness_homogeneity"] = scaling
                del output, repeated, changed, common, original_common
            model.train()
            output, _ = experiment.forward(model, item, operator, config=config)
            terms = experiment.loss(output, item, operator, variance, config)
            # Measure the actually added q derivative independently of the
            # coefficient diagnostic: total minus original var/TV derivatives.
            q = output._shape
            grad_total = torch.autograd.grad(terms.total, q, retain_graph=True)[0]
            grad_var = torch.autograd.grad(terms.weighted_var, q, retain_graph=True)[0]
            grad_tv = torch.autograd.grad(terms.weighted_tv, q, retain_graph=True)[0]
            added = grad_total - grad_var - grad_tv
            actual_ratio = float(added.norm() / grad_var.norm().clamp_min(1e-30))
            if terms.weighted_mean.requires_grad:
                mean_to_shape = torch.autograd.grad(terms.weighted_mean, q, allow_unused=True, retain_graph=True)[0]
            else:
                mean_to_shape = None
            row["actual_added_q_gradient_ratio"] = actual_ratio
            row["gain_mean_gradient_to_shape_is_none"] = mean_to_shape is None
            row["gradient_diagnostics"] = output._round2_diagnostics
            terms.total.backward()
            row["all_parameter_gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            row["gamma_gradient"] = None if model.mean_gain_gamma.grad is None else float(model.mean_gain_gamma.grad)
            row["network_gradient_norm"] = sum(float(p.grad.detach().square().sum()) for n, p in model.named_parameters() if n != "mean_gain_gamma" and p.grad is not None) ** .5
            row["r1_training_mean_is_zero"] = kind != "r1_no_mean" or float(terms.weighted_mean) == 0.
            budget = float(config["round2"]["shape_gradient_budget"])
            row["passed"] = (
                row["target_gt_contamination_relative_l2"] <= 1e-6
                and row["common_evaluation_loss_absolute_error"] <= 1e-6
                and row["common_evaluation_mean_is_included"]
                and row["prediction_finite_nonnegative"]
                and row.get("E3_gamma0_reproduction_relative_l2", 0.) <= 1e-6
                and max((x["relative_l2_or_zero_max"] for x in row.get("brightness_homogeneity", [])), default=0.) <= 1e-6
                and row["gain_mean_gradient_to_shape_is_none"]
                and row["all_parameter_gradients_finite"]
                and row["network_gradient_norm"] > 0
                and row["r1_training_mean_is_zero"]
                and actual_ratio <= budget + 1e-6
                and (kind != "r1_no_mean" or row["gamma_gradient"] is None)
            )
            rows.append(row)
            print(json.dumps({"gpu_check_arm": kind, "passed": row["passed"], "actual_q_gradient_ratio": actual_ratio}), flush=True)
            del model, output, terms, grad_total, grad_var, grad_tv, added, q, reference, repeated_value
            torch.cuda.empty_cache()
    finally:
        experiment._STATE.clear()
        experiment._STATE.update(original_state)
    result = {"complete": True, "rows": rows, "initial_state_hashes": initialized_hashes,
              "all_initial_states_identical": len(set(initialized_hashes.values())) == 1,
              "gradient_budget_location": "normalized output shape q; not Adam parameter updates",
              "ramp_steps_at_check": 50, "precision": {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}, "seconds": time.monotonic() - started}
    result["passed"] = result["all_initial_states_identical"] and all(row["passed"] for row in rows)
    old.write_json(experiment.OUTPUT / "gpu_checks.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Round-2 GPU numerical acceptance failed")
