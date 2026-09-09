"""Four-way E3 continuation with optional mean-shape and local-depth protection.

The historical trainer and V3 experiment stay untouched.  Every arm starts from
the same frozen V3 E3 step-400 weights and a fresh Adam optimizer.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import LossBreakdown, TaylorH2VarianceModel, total_variation_3d
from tools import three_way_experiment as old
from tools import v3_compare_experiment as v3


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/mean_depth_protection_20260908_run02"
DATA = ROOT / "data/speckle_dataset_v3_full_20260907_run01"
SOURCE_OUTPUT = ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01"
SOURCE_CHECKPOINT = SOURCE_OUTPUT / "e3/checkpoint_last.pt"
ARMS = ("r0_continue", "r1_mean005", "r2_depth", "r3_mean005_depth")

_ORIGINAL_EVALUATE = trainer._evaluate
_ORIGINAL_PREFLIGHT = trainer._preflight
_ORIGINAL_CONFIGURE = trainer._configure_trainable_parameters
_ORIGINAL_ATOMIC_SAVE = trainer._atomic_torch_save
_ORIGINAL_WRITE_JSON = trainer._write_json
_STATE: dict[str, Any] = {
    "completed_steps": 0,
    "micro_call": 0,
    "phase": "standalone",
    "evaluation": False,
    "teacher": None,
    "teacher_device": None,
}


def configure_precision() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def spec(config: dict[str, Any]) -> dict[str, Any]:
    return config["mean_depth"]


def validate_config(config: dict[str, Any]) -> None:
    current = spec(config)
    kind = str(current["kind"])
    if kind not in ARMS:
        raise ValueError(f"Unknown mean/depth arm: {kind}")
    if Path(current["source_checkpoint"]).resolve() != SOURCE_CHECKPOINT.resolve():
        raise ValueError("Unexpected continuation checkpoint")
    if int(current["source_total_steps"]) != 400 or int(current["additional_steps"]) != 200:
        raise ValueError("This comparison must continue V3 E3 step 400 for exactly 200 updates")
    if int(config["optimization"]["max_steps"]) != 200:
        raise ValueError("Every continuation arm must run exactly 200 optimizer updates")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("Global batch size must remain 8")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("Validation interval must remain 20 updates")
    if bool(config["runtime"]["amp"]):
        raise ValueError("AMP must remain disabled")
    if config["data"].get("var_feature_representation", "sqrt") != "sqrt":
        raise ValueError("Taylor input must use the saved square root exactly once")
    expect_mean = kind in ("r1_mean005", "r3_mean005_depth")
    expect_depth = kind in ("r2_depth", "r3_mean005_depth")
    if bool(current["use_mean_shape"]) != expect_mean:
        raise ValueError("Mean-shape switch does not match the arm")
    if bool(current["use_depth_protection"]) != expect_depth:
        raise ValueError("Depth-protection switch does not match the arm")
    if float(current["mean_gradient_budget"]) != (0.05 if expect_mean else 0.0):
        raise ValueError("Mean q-gradient budget does not match the arm")
    if tuple(int(v) for v in current["depth_window_pixels"]) != (9, 17):
        raise ValueError("Depth protection requires the frozen 9x9 and 17x17 windows")
    if int(current["depth_window_stride"]) != 4 or float(current["depth_tolerance_um"]) != 1.0:
        raise ValueError("Depth stride/tolerance differs from the frozen plan")
    if float(current["depth_gradient_budget"]) != (0.25 if expect_depth else 0.0):
        raise ValueError("Depth q-gradient budget does not match the arm")
    if int(current["depth_ramp_steps"]) != 50:
        raise ValueError("Depth q-gradient ramp must contain 50 optimizer updates")
    compatibility = dict(config)
    v3.validate_config.native_validation(compatibility)


def build_model(config: dict[str, Any], *, initial: bool = False) -> torch.nn.Module:
    configure_precision()
    model = v3.build_model(config, initial=False)
    if initial:
        source = Path(spec(config)["source_checkpoint"])
        if old.sha256(source) != spec(config)["source_checkpoint_sha256"]:
            raise ValueError("Frozen source checkpoint changed")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if int(checkpoint["completed_steps"]) != 400:
            raise ValueError("Source checkpoint is not the actual V3 E3 step-400 final")
        model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def load_operator(config: dict[str, Any], device: torch.device):
    configure_precision()
    return old.load_operator(config, device, sparse=True)


def _teacher(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    cached = _STATE.get("teacher")
    if cached is not None and _STATE.get("teacher_device") == str(device):
        return cached
    teacher = build_model(config, initial=True).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    _STATE["teacher"] = teacher
    _STATE["teacher_device"] = str(device)
    return teacher


def forward(model, item, operator, beta0=None, *, config):
    output, beta0 = v3.forward(model, item, operator, beta0, config=config)
    output._mean_depth_training = bool(model.training and torch.is_grad_enabled())
    output._reference_q = None
    if (
        bool(spec(config)["use_depth_protection"])
        and output._mean_depth_training
        and not _STATE["evaluation"]
    ):
        base = model.module if hasattr(model, "module") else model
        teacher = _teacher(config, next(base.parameters()).device)
        with torch.no_grad():
            reference, _ = v3.forward(teacher, item, operator, beta0, config=config)
        output._reference_q = reference._shape.detach()
    return output, beta0


def _window_depth_mass(q: torch.Tensor, kernel: int, stride: int):
    if q.ndim != 5 or q.shape[1] != 1:
        raise ValueError(f"Expected q [B,1,Z,H,W], got {tuple(q.shape)}")
    batch, _, depths, height, width = q.shape
    pooled = F.avg_pool2d(
        q[:, 0].reshape(batch * depths, 1, height, width),
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
        count_include_pad=False,
    ).reshape(batch, depths, -1)
    mass = pooled.sum(dim=1)
    return pooled, mass


def local_depth_protection(
    q: torch.Tensor,
    reference_q: torch.Tensor,
    *,
    windows: tuple[int, ...] = (9, 17),
    stride: int = 4,
    z_spacing_um: float = 10.0,
    tolerance_um: float = 1.0,
    reference_floor_fraction: float = 0.01,
):
    """Soft local W1 trust region around a frozen input-only E3 reference."""
    if q.shape != reference_q.shape:
        raise ValueError("Student and reference volumes must have the same shape")
    if tolerance_um <= 0 or not 0 < reference_floor_fraction < 1:
        raise ValueError("Invalid local-depth protection constants")
    losses = []
    detached_distances = []
    detached_weights = []
    for kernel in windows:
        student_mass_by_depth, student_mass = _window_depth_mass(q, kernel, stride)
        reference_mass_by_depth, reference_mass = _window_depth_mass(reference_q, kernel, stride)
        maximum = reference_mass.amax(dim=1, keepdim=True)
        tau = float(reference_floor_fraction) * maximum
        # A normalized depth profile has an ill-conditioned derivative when a
        # window contains only floating-point background.  The 1% signal floor
        # is therefore also an explicit active-window cutoff, not merely a soft
        # weight.  The student denominator uses the same detached reference
        # floor so a small first update cannot create an unbounded 1/mass term.
        active = (reference_mass >= tau) & (maximum > 0)
        stable_tau = tau.detach().clamp_min(1e-30)
        student = student_mass_by_depth / student_mass[:, None].clamp_min(
            stable_tau[:, None]
        )
        reference = reference_mass_by_depth / reference_mass[:, None].clamp_min(
            stable_tau[:, None]
        )
        distance = float(z_spacing_um) * (
            student.cumsum(dim=1)[:, :-1] - reference.cumsum(dim=1)[:, :-1]
        ).abs().sum(dim=1)
        weight = torch.where(
            active,
            reference_mass / (reference_mass + tau).clamp_min(1e-30),
            torch.zeros_like(reference_mass),
        )
        violation = F.relu(distance / float(tolerance_um) - 1.0)
        losses.append((weight * violation.square()).sum() / weight.sum().clamp_min(1e-30))
        detached_distances.append(distance.detach().reshape(-1))
        detached_weights.append(weight.detach().reshape(-1))
    combined = torch.stack(losses).mean()
    distances = torch.cat(detached_distances)
    weights = torch.cat(detached_weights)
    active = weights > 0
    if bool(active.any()):
        selected = distances[active]
        statistics = {
            "depth_distance_median_um": float(selected.median()),
            "depth_distance_p95_um": float(torch.quantile(selected, 0.95)),
            "depth_distance_max_um": float(selected.max()),
            "depth_violation_fraction": float((selected > float(tolerance_um)).float().mean()),
        }
    else:
        statistics = {
            "depth_distance_median_um": 0.0,
            "depth_distance_p95_um": 0.0,
            "depth_distance_max_um": 0.0,
            "depth_violation_fraction": 0.0,
        }
    return combined, statistics


class MeanDepthLossBreakdown(LossBreakdown):
    def scalar_metrics(self) -> dict[str, float]:
        result = super().scalar_metrics()
        result.update(getattr(self, "mean_depth_metrics", {}))
        return result


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def loss(output, item, operator, variance_model, config):
    current = spec(config)
    q = output._shape
    mu = item["measured_mean"]
    target = item["measured_variance"]
    pred_mu = output._mean_prediction
    mean_scale = mu.detach().abs().mean().clamp_min(1e-8)
    raw_mean = F.smooth_l1_loss(pred_mu, mu)
    normalized_mean = F.smooth_l1_loss(pred_mu / mean_scale, mu / mean_scale)

    predicted_shape_variance = variance_model(q, mu)
    eps = float(config["loss"]["var_log_eps"])
    normalized_prediction = predicted_shape_variance / predicted_shape_variance.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-30)
    normalized_target = target / target.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    normalized_var = F.smooth_l1_loss(
        torch.log(normalized_prediction.clamp_min(0) + eps),
        torch.log(normalized_target + eps),
    )
    physical_var = predicted_shape_variance * output._gain.detach()[:, None, None, None].square()
    raw_var = F.smooth_l1_loss(physical_var, target)
    tv = total_variation_3d(
        q * output._gain.detach()[:, None, None, None, None],
        z_weight=float(config["loss"]["lambda_tv_z"]),
    )
    weighted_tv = tv * float(config["loss"]["lambda_tv"])
    zero = q.new_zeros(())
    shape_mean = zero
    mean_coefficient = zero
    depth_raw = zero
    weighted_depth = zero
    metrics: dict[str, float] = {
        "mean_shape_loss": 0.0,
        "weighted_mean_shape_loss": 0.0,
        "mean_shape_coefficient": 0.0,
        "mean_shape_gradient_ratio": 0.0,
        "mean_var_q_gradient_cosine": 0.0,
        "depth_protection_loss": 0.0,
        "weighted_depth_protection_loss": 0.0,
        "depth_coefficient": 0.0,
        "depth_q_gradient_norm": 0.0,
        "depth_raw_to_var_q_gradient_ratio": 0.0,
        "depth_to_var_q_gradient_ratio": 0.0,
        "depth_distance_median_um": 0.0,
        "depth_distance_p95_um": 0.0,
        "depth_distance_max_um": 0.0,
        "depth_violation_fraction": 0.0,
    }
    training = bool(output._mean_depth_training and torch.is_grad_enabled() and not _STATE["evaluation"])
    grad_var = None
    if training and (bool(current["use_mean_shape"]) or bool(current["use_depth_protection"])):
        grad_var = torch.autograd.grad(normalized_var, q, retain_graph=True, create_graph=False)[0]
        norm_var = torch.linalg.vector_norm(grad_var.detach())
    else:
        norm_var = q.new_zeros(())

    if training and bool(current["use_mean_shape"]):
        predicted_mean_shape = operator(q)
        predicted_mean_shape = predicted_mean_shape / predicted_mean_shape.mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-30)
        target_mean_shape = mu / mu.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        shape_mean = F.smooth_l1_loss(predicted_mean_shape, target_mean_shape)
        grad_mean = torch.autograd.grad(shape_mean, q, retain_graph=True, create_graph=False)[0]
        norm_mean = torch.linalg.vector_norm(grad_mean.detach())
        floor = float(current["gradient_eps"])
        ramp = min((int(_STATE["completed_steps"]) + 1) / int(current["mean_ramp_steps"]), 1.0)
        budget = float(current["mean_gradient_budget"]) * ramp
        mean_coefficient = torch.minimum(
            q.new_tensor(1.0), q.new_tensor(budget) * norm_var / (norm_mean + floor)
        ).detach()
        ratio = mean_coefficient * norm_mean / (norm_var + floor)
        cosine = (grad_var.detach() * grad_mean.detach()).sum() / (
            norm_var * norm_mean
        ).clamp_min(floor)
        metrics.update(
            mean_shape_loss=float(shape_mean.detach()),
            weighted_mean_shape_loss=float((mean_coefficient * shape_mean).detach()),
            mean_shape_coefficient=float(mean_coefficient),
            mean_shape_gradient_ratio=float(ratio),
            mean_var_q_gradient_cosine=float(cosine),
            effective_mean_gradient_budget=float(budget),
        )
        if float(ratio) > budget + max(1e-7, budget * 1e-5):
            raise FloatingPointError("Mean-shape q-gradient exceeded the frozen budget")
        del grad_mean

    if training and bool(current["use_depth_protection"]):
        if output._reference_q is None:
            raise RuntimeError("Depth-protected training has no frozen reference volume")
        depth_raw, depth_statistics = local_depth_protection(
            q,
            output._reference_q,
            windows=tuple(int(v) for v in current["depth_window_pixels"]),
            stride=int(current["depth_window_stride"]),
            z_spacing_um=float(current["z_spacing_um"]),
            tolerance_um=float(current["depth_tolerance_um"]),
            reference_floor_fraction=float(current["reference_floor_fraction"]),
        )
        unbounded_depth = float(current["lambda_depth"]) * depth_raw
        grad_depth = torch.autograd.grad(unbounded_depth, q, retain_graph=True, create_graph=False)[0]
        norm_depth = torch.linalg.vector_norm(grad_depth.detach())
        floor = float(current["gradient_eps"])
        ramp = min(
            (int(_STATE["completed_steps"]) + 1) / int(current["depth_ramp_steps"]),
            1.0,
        )
        budget = float(current["depth_gradient_budget"]) * ramp
        depth_coefficient = torch.minimum(
            q.new_tensor(1.0),
            q.new_tensor(budget) * norm_var / (norm_depth + floor),
        ).detach()
        weighted_depth = depth_coefficient * unbounded_depth
        actual_ratio = depth_coefficient * norm_depth / (norm_var + floor)
        metrics.update(
            depth_protection_loss=float(depth_raw.detach()),
            weighted_depth_protection_loss=float(weighted_depth.detach()),
            depth_coefficient=float(depth_coefficient),
            depth_q_gradient_norm=float(norm_depth),
            depth_raw_to_var_q_gradient_ratio=float(
                norm_depth / norm_var.clamp_min(floor)
            ),
            depth_to_var_q_gradient_ratio=float(actual_ratio),
            effective_depth_gradient_budget=float(budget),
            **depth_statistics,
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Depth protection produced a non-finite diagnostic")
        if float(actual_ratio) > budget + max(1e-7, budget * 1e-5):
            raise FloatingPointError("Depth-protection q-gradient exceeded the frozen budget")
        del grad_depth

    total = normalized_mean + normalized_var + weighted_tv + mean_coefficient * shape_mean + weighted_depth
    result = MeanDepthLossBreakdown(
        total, raw_mean, normalized_mean, normalized_mean, raw_var, normalized_var,
        normalized_var, zero, zero, tv, weighted_tv, pred_mu, physical_var,
    )
    result.mean_depth_metrics = metrics
    if training:
        rank = int(os.environ.get("RANK", "0"))
        record = {
            "rank": rank,
            "optimizer_completed_steps": int(_STATE["completed_steps"]),
            "update_step": int(_STATE["completed_steps"]) + 1,
            "sample_id": str(item.get("sample_id", "")),
            "subset_index": int(item.get("subset_index", 0)),
            **metrics,
        }
        _append_jsonl(Path(config["experiment"]["output_dir"]) / f"mean_depth_rank{rank}.jsonl", record)
    if grad_var is not None:
        del grad_var
    return result


def optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    named = dict(model.named_parameters())
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if n not in ("raw_beta", "mean_gain_gamma") and p.requires_grad],
            "lr": float(config["optimization"]["lr_network"]),
            "group_name": "network",
        },
        {"params": [named["raw_beta"]], "lr": float(config["optimization"]["lr_beta"]), "group_name": "beta"},
        {"params": [named["mean_gain_gamma"]], "lr": float(config["three_way"]["lr_gain"]), "group_name": "mean_gain_gamma"},
    ]
    instance = torch.optim.Adam(groups)

    def after_step(_optimizer, *_args, **_kwargs):
        _STATE["completed_steps"] = int(_STATE["completed_steps"]) + 1
        if _STATE["phase"] == "training":
            rank = int(os.environ.get("RANK", "0"))
            _append_jsonl(
                Path(config["experiment"]["output_dir"]) / f"learning_rate_rank{rank}.jsonl",
                {
                    "rank": rank,
                    "additional_step": int(_STATE["completed_steps"]),
                    "total_step": int(spec(config)["source_total_steps"]) + int(_STATE["completed_steps"]),
                    "used_lrs": {g["group_name"]: float(g["lr"]) for g in instance.param_groups},
                },
            )

    instance.register_step_post_hook(after_step)
    return instance


def install_training(config: dict[str, Any]) -> None:
    configure_precision()
    validate_config(config)
    _STATE.update(completed_steps=0, micro_call=0, phase="training", evaluation=False, teacher=None, teacher_device=None)
    trainer._validate_config = validate_config
    trainer._model_from_config = lambda current: build_model(current, initial=True)
    trainer._load_operator = lambda current, _path, device, **_kwargs: load_operator(current, device)
    trainer._forward = lambda model, item, operator, beta0=None: forward(model, item, operator, beta0, config=config)
    trainer._loss = loss
    trainer._optimizer = optimizer
    trainer.TaylorH2VarianceModel = TaylorH2VarianceModel
    trainer._configure_trainable_parameters = lambda model: _ORIGINAL_CONFIGURE(model)

    def evaluate(**kwargs):
        _STATE["evaluation"] = True
        try:
            summary, rows = _ORIGINAL_EVALUATE(**kwargs)
        finally:
            _STATE["evaluation"] = False
        if kwargs["rank"] == 0:
            by_object: dict[str, list[float]] = {}
            for row in rows:
                row["selection_score"] = float(row["total_loss"])
                by_object.setdefault(str(row["sample_id"]), []).append(row["selection_score"])
            summary["selection_score"] = float(np.mean([np.mean(values) for values in by_object.values()]))
        summary = trainer._broadcast_object(
            summary if kwargs["rank"] == 0 else None, kwargs["rank"], kwargs["world_size"]
        )
        return summary, rows

    trainer._evaluate = evaluate

    def preflight(*args, **kwargs):
        previous = _STATE["phase"]
        _STATE["phase"] = "trainer_preflight"
        try:
            return _ORIGINAL_PREFLIGHT(*args, **kwargs)
        finally:
            _STATE["phase"] = previous

    trainer._preflight = preflight

    def atomic_save(payload, path):
        _ORIGINAL_ATOMIC_SAVE(payload, path)
        if path.name == "checkpoint_last.pt" and int(payload["completed_steps"]) == 100:
            _ORIGINAL_ATOMIC_SAVE(payload, path.with_name("checkpoint_additional_000100.pt"))

    trainer._atomic_torch_save = atomic_save

    def write_json(path, value):
        if path.name == "run_contract.json":
            value = dict(value)
            value.update(
                train_objects=[f"P{i:02d}" for i in range(1, 12)],
                validation_objects=["V01", "V02", "V03"],
                test_objects=["T02", "T03", "T04"],
                source_total_steps=400,
                additional_steps=200,
                final_total_steps=600,
                continuation_arm=str(spec(config)["kind"]),
                frozen_reference_uses_input_only=True,
                numerical_precision="fp32_tf32_disabled",
            )
        _ORIGINAL_WRITE_JSON(path, value)

    trainer._write_json = write_json


def run_train(config_path: str | Path, *, resume: bool = False) -> None:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    contract = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    if old.sha256(config_path) != contract["config_hashes"][str(config_path)]:
        raise ValueError("Configuration changed after preflight")
    for path, digest in contract["training_source_hashes"].items():
        if old.sha256(path) != digest:
            raise ValueError(f"Frozen training source changed: {path}")
    from datasets.matlab_multivolume_dataset import load_dataset_index
    _index, fingerprint = load_dataset_index(DATA)
    if fingerprint != contract["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint changed")
    folder = Path(config["experiment"]["output_dir"]).resolve()
    if folder.parent != OUTPUT.resolve():
        raise ValueError("Training output escaped the isolated experiment directory")
    install_training(config)
    checkpoint_path = folder / "checkpoint_last.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _STATE["completed_steps"] = int(checkpoint["completed_steps"])
        del checkpoint
    torch.set_num_threads(4)
    arguments = SimpleNamespace(
        config=str(config_path), max_steps=None, global_batch_size=None,
        validate_every=None, phase_chunk_size=None,
        resume=str(checkpoint_path) if resume else None,
        output_dir=str(folder), limit_validation_items=None, limit_test_items=None,
    )
    trainer.run_training(arguments)
