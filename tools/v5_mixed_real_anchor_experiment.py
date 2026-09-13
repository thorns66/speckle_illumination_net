"""V5 two-arm experiment: P01-P11 plus full-field real 45/55, no P12."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import TaylorH2VarianceModel
from models.configurable_anchor_lfm_net import model_from_config
from tools import three_way_experiment as old
from tools import v3_compare_experiment as base
from tools.mixed_resolution_lfm import MixedResolutionLFM
from tools.v5_mixed_dataset import MixedExperimentDataset, MixedSixOneOneScheduler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "V5_MIXED_OUTPUT",
    ROOT / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run01",
))
SIMULATION_DATA = ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/dataset_v3_frozen_view"
REAL_DATA = Path(os.environ.get(
    "V5_REAL_DATA", ROOT / "data/spinach_real_mixed_training_20260910_run01"
))
SHARED_INITIAL_STATE = ROOT / "outputs/v3_baseline_e3_mean005_400_20260908_run01/shared_initial_state.pt"
SPARSE_CACHE = ROOT / "outputs/parallel_validation_scale_cov_mean_20260907/sparse_operator"
ARMS = ("taylor_anchor_e3_mean100", "mean_anchor_e3_mean100")
ANCHORS = {ARMS[0]: "taylor_sqrt", ARMS[1]: "mean_rl3"}
REAL_RANKS = (0, 1)
_NATIVE_VALIDATION = trainer._validate_config


def _spec(config: dict[str, Any]) -> dict[str, Any]:
    return config["v5_mixed"]


def _arm(config: dict[str, Any]) -> str:
    return str(_spec(config)["kind"])


def _anchor(config: dict[str, Any]) -> str:
    return ANCHORS[_arm(config)]


def validate_config(config: dict[str, Any]) -> None:
    arm = _arm(config)
    if arm not in ARMS or config["model"].get("reconstruction_anchor") != ANCHORS[arm]:
        raise ValueError("V5 arm and reconstruction anchor disagree")
    spec = _spec(config)
    required = {
        "simulation_per_batch": 6,
        "field45_per_batch": 1,
        "field55_per_batch": 1,
        "lr_drop_after_steps": 200,
        "ramp_steps": 50,
    }
    for name, expected in required.items():
        if int(spec[name]) != expected:
            raise ValueError(f"V5 {name} must remain {expected}")
    if bool(spec.get("p12_in_training", True)):
        raise ValueError("P12 must be excluded from V5 training")
    if float(spec["shape_gradient_budget"]) != 1.0:
        raise ValueError("V5 keeps the mean-shape gradient budget at 100%")
    if Path(spec["shared_initial_state"]).resolve() != SHARED_INITIAL_STATE.resolve():
        raise ValueError("V5 must use the frozen common random initialization")
    if Path(spec["simulation_manifest_root"]).resolve() != SIMULATION_DATA.resolve():
        raise ValueError("V5 must use the frozen P01-P11 simulation view")
    if Path(spec["real_final_manifest"]).resolve() != (REAL_DATA / "final_manifest.json").resolve():
        raise ValueError("V5 must record the finalized 45/55 real-data manifest")
    if int(config["optimization"]["max_steps"]) != 600:
        raise ValueError("V5 requires exactly 600 optimizer updates")
    if int(config["optimization"]["global_batch_size"]) != 8:
        raise ValueError("V5 global batch must remain eight")
    if int(config["optimization"]["validate_every"]) != 20:
        raise ValueError("V5 validates every 20 updates")
    if bool(config["runtime"]["amp"]) or not bool(config["model"].get("activation_checkpoint_segments")):
        raise ValueError("V5 requires FP32 and segment activation recomputation")
    if config["data"].get("var_feature_representation") != "sqrt":
        raise ValueError("The VAR encoder must keep Taylor-RL3-sqrt")
    if not bool(config["ablation"].get("use_set_branch")):
        raise ValueError("The Set branch must remain enabled")
    if any(float(config["noise"][name]) != 0.0 for name in ("alpha_noise", "sigma_read")):
        raise ValueError("Uncalibrated real-noise terms must remain disabled")
    compatibility = copy.deepcopy(config)
    _NATIVE_VALIDATION(compatibility)


def configure_precision() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def build_model(config: dict[str, Any], *, initial: bool) -> torch.nn.Module:
    configure_precision()
    model = model_from_config(config)
    if initial:
        payload = torch.load(SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        if old.state_hash(model.state_dict()) != payload["common_state_sha256"]:
            raise ValueError("Common network initialization changed")
    model.register_parameter("mean_gain_gamma", torch.nn.Parameter(torch.zeros(())))
    return model


def load_operator(
    config: dict[str, Any], device: torch.device, *, selected_h_cache: Path
) -> MixedResolutionLFM:
    configure_precision()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    needs_real = world_size == 1 or rank in REAL_RANKS
    full_h = None
    if needs_real:
        selected = np.load(selected_h_cache, mmap_mode="c", allow_pickle=False)
        full_h = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
    return MixedResolutionLFM(
        SPARSE_CACHE, device, full_h=full_h,
        phase_chunk_size=int(config["runtime"]["operator_phase_chunk_size"]),
        real_shape=(1029, 1421),
        load_sparse_simulation=not needs_real,
    )


def anchor_volume(item: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    return item["g_mean"] if _anchor(config) == "mean_rl3" else item["f_var"]


@torch.no_grad()
def analytic_beta0(operator: MixedResolutionLFM, anchor: torch.Tensor, input_mean: torch.Tensor) -> torch.Tensor:
    prediction = operator(anchor)
    numerator = (prediction * input_mean).flatten(1).sum(1)
    denominator = prediction.square().flatten(1).sum(1).add(1e-8)
    value = (numerator / denominator).clamp_min(1e-8)
    if not torch.isfinite(value).all():
        raise FloatingPointError("V5 analytic beta0 is non-finite")
    return value


def cached_beta0(cache: dict, item: dict[str, Any], operator: MixedResolutionLFM, config: dict[str, Any]) -> torch.Tensor:
    volume = anchor_volume(item, config)
    key = (
        str(item["sample_id"]), int(item["subset_index"]), _anchor(config),
        str(item.get("domain", "simulation")), tuple(volume.shape[-2:]),
    )
    if key not in cache:
        cache[key] = float(analytic_beta0(operator, volume, item["input_mean"]).item())
    return volume.new_tensor([cache[key]])


def forward(model, item, operator, beta0, config):
    volume = anchor_volume(item, config)
    if beta0 is None:
        beta0 = analytic_beta0(operator, volume, item["input_mean"])
    output, beta0 = base._ORIGINAL_FORWARD(model, item, operator, beta0)
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
    gain = a0 * (1.0 + float(config["three_way"]["gain_bound"]) * torch.tanh(instance.mean_gain_gamma))
    reconstruction = q * gain[:, None, None, None, None]
    output.reconstruction = reconstruction
    output._shape = q
    output._gain = gain
    output._mean_prediction = hq * gain[:, None, None, None]
    output._network_anchor = network_anchor
    output._physical_anchor = network_anchor * (gain[:, None, None, None, None] / total)
    output._pre_gain_reconstruction = pre_gain
    output._effective_correction = reconstruction - output._physical_anchor
    output._anchor_kind = _anchor(config)
    output._v3_training = bool(model.training and torch.is_grad_enabled())
    return output, beta0


def optimizer(model: torch.nn.Module, config: dict[str, Any]):
    return base._optimizer(model, config)


def loss(output, item, operator, variance_model, config):
    result = base.loss(output, item, operator, variance_model, config)
    if base._STATE["phase"] == "training" and not base._STATE["evaluation"]:
        base._append_jsonl(
            Path(config["experiment"]["output_dir"])
            / f"domain_training_rank{int(os.environ.get('RANK', '0'))}.jsonl",
            {
                "update_step": int(base._STATE["completed_steps"]) + 1,
                "sample_id": str(item["sample_id"]),
                "subset_index": int(item["subset_index"]),
                "domain": str(item.get("domain", "simulation")),
                **result.scalar_metrics(),
            },
        )
    return result


def _allreduce_max(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def preflight(model, dataset, operator, variance_model, config, device, rank, world_size) -> float:
    previous = base._STATE["phase"]
    rng_state = trainer._rng_state(device)
    state_before = old.state_hash(model.state_dict())
    base._STATE["phase"] = "trainer_preflight"
    try:
        index = 110 if rank == 0 else 120 if rank == 1 else rank - 2
        item = trainer._to_device(dataset[index], device)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        free_before, _total = torch.cuda.mem_get_info(device)
        baseline_allocated = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        beta0 = cached_beta0({}, item, operator, config)
        output, _ = forward(model, item, operator, beta0, config)
        losses = loss(output, item, operator, variance_model, config)
        losses.total.backward()
        torch.cuda.synchronize(device)
        peak = float(torch.cuda.max_memory_allocated(device))
        model.zero_grad(set_to_none=True)
        state_after = old.state_hash(model.state_dict())
        if state_after != state_before:
            raise RuntimeError(f"V5 preflight changed model parameters or buffers on rank {rank}")
        limit = float(config["runtime"]["max_peak_memory_gib"]) * 1024**3
        reserve = float(config["runtime"]["required_free_reserve_gib"]) * 1024**3
        capacity_for_process = float(free_before) + baseline_allocated
        if peak > limit or peak + reserve > capacity_for_process:
            raise RuntimeError(
                f"V5 full-field preflight violates memory contract on rank {rank}: "
                f"peak={peak/1024**3:.2f}GiB capacity={capacity_for_process/1024**3:.2f}GiB"
            )
        base._ORIGINAL_WRITE_JSON(
            Path(config["experiment"]["output_dir"]) / f"preflight_rank_{rank}.json",
            {
                "complete": True, "rank": rank, "domain": item["domain"],
                "sample_id": item["sample_id"], "subset_index": int(item["subset_index"]),
                "peak_memory_gib": peak / 1024**3,
                "free_before_gib": free_before / 1024**3,
                "required_reserve_gib": reserve / 1024**3,
                "phase_chunk_size": operator.phase_chunk_size,
                "model_state_sha256_before": state_before,
                "model_state_sha256_after": state_after,
                "model_state_unchanged": True,
                "rng_restored_after_preflight": True,
            },
        )
        return _allreduce_max(peak / 1024**3, device, world_size)
    finally:
        trainer._restore_rng(rng_state, device)
        base._STATE["phase"] = previous


def install_training(config: dict[str, Any]) -> None:
    validate_config(config)
    configure_precision()
    base._STATE.update(completed_steps=0, micro_call=0, phase="training", evaluation=False)
    trainer._validate_config = validate_config

    real_root = Path(config["data"]["real_root"])
    def dataset_factory(root, split, **kwargs):
        # MixedExperimentDataset itself uses the unpatched class imported in its module.
        return MixedExperimentDataset(root, split, real_root=real_root, **kwargs)

    trainer.MatlabMultiVolumeDataset = dataset_factory
    trainer.FixedGlobalBatchScheduler = MixedSixOneOneScheduler
    trainer._model_from_config = lambda current: build_model(current, initial=True)
    trainer._load_operator = lambda current, _path, device, selected_h_cache=None, **_kwargs: load_operator(
        current, device, selected_h_cache=Path(selected_h_cache)
    )
    trainer._cached_beta0 = lambda cache, item, operator: cached_beta0(cache, item, operator, config)
    trainer._forward = lambda model, item, operator, beta0=None: forward(model, item, operator, beta0, config)
    trainer._loss = loss
    trainer.TaylorH2VarianceModel = TaylorH2VarianceModel
    trainer._optimizer = optimizer
    trainer._configure_trainable_parameters = base._ORIGINAL_CONFIGURE
    trainer._preflight = preflight

    def evaluate(**kwargs):
        base._STATE["evaluation"] = True
        try:
            summary, rows = base._ORIGINAL_EVALUATE(**kwargs)
        finally:
            base._STATE["evaluation"] = False
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

    def atomic_save(payload, path):
        base._ORIGINAL_ATOMIC_SAVE(payload, path)
        if path.name == "checkpoint_last.pt" and int(payload["completed_steps"]) in (200, 400, 600):
            base._ORIGINAL_ATOMIC_SAVE(
                payload, path.with_name(f"checkpoint_step_{int(payload['completed_steps']):06d}.pt")
            )

    trainer._atomic_torch_save = atomic_save

    def write_json(path, value):
        if path.name == "run_contract.json":
            value = dict(value)
            value.update(
                train_objects=[f"P{i:02d}" for i in range(1, 12)],
                excluded_training_objects=["P12"], real_training_fields=["45", "55"],
                real_ground_truth=False, validation_objects=["V01", "V02", "V03"],
                test_objects=["T02", "T03", "T04"], mixed_batch="6 simulation + 1 field45 + 1 field55",
                real_shape_yx=[1029, 1421], spatial_policy="full field; no resize; no tiles",
                activation_checkpoint_segments=True, reconstruction_anchor=_anchor(config),
                beta_cache_includes_anchor_domain_shape=True, numerical_precision="FP32; TF32/AMP disabled",
                operator_residency="ranks0-1 full PSF FFT for simulation+real; ranks2-5 validated sparse simulation operator",
                real_final_manifest=str(REAL_DATA / "final_manifest.json"),
            )
        base._ORIGINAL_WRITE_JSON(path, value)

    trainer._write_json = write_json


def run_train(config_path: str | Path, *, resume: bool = False) -> None:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    contract = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    if old.sha256(config_path) != contract["config_hashes"][str(config_path)]:
        raise ValueError("V5 configuration changed after freezing")
    for path, digest in contract["source_hashes"].items():
        if old.sha256(path) != digest:
            raise ValueError(f"V5 source changed after freezing: {path}")
    if old.sha256(REAL_DATA / "final_manifest.json") != contract["real_final_manifest_sha256"]:
        raise ValueError("Real training data manifest changed after freezing")
    if int(os.environ.get("WORLD_SIZE", "1")) != 6:
        raise ValueError("V5 exact mixed schedule requires six DDP ranks")
    torch.set_num_threads(4)
    install_training(config)
    checkpoint = Path(config["experiment"]["output_dir"]) / "checkpoint_last.pt"
    if resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base._STATE["completed_steps"] = int(payload["completed_steps"])
        del payload
    arguments = SimpleNamespace(
        config=str(config_path), max_steps=None, global_batch_size=None,
        validate_every=None, phase_chunk_size=None,
        resume=str(checkpoint) if resume else None,
        output_dir=str(Path(config["experiment"]["output_dir"])),
        limit_validation_items=None, limit_test_items=None,
    )
    trainer.run_training(arguments)
