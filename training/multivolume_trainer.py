from __future__ import annotations

import contextlib
import copy
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset
from losses.self_supervised_losses import TaylorH2VarianceModel, compute_self_supervised_loss
from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from train_volume import _model_from_config
from training.global_batch_schedule import FixedGlobalBatchScheduler, slots_for_rank
from utils.checkpoint import cpu_state_dict
from utils.io import save_volume_tiff
from utils.reconstruction_metrics import reconstruction_metrics


LOGGER = logging.getLogger("multivolume")
CHECKPOINT_FORMAT = "multivolume_shared_no_mean_v1"


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    # Bind the rank before NCCL initialization. Otherwise the first collective
    # cannot reliably infer which visible A40 belongs to this process.
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
            device_id=device,
        )
    return world_size, rank, local_rank, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _all_gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: Any, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    ablation = config["ablation"]
    loss = config["loss"]
    optimization = config["optimization"]
    if data["stream"] != "physics" or int(data["input_frames"]) != 10 or int(
        data["target_frames"]
    ) != 90:
        raise ValueError("The frozen training protocol requires physics 10/90 data")
    if data.get("var_feature_representation", "sqrt") not in {"sqrt", "raw"}:
        raise ValueError("data.var_feature_representation must be 'sqrt' or 'raw'")
    if model.get("set_encoder_type", "mean_std") != "mean_std":
        raise ValueError("The first shared-training experiment requires mean/std SetBranch")
    if model.get("coarse_application", "legacy_anchor_positive") != "legacy_anchor_positive":
        raise ValueError("The first shared-training experiment requires the V1 anchor map")
    required_branches = (
        "use_network_refinement",
        "use_variance_branch",
        "use_mean_branch",
        "use_gate",
    )
    if not all(bool(ablation.get(name, False)) for name in required_branches):
        raise ValueError("Shared training requires VAR+Mean refinement and the gate configuration")
    if not isinstance(ablation.get("use_set_branch"), bool):
        raise ValueError("ablation.use_set_branch must explicitly be true or false")
    if float(loss["lambda_mean"]) != 0.0 or float(loss["lambda_var"]) != 1.0:
        raise ValueError("The shared-training baseline is the no_mean absolute variance objective")
    if int(optimization["micro_batch_per_gpu"]) != 1:
        raise ValueError("Only micro_batch_per_gpu=1 is supported for full 260x260 physics")
    if int(optimization["global_batch_size"]) < 1:
        raise ValueError("global_batch_size must be positive")
    if int(optimization["validate_every"]) != int(optimization["checkpoint_every"]):
        raise ValueError(
            "validate_every and checkpoint_every must match so every exact-resume "
            "checkpoint has a validation score"
        )


def _resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable part of a run configuration.

    Extending max_steps and changing the concrete output directory do not alter the
    mathematical experiment. Every other field is part of exact-resume semantics.
    """
    contract = copy.deepcopy(config)
    contract.get("experiment", {}).pop("output_dir", None)
    contract.get("optimization", {}).pop("max_steps", None)
    return contract


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _psf_cache_spec(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    psf_config = config["psf"]
    source = _resolve(config_path, psf_config["H_path"])
    stat = source.stat()
    return {
        "cache_format": "selected_lfm_h_float32_v1",
        "source": str(source),
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "H_variable_name": psf_config["H_variable_name"],
        "Ht_variable_name": psf_config["Ht_variable_name"],
        "psf_z_all_um": psf_config.get("psf_z_all_um"),
        "z_values_um": [float(value) for value in psf_config["z_values_um"]],
        "depth_unit": psf_config.get("depth_unit", "auto"),
    }


def _prepare_psf_cache(
    config: dict[str, Any],
    config_path: Path,
    rank: int,
    world_size: int,
) -> Path:
    spec = _psf_cache_spec(config, config_path)
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = _resolve(
        config_path, config["runtime"].get("psf_cache_dir", "data/.cache/psf")
    )
    cache_path = cache_dir / f"selected_H_{digest}.npy"
    metadata_path = cache_dir / f"selected_H_{digest}.json"
    if rank == 0:
        valid = False
        if cache_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                valid = (
                    metadata.get("complete") is True
                    and metadata.get("spec") == spec
                    and array.dtype == np.float32
                    and array.ndim == 5
                    and array.shape[0] == len(spec["z_values_um"])
                )
                del array
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
        if not valid:
            LOGGER.info("Building one-time uncompressed selected-PSF cache: %s", cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            psf_config = config["psf"]
            psf = load_psf(
                spec["source"],
                np.asarray(psf_config["z_values_um"], dtype=np.float32),
                h_variable_name=psf_config["H_variable_name"],
                ht_variable_name=psf_config["Ht_variable_name"],
                psf_z_all_um=psf_config.get("psf_z_all_um"),
                depth_unit=psf_config.get("depth_unit", "auto"),
                load_h=True,
                load_ht=False,
            )
            assert psf.H is not None
            temporary = cache_path.with_name(cache_path.stem + f".tmp-{os.getpid()}.npy")
            np.save(temporary, psf.H, allow_pickle=False)
            os.replace(temporary, cache_path)
            _write_json(
                metadata_path,
                {
                    "complete": True,
                    "spec": spec,
                    "shape": list(psf.H.shape),
                    "dtype": str(psf.H.dtype),
                    "selected_indices": psf.metadata.selected_indices.tolist(),
                    "selected_z_um": psf.metadata.selected_z_um.tolist(),
                },
            )
        LOGGER.info("Using selected-PSF cache: %s", cache_path)
    _barrier(world_size)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Rank {rank} cannot see PSF cache {cache_path}")
    return cache_path


def _load_operator(
    config: dict[str, Any],
    config_path: Path,
    device: torch.device,
    *,
    selected_h_cache: Path | None = None,
) -> LFMOperator:
    psf_config = config["psf"]
    if selected_h_cache is None:
        psf = load_psf(
            _resolve(config_path, psf_config["H_path"]),
            np.asarray(psf_config["z_values_um"], dtype=np.float32),
            h_variable_name=psf_config["H_variable_name"],
            ht_variable_name=psf_config["Ht_variable_name"],
            psf_z_all_um=psf_config.get("psf_z_all_um"),
            depth_unit=psf_config.get("depth_unit", "auto"),
            load_h=True,
            load_ht=False,
        )
        assert psf.H is not None
        selected_h = psf.H
    else:
        # mmap_mode='c' is writable copy-on-write, avoiding PyTorch's warning
        # while still sharing the OS page cache across DDP worker processes.
        selected_h = np.load(selected_h_cache, mmap_mode="c", allow_pickle=False)
    h = torch.from_numpy(selected_h).to(device=device, dtype=torch.float32)
    return LFMOperator(
        h,
        mode=config["runtime"]["operator_mode"],
        phase_chunk_size=int(config["runtime"]["operator_phase_chunk_size"]),
    )


def _to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(item)
    for name in (
        "f_var",
        "f_var_feature",
        "g_mean",
        "input_mean",
        "residual_frames",
        "measured_mean",
        "measured_variance",
        "ground_truth",
        "z_values_um",
    ):
        if name in item:
            output[name] = item[name].unsqueeze(0).to(device=device, dtype=torch.float32)
    return output


@torch.no_grad()
def _analytic_beta0(operator: LFMOperator, f_var: Tensor, input_mean: Tensor) -> Tensor:
    prediction = operator(f_var)
    numerator = (prediction * input_mean).flatten(1).sum(dim=1)
    denominator = prediction.square().flatten(1).sum(dim=1).add(1e-8)
    value = (numerator / denominator).clamp_min(1e-8)
    if not torch.isfinite(value).all():
        raise FloatingPointError("Analytic beta0 is non-finite")
    return value


def _configure_trainable_parameters(model: torch.nn.Module) -> dict[str, Any]:
    """Freeze bypassed Set parameters without changing module initialization order."""
    if not model.use_set_branch:
        for name, parameter in model.named_parameters():
            if name.startswith(("set_encoder.", "fusion.set_projection.", "fusion.gate.")) or name == "fusion.raw_alpha":
                parameter.requires_grad_(False)
    return {
        "use_set_branch": model.use_set_branch,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameter_names": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
    }


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    options = config["optimization"]
    network = [
        parameter for name, parameter in model.named_parameters()
        if name != "raw_beta" and parameter.requires_grad
    ]
    groups = [
        {"params": network, "lr": float(options["lr_network"])},
        {"params": [dict(model.named_parameters())["raw_beta"]], "lr": float(options["lr_beta"])},
    ]
    if str(options["optimizer"]).lower() != "adam":
        raise ValueError("The frozen first experiment uses Adam")
    return torch.optim.Adam(groups)


def _loss(
    output: Any,
    item: dict[str, Any],
    operator: LFMOperator,
    variance_model: TaylorH2VarianceModel,
    config: dict[str, Any],
) -> Any:
    return compute_self_supervised_loss(
        output.reconstruction.to(operator.H.dtype),
        item["measured_mean"],
        item["measured_variance"],
        operator,
        variance_model,
        physics_use_checkpoint=bool(config["runtime"]["physics_use_checkpoint"]),
        **config["loss"],
    )


def _forward(
    model: torch.nn.Module,
    item: dict[str, Any],
    operator: LFMOperator,
    beta0: Tensor | None = None,
) -> tuple[Any, Tensor]:
    if beta0 is None:
        beta0 = _analytic_beta0(operator, item["f_var"], item["input_mean"])
    output = model(
        item["f_var"],
        item["g_mean"],
        item["residual_frames"],
        item["z_values_um"],
        var_feature_volume=item["f_var_feature"],
        beta0=beta0,
    )
    return output, beta0


def _cached_beta0(
    cache: dict[tuple[str, int], float],
    item: dict[str, Any],
    operator: LFMOperator,
) -> Tensor:
    key = (str(item["sample_id"]), int(item["subset_index"]))
    if key not in cache:
        cache[key] = float(
            _analytic_beta0(operator, item["f_var"], item["input_mean"]).item()
        )
    return item["f_var"].new_tensor([cache[key]])


def _preflight(
    model: torch.nn.Module,
    dataset: MatlabMultiVolumeDataset,
    operator: LFMOperator,
    variance_model: TaylorH2VarianceModel,
    config: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> float:
    limit_bytes = int(float(config["runtime"]["max_peak_memory_gib"]) * 1024**3)
    item = _to_device(dataset[rank % len(dataset)], device)

    def attempt() -> float:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        output, _ = _forward(model, item, operator)
        losses = _loss(output, item, operator, variance_model, config)
        losses.total.backward()
        torch.cuda.synchronize(device)
        peak = float(torch.cuda.max_memory_allocated(device))
        model.zero_grad(set_to_none=True)
        return peak

    try:
        peak = attempt()
    except torch.OutOfMemoryError:
        if operator.phase_chunk_size <= 16:
            raise
        LOGGER.warning("Rank %d OOM with phase chunk %d; retry chunk 16", rank, operator.phase_chunk_size)
        operator.phase_chunk_size = 16
        peak = attempt()
    if peak > limit_bytes and operator.phase_chunk_size > 16:
        LOGGER.warning(
            "Rank %d preflight peak %.2f GiB exceeds limit; train with phase chunk 16",
            rank,
            peak / 1024**3,
        )
        operator.phase_chunk_size = 16
    selected = torch.tensor(operator.phase_chunk_size, device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(selected, op=dist.ReduceOp.MIN)
    operator.phase_chunk_size = int(selected.item())
    return peak / 1024**3


def _average_training_metrics(
    local: dict[str, float],
    count: int,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    names = sorted(local)
    values = torch.tensor([local[name] for name in names] + [float(count)], device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total = max(float(values[-1].item()), 1.0)
    return {name: float(values[index].item() / total) for index, name in enumerate(names)}


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(
    *,
    model: torch.nn.Module,
    dataset: MatlabMultiVolumeDataset,
    operator: LFMOperator,
    variance_model: TaylorH2VarianceModel,
    config: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    step: int,
    split: str,
    output_dir: Path,
    beta0_cache: dict[tuple[str, int], float],
    limit_items: int | None,
    save_predictions: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    selected = list(range(len(dataset)))
    if limit_items is not None:
        selected = selected[: max(0, min(int(limit_items), len(selected)))]
    local_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for position, dataset_index in enumerate(selected):
            if position % world_size != rank:
                continue
            item = _to_device(dataset[dataset_index], device)
            beta0 = _cached_beta0(beta0_cache, item, operator)
            output, beta0 = _forward(model, item, operator, beta0)
            losses = _loss(output, item, operator, variance_model, config)
            prediction = output.reconstruction[0, 0].detach().cpu().numpy()
            truth = item["ground_truth"][0, 0].detach().cpu().numpy()
            row: dict[str, Any] = {
                "step": step,
                "split": split,
                "sample_id": item["sample_id"],
                "subset_index": int(item["subset_index"]),
                "selection_score": float(
                    (losses.weighted_var + losses.weighted_tv).detach().item()
                ),
                "beta0": float(beta0.item()),
                "beta": float(output.beta.item()),
            }
            row.update(losses.scalar_metrics())
            row.update(
                reconstruction_metrics(
                    prediction,
                    truth,
                    item["z_values_um"][0].detach().cpu().numpy(),
                )
            )
            local_rows.append(row)
            if save_predictions:
                destination = (
                    output_dir
                    / split
                    / f"step_{step:06d}"
                    / str(item["sample_id"])
                    / f"subset_{int(item['subset_index']):02d}"
                )
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / "reconstruction.npy", prediction)
                save_volume_tiff(destination / "reconstruction.tif", prediction)
    gathered = _all_gather_objects(local_rows, world_size)
    rows = [row for part in gathered for row in part]
    if not rows:
        raise RuntimeError(f"No {split} items were evaluated")
    object_scores: dict[str, list[float]] = {}
    for row in rows:
        object_scores.setdefault(str(row["sample_id"]), []).append(float(row["selection_score"]))
    summary = {
        "selection_score": float(
            np.mean([np.mean(values) for values in object_scores.values()])
        ),
        "objects": float(len(object_scores)),
        "items": float(len(rows)),
    }
    numeric_names = [
        name
        for name, value in rows[0].items()
        if isinstance(value, (int, float))
        and name not in {"step", "subset_index", "selection_score"}
    ]
    for name in numeric_names:
        summary[name] = float(np.mean([float(row[name]) for row in rows]))
    summary = _broadcast_object(summary if rank == 0 else None, rank, world_size)
    model.train()
    return summary, rows if rank == 0 else []


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: FixedGlobalBatchScheduler,
    completed_steps: int,
    best_score: float,
    best_step: int,
    config: dict[str, Any],
    fingerprint: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    rng_states = _all_gather_objects(_rng_state(device), world_size)
    if rank != 0:
        return None
    return {
        "format": CHECKPOINT_FORMAT,
        "completed_steps": completed_steps,
        "best_validation_score": best_score,
        "best_step": best_step,
        "model_state": cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "dataset_fingerprint": fingerprint,
        "world_size": world_size,
        "physical_gpus": os.environ.get("SPECKLE_PHYSICAL_GPUS", ""),
        "rng_states": rng_states,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_training(arguments: Any) -> None:
    world_size, rank, local_rank, device = _distributed()
    tensorboard_writer: Any | None = None
    try:
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        )
        config_path = Path(arguments.config).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if arguments.max_steps is not None:
            config["optimization"]["max_steps"] = int(arguments.max_steps)
        if arguments.global_batch_size is not None:
            config["optimization"]["global_batch_size"] = int(arguments.global_batch_size)
        if arguments.validate_every is not None:
            config["optimization"]["validate_every"] = int(arguments.validate_every)
            config["optimization"]["checkpoint_every"] = int(arguments.validate_every)
        if arguments.phase_chunk_size is not None:
            config["runtime"]["operator_phase_chunk_size"] = int(arguments.phase_chunk_size)
        if arguments.resume:
            resume_path = Path(arguments.resume).expanduser().resolve()
            output_dir = resume_path.parent
        elif arguments.output_dir:
            resume_path = None
            output_dir = Path(arguments.output_dir).expanduser().resolve()
        else:
            resume_path = None
            output_dir = _resolve(config_path, config["experiment"]["output_dir"])

        checkpoint: dict[str, Any] | None = None
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("Checkpoint is not a shared multi-volume checkpoint")
            if _resume_contract(checkpoint["config"]) != _resume_contract(config):
                raise ValueError(
                    "Resume configuration differs from the checkpoint. Only max_steps "
                    "and the concrete output directory may change."
                )
            if int(checkpoint["world_size"]) != world_size:
                raise ValueError("Exact resume requires the same number of GPUs")

        _validate_config(config)
        if int(config["optimization"]["global_batch_size"]) < world_size:
            raise ValueError("global_batch_size must be at least the selected GPU count")
        seed = int(config["experiment"]["seed"])
        _seed_everything(seed)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            config_name = "config_resume_used.yaml" if checkpoint is not None else "config_used.yaml"
            (output_dir / config_name).write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        _barrier(world_size)

        data_root = _resolve(config_path, config["data"]["root"])
        cache_dir = _resolve(config_path, config["data"]["cache_dir"])
        dataset_options = {
            "cache_dir": cache_dir,
            "var_feature_representation": config["data"].get(
                "var_feature_representation", "sqrt"
            ),
        }
        if rank == 0 and bool(config["data"].get("precompute_cache", True)):
            LOGGER.info("Preparing validated MAT cache")
            for split in ("train", "validation", "test"):
                MatlabMultiVolumeDataset(data_root, split, **dataset_options).precompute()
        _barrier(world_size)
        train_data = MatlabMultiVolumeDataset(data_root, "train", **dataset_options)
        validation_data = MatlabMultiVolumeDataset(
            data_root, "validation", **dataset_options
        )
        test_data = MatlabMultiVolumeDataset(data_root, "test", **dataset_options)
        fingerprints = {train_data.dataset_fingerprint, validation_data.dataset_fingerprint, test_data.dataset_fingerprint}
        if len(fingerprints) != 1:
            raise RuntimeError("Dataset splits produced different fingerprints")
        fingerprint = fingerprints.pop()

        psf_cache_path = _prepare_psf_cache(config, config_path, rank, world_size)
        operator = _load_operator(
            config, config_path, device, selected_h_cache=psf_cache_path
        )
        variance_model = TaylorH2VarianceModel(operator, **config["noise"])
        base_model = _model_from_config(config).to(device)
        parameter_contract = _configure_trainable_parameters(base_model)
        optimizer = _optimizer(base_model, config)
        scheduler = FixedGlobalBatchScheduler(
            len(train_data),
            int(config["optimization"]["global_batch_size"]),
            seed,
        )
        completed_steps = 0
        best_score = math.inf
        best_step = -1
        if checkpoint is not None:
            if checkpoint["dataset_fingerprint"] != fingerprint:
                raise ValueError("Dataset changed since the checkpoint was created")
            base_model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            completed_steps = int(checkpoint["completed_steps"])
            best_score = float(checkpoint["best_validation_score"])
            best_step = int(checkpoint["best_step"])

        preflight_peak = _preflight(
            base_model,
            train_data,
            operator,
            variance_model,
            config,
            device,
            rank,
            world_size,
        )
        if not base_model.use_set_branch and operator.phase_chunk_size != int(
            config["runtime"]["operator_phase_chunk_size"]
        ):
            raise RuntimeError("No-Set comparison requires the configured physics phase chunk unchanged")
        if checkpoint is not None:
            _restore_rng(checkpoint["rng_states"][rank], device)
        if rank == 0:
            LOGGER.info(
                "Training on physical GPUs %s | world=%d | train=%d val=%d test=%d | "
                "preflight_peak=%.2f GiB | phase_chunk=%d",
                os.environ.get("SPECKLE_PHYSICAL_GPUS", ""),
                world_size,
                len(train_data),
                len(validation_data),
                len(test_data),
                preflight_peak,
                operator.phase_chunk_size,
            )
            _write_json(
                output_dir / "run_contract.json",
                {
                    "dataset_root": str(data_root),
                    "dataset_fingerprint": fingerprint,
                    "train_objects": list(("P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10")),
                    "validation_objects": ["P09", "V01", "V02"],
                    "test_objects": ["P07", "T01", "T02"],
                    "input_frames": 10,
                    "target_frames": 90,
                    "target_use": "training constraint, not an independent object-level test",
                    "physical_gpus": os.environ.get("SPECKLE_PHYSICAL_GPUS", ""),
                    "world_size": world_size,
                    "phase_chunk_size": operator.phase_chunk_size,
                    "selected_psf_cache": str(psf_cache_path),
                    "parameter_contract": parameter_contract,
                    "preflight_peak_memory_gib": preflight_peak,
                },
            )

        model: torch.nn.Module
        if world_size > 1:
            model = DistributedDataParallel(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        else:
            model = base_model

        beta0_cache: dict[tuple[str, int], float] = {}

        training_rows = _read_csv(output_dir / "training_metrics.csv") if rank == 0 else []
        validation_rows = _read_csv(output_dir / "validation_metrics.csv") if rank == 0 else []
        if rank == 0 and bool(config["runtime"].get("tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError as exception:
                raise RuntimeError(
                    "runtime.tensorboard=true but tensorboard is not installed"
                ) from exception
            tensorboard_writer = SummaryWriter(
                log_dir=str(output_dir / "tensorboard"), purge_step=completed_steps
            )
        max_steps = int(config["optimization"]["max_steps"])
        validate_every = int(config["optimization"]["validate_every"])
        checkpoint_every = int(config["optimization"]["checkpoint_every"])
        if validate_every < 1 or checkpoint_every < 1:
            raise ValueError("Validation/checkpoint intervals must be positive")
        while completed_steps < max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_indices = scheduler.next_batch()
            slots, backward_scale = slots_for_rank(batch_indices, world_size, rank)
            local_sums: dict[str, float] = {}
            local_count = 0
            for slot_index, slot in enumerate(slots):
                item = _to_device(train_data[slot.dataset_index], device)
                synchronize = slot_index == len(slots) - 1
                context = (
                    contextlib.nullcontext()
                    if synchronize or world_size == 1
                    else model.no_sync()  # type: ignore[attr-defined]
                )
                with context:
                    beta0 = _cached_beta0(beta0_cache, item, operator)
                    output, beta0 = _forward(model, item, operator, beta0)
                    losses = _loss(output, item, operator, variance_model, config)
                    weighted = losses.total * (backward_scale if slot.valid else 0.0)
                    weighted.backward()
                if slot.valid:
                    metrics = losses.scalar_metrics()
                    metrics["beta0"] = float(beta0.item())
                    metrics["beta"] = float(output.beta.item())
                    for name, value in metrics.items():
                        local_sums[name] = local_sums.get(name, 0.0) + float(value)
                    local_count += 1
            optimizer.step()
            completed_steps += 1
            averaged = _average_training_metrics(local_sums, local_count, device, world_size)
            if rank == 0:
                row = {
                    "step": completed_steps,
                    "epoch": scheduler.epoch,
                    "global_batch_size": len(batch_indices),
                    **averaged,
                }
                training_rows.append(row)
                if tensorboard_writer is not None:
                    for name, value in averaged.items():
                        tensorboard_writer.add_scalar(f"train/{name}", value, completed_steps)
                LOGGER.info(
                    "step=%d/%d loss=%.6g var=%.6g beta=%.6g epoch=%d",
                    completed_steps,
                    max_steps,
                    averaged["total_loss"],
                    averaged["normalized_var_loss"],
                    averaged["beta"],
                    scheduler.epoch,
                )

            should_validate = completed_steps % validate_every == 0 or completed_steps == max_steps
            if not should_validate:
                continue
            validation_summary, rows = _evaluate(
                model=model,
                dataset=validation_data,
                operator=operator,
                variance_model=variance_model,
                config=config,
                device=device,
                rank=rank,
                world_size=world_size,
                step=completed_steps,
                split="validation",
                output_dir=output_dir,
                beta0_cache=beta0_cache,
                limit_items=arguments.limit_validation_items,
                save_predictions=True,
            )
            improved = validation_summary["selection_score"] < best_score
            if improved:
                best_score = validation_summary["selection_score"]
                best_step = completed_steps
            payload = _checkpoint_payload(
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_steps=completed_steps,
                best_score=best_score,
                best_step=best_step,
                config=config,
                fingerprint=fingerprint,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                validation_rows.extend(rows)
                _append_csv(output_dir / "training_metrics.csv", training_rows)
                _append_csv(output_dir / "validation_metrics.csv", validation_rows)
                _write_json(
                    output_dir / f"validation_step_{completed_steps:06d}.json",
                    validation_summary,
                )
                if tensorboard_writer is not None:
                    for name, value in validation_summary.items():
                        tensorboard_writer.add_scalar(
                            f"validation/{name}", value, completed_steps
                        )
                    tensorboard_writer.flush()
                assert payload is not None
                if completed_steps % checkpoint_every == 0 or completed_steps == max_steps:
                    _atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
                if improved:
                    _atomic_torch_save(payload, output_dir / "checkpoint_best.pt")
                LOGGER.info(
                    "validation step=%d score=%.6g best_step=%d",
                    completed_steps,
                    validation_summary["selection_score"],
                    best_step,
                )
            _barrier(world_size)

        best_payload = torch.load(
            output_dir / "checkpoint_best.pt", map_location=device, weights_only=False
        )
        base_model.load_state_dict(best_payload["model_state"])
        test_summary, test_rows = _evaluate(
            model=model,
            dataset=test_data,
            operator=operator,
            variance_model=variance_model,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
            step=best_step,
            split="test",
            output_dir=output_dir,
            beta0_cache=beta0_cache,
            limit_items=arguments.limit_test_items,
            save_predictions=True,
        )
        if rank == 0:
            _append_csv(output_dir / "test_metrics.csv", test_rows)
            _write_json(
                output_dir / "training_complete.json",
                {
                    "complete": True,
                    "completed_steps": completed_steps,
                    "best_step": best_step,
                    "best_validation_score": best_score,
                    "test": test_summary,
                    "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
                    "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
                },
            )
            if tensorboard_writer is not None:
                for name, value in test_summary.items():
                    tensorboard_writer.add_scalar(f"test/{name}", value, best_step)
                tensorboard_writer.flush()
            LOGGER.info("TRAINING COMPLETE best_step=%d test_score=%.6g", best_step, test_summary["selection_score"])
        _barrier(world_size)
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
