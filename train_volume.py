from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from losses.self_supervised_losses import TaylorH2VarianceModel, compute_self_supervised_loss
from models.variance_anchored_lfm_net import VarianceAnchoredLFMNet
from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from utils.checkpoint import cpu_state_dict, load_model_checkpoint, save_checkpoint
from utils.io import load_tiff_stack, load_volume_tiff, save_volume_tiff, select_frame_indices
from utils.visualization import save_gate_outputs, save_loss_curve, save_volume_visualizations

LOGGER = logging.getLogger("train_volume")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent.parent / path).resolve()
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_from_config(config: dict[str, Any]) -> VarianceAnchoredLFMNet:
    model = config["model"]
    ablation = config["ablation"]
    return VarianceAnchoredLFMNet(
        var_channels=model["var_channels"],
        mean_channels=model["mean_channels"],
        set_channels=model["set_channels"],
        decoder_channels=model["decoder_channels"],
        set_frame_chunk_size=model["set_frame_chunk_size"],
        alpha_init=model["alpha_init"],
        beta_range=model["beta_range"],
        positivity_eps=model.get("positivity_eps", 1e-8),
        z_scale_um=model["z_scale_um"],
        set_use_checkpoint=model.get("set_use_checkpoint", True),
        use_variance_branch=ablation["use_variance_branch"],
        use_mean_branch=ablation["use_mean_branch"],
        use_set_branch=ablation["use_set_branch"],
        use_gate=ablation["use_gate"],
        use_network_refinement=ablation.get("use_network_refinement", True),
    )


def _optimizer_from_config(model: VarianceAnchoredLFMNet, config: dict[str, Any], warm: bool):
    options = config["optimization"]
    factor = options["warm_start_lr_factor"] if warm else 1.0
    network_parameters = [parameter for name, parameter in model.named_parameters() if name != "raw_beta"]
    groups = [
        {"params": network_parameters, "lr": options["lr_network"] * factor},
        {"params": [model.raw_beta], "lr": options["lr_beta"] * factor},
    ]
    name = options["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "adamw":
        return torch.optim.AdamW(groups)
    raise ValueError(f"Unsupported optimizer {options['optimizer']!r}")


def _save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-volume variance-anchored LFM optimization")
    parser.add_argument("--config", default="configs/v1_100_simulation.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--init", choices=("random", "checkpoint"), default="random")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.max_steps is not None:
        config["optimization"]["max_steps"] = args.max_steps
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)

    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = _resolve(config_path, config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    data_config = config["data"]
    frame_count = int(data_config["num_speckle_frames"])
    offline_count = data_config.get("offline_statistics_num_frames")
    if offline_count is not None and int(offline_count) != frame_count:
        raise ValueError("g_mean/F_var frame count does not match num_speckle_frames")
    intensity_mode = data_config["tiff_intensity_mode"]
    raw_all = load_tiff_stack(
        _resolve(config_path, data_config["raw_tiff_path"]), intensity_mode=intensity_mode
    )
    frame_indices = select_frame_indices(
        raw_all.shape[0], frame_count, data_config.get("frame_indices"), seed=seed
    )
    raw = raw_all[frame_indices]
    measured_mean_np = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    measured_variance_np = raw.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    residual_np = raw - measured_mean_np[None]
    del raw_all, raw

    z_values = np.asarray(config["psf"]["z_values_um"], dtype=np.float32)
    f_var_np = load_volume_tiff(
        _resolve(config_path, data_config["f_var_tiff_path"]), len(z_values), intensity_mode=intensity_mode
    )
    g_mean_np = load_volume_tiff(
        _resolve(config_path, data_config["g_mean_tiff_path"]), len(z_values), intensity_mode=intensity_mode
    )
    if f_var_np.shape[1:] != measured_mean_np.shape or g_mean_np.shape != f_var_np.shape:
        raise ValueError("Raw sensor, F_var, and g_mean lateral dimensions must match exactly")

    psf_config = config["psf"]
    psf_data = load_psf(
        _resolve(config_path, psf_config["H_path"]),
        z_values,
        h_variable_name=psf_config["H_variable_name"],
        ht_variable_name=psf_config["Ht_variable_name"],
        psf_z_all_um=psf_config.get("psf_z_all_um"),
        depth_unit=psf_config.get("depth_unit", "auto"),
        load_h=True,
        load_ht=False,
    )
    LOGGER.info(
        "Selected PSF indices=%s z_um=%s canonical_shape=%s",
        psf_data.metadata.selected_indices.tolist(),
        psf_data.metadata.selected_z_um.tolist(),
        psf_data.H.shape,
    )
    selected_psf_indices = psf_data.metadata.selected_indices.tolist()
    h_tensor = torch.from_numpy(psf_data.H).to(device=device, dtype=torch.float32)
    del psf_data
    gc.collect()
    operator = LFMOperator(
        h_tensor,
        mode=config["runtime"]["operator_mode"],
        phase_chunk_size=config["runtime"]["operator_phase_chunk_size"],
    )
    del h_tensor

    f_var = torch.from_numpy(f_var_np)[None, None].to(device)
    g_mean = torch.from_numpy(g_mean_np)[None, None].to(device)
    residual_frames = torch.from_numpy(residual_np)[None, :, None].to(device)
    measured_mean = torch.from_numpy(measured_mean_np)[None, None].to(device)
    measured_variance = torch.from_numpy(measured_variance_np)[None, None].to(device)
    z_tensor = torch.from_numpy(z_values).to(device)

    model = _model_from_config(config).to(device)
    warm = args.init == "checkpoint"
    if warm:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --init checkpoint")
        load_model_checkpoint(model, args.checkpoint)
    model.initialize_beta(f_var, measured_mean, operator)
    optimizer = _optimizer_from_config(model, config, warm)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])

    writer = None
    if config["runtime"].get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output_dir / "tensorboard")

    rows: list[dict[str, float]] = []
    best_total = float("inf")
    best_step = -1
    best_state = None
    best_reconstruction = None
    best_gates = None
    best_metrics: dict[str, float] = {}
    max_steps = int(config["optimization"]["max_steps"])
    save_every = int(config["optimization"]["save_every"])
    if max_steps < 1 or save_every < 1:
        raise ValueError("max_steps and save_every must both be positive")
    amp_enabled = bool(config["runtime"]["amp"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(max_steps):
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(f_var, g_mean, residual_frames, z_tensor)
        reconstruction_physics = output.reconstruction.to(dtype=operator.H.dtype)
        losses = compute_self_supervised_loss(
            reconstruction_physics,
            measured_mean,
            measured_variance,
            operator,
            variance_model,
            **config["loss"],
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"Non-finite total loss at step {step}")
        metrics = losses.scalar_metrics()
        metrics.update(
            {
                "step": float(step),
                "beta": float(output.beta.detach().item()),
                "alpha0": float(output.alphas[0].detach().item()),
                "alpha1": float(output.alphas[1].detach().item()),
                "alpha2": float(output.alphas[2].detach().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if metrics["total_loss"] < best_total:
            best_total = metrics["total_loss"]
            best_step = step
            best_state = cpu_state_dict(model)
            best_reconstruction = output.reconstruction.detach().cpu().numpy()[0, 0]
            best_gates = [gate.detach().cpu().numpy() for gate in output.gates]
            best_metrics = dict(metrics)
        losses.total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite model gradient at step {step}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        else:
            metrics["gpu_memory_mb"] = 0.0
        metrics["step_time_s"] = time.perf_counter() - start_time
        rows.append(metrics)
        if writer is not None:
            for key, value in metrics.items():
                if key != "step":
                    writer.add_scalar(key, value, step)
        if step % save_every == 0 or step == max_steps - 1:
            snapshot = output.reconstruction.detach().cpu().numpy()[0, 0]
            save_volume_tiff(output_dir / f"reconstruction_step_{step:04d}.tif", snapshot)
            _save_csv(output_dir / "losses.csv", rows)
        LOGGER.info(
            "step=%d total=%.6g mean=%.6g var=%.6g beta=%.4g time=%.3fs",
            step,
            metrics["total_loss"],
            metrics["normalized_mean_loss"],
            metrics["normalized_var_loss"],
            metrics["beta"],
            metrics["step_time_s"],
        )

    assert best_state is not None and best_reconstruction is not None and best_gates is not None
    save_volume_tiff(output_dir / "reconstruction_best.tif", best_reconstruction)
    np.save(output_dir / "reconstruction_best.npy", best_reconstruction)
    save_volume_visualizations(output_dir, best_reconstruction, z_values)
    save_gate_outputs(output_dir, best_gates)
    save_loss_curve(output_dir / "loss_curve.png", rows)
    _save_csv(output_dir / "losses.csv", rows)

    save_checkpoint(
        output_dir / "checkpoint_best.pt",
        model_state=best_state,
        step=best_step,
        metrics=best_metrics,
        config=config,
    )
    summary = {
        "selected_frame_indices_zero_based": frame_indices.tolist(),
        "selected_psf_indices_zero_based": selected_psf_indices,
        "z_values_um": z_values.tolist(),
        "best_step": best_step,
        "best_total_loss": best_total,
        "best_mean_loss": best_metrics["normalized_mean_loss"],
        "best_var_loss": best_metrics["normalized_var_loss"],
        "best_metrics": best_metrics,
        "artifact_source_step": best_step,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
