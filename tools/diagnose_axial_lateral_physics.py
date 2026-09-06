from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import yaml

from losses.self_supervised_losses import TaylorH2VarianceModel
from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from utils.io import load_tiff_stack, load_volume_tiff, save_volume_tiff, select_frame_indices


LOGGER = logging.getLogger("diagnose_axial_lateral_physics")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent.parent / path).resolve()
    return path


def _variance_loss(
    reconstruction: torch.Tensor,
    measured_mean: torch.Tensor,
    measured_variance: torch.Tensor,
    variance_model: TaylorH2VarianceModel,
    log_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled() and reconstruction.requires_grad:
        predicted = checkpoint(
            variance_model,
            reconstruction,
            measured_mean,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        predicted = variance_model(reconstruction, measured_mean)
    loss = F.smooth_l1_loss(
        torch.log(predicted.clamp_min(0.0) + float(log_eps)),
        torch.log(measured_variance.clamp_min(0.0) + float(log_eps)),
    )
    return loss, predicted


def _reconstruction(
    normalized_layer_shapes: torch.Tensor,
    f_var_sum: torch.Tensor,
    beta0: torch.Tensor,
    beta_range: float,
    raw_beta: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = beta0 * (1.0 + float(beta_range) * torch.tanh(raw_beta))
    axial_mass = torch.softmax(logits - logits.mean(), dim=0)
    reconstruction = (
        beta
        * f_var_sum
        * axial_mass.view(1, 1, -1, 1, 1)
        * normalized_layer_shapes
    )
    return reconstruction, axial_mass, beta


def _depth_metrics(axial_mass: torch.Tensor, z_values_um: np.ndarray) -> dict[str, float]:
    axial = axial_mass.detach().cpu().numpy().astype(np.float64)
    peak_index = int(np.argmax(axial))
    return {
        "depth_peak_um": float(z_values_um[peak_index]),
        "depth_centroid_um": float(np.sum(axial * z_values_um)),
        "mass_40_60_fraction": float(axial[(z_values_um >= 40) & (z_values_um <= 60)].sum()),
        "mass_first_fraction": float(axial[0]),
        "mass_last_fraction": float(axial[-1]),
    }


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-physics axial identifiability and direct-logit diagnostic"
    )
    parser.add_argument("--config", default="configs/depth50_c1_context_reflect.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-dir", default="outputs/axial_lateral_50um/physical_diagnostic"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr-logits", type=float, default=0.05)
    args = parser.parse_args()

    if args.steps != 200:
        raise ValueError("This approved diagnostic must run exactly 200 steps")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    data = config["data"]
    raw_all = load_tiff_stack(
        _resolve(config_path, data["raw_tiff_path"]),
        intensity_mode=data["tiff_intensity_mode"],
    )
    frame_indices = select_frame_indices(
        raw_all.shape[0], int(data["num_speckle_frames"]), data.get("frame_indices"), seed=seed
    )
    raw = raw_all[frame_indices]
    measured_mean_np = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    measured_variance_np = raw.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    del raw_all, raw

    z_values_um = np.asarray(config["psf"]["z_values_um"], dtype=np.float32)
    f_var_np = load_volume_tiff(
        _resolve(config_path, data["f_var_tiff_path"]),
        len(z_values_um),
        intensity_mode=data["tiff_intensity_mode"],
    )
    psf = config["psf"]
    psf_data = load_psf(
        _resolve(config_path, psf["H_path"]),
        z_values_um,
        h_variable_name=psf["H_variable_name"],
        ht_variable_name=psf["Ht_variable_name"],
        psf_z_all_um=psf.get("psf_z_all_um"),
        depth_unit=psf.get("depth_unit", "auto"),
        load_h=True,
        load_ht=False,
    )
    operator = LFMOperator(
        torch.from_numpy(psf_data.H).to(device=device, dtype=torch.float32),
        mode=config["runtime"]["operator_mode"],
        phase_chunk_size=int(config["runtime"]["operator_phase_chunk_size"]),
    )
    del psf_data
    gc.collect()

    f_var = torch.from_numpy(f_var_np)[None, None].to(device)
    measured_mean = torch.from_numpy(measured_mean_np)[None, None].to(device)
    measured_variance = torch.from_numpy(measured_variance_np)[None, None].to(device)
    eps = float(config["model"].get("positivity_eps", 1e-8))
    safe_f_var = f_var.clamp_min(eps)
    normalized_layer_shapes = safe_f_var / safe_f_var.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(eps)
    f_var_sum = f_var.clamp_min(0.0).sum()
    with torch.no_grad():
        unit_prediction = operator(f_var)
        beta0 = ((unit_prediction * measured_mean).sum() / unit_prediction.square().sum().clamp_min(eps)).clamp_min(eps)
    variance_model = TaylorH2VarianceModel(operator, **config["noise"])
    log_eps = float(config["loss"].get("var_log_eps", 1e-6))
    beta_range = float(config["model"]["beta_range"])

    scan_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for index, z_um in enumerate(z_values_um):
            axial = torch.zeros(len(z_values_um), device=device)
            axial[index] = 1.0
            candidate = beta0 * f_var_sum * axial.view(1, 1, -1, 1, 1) * normalized_layer_shapes
            loss, _ = _variance_loss(
                candidate, measured_mean, measured_variance, variance_model, log_eps
            )
            scan_rows.append({"z_um": float(z_um), "normalized_var_loss": float(loss.item())})
    scan_rows.sort(key=lambda row: row["z_um"])
    scan_by_loss = sorted(scan_rows, key=lambda row: row["normalized_var_loss"])
    best_scan = scan_by_loss[0]
    second_scan = scan_by_loss[1]
    relative_margin = (
        second_scan["normalized_var_loss"] / max(best_scan["normalized_var_loss"], 1e-12) - 1.0
    )
    _write_csv(output_dir / "single_depth_scan.csv", scan_rows)

    initial_logits = torch.zeros(len(z_values_um), device=device, requires_grad=True)
    initial_raw_beta = torch.zeros((), device=device)
    initial_reconstruction, initial_axial, _ = _reconstruction(
        normalized_layer_shapes,
        f_var_sum,
        beta0,
        beta_range,
        initial_raw_beta,
        initial_logits,
    )
    initial_loss, _ = _variance_loss(
        initial_reconstruction, measured_mean, measured_variance, variance_model, log_eps
    )
    initial_gradient = torch.autograd.grad(initial_loss, initial_logits)[0]
    descent_index = int(initial_gradient.argmin().item())

    logits = torch.nn.Parameter(torch.zeros(len(z_values_um), device=device))
    raw_beta = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [logits], "lr": float(args.lr_logits)},
            {"params": [raw_beta], "lr": float(config["optimization"]["lr_beta"])},
        ]
    )
    rows: list[dict[str, float]] = []
    best_loss = float("inf")
    best_step = -1
    best_reconstruction: np.ndarray | None = None
    best_axial: np.ndarray | None = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, axial, beta = _reconstruction(
            normalized_layer_shapes,
            f_var_sum,
            beta0,
            beta_range,
            raw_beta,
            logits,
        )
        loss, _ = _variance_loss(
            reconstruction, measured_mean, measured_variance, variance_model, log_eps
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite direct-logit loss at step {step}")
        row = {
            "step": float(step),
            "normalized_var_loss": float(loss.detach().item()),
            "beta": float(beta.detach().item()),
            **_depth_metrics(axial, z_values_um),
        }
        for index, z_um in enumerate(z_values_um):
            row[f"mass_z{float(z_um):g}_um"] = float(axial[index].detach().item())
        rows.append(row)
        if row["normalized_var_loss"] < best_loss:
            best_loss = row["normalized_var_loss"]
            best_step = step
            best_reconstruction = reconstruction.detach().cpu().numpy()[0, 0]
            best_axial = axial.detach().cpu().numpy()
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError(f"Invalid direct-logit gradient at step {step}")
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            LOGGER.info(
                "step=%d loss=%.6g peak=%.1f centroid=%.3f mass40_60=%.4f beta=%.6g",
                step,
                row["normalized_var_loss"],
                row["depth_peak_um"],
                row["depth_centroid_um"],
                row["mass_40_60_fraction"],
                row["beta"],
            )
            save_volume_tiff(
                output_dir / f"direct_reconstruction_step_{step:04d}.tif",
                reconstruction.detach().cpu().numpy()[0, 0],
            )

    assert best_reconstruction is not None and best_axial is not None
    _write_csv(output_dir / "direct_logits.csv", rows)
    save_volume_tiff(output_dir / "direct_reconstruction_best.tif", best_reconstruction)
    np.save(output_dir / "direct_reconstruction_best.npy", best_reconstruction)
    final_metrics = rows[-1]
    direct_pass = bool(
        final_metrics["depth_peak_um"] == 50.0
        and final_metrics["mass_40_60_fraction"] >= 0.8
        and abs(final_metrics["depth_centroid_um"] - 50.0) <= 5.0
        and final_metrics["mass_first_fraction"] < 0.05
        and final_metrics["mass_last_fraction"] < 0.05
    )
    summary = {
        "steps": args.steps,
        "beta0": float(beta0.item()),
        "single_depth_scan_best_um": best_scan["z_um"],
        "single_depth_scan_best_loss": best_scan["normalized_var_loss"],
        "single_depth_scan_second_um": second_scan["z_um"],
        "single_depth_scan_relative_margin": relative_margin,
        "single_depth_scan_unique_50um_with_10pct_margin": bool(
            best_scan["z_um"] == 50.0 and relative_margin >= 0.1
        ),
        "initial_uniform_loss": float(initial_loss.detach().item()),
        "initial_logit_gradient": initial_gradient.detach().cpu().tolist(),
        "initial_logit_gradient_norm": float(initial_gradient.norm().item()),
        "initial_max_descent_depth_um": float(z_values_um[descent_index]),
        "initial_gradient_finite_nonzero": bool(
            torch.isfinite(initial_gradient).all().item() and initial_gradient.norm().item() > 0.0
        ),
        "best_step": best_step,
        "best_loss": best_loss,
        "best_axial_mass": best_axial.tolist(),
        "final_metrics": final_metrics,
        "direct_logits_depth_gate_pass": direct_pass,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Diagnostic summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
