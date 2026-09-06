from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml

from physics.covariance_sketch import (
    load_preprocess_cs,
    theoretical_covariance_action,
)
from physics.lfm_operator import LFMOperator
from physics.psf_loader import load_psf
from utils.io import load_tiff_stack, load_volume_tiff


LOGGER = logging.getLogger("scan_covariance_sketch_depth")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _unit_norm(probes: np.ndarray) -> np.ndarray:
    flattened = probes.reshape(probes.shape[0], -1)
    return (
        flattened
        / np.linalg.norm(flattened, axis=1, keepdims=True).clip(
            min=np.finfo(np.float64).eps
        )
    ).reshape(probes.shape).astype(np.float32)


def _lowpass_probes(
    count: int,
    shape: tuple[int, int],
    *,
    sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.choice((-1.0, 1.0), size=(count, *shape)).astype(np.float32)
    probes = np.stack(
        [gaussian_filter(probe, sigma=sigma, mode="reflect") for probe in raw]
    )
    probes -= probes.mean(axis=(1, 2), keepdims=True)
    return _unit_norm(probes)


def _empirical_action(frames: np.ndarray, probes: np.ndarray) -> np.ndarray:
    flattened = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=False)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    flat_probes = probes.reshape(probes.shape[0], -1)
    weights = centered @ flat_probes.T
    return ((weights.T @ centered) / float(frames.shape[0] - 1)).reshape(probes.shape)


def _numbered_frame_paths(root: Path, depth_um: int) -> list[Path]:
    paths = list(root.glob(f"img_detph{depth_um}_*.tif"))

    def number(path: Path) -> int:
        match = re.search(r"_(\d+)\.tif$", path.name)
        if match is None:
            raise ValueError(f"Cannot extract frame number from {path}")
        return int(match.group(1))

    return sorted(paths, key=number)


def _load_raw_depth(config_path: Path, config: dict, depth_um: int) -> np.ndarray:
    prepared = config_path.parent.parent / f"data/分辨率图仿真/prepared/depth{depth_um}_raw_100.tif"
    if prepared.is_file():
        return load_tiff_stack(prepared, intensity_mode=config["data"]["tiff_intensity_mode"])
    source_root = config_path.parent.parent / "data/分辨率图仿真"
    paths = _numbered_frame_paths(source_root, depth_um)[:100]
    if len(paths) != 100:
        raise FileNotFoundError(f"Expected 100 raw frames for {depth_um} um, found {len(paths)}")
    frames = [
        load_tiff_stack(path, intensity_mode=config["data"]["tiff_intensity_mode"])[0]
        for path in paths
    ]
    return np.stack(frames)


def _load_f_var(config_path: Path, config: dict, depth_um: int) -> np.ndarray:
    z_count = len(config["psf"]["z_values_um"])
    configured = _resolve(config_path, config["data"]["f_var_tiff_path"])
    if depth_um == 50:
        path = configured
    else:
        path = configured.with_name(configured.name.replace("detph50", f"detph{depth_um}"))
    return load_volume_tiff(
        path,
        z_count,
        intensity_mode=config["data"]["tiff_intensity_mode"],
    )


def _common_template(f_var: np.ndarray) -> np.ndarray:
    template = np.maximum(f_var, 0.0).sum(axis=0, dtype=np.float64)
    total = float(template.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("F_var projection has no positive mass")
    return (template / total).astype(np.float32)


@torch.inference_mode()
def _theory_for_depth(
    h_depth: torch.Tensor,
    template: np.ndarray,
    measured_mean: np.ndarray,
    probes: np.ndarray,
    cs_kernel: torch.Tensor,
    *,
    phase_chunk_size: int,
    probe_chunk_size: int,
) -> tuple[np.ndarray, float, float]:
    operator = LFMOperator(
        h_depth,
        mode="optimized",
        phase_chunk_size=phase_chunk_size,
    )
    device = h_depth.device
    shape = torch.from_numpy(template)[None, None, None].to(device=device, dtype=h_depth.dtype)
    target_mean = torch.from_numpy(measured_mean)[None, None].to(
        device=device, dtype=h_depth.dtype
    )
    unit_mean = operator(shape)
    gain = (
        (unit_mean * target_mean).sum()
        / unit_mean.square().sum().clamp_min(torch.finfo(unit_mean.dtype).eps)
    ).clamp_min(torch.finfo(unit_mean.dtype).eps)
    mean_relative_error = (
        torch.linalg.vector_norm(gain * unit_mean - target_mean)
        / torch.linalg.vector_norm(target_mean).clamp_min(torch.finfo(unit_mean.dtype).eps)
    )
    reconstruction = gain * shape
    typed_cs = cs_kernel.to(device=device, dtype=h_depth.dtype)
    actions: list[np.ndarray] = []
    for start in range(0, probes.shape[0], probe_chunk_size):
        probe_chunk = torch.from_numpy(probes[start : start + probe_chunk_size]).to(
            device=device, dtype=h_depth.dtype
        )
        action = theoretical_covariance_action(
            reconstruction,
            probe_chunk,
            operator,
            typed_cs,
        )
        actions.append(action.cpu().numpy())
    return (
        np.concatenate(actions, axis=0),
        float(gain.item()),
        float(mean_relative_error.item()),
    )


def _fit_nonnegative_scale(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.vdot(prediction.reshape(-1), target.reshape(-1)).real)
    denominator = float(np.vdot(prediction.reshape(-1), prediction.reshape(-1)).real)
    return max(numerator / max(denominator, np.finfo(np.float64).eps), 0.0)


def _sketch_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    correlation_weight: float,
) -> dict[str, float]:
    pred = prediction.reshape(prediction.shape[0], -1).astype(np.float64, copy=False)
    truth = target.reshape(target.shape[0], -1).astype(np.float64, copy=False)
    denominator = np.einsum("ij,ij->i", truth, truth).clip(
        min=np.finfo(np.float64).eps
    )
    relative_mse = np.einsum("ij,ij->i", pred - truth, pred - truth) / denominator
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    correlation = np.einsum("ij,ij->i", pred_centered, truth_centered) / np.sqrt(
        np.einsum("ij,ij->i", pred_centered, pred_centered)
        * np.einsum("ij,ij->i", truth_centered, truth_centered)
    ).clip(min=np.finfo(np.float64).eps)
    rel = float(relative_mse.mean())
    corr = float(correlation.mean())
    return {
        "relative_mse": rel,
        "correlation": corr,
        "correlation_loss": 1.0 - corr,
        "combined_loss": rel + float(correlation_weight) * (1.0 - corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-B no-network covariance-sketch depth scan"
    )
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument(
        "--cs",
        default=(
            "/workspace/xyx/.codex/attachments/"
            "2b2936ce-408b-4e55-8025-f006f69f19e5/psf_NA0.04479.tif"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-sigma", type=float, default=16.0)
    parser.add_argument("--probe-seed", type=int, default=20260901)
    parser.add_argument("--probe-chunk-size", type=int, default=1)
    parser.add_argument("--phase-chunk-size", type=int, default=16)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="outputs/covariance_sketch_validation/stage_b_depth_scan"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-size PSF depth scan")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_depths = (20, 50, 80)
    datasets: dict[int, dict[str, np.ndarray]] = {}
    for depth_um in calibration_depths:
        LOGGER.info("Loading %d um raw frames and F_var", depth_um)
        frames = _load_raw_depth(config_path, config, depth_um)
        f_var = _load_f_var(config_path, config, depth_um)
        datasets[depth_um] = {
            "frames": frames,
            "mean": frames.mean(axis=0, dtype=np.float64).astype(np.float32),
            "template": _common_template(f_var),
        }
    sensor_shape = tuple(int(value) for value in datasets[50]["mean"].shape)
    probes = _lowpass_probes(
        args.num_probes,
        sensor_shape,
        sigma=args.probe_sigma,
        seed=args.probe_seed,
    )
    empirical_full = {
        depth: _empirical_action(data["frames"], probes)
        for depth, data in datasets.items()
    }
    permutation = np.random.default_rng(args.probe_seed + 2000).permutation(100)
    split_indices = (permutation[:50], permutation[50:])
    empirical_50_splits = [
        _empirical_action(datasets[50]["frames"][indices], probes)
        for indices in split_indices
    ]

    cs_kernel, cs_report = load_preprocess_cs(args.cs)
    psf_config = config["psf"]
    LOGGER.info("Loading selected PSF depths; this is the slow initialization step")
    psf_data = load_psf(
        _resolve(config_path, psf_config["H_path"]),
        psf_config["z_values_um"],
        h_variable_name=psf_config["H_variable_name"],
        ht_variable_name=psf_config["Ht_variable_name"],
        psf_z_all_um=psf_config.get("psf_z_all_um"),
        depth_unit=psf_config.get("depth_unit", "auto"),
        load_h=True,
        load_ht=False,
    )
    h_all = torch.from_numpy(psf_data.H).to(device=device, dtype=torch.float32)
    z_values = [float(value) for value in psf_data.metadata.selected_z_um]
    del psf_data
    gc.collect()
    LOGGER.info("PSF ready on %s; starting system-scale calibration", device)

    theory_cache: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}

    def theory(dataset_depth: int, candidate_index: int) -> tuple[np.ndarray, float, float]:
        key = (dataset_depth, candidate_index)
        if key not in theory_cache:
            theory_cache[key] = _theory_for_depth(
                h_all[candidate_index : candidate_index + 1],
                datasets[dataset_depth]["template"],
                datasets[dataset_depth]["mean"],
                probes,
                cs_kernel,
                phase_chunk_size=args.phase_chunk_size,
                probe_chunk_size=args.probe_chunk_size,
            )
        return theory_cache[key]

    depth_to_index = {
        int(round(depth)): index for index, depth in enumerate(z_values)
    }
    calibration_rows: list[dict[str, float]] = []
    for depth_um in calibration_depths:
        prediction, gain, mean_error = theory(depth_um, depth_to_index[depth_um])
        scale = _fit_nonnegative_scale(prediction, empirical_full[depth_um])
        calibration_rows.append(
            {
                "depth_um": float(depth_um),
                "covariance_scale": scale,
                "photometric_gain": gain,
                "mean_relative_error": mean_error,
            }
        )
        LOGGER.info(
            "Calibration %d um: kappa=%.7g gain=%.7g mean_relerr=%.5f",
            depth_um,
            scale,
            gain,
            mean_error,
        )
    scales = np.asarray([row["covariance_scale"] for row in calibration_rows])
    if np.any(scales <= 0.0):
        raise RuntimeError("At least one calibration depth produced a nonpositive scale")
    system_scale = float(np.exp(np.median(np.log(scales))))
    scale_cv = float(scales.std(ddof=1) / scales.mean())
    LOGGER.info("Fixed system kappa=%.7g; three-depth CV=%.3f", system_scale, scale_cv)

    scan_rows: list[dict[str, float]] = []
    for candidate_index, z_um in enumerate(z_values):
        unscaled, gain, mean_error = theory(50, candidate_index)
        scaled = system_scale * unscaled
        full_loss = _sketch_loss(
            scaled,
            empirical_full[50],
            correlation_weight=args.correlation_weight,
        )
        split_a_loss = _sketch_loss(
            scaled,
            empirical_50_splits[0],
            correlation_weight=args.correlation_weight,
        )
        split_b_loss = _sketch_loss(
            scaled,
            empirical_50_splits[1],
            correlation_weight=args.correlation_weight,
        )
        row = {
            "candidate_depth_um": z_um,
            "photometric_gain": gain,
            "mean_relative_error": mean_error,
            **{f"full_{key}": value for key, value in full_loss.items()},
            **{f"split_a_{key}": value for key, value in split_a_loss.items()},
            **{f"split_b_{key}": value for key, value in split_b_loss.items()},
        }
        scan_rows.append(row)
        LOGGER.info(
            "Depth %.0f um: full=%.6f splitA=%.6f splitB=%.6f corr=%.4f",
            z_um,
            row["full_combined_loss"],
            row["split_a_combined_loss"],
            row["split_b_combined_loss"],
            row["full_correlation"],
        )

    minima = {
        subset: min(scan_rows, key=lambda row: row[f"{subset}_combined_loss"])[
            "candidate_depth_um"
        ]
        for subset in ("full", "split_a", "split_b")
    }
    true_row = next(row for row in scan_rows if round(row["candidate_depth_um"]) == 50)
    wrong_rows = [row for row in scan_rows if round(row["candidate_depth_um"]) != 50]
    margins = {
        subset: (
            min(row[f"{subset}_combined_loss"] for row in wrong_rows)
            / max(true_row[f"{subset}_combined_loss"], np.finfo(np.float64).eps)
            - 1.0
        )
        for subset in ("full", "split_a", "split_b")
    }
    passed = (
        all(round(value) == 50 for value in minima.values())
        and all(value >= 0.10 for value in margins.values())
        and scale_cv <= 0.10
    )
    summary = {
        "model": "independent_lateral_cs_covariance_vector",
        "probe_family": f"lowpass_rademacher_sigma{args.probe_sigma:g}",
        "probe_count": args.num_probes,
        "probe_seed": args.probe_seed,
        "cs_preprocess": cs_report.to_dict(),
        "system_covariance_scale": system_scale,
        "calibration_scale_cv": scale_cv,
        "calibration": calibration_rows,
        "minima_um": minima,
        "wrong_depth_relative_margin": margins,
        "stage_b_passed": passed,
        "pass_requirements": {
            "all_full_and_split_minima_are_50_um": True,
            "minimum_wrong_depth_margin": 0.10,
            "maximum_three_depth_scale_cv": 0.10,
        },
    }
    with (output_dir / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        print("STAGE_B_BLOCKED: covariance-sketch training is not authorized", flush=True)
        raise SystemExit(2)
    print("STAGE_B_PASSED: a controlled 200-step training ablation is authorized", flush=True)


if __name__ == "__main__":
    main()
