"""Dated full-field spinach-root inference for V5 Taylor step 600 and controls."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/speckle_spinach_v5_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools import three_way_experiment as common
from tools import v5_mixed_real_anchor_experiment as v5
from tools.spinach_root_network_v2 import execute, mat_volume
from utils.io import save_volume_tiff


OUTPUT = ROOT / "outputs/spinach_root_v5_taylor600_compare_20260911_run01"
REFERENCE = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
SPECIFIED_OLD = (
    ROOT
    / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/spinach_root_transfer"
)
CHECKPOINT = (
    ROOT
    / "outputs/v5_sim_real_no_p12_anchor_compare_e3_mean100_600_20260910_run02"
    / "taylor_anchor_e3_mean100/checkpoint_step_000600.pt"
)
MATLAB = Path("/workspace/xyx/MATLAB/R2023b/bin/matlab")
PHYSICAL_GPU = 3

METHODS = (
    "mean_rl3",
    "taylor_rl3_sqrt",
    "old_taylor_anchor_400",
    "specified_old_mean_anchor_400",
    "new_taylor_anchor_mixed_real_600",
)
LABELS = {
    "mean_rl3": "Mean-RL3",
    "taylor_rl3_sqrt": "Taylor-RL3-sqrt",
    "old_taylor_anchor_400": "Old Taylor anchor · 400",
    "specified_old_mean_anchor_400": "Specified old Mean anchor · 400",
    "new_taylor_anchor_mixed_real_600": "New Taylor anchor + real · 600",
}
NETWORK_METHODS = METHODS[2:]
ROIS = {
    "center": (7 * 49, 14 * 49, 11 * 49, 18 * 49),
    "upper_left": (2 * 49, 9 * 49, 2 * 49, 9 * 49),
    "lower_right": (12 * 49, 19 * 49, 20 * 49, 27 * 49),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def h5_array(path: Path, name: str, *, volume: bool = False) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        value = np.asarray(handle[name], dtype=np.float32)
    return value.transpose(0, 2, 1).copy() if volume else value.T.copy()


def normalized(volume: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(volume, np.float64), 0)
    return value / max(value.sum(), 1e-30)


def scale_nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    p = prediction.astype(np.float64).ravel()
    t = target.astype(np.float64).ravel()
    gain = float(np.dot(p, t) / max(np.dot(p, p), 1e-30))
    return float(np.linalg.norm(gain * p - t) / max(np.linalg.norm(t), 1e-30))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64).ravel()
    y = right.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def gpu_inventory() -> dict[int, dict[str, Any]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, dict[str, Any]] = {}
    for row in rows.splitlines():
        index, uuid, free, total, utilization = [value.strip() for value in row.split(",")]
        result[int(index)] = {
            "uuid": uuid,
            "free_mib": int(free),
            "total_mib": int(total),
            "utilization": int(utilization),
        }
    return result


def prepare() -> Path:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = payload["config"]
    v5.validate_config(config)
    if int(payload["completed_steps"]) != 600:
        raise ValueError("The requested V5 checkpoint is not step 600")
    if config["model"].get("reconstruction_anchor") != "taylor_sqrt":
        raise ValueError("The requested V5 checkpoint is not Taylor-anchor")
    source = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    if source["input_indices"] != [9, 10, 27, 44, 54, 74, 77, 83, 85, 92]:
        raise ValueError("Frozen spinach-root frame selection changed")
    inventory = gpu_inventory()
    card = inventory[PHYSICAL_GPU]
    if card["free_mib"] < 20 * 1024:
        raise RuntimeError(f"GPU {PHYSICAL_GPU} has insufficient free memory: {card}")

    OUTPUT.mkdir(parents=True)
    manifest = OUTPUT / "manifest.json"
    value = {
        **source,
        "experiment": "spinach_root_v5_taylor600_compare",
        "output": str(OUTPUT),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": common.sha256(CHECKPOINT),
        "checkpoint_step": 600,
        "checkpoint_role": "final_and_best",
        "reconstruction_anchor": "taylor_sqrt",
        "new_training_objects": [f"P{index:02d}" for index in range(1, 12)],
        "new_real_training_fields": ["45", "55"],
        "p12_in_training": False,
        "specified_old_checkpoint": str(
            SPECIFIED_OLD.parent / "mean_anchor_e3_mean100/checkpoint_last.pt"
        ),
        "specified_old_actual_anchor": "mean_rl3",
        "specified_old_training_objects": [f"P{index:02d}" for index in range(1, 12)],
        "specified_old_real_training_fields": [],
        "old_taylor_source": str(REFERENCE / "e3_mean100.mat"),
        "gpu_uuid": {"network": card["uuid"], "gain": card["uuid"]},
        "physical_gpu": PHYSICAL_GPU,
        "gpu_inventory_at_start": card,
        "network_base_mat": str(OUTPUT / "new_taylor_step600_base.mat"),
        "network_final_mat": str(OUTPUT / "new_taylor_step600.mat"),
        "network_record": str(OUTPUT / "network_record.json"),
        "comparison_methods": list(METHODS),
        "interpretation_boundary": (
            "old-to-new Taylor changes both real-field training inclusion and 400-to-600 "
            "training budget; the specified old path is Mean-anchor, not Taylor-anchor"
        ),
    }
    write_json(manifest, value)
    snapshot = OUTPUT / "source_snapshot"
    for path in (
        Path(__file__).resolve(),
        ROOT / "tools/spinach_root_network_v2.py",
        ROOT / "tools/v5_mixed_real_anchor_experiment.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "models/configurable_anchor_lfm_net.py",
        ROOT / "matlab_code/real_data/spinach_root_gain_worker_v2.m",
    ):
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(
        OUTPUT / "preflight.json",
        {
            "complete": True,
            "manifest": str(manifest),
            "checkpoint_sha256": common.sha256(CHECKPOINT),
            "reference_manifest_sha256": common.sha256(REFERENCE / "manifest.json"),
            "specified_old_result_sha256": common.sha256(
                SPECIFIED_OLD / "mean_anchor_e3_mean100.mat"
            ),
            "gpu": card,
        },
    )
    return manifest


def network(manifest: Path) -> None:
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]["network"]:
        raise ValueError("Network worker GPU differs from the frozen manifest")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    v5.configure_precision()
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    config = payload["config"]
    v5.validate_config(config)
    if int(payload["completed_steps"]) != 600:
        raise ValueError("Checkpoint step changed")
    model = v5.build_model(config, initial=False).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    mean = mat_volume(REFERENCE / "mean_rl3.mat", "reconstruction_raw")
    taylor = mat_volume(REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt")
    with h5py.File(REFERENCE / "taylor_rl3.mat", "r") as handle:
        projection = np.asarray(handle["projection_of_sqrt"], dtype=np.float32).T.copy()
    source = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    frames = np.stack(
        [tifffile.imread(path).astype(np.float32) / 255 for path in source["selected_files"]]
    )
    image_mean = frames.mean(0, dtype=np.float64).astype(np.float32)
    residual = frames - image_mean
    beta0 = float(
        np.sum(projection.astype(np.float64) * image_mean)
        / max(np.sum(projection.astype(np.float64) ** 2) + 1e-8, 1e-30)
    )
    if not np.isfinite(beta0) or beta0 <= 0:
        raise ValueError("Invalid full-field analytic Taylor beta0")
    raw, mode = execute(
        model,
        taylor,
        mean,
        residual,
        np.asarray(source["z_um"], np.float32),
        beta0,
        device,
    )
    if raw.shape != (10, 1029, 1421) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Invalid V5 full-field output")
    savemat(
        cfg["network_base_mat"],
        {"base_reconstruction": raw.transpose(1, 2, 0)},
        do_compression=False,
    )
    factor = 1 + float(config["three_way"]["gain_bound"]) * math.tanh(
        float(model.mean_gain_gamma.detach().cpu())
    )
    cfg["e3_gain_factor"] = factor
    write_json(manifest, cfg)
    write_json(
        Path(cfg["network_record"]),
        {
            "complete": True,
            "checkpoint_step": 600,
            "anchor": "taylor_sqrt",
            "beta0": beta0,
            "raw_beta": float(model.raw_beta.detach().cpu()),
            "mean_gain_gamma": float(model.mean_gain_gamma.detach().cpu()),
            "e3_gain_factor": factor,
            **mode,
        },
    )


def load_products(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    new_path = Path(cfg["network_final_mat"])
    volumes = {
        "mean_rl3": h5_array(REFERENCE / "mean_rl3.mat", "reconstruction_raw", volume=True),
        "taylor_rl3_sqrt": h5_array(
            REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt", volume=True
        ),
        "old_taylor_anchor_400": h5_array(
            REFERENCE / "e3_mean100.mat", "reconstruction", volume=True
        ),
        "specified_old_mean_anchor_400": h5_array(
            SPECIFIED_OLD / "mean_anchor_e3_mean100.mat", "reconstruction", volume=True
        ),
        "new_taylor_anchor_mixed_real_600": h5_array(
            new_path, "reconstruction", volume=True
        ),
    }
    sources = {
        "mean_rl3": REFERENCE / "mean_rl3.mat",
        "taylor_rl3_sqrt": REFERENCE / "taylor_rl3.mat",
        "old_taylor_anchor_400": REFERENCE / "e3_mean100.mat",
        "specified_old_mean_anchor_400": SPECIFIED_OLD / "mean_anchor_e3_mean100.mat",
        "new_taylor_anchor_mixed_real_600": new_path,
    }
    predictions = {
        name: (h5_array(path, "predicted_mean"), h5_array(path, "predicted_variance"))
        for name, path in sources.items()
    }
    for name, volume in volumes.items():
        if volume.shape != (10, 1029, 1421) or not np.isfinite(volume).all():
            raise ValueError(f"Invalid comparison volume: {name}")
    return volumes, predictions


def show(axis, image: np.ndarray, scale: float, title: str, *, aspect: str = "equal") -> None:
    axis.imshow(
        np.clip(image / max(scale, 1e-30), 0, 1),
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect=aspect,
    )
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_projections(volumes: dict[str, np.ndarray], figure_dir: Path) -> dict[str, float]:
    views = {
        name: (volume.max(0), volume.max(1), volume.max(2))
        for name, volume in volumes.items()
    }
    own = {
        name: max(float(np.quantile(volume, 0.999)), 1e-30)
        for name, volume in volumes.items()
    }
    figure, axes = plt.subplots(3, len(METHODS), figsize=(16.5, 9.2), layout="constrained")
    for column, name in enumerate(METHODS):
        for row, suffix in enumerate(("XY MIP", "XZ MIP", "YZ MIP")):
            show(
                axes[row, column],
                views[name][row],
                own[name],
                f"{LABELS[name]} · {suffix}",
                aspect="equal" if row == 0 else "auto",
            )
    figure.savefig(figure_dir / "five_method_projections_volume_normalized_p999.png", dpi=180)
    plt.close(figure)

    common_network = max(own[name] for name in NETWORK_METHODS)
    figure, axes = plt.subplots(3, len(NETWORK_METHODS), figsize=(10.5, 9.2), layout="constrained")
    for column, name in enumerate(NETWORK_METHODS):
        for row, suffix in enumerate(("XY MIP", "XZ MIP", "YZ MIP")):
            show(
                axes[row, column],
                views[name][row],
                common_network,
                f"{LABELS[name]} · {suffix}",
                aspect="equal" if row == 0 else "auto",
            )
    figure.savefig(figure_dir / "three_network_projections_common_p999.png", dpi=180)
    plt.close(figure)
    return {**{f"{name}_p999": scale for name, scale in own.items()}, "network_common_p999": common_network}


def plot_layers(volumes: dict[str, np.ndarray], figure_dir: Path, scales: dict[str, float]) -> None:
    figure, axes = plt.subplots(10, len(METHODS), figsize=(16, 29), layout="constrained")
    for column, name in enumerate(METHODS):
        for depth in range(10):
            show(
                axes[depth, column],
                volumes[name][depth],
                scales[f"{name}_p999"],
                f"{LABELS[name]} · {(depth + 1) * 10} µm",
            )
    figure.savefig(figure_dir / "five_method_native_layers_volume_normalized_p999.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(10, len(NETWORK_METHODS), figsize=(10.5, 29), layout="constrained")
    for column, name in enumerate(NETWORK_METHODS):
        for depth in range(10):
            show(
                axes[depth, column],
                volumes[name][depth],
                scales["network_common_p999"],
                f"{LABELS[name]} · {(depth + 1) * 10} µm",
            )
    figure.savefig(figure_dir / "three_network_native_layers_common_p999.png", dpi=120)
    plt.close(figure)


def plot_rois_and_profiles(volumes: dict[str, np.ndarray], figure_dir: Path, scales: dict[str, float]) -> None:
    for roi_name, (y0, y1, x0, x1) in ROIS.items():
        figure, axes = plt.subplots(1, len(METHODS), figsize=(16, 3.4), layout="constrained")
        for column, name in enumerate(METHODS):
            show(
                axes[column],
                volumes[name][:, y0:y1, x0:x1].max(0),
                scales[f"{name}_p999"],
                LABELS[name],
            )
        figure.suptitle(f"Fixed ROI: {roi_name} · whole-volume p99.9 scale", fontsize=10)
        figure.savefig(figure_dir / f"roi_{roi_name}_xy_mip.png", dpi=180)
        plt.close(figure)

    z_um = np.arange(10, 101, 10)
    figure, axis = plt.subplots(figsize=(9, 5.4), layout="constrained")
    for name in METHODS:
        profile = normalized(volumes[name]).sum((1, 2))
        axis.plot(z_um, profile, ".-", label=LABELS[name])
    axis.set_xlabel("Depth (µm)")
    axis.set_ylabel("Mass fraction")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.savefig(figure_dir / "five_method_axial_mass_profiles.png", dpi=190)
    plt.close(figure)


def plot_differences(volumes: dict[str, np.ndarray], figure_dir: Path) -> None:
    new = normalized(volumes["new_taylor_anchor_mixed_real_600"])
    comparisons = (
        ("old_taylor_anchor_400", "new_minus_old_taylor"),
        ("specified_old_mean_anchor_400", "new_taylor_minus_specified_old_mean"),
    )
    for reference, stem in comparisons:
        delta = new - normalized(volumes[reference])
        projections = (delta.sum(0), delta.sum(1), delta.sum(2))
        scale = max(float(np.quantile(np.abs(view), 0.999)) for view in projections)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for axis, image, suffix in zip(axes, projections, ("XY", "XZ", "YZ")):
            axis.imshow(
                np.clip(image / max(scale, 1e-30), -1, 1),
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
                aspect="equal" if suffix == "XY" else "auto",
            )
            axis.set_title(f"{stem} · {suffix}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.savefig(figure_dir / f"{stem}_normalized_difference.png", dpi=190)
        plt.close(figure)


def report(manifest: Path) -> dict[str, Any]:
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    volumes, predictions = load_products(cfg)
    selected = np.stack(
        [tifffile.imread(path).astype(np.float32) / 255 for path in cfg["selected_files"]]
    )
    holdout = np.stack(
        [tifffile.imread(path).astype(np.float32) / 255 for path in cfg["holdout_files"]]
    )
    targets = {
        "input10": (selected.mean(0), selected.var(0, ddof=1)),
        "holdout90": (holdout.mean(0), holdout.var(0, ddof=1)),
    }
    metrics: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    z_um = np.arange(10, 101, 10)
    volume_dir = OUTPUT / "volumes"
    volume_dir.mkdir(parents=True, exist_ok=True)
    for name in METHODS:
        volume = volumes[name]
        q = normalized(volume)
        profile = q.sum((1, 2))
        np.save(volume_dir / f"{name}.npy", volume.astype(np.float32, copy=False))
        save_volume_tiff(volume_dir / f"{name}.tif", volume.astype(np.float32, copy=False))
        statistics.append(
            {
                "method": name,
                "total_mass": float(np.maximum(volume, 0).sum(dtype=np.float64)),
                "maximum": float(volume.max()),
                "axial_centroid_um": float(np.dot(z_um, profile)),
                "axial_peak_um": int(z_um[int(np.argmax(profile))]),
                "axial_entropy": float(-(profile * np.log(profile + 1e-30)).sum()),
                "normalized_lateral_tv": float(
                    (np.abs(np.diff(volume, axis=1)).sum() + np.abs(np.diff(volume, axis=2)).sum())
                    / max(np.maximum(volume, 0).sum(), 1e-30)
                ),
            }
        )
        predicted_mean, predicted_variance = predictions[name]
        for target_name, (target_mean, target_variance) in targets.items():
            metrics.append(
                {
                    "method": name,
                    "target": target_name,
                    "mean_scale_aligned_nrmse": scale_nrmse(predicted_mean, target_mean),
                    "variance_scale_aligned_nrmse": scale_nrmse(
                        predicted_variance, target_variance
                    ),
                    "mean_pearson": pearson(predicted_mean, target_mean),
                    "variance_pearson": pearson(predicted_variance, target_variance),
                }
            )
    write_csv(OUTPUT / "physics_metrics.csv", metrics)
    write_csv(OUTPUT / "volume_statistics.csv", statistics)

    pairwise: list[dict[str, Any]] = []
    new_q = normalized(volumes["new_taylor_anchor_mixed_real_600"])
    new_profile = new_q.sum((1, 2))
    new_mip = new_q.max(0)
    for reference in ("old_taylor_anchor_400", "specified_old_mean_anchor_400"):
        old_q = normalized(volumes[reference])
        old_profile = old_q.sum((1, 2))
        old_mip = old_q.max(0)
        pairwise.append(
            {
                "new_method": "new_taylor_anchor_mixed_real_600",
                "reference_method": reference,
                "unit_mass_relative_l2": float(
                    np.linalg.norm(new_q - old_q) / max(np.linalg.norm(old_q), 1e-30)
                ),
                "axial_w1_um": float(
                    np.abs(np.cumsum(new_profile) - np.cumsum(old_profile)).sum() * 10
                ),
                "xy_mip_pearson": pearson(new_mip, old_mip),
                "xy_mip_scale_aligned_nrmse": scale_nrmse(new_mip, old_mip),
            }
        )
    write_csv(OUTPUT / "network_pairwise_metrics.csv", pairwise)

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    scales = plot_projections(volumes, figure_dir)
    plot_layers(volumes, figure_dir, scales)
    plot_rois_and_profiles(volumes, figure_dir, scales)
    plot_differences(volumes, figure_dir)
    write_json(OUTPUT / "display_scales.json", scales)

    primary = {row["method"]: row for row in metrics if row["target"] == "holdout90"}
    lines = [
        "# 菠菜根 Taylor step 600 完整探索性对比",
        "",
        "固定使用帧 9、10、27、44、54、74、77、83、85、92，以及完全相同的 Mean/Taylor-RL3。",
        "",
        "| 方法 | 90帧均值NRMSE↓ | 90帧方差NRMSE↓ | 方差相关↑ | 轴向质心(µm) |",
        "|---|---:|---:|---:|---:|",
    ]
    by_stat = {row["method"]: row for row in statistics}
    for name in METHODS:
        row = primary[name]
        lines.append(
            f"| {LABELS[name]} | {row['mean_scale_aligned_nrmse']:.4f} | "
            f"{row['variance_scale_aligned_nrmse']:.4f} | {row['variance_pearson']:.4f} | "
            f"{by_stat[name]['axial_centroid_um']:.2f} |"
        )
    lines += [
        "",
        "## 对比边界",
        "",
        "- 用户指定目录中的模型实际为 Mean-anchor 400，并非 Taylor-anchor。",
        "- 指定旧模型和旧 Taylor 模型的训练清单均包含 P01–P11，但不包含真实视场45/55。",
        "- 新旧 Taylor 对比同时改变了真实训练数据和训练步数（400→600），不能把全部差异单独归因于45/55。",
        "- 菠菜根没有GT；这些指标表示10帧输入及90帧留出数据的前向统计一致性，不证明真实深度一定正确。",
        "- 原菠菜根输入经过逐帧最大值归一化和uint8量化，与保留强度的45/55训练流并不完全一致。",
        "",
        "图像均按整个三维体确定尺度，未逐层归一化。",
    ]
    (OUTPUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete = {
        "complete": True,
        "checkpoint_step": 600,
        "methods": list(METHODS),
        "volumes": len(METHODS),
        "physics_metric_rows": len(metrics),
        "figures": len(list(figure_dir.glob("*.png"))),
        "report": str(OUTPUT / "REPORT_ZH.md"),
    }
    write_json(OUTPUT / "complete.json", complete)
    return complete


def supervisor() -> None:
    manifest = prepare()
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=cfg["gpu_uuid"]["network"],
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
    )
    with (OUTPUT / "network.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--network", str(manifest)],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("V5 Taylor step-600 spinach inference failed; inspect network.log")

    matlab_paths = [
        ROOT / "matlab_code/real_data",
        ROOT / "matlab_code/pilot_dataset",
        ROOT / "matlab_code/Util",
        ROOT / "matlab_code/Solver",
    ]
    quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in matlab_paths)
    expression = f"addpath({quoted});spinach_root_gain_worker_v2('{manifest}');"
    environment["CUDA_VISIBLE_DEVICES"] = cfg["gpu_uuid"]["gain"]
    environment["MATLAB_PREFDIR"] = str(OUTPUT / "matlab_preferences")
    with (OUTPUT / "gain.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(MATLAB), "-singleCompThread", "-softwareopengl", "-batch", expression],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("V5 Taylor step-600 global gain failed; inspect gain.log")
    print(json.dumps(report(manifest), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path)
    arguments = parser.parse_args()
    if arguments.network:
        network(arguments.network)
    else:
        supervisor()


if __name__ == "__main__":
    main()
