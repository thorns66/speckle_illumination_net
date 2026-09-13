"""Full-field spinach-root transfer for both P12-trained anchor networks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from scipy.io import savemat

from tools import three_way_experiment as old
from tools import v4_p12_anchor_compare_experiment as exp
from tools.spinach_root_network_v2 import execute, mat_volume
from tools.three_way_checks import input_item


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
METHODS = (
    "mean_rl3", "taylor_rl3_sqrt", "old_taylor_e3_mean100",
    "taylor_anchor_e3_mean100", "mean_anchor_e3_mean100",
)


def network(manifest_path: Path) -> None:
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]["network"]:
        raise ValueError("Worker GPU differs from the transfer manifest")
    arm = cfg["arm"]
    anchor_kind = exp.ANCHORS[arm]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    exp.configure_precision()
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    exp.validate_config(config)
    if int(checkpoint["completed_steps"]) != 400 or config["v3_compare"]["kind"] != arm:
        raise ValueError("Spinach transfer requires the matching step-400 final checkpoint")
    model = exp.build_model(config, initial=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    operator = exp.load_operator(config, device)
    item = input_item(exp.DATA / "T03", 1, device)
    with torch.inference_mode():
        observed, _ = exp.forward(model, item, operator, config=config)
    reference = np.load(
        exp.OUTPUT / "evaluation" / arm / "final/T03_subset_01/reconstruction.npy",
        allow_pickle=False,
    )
    regression = float(
        np.linalg.norm(observed.reconstruction[0, 0].cpu().numpy() - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )
    if regression > 1e-5:
        raise ValueError(f"Frozen V4 regression failed: {regression}")
    del operator, item, observed
    torch.cuda.empty_cache()

    mean = mat_volume(REFERENCE / "mean_rl3.mat", "reconstruction_raw")
    taylor = mat_volume(REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt")
    projection_file = REFERENCE / ("taylor_rl3.mat" if anchor_kind == "taylor_sqrt" else "mean_rl3.mat")
    projection_name = "projection_of_sqrt" if anchor_kind == "taylor_sqrt" else "predicted_mean"
    with h5py.File(projection_file, "r") as handle:
        projection = np.asarray(handle[projection_name], dtype=np.float32).T.copy()
    source_manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    frames = np.stack([
        tifffile.imread(path).astype(np.float32) / 255 for path in source_manifest["selected_files"]
    ])
    image_mean = frames.mean(0, dtype=np.float64).astype(np.float32)
    residual = frames - image_mean
    beta0 = float(
        np.sum(projection.astype(np.float64) * image_mean)
        / max(np.sum(projection.astype(np.float64) ** 2) + 1e-8, 1e-30)
    )
    if not np.isfinite(beta0) or beta0 <= 0:
        raise ValueError("Invalid full-field analytic beta0")
    base, mode = execute(
        model, taylor, mean, residual, np.asarray(source_manifest["z_um"], np.float32),
        beta0, device,
    )
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("Invalid full-field anchor-network output")
    savemat(
        cfg["network_base_mat"], {"base_reconstruction": base.transpose(1, 2, 0)},
        do_compression=False,
    )
    factor = 1 + float(config["three_way"]["gain_bound"]) * math.tanh(
        float(model.mean_gain_gamma.detach().cpu())
    )
    old.write_json(Path(cfg["network_record"]), {
        "complete": True, "arm": arm, "anchor_kind": anchor_kind,
        "beta0": beta0, "regression_relative_l2": regression,
        "raw_beta": float(model.raw_beta.detach().cpu()),
        "mean_gain_gamma": float(model.mean_gain_gamma.detach().cpu()),
        "e3_gain_factor": factor, **mode,
    })
    cfg["e3_gain_factor"] = factor
    old.write_json(manifest_path, cfg)


def _mat(path: Path, name: str, *, volume: bool = False) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        value = np.asarray(handle[name], dtype=np.float32)
    return value.transpose(0, 2, 1).copy() if volume else value.T.copy()


def _scale_nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    p = prediction.astype(np.float64)
    t = target.astype(np.float64)
    gain = np.vdot(p, t).real / max(np.vdot(p, p).real, 1e-30)
    return float(np.linalg.norm(gain * p - t) / max(np.linalg.norm(t), 1e-30))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64).ravel()
    y = right.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def report(root: Path) -> dict:
    manifests = {
        arm: json.loads((root / arm / "manifest.json").read_text(encoding="utf-8"))
        for arm in exp.ARMS
    }
    volumes = {
        "mean_rl3": _mat(REFERENCE / "mean_rl3.mat", "reconstruction_raw", volume=True),
        "taylor_rl3_sqrt": _mat(REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt", volume=True),
        "old_taylor_e3_mean100": _mat(REFERENCE / "e3_mean100.mat", "reconstruction", volume=True),
        **{
            arm: _mat(Path(manifests[arm]["network_final_mat"]), "reconstruction", volume=True)
            for arm in exp.ARMS
        },
    }
    predictions = {
        "mean_rl3": (
            _mat(REFERENCE / "mean_rl3.mat", "predicted_mean"),
            _mat(REFERENCE / "mean_rl3.mat", "predicted_variance"),
        ),
        "taylor_rl3_sqrt": (
            _mat(REFERENCE / "taylor_rl3.mat", "predicted_mean"),
            _mat(REFERENCE / "taylor_rl3.mat", "predicted_variance"),
        ),
        "old_taylor_e3_mean100": (
            _mat(REFERENCE / "e3_mean100.mat", "predicted_mean"),
            _mat(REFERENCE / "e3_mean100.mat", "predicted_variance"),
        ),
        **{
            arm: (
                _mat(Path(manifests[arm]["network_final_mat"]), "predicted_mean"),
                _mat(Path(manifests[arm]["network_final_mat"]), "predicted_variance"),
            )
            for arm in exp.ARMS
        },
    }
    source = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    selected = np.stack([
        tifffile.imread(path).astype(np.float32) / 255 for path in source["selected_files"]
    ])
    holdout = np.stack([
        tifffile.imread(path).astype(np.float32) / 255 for path in source["holdout_files"]
    ])
    targets = {
        "input10": (selected.mean(0), selected.var(0, ddof=1)),
        "holdout90": (holdout.mean(0), holdout.var(0, ddof=1)),
    }
    rows = []
    for method, (predicted_mean, predicted_variance) in predictions.items():
        volume = volumes[method]
        mass = volume.sum((1, 2), dtype=np.float64)
        fraction = mass / max(mass.sum(), 1e-30)
        for target_name, (target_mean, target_variance) in targets.items():
            rows.append({
                "method": method, "target": target_name,
                "mean_scale_aligned_nrmse": _scale_nrmse(predicted_mean, target_mean),
                "variance_scale_aligned_nrmse": _scale_nrmse(predicted_variance, target_variance),
                "mean_pearson": _correlation(predicted_mean, target_mean),
                "variance_pearson": _correlation(predicted_variance, target_variance),
                "axial_centroid_um": float(np.dot(np.arange(10, 101, 10), fraction)),
                "axial_peak_um": int(np.arange(10, 101, 10)[np.argmax(fraction)]),
                "axial_entropy": float(-(fraction * np.log(fraction + 1e-30)).sum()),
                "normalized_lateral_total_variation": float(
                    (np.abs(np.diff(volume, axis=1)).sum() + np.abs(np.diff(volume, axis=2)).sum())
                    / max(volume.sum(), 1e-30)
                ),
            })
    metrics = root / "physics_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    scales = {method: max(float(np.quantile(volume, .999)), 1e-30) for method, volume in volumes.items()}
    figure, axes = plt.subplots(3, 5, figsize=(19, 10), layout="constrained")
    for column, method in enumerate(METHODS):
        volume = volumes[method]
        for row, view in enumerate((volume.max(0), volume.max(1), volume.max(2))):
            axes[row, column].imshow(
                np.clip(view / scales[method], 0, 1), cmap="magma", vmin=0, vmax=1,
                interpolation="nearest", aspect="auto" if row else "equal",
            )
            axes[row, column].set_title(method, fontsize=8)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    figure.savefig(figure_dir / "five_method_projections_volume_normalized.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(10, 5, figsize=(16, 30), layout="constrained")
    for column, method in enumerate(METHODS):
        for depth in range(10):
            axes[depth, column].imshow(
                np.clip(volumes[method][depth] / scales[method], 0, 1),
                cmap="magma", vmin=0, vmax=1, interpolation="nearest",
            )
            axes[depth, column].set_title(f"{method} · {(depth + 1) * 10} um", fontsize=7)
            axes[depth, column].set_xticks([]); axes[depth, column].set_yticks([])
    figure.savefig(figure_dir / "five_method_native_layers.png", dpi=115)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for method, volume in volumes.items():
        profile = volume.sum((1, 2), dtype=np.float64)
        profile /= max(profile.sum(), 1e-30)
        ax.plot(np.arange(10, 101, 10), profile, ".-", label=method)
    ax.grid(alpha=.25); ax.set_xlabel("depth (um)"); ax.set_ylabel("mass fraction")
    ax.legend(fontsize=7)
    figure.savefig(figure_dir / "five_method_axial_profiles.png", dpi=180)
    plt.close(figure)

    primary = {row["method"]: row for row in rows if row["target"] == "holdout90"}
    lines = [
        "# 菠菜根 P12 双主分支探索性迁移", "",
        "固定使用帧 9、10、27、44、54、74、77、83、85、92；所有方法共用同一 Mean/Taylor-RL3。真实数据无 GT，以下只衡量前向统计一致性。", "",
        "| 方法 | 90帧均值NRMSE↓ | 90帧方差NRMSE↓ | 方差相关↑ | 轴向质心(µm) |", "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = primary[method]
        lines.append(
            f"| {method} | {row['mean_scale_aligned_nrmse']:.4f} | "
            f"{row['variance_scale_aligned_nrmse']:.4f} | {row['variance_pearson']:.4f} | "
            f"{row['axial_centroid_um']:.2f} |"
        )
    lines += ["", "这部分不能单独证明真实深度正确，只用于观察仿真训练网络迁移后的物理统计与图像形态。"]
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"complete": True, "methods": list(METHODS), "metric_rows": len(rows), "figures": 3}
    old.write_json(root / "complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("network", "report"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    if arguments.stage == "network":
        network(arguments.path)
    else:
        print(json.dumps(report(arguments.path), ensure_ascii=False))


if __name__ == "__main__":
    main()
