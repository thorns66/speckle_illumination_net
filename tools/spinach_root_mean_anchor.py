"""Full-field Mean-anchor transfer and four-method report for the spinach root."""
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

from tools import mean_anchor_experiment as exp
from tools import three_way_experiment as old
from tools.spinch_root_network_v2 import execute, mat_volume
from tools.three_way_checks import input_item


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
METHODS = ("mean_rl3", "taylor_rl3_sqrt", "e3_mean100", exp.ARM)


def _mat(path, name, volume=False):
    with h5py.File(path, "r") as handle:
        value = np.asarray(handle[name], dtype=np.float32)
    return value.transpose(0, 2, 1).copy() if volume else value.T.copy()


def network(manifest_path: Path) -> None:
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]:
        raise ValueError("Worker GPU differs from the frozen transfer manifest")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    exp.configure_precision()
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    exp.validate_config(config)
    if checkpoint["completed_steps"] != 400:
        raise ValueError("Spinach transfer must use the step-400 final checkpoint")
    model = exp.build_model(config, initial=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    operator = exp.load_operator(config, device)
    item = input_item(exp.DATA / "T03", 1, device)
    with torch.inference_mode():
        observed, _ = exp.forward(model, item, operator, config=config)
    reference = np.load(
        exp.OUTPUT / "evaluation" / exp.ARM / "final/T03_subset_01/reconstruction.npy",
        allow_pickle=False,
    )
    regression = float(np.linalg.norm(observed.reconstruction[0, 0].cpu().numpy() - reference) / max(np.linalg.norm(reference), 1e-30))
    if regression > 1e-5:
        raise ValueError(f"Frozen Mean-anchor regression failed: {regression}")
    del operator, item, observed
    torch.cuda.empty_cache()

    mean = mat_volume(REFERENCE / "mean_rl3.mat", "reconstruction_raw")
    taylor = mat_volume(REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt")
    with h5py.File(REFERENCE / "mean_rl3.mat", "r") as handle:
        projection = np.asarray(handle["predicted_mean"], dtype=np.float32).T.copy()
    source_manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    frames = np.stack([tifffile.imread(path).astype(np.float32) / 255 for path in source_manifest["selected_files"]])
    image_mean = frames.mean(0, dtype=np.float64).astype(np.float32)
    residual = frames - image_mean
    beta0 = float(np.sum(projection.astype(np.float64) * image_mean) / max(np.sum(projection.astype(np.float64) ** 2) + 1e-8, 1e-30))
    if not np.isfinite(beta0) or beta0 <= 0:
        raise ValueError("Invalid full-field Mean-anchor beta0")
    base, mode = execute(
        model, taylor, mean, residual, np.asarray(source_manifest["z_um"], np.float32), beta0, device
    )
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("Invalid full-field Mean-anchor output")
    savemat(cfg["network_base_mat"], {"base_reconstruction": base.transpose(1, 2, 0)}, do_compression=False)
    factor = 1 + float(config["three_way"]["gain_bound"]) * math.tanh(float(model.mean_gain_gamma.detach().cpu()))
    cfg["e3_gain_factor"] = factor
    manifest_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    old.write_json(Path(cfg["network_record"]), {
        "complete": True, "beta0": beta0, "regression_relative_l2": regression,
        "raw_beta": float(model.raw_beta.detach().cpu()),
        "mean_gain_gamma": float(model.mean_gain_gamma.detach().cpu()),
        "e3_gain_factor": factor, **mode,
    })


def _scale_nrmse(pred, target):
    p = pred.astype(np.float64); t = target.astype(np.float64)
    gain = np.vdot(p, t).real / max(np.vdot(p, p).real, 1e-30)
    return float(np.linalg.norm(gain * p - t) / max(np.linalg.norm(t), 1e-30))


def _corr(left, right):
    x = left.astype(np.float64).ravel(); y = right.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def _show(ax, image, scale, title, aspect="equal"):
    ax.imshow(np.clip(image / max(scale, 1e-30), 0, 1), cmap="magma", vmin=0, vmax=1, interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def report(manifest_path: Path) -> dict:
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest_path.parent
    volumes = {
        "mean_rl3": _mat(REFERENCE / "mean_rl3.mat", "reconstruction_raw", True),
        "taylor_rl3_sqrt": _mat(REFERENCE / "taylor_rl3.mat", "reconstruction_sqrt", True),
        "e3_mean100": _mat(REFERENCE / "e3_mean100.mat", "reconstruction", True),
        exp.ARM: _mat(cfg["network_final_mat"], "reconstruction", True),
    }
    predictions = {
        "mean_rl3": (_mat(REFERENCE / "mean_rl3.mat", "predicted_mean"), _mat(REFERENCE / "mean_rl3.mat", "predicted_variance")),
        "taylor_rl3_sqrt": (_mat(REFERENCE / "taylor_rl3.mat", "predicted_mean"), _mat(REFERENCE / "taylor_rl3.mat", "predicted_variance")),
        "e3_mean100": (_mat(REFERENCE / "e3_mean100.mat", "predicted_mean"), _mat(REFERENCE / "e3_mean100.mat", "predicted_variance")),
        exp.ARM: (_mat(cfg["network_final_mat"], "predicted_mean"), _mat(cfg["network_final_mat"], "predicted_variance")),
    }
    source_manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    selected = np.stack([tifffile.imread(path).astype(np.float32) / 255 for path in source_manifest["selected_files"]])
    holdout = np.stack([tifffile.imread(path).astype(np.float32) / 255 for path in source_manifest["holdout_files"]])
    targets = {
        "input10": (selected.mean(0), selected.var(0, ddof=1)),
        "holdout90": (holdout.mean(0), holdout.var(0, ddof=1)),
    }
    rows = []
    for method, (pm, pv) in predictions.items():
        volume = volumes[method]
        mass = volume.sum((1, 2), dtype=np.float64); fraction = mass / max(mass.sum(), 1e-30)
        for target, (tm, tv) in targets.items():
            rows.append({
                "method": method, "target": target,
                "mean_scale_aligned_nrmse": _scale_nrmse(pm, tm),
                "variance_scale_aligned_nrmse": _scale_nrmse(pv, tv),
                "mean_pearson": _corr(pm, tm), "variance_pearson": _corr(pv, tv),
                "axial_centroid_um": float(np.dot(np.arange(10, 101, 10), fraction)),
                "axial_peak_um": int(np.arange(10, 101, 10)[np.argmax(fraction)]),
                "axial_entropy": float(-(fraction * np.log(fraction + 1e-30)).sum()),
                "normalized_lateral_total_variation": float((np.abs(np.diff(volume, axis=1)).sum() + np.abs(np.diff(volume, axis=2)).sum()) / max(volume.sum(), 1e-30)),
            })
    with (output / "physics_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    own = {method: max(float(np.quantile(value, .999)), 1e-30) for method, value in volumes.items()}
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
    for column, method in enumerate(METHODS):
        value = volumes[method]
        for row, view in enumerate((value.max(0), value.max(1), value.max(2))):
            _show(axes[row, column], view, own[method], method, "auto" if row else "equal")
    fig.savefig(figure_dir / "four_method_projections_shape_p999.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(10, 4, figsize=(13, 30), layout="constrained")
    for column, method in enumerate(METHODS):
        for depth in range(10):
            _show(axes[depth, column], volumes[method][depth], own[method], f"{method} · {10 * (depth + 1)} um")
    fig.savefig(figure_dir / "four_method_native_layers.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for method, value in volumes.items():
        profile = value.sum((1, 2), dtype=np.float64); profile /= max(profile.sum(), 1e-30)
        ax.plot(np.arange(10, 101, 10), profile, ".-", label=method)
    ax.grid(alpha=.25); ax.set_xlabel("depth (um)"); ax.set_ylabel("mass fraction"); ax.legend(fontsize=8)
    fig.savefig(figure_dir / "four_method_axial_profiles.png", dpi=170); plt.close(fig)
    primary = {row["method"]: row for row in rows if row["target"] == "holdout90"}
    lines = ["# 菠菜根 Mean-anchor 探索性迁移", "", "固定使用原实验10帧和同一组Mean/Taylor-RL3；新增网络采用Mean重建基础及全幅E3亮度校准。", "",
             "| 方法 | 90帧均值NRMSE↓ | 90帧方差NRMSE↓ | 方差相关↑ | 轴向质心μm |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        row = primary[method]; lines.append(f"| {method} | {row['mean_scale_aligned_nrmse']:.4f} | {row['variance_scale_aligned_nrmse']:.4f} | {row['variance_pearson']:.4f} | {row['axial_centroid_um']:.2f} |")
    lines += ["", "真实数据没有GT，这些指标只表示前向统计一致性，不能单独证明深度正确。"]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"complete": True, "methods": list(METHODS), "metric_rows": len(rows), "figures": 3}
    old.write_json(output / "report_complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("network", "report")); parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.stage == "network": network(args.manifest)
    else: print(json.dumps(report(args.manifest), ensure_ascii=False))


if __name__ == "__main__":
    main()

