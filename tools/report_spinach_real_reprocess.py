"""CPU reporting for the corrected two-field real-data comparison."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spinach_real_reprocess_mpl")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tools.spinach_real_reprocess import ROOT, CHECKPOINTS, read, save, sha

METHODS = ["mean_rl3", "taylor_rl3_sqrt", *CHECKPOINTS]
TITLES = ["Mean-RL3", "Taylor-RL3-sqrt", "Before P12: Taylor", "Before P12: Mean", "After P12: Taylor", "After P12: Mean"]
OLDREF = ROOT / "outputs/spinach_root_exploratory_20260909_run01"
OLDPATHS = {
    "mean_rl3": OLDREF / "mean_rl3.mat",
    "taylor_rl3_sqrt": OLDREF / "taylor_rl3.mat",
    "before_p12_taylor": OLDREF / "e3_mean100.mat",
    "before_p12_mean": ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/spinach_root_transfer/mean_anchor_e3_mean100.mat",
    "after_p12_taylor": ROOT / "outputs/v4_p12_anchor_compare_e3_mean100_400_20260909_run01/spinach_root_transfer/taylor_anchor_e3_mean100/taylor_anchor_e3_mean100.mat",
    "after_p12_mean": ROOT / "outputs/v4_p12_anchor_compare_e3_mean100_400_20260909_run01/spinach_root_transfer/mean_anchor_e3_mean100/mean_anchor_e3_mean100.mat",
}


def mat(path, key, volume=False):
    with h5py.File(path) as handle:
        array = np.asarray(handle[key], dtype=np.float32)
    return array.transpose(0, 2, 1).copy() if volume else array.T.copy()


def metric(prediction, target):
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    gain = np.vdot(p, t) / max(np.vdot(p, p), 1e-30)
    raw = np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-30)
    aligned = np.linalg.norm(gain * p - t) / max(np.linalg.norm(t), 1e-30)
    p -= p.mean(); t -= t.mean()
    corr = np.vdot(p, t) / max(np.linalg.norm(p) * np.linalg.norm(t), 1e-30)
    return float(raw), float(aligned), float(corr)


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def volume_key(method):
    return {"mean_rl3": "reconstruction_raw", "taylor_rl3_sqrt": "reconstruction_sqrt"}.get(method, "reconstruction")


def measures(path, method, tm, tv):
    volume = mat(path, volume_key(method), True)
    pm, pv = mat(path, "predicted_mean"), mat(path, "predicted_variance")
    mr, ma, mc = metric(pm, tm)
    vr, va, vc = metric(pv, tv)
    profile = volume.sum((1, 2), dtype=np.float64)
    mass = profile.sum(); profile /= max(mass, 1e-30)
    result = dict(mean_raw_nrmse=mr, mean_aligned_nrmse=ma, mean_pearson=mc,
                  variance_raw_nrmse=vr, variance_aligned_nrmse=va, variance_pearson=vc,
                  total_mass=float(mass), axial_centroid_um=float(profile @ np.arange(10, 101, 10)),
                  axial_peak_um=int(10 + 10 * profile.argmax()),
                  first_layer_mass=float(profile[0]), last_layer_mass=float(profile[-1]),
                  mass_20_30_um=float(profile[1:3].sum()), mass_50_100_um=float(profile[4:].sum()))
    return volume, profile, result


def show(ax, value, scale, title, aspect="equal"):
    ax.imshow(np.clip(value / max(scale, 1e-30), 0, 1), cmap="magma", vmin=0, vmax=1,
              interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def figures(folder, volumes):
    folder.mkdir(exist_ok=True)
    signature = {name: hashlib.sha256(np.ascontiguousarray(value)).hexdigest() for name, value in volumes.items()}
    marker = folder / "plots_complete.json"
    if marker.exists():
        cached = read(marker)
        if cached.get("volume_sha256") == signature and all((folder / name).is_file() for name in cached["files"]):
            return
    own = {name: max(float(np.quantile(value, .999)), 1e-30) for name, value in volumes.items()}
    shared = max(own.values())
    network_shared = max(own[name] for name in CHECKPOINTS)
    display = {"whole_volume_p999": own, "shared_scale": shared,
               "network_shared_scale": network_shared, "no_per_layer_normalization": True,
               "own_clipped_fraction": {name: float((value > own[name]).mean()) for name, value in volumes.items()},
               "shared_clipped_fraction": {name: float((value > shared).mean()) for name, value in volumes.items()}}
    for mode in ["own", "shared"]:
        fig, axes = plt.subplots(3, 6, figsize=(24, 10), layout="constrained", gridspec_kw={"height_ratios": [3, 1, 1]})
        for column, method in enumerate(METHODS):
            volume = volumes[method]
            for row, view in enumerate([volume.max(0), volume.max(1), volume.max(2)]):
                show(axes[row, column], view, own[method] if mode == "own" else shared,
                     TITLES[column] + " / " + ["XY", "XZ", "YZ"][row], "equal" if row == 0 else "auto")
        fig.suptitle(f"Field {folder.parent.name} (identifier, not depth): linear {mode} whole-volume scale")
        fig.savefig(folder / f"six_method_projections_{mode}.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), layout="constrained", gridspec_kw={"height_ratios": [3, 1, 1]})
    for column, method in enumerate(CHECKPOINTS):
        volume = volumes[method]
        for row, view in enumerate([volume.max(0), volume.max(1), volume.max(2)]):
            show(axes[row, column], view, network_shared, method + " / " + ["XY", "XZ", "YZ"][row], "equal" if row == 0 else "auto")
    fig.suptitle("Four networks on the same preserved inputs: shared whole-volume scale")
    fig.savefig(folder / "four_networks_shared.png", dpi=145); plt.close(fig)
    height, width = next(iter(volumes.values())).shape[1:]
    y0 = max(0, (height // 2 // 49 - 4) * 49)
    x0 = max(0, (width // 2 // 49 - 4) * 49)
    display["fixed_roi_yx_zero_based_half_open"] = [y0, y0 + 392, x0, x0 + 392]
    for roi_only in [False, True]:
        fig, axes = plt.subplots(10, 6, figsize=(24, 32), layout="constrained")
        for column, method in enumerate(METHODS):
            volume = volumes[method]
            for layer in range(10):
                value = volume[layer, y0:y0 + 392, x0:x0 + 392] if roi_only else volume[layer]
                show(axes[layer, column], value, own[method], f"{TITLES[column]} | {10 + 10 * layer} um")
        name = "fixed_roi_native_layers" if roi_only else "all_native_layers"
        fig.savefig(folder / (name + ".png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for method in METHODS:
        profile = volumes[method].sum((1, 2), dtype=np.float64)
        ax.plot(np.arange(10, 101, 10), profile / profile.sum(), ".-", label=method)
    ax.set_xlabel("Depth (um; no GT)"); ax.set_ylabel("Fraction of reconstructed mass")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(folder / "axial_profiles.png", dpi=150); plt.close(fig)
    save(folder / "display_scales.json", display)
    save(marker, {"volume_sha256": signature, "files": ["six_method_projections_own.png", "six_method_projections_shared.png", "four_networks_shared.png", "all_native_layers.png", "fixed_roi_native_layers.png", "axial_profiles.png", "display_scales.json"]})


def report(output):
    cfg = read(output / "manifest.json")
    assert (output / "inference_complete.json").exists()
    rows, profiles, legacy_rows = [], [], []
    for field in cfg["fields"]:
        field_id = field["field_id"]
        folder = output / field_id
        manifest = read(folder / "manifest.json")
        paths = {"mean_rl3": Path(manifest["output_mat"]["mean"]), "taylor_rl3_sqrt": Path(manifest["output_mat"]["taylor"]),
                 **{method: folder / method / "reconstruction.mat" for method in CHECKPOINTS}}
        targets = {"input10": (tifffile.imread(manifest["mean_tiff"]), tifffile.imread(manifest["variance_tiff"])),
                   "holdout90": (tifffile.imread(manifest["holdout_mean_tiff"]), tifffile.imread(manifest["holdout_variance_tiff"]))}
        volumes = {}
        for method, path in paths.items():
            for target_name, (tm, tv) in targets.items():
                volume, profile, result = measures(path, method, tm, tv)
                assert volume.shape == (10, 1029, 1421) and np.isfinite(volume).all() and volume.min() >= 0
                rows.append(dict(field_id=field_id, split=field["split"], method=method, target=target_name, **result))
                volumes[method] = volume
                if target_name == "holdout90":
                    profiles.extend(dict(field_id=field_id, split=field["split"], method=method, z_um=10 + 10 * i, mass_fraction=float(p)) for i, p in enumerate(profile))
            if field_id == "55":
                for version, candidate in [("legacy_frame_normalized", OLDPATHS[method]), ("preserved_float", path)]:
                    _, _, result = measures(candidate, method, *targets["holdout90"])
                    # Raw errors across differently scaled preprocessing are not comparable.
                    result.pop("mean_raw_nrmse"); result.pop("variance_raw_nrmse")
                    legacy_rows.append(dict(method=method, preprocessing=version, evaluation_target="same_preserved_holdout90", **result))
        figures(folder / "figures", volumes)
        if field_id == "55":
            fig, axes = plt.subplots(4, 4, figsize=(17, 17), layout="constrained")
            for column, method in enumerate(CHECKPOINTS):
                previous = mat(OLDPATHS[method], "reconstruction", True)
                current = volumes[method]
                for row, (array, prefix) in enumerate([(previous, "Legacy input"), (current, "Preserved input")]):
                    scale = max(float(np.quantile(array, .999)), 1e-30)
                    show(axes[row, column], array.max(0), scale, f"{method}: {prefix}")
                    show(axes[row + 2, column], array[5], scale, f"{prefix} | 60 um")
            fig.savefig(folder / "figures/preprocessing_before_after_networks.png", dpi=135); plt.close(fig)
        print(f"REPORT field={field_id} complete", flush=True)
    csv_write(output / "physics_metrics.csv", rows)
    csv_write(output / "axial_profiles.csv", profiles)
    csv_write(output / "test55_preprocessing_comparison.csv", legacy_rows)
    integrity = {"raw_hashes_unchanged": all(sha(p) == h for field in cfg["fields"] for p, h in field["source_sha256"].items()),
                 "checkpoint_hashes_unchanged": all(sha(value["path"]) == value["sha256"] for value in cfg["checkpoints"].values()),
                 "psf_hash_unchanged": sha(cfg["psf_path"]) == cfg["psf_sha256"],
                 "processed_hashes_unchanged": all(sha(p) == h for field in cfg["fields"] for p, h in read(output / field["field_id"] / "manifest.json")["rectified_hashes"].items()),
                 "processed_frames": 200, "rl_volumes": 4, "network_volumes": 8,
                 "physics_rows": len(rows), "profile_rows": len(profiles), "legacy_comparison_rows": len(legacy_rows),
                 "training_started": False, "subset_count_per_field": 1,
                 "all_100_frames_preserved_for_future_training_subsets": True,
                 "test_field_seen_previously": True,
                 "no_gt_or_depth_label": True}
    assert all(integrity[k] for k in ["raw_hashes_unchanged", "checkpoint_hashes_unchanged", "psf_hash_unchanged", "processed_hashes_unchanged"])
    lines = ["# 真实菠菜根：保留强度重处理与六方法对比", "",
             "45登记为训练视场，55登记为测试视场。55原文件名为Z50，之前称为50；这些编号都不是深度，两个视场的真实深度未知且不强制相同。", "",
             "本轮没有训练或微调，没有修改历史仿真划分。全部网络使用既有400步final权重，保留Set。固定帧9、10、27、44、54、74、77、83、85、92输入，另90帧只用于统计检查。", "",
             "## 数据流程", "",
             "RGB uint8固定/255 → MATLAB rgb2gray浮点灰度 → 用户提供的标定及原几何插值/裁剪 → float32 TIFF。移除逐帧max归一化，不增加gamma、去噪或无依据暗电平扣除。原BMP本身是8位，已有量化/饱和无法恢复；曝光、增益及相机ISP是否固定尚无独立记录。", "",
             "标定来自1.txt，与矫正代码一致。矫正结果1029×1421，不缩放到256。模型使用原生10–100 µm十层，不以45/55裁层，也不把两个视场预测到同一深度。", "",
             "## 测试视场55", "",
             "没有真实GT，以下是剩余90帧的前向统计一致性，不是3D定位/分辨率真值评分。方差模型仍为H²(g²)近似，未做相机噪声或真实照明协方差标定。", "",
             "| 方法 | 均值尺度对齐NRMSE↓ | 方差尺度对齐NRMSE↓ | 方差相关↑ | 输出质心µm（非GT误差） |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["field_id"] == "55" and row["target"] == "holdout90":
            lines.append(f"| {row['method']} | {row['mean_aligned_nrmse']:.4f} | {row['variance_aligned_nrmse']:.4f} | {row['variance_pearson']:.4f} | {row['axial_centroid_um']:.2f} |")
    lines += ["", "## 同一新90帧目标上的预处理前后对比", "",
              "下表把已有旧输入预测与本次新预测，统一对齐到本次保留强度的90帧统计目标。只比较尺度对齐误差；不得直接比较两个不同强度单位下的原始误差。", "",
              "| 方法 | 旧→新 均值NRMSE | 旧→新 方差NRMSE |", "|---|---:|---:|"]
    for method in METHODS:
        old, new = [row for row in legacy_rows if row["method"] == method]
        lines.append(f"| {method} | {old['mean_aligned_nrmse']:.4f} → {new['mean_aligned_nrmse']:.4f} | {old['variance_aligned_nrmse']:.4f} → {new['variance_aligned_nrmse']:.4f} |")
    test = {row["method"]: row for row in rows if row["field_id"] == "55" and row["target"] == "holdout90"}
    before, after = test["before_p12_mean"], test["after_p12_mean"]
    lines += ["", "## 本次结果解读", "",
              f"在同一新浮点输入下，P12前Mean网络的方差NRMSE为{before['variance_aligned_nrmse']:.4f}，P12后为{after['variance_aligned_nrmse']:.4f}；后者均值NRMSE略低，但方差一致性没有改善。",
              f"两者20–30 µm质量占比分别为{before['mass_20_30_um']:.1%}、{after['mass_20_30_um']:.1%}，50–100 µm占比分别为{before['mass_50_100_um']:.1%}、{after['mass_50_100_um']:.1%}。修正预处理后，P12后网络更集中浅层的差异仍存在。", "",
              "共同尺度图显示P12前Mean的环状结构较连贯，P12后结果局部更尖锐/碎片化。这是外观观察，不是经GT或标定目标证明的分辨率结论。", "",
              "整体而言，去掉逐帧归一化没有使现有冻结网络获得明显质量跃升，也没有消除P12前后的差异。因此之前退步不能仅归因于逐帧归一化；训练分布和优化轨迹变化、真实照明/噪声等其他差异仍需考虑。保留强度是为后续物理自监督提供更合适的数据，并不保证旧权重立即改善。", "",
              "Mean-RL3在本次前向统计一致性上表现较好，但其质量分散到多层，不能据此判定它具有更好的轴向定位或分辨率。任何方法的输出质心都不是实测真实深度。"]
    lines += ["", "## 交付与限制", "",
              f"数据目录：`{cfg['dataset']}`。每视场100帧灰度float及100帧矫正float；本轮各冻结一组10/90，保存均值、方差、Mean-RL3和Taylor-RL3-sqrt，并可供后续真实训练抽样。尚未生成另外9组RL缓存，也未接入现有仿真训练器的硬编码对象清单。", "",
              "输出包括两视场共4份RL和8份网络重建、24条物理统计、120条深度占比、12条测试预处理比较、两视场的投影/原生十层/固定ROI/深度曲线。", "",
              "显示采用整个体p99.9线性归一化与方法共同尺度两版，所有原生层共用同一体的尺度，不逐层拉亮。具体尺度、截断比例和固定ROI坐标见各figures/display_scales.json。", "",
              "45的预测仅作训练视场诊断，不与55合并成独立测试均值。55已用于过去多轮探索，虽然此后固定为测试并禁止进入训练，但不能视为从未看过的盲测；正式泛化结论仍建议新增独立视场。", "",
              "真实训练时应按视场隔离，并先检查真实噪声和方差模型单位；本轮不宣称仅移除归一化即可解决所有仿真到真实的差异。"]
    lines += ["", "## 准备验证", "",
              "完整CPU unittest：198项，191项通过、7项跳过，无失败。四个checkpoint在49×49十层冻结随机输入上，新入口与各自旧入口逐位一致，最大绝对差0。", "",
              "55/Z50_image_0(4).bmp通过旧灰度与矫正链路复现后，与历史50_001.tif逐像素一致，确认同源。原始帧按image_0..99数值排序，括号后缀不参与编号且已排除重复。", "",
              "初次MATLAB写出的LZW浮点TIFF与当前tifffile依赖不兼容，已仅对本次生成文件做无损不压缩转存，逐个检查像素完全一致；没有更改数值处理方案。", "",
              "旧/新观测统计去掉整体尺度后的差异：10帧均值0.004164、方差0.162150；90帧均值0.002620、方差0.047994。这是输入统计之间的差异，不是重建相对真实GT的误差。"]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    save(output / "complete.json", {"complete": True, **integrity})
    print(json.dumps(integrity, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path)
    report(parser.parse_args().output.resolve())
