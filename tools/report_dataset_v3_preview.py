"""Scientific plotting of saved truth; no AI images or changes to truth arrays."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from tools.run_dataset_v3_preview import NEW_OBJECTS, read_json, sha256, write_json
from utils.display_normalization import normalize_display_volume


def load_truth(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(folder / "truth.mat", "r") as handle:
        # MATLAB YXZ is stored by HDF5 as ZXY; expose ZYX consistently.
        coarse = np.asarray(handle["ground_truth"][()]).transpose(0, 2, 1)
        fine = np.asarray(handle["ground_truth_fine"][()]).transpose(0, 2, 1)
        inputs = np.asarray(handle["input_indices"][()]).T.astype(int)
        holdouts = np.asarray(handle["holdout_indices"][()]).T.astype(int)
    if coarse.shape != (10, 260, 260) or fine.shape != (100, 260, 260):
        raise ValueError(f"Unexpected volume shape: {folder}")
    if inputs.shape != (10, 10) or holdouts.shape != (10, 90):
        raise ValueError("Unexpected frozen subset shapes")
    if sorted(inputs.ravel().tolist()) != list(range(1, 101)):
        raise ValueError("Ten input subsets do not partition all hundred frames")
    for left, right in zip(inputs, holdouts):
        if set(left) & set(right) or set(left) | set(right) != set(range(1, 101)):
            raise ValueError("Invalid 10/90 complement")
    return coarse, fine


def depth_rgba(fine: np.ndarray, z: np.ndarray) -> np.ndarray:
    normalized, _ = normalize_display_volume(fine)
    peak = normalized.max(axis=0)
    depth = z[np.argmax(normalized, axis=0)]
    color = plt.get_cmap("turbo")(Normalize(10, 100)(depth))
    # Alpha encodes one globally normalized volume, not per-layer gain.
    color[..., 3] = peak
    return color


def render(root: Path) -> None:
    preview = read_json(root / "preview_report.json")
    if (
        not preview["preview_complete"]
        or preview["dataset_complete"]
        or preview["morphology_approved"]
    ):
        raise ValueError(
            "Report requires complete previews, not an approved or completed dataset"
        )
    source_paths = [
        root / sample / name
        for sample in NEW_OBJECTS
        for name in (
            "truth.mat",
            "ground_truth_float.tif",
            "ground_truth_fine_float.tif",
            "geometry.json",
            "config.json",
            "truth_metrics.json",
        )
    ]
    before = {str(p.relative_to(root)): sha256(p) for p in source_paths}
    figures = root / "previews"
    figures.mkdir(exist_ok=True)
    overview, axes = plt.subplots(3, 4, figsize=(16, 10.8), constrained_layout=True)
    descriptions = {
        "T03": "single-layer resolution chart",
        "T04": "sparse diagonal 3D mesh",
        "V03": "60 independent solid beads",
    }
    records = []
    for row, sample in enumerate(NEW_OBJECTS):
        folder = root / sample
        cfg, metrics = read_json(folder / "config.json"), read_json(
            folder / "truth_metrics.json"
        )
        g, fine = load_truth(folder)
        normalized, divisor = normalize_display_volume(g)
        pitch = cfg["object_pixel_pitch_um"]
        xy_extent = [-pitch / 2, 259.5 * pitch, 259.5 * pitch, -pitch / 2]
        z_extent = [-pitch / 2, 259.5 * pitch, 105, 5]
        xy, xz, yz = (
            normalized.max(axis=0),
            normalized.max(axis=1),
            normalized.max(axis=2),
        )
        for col, value, extent, title in (
            (0, xy, xy_extent, "XY"),
            (1, xz, z_extent, "XZ"),
            (2, yz, z_extent, "YZ"),
        ):
            ax = axes[row, col]
            ax.imshow(
                value,
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
                interpolation="nearest",
            )
            ax.set_title(f"{sample} | {title} MIP")
            ax.set_xlabel("Y (um)" if col == 2 else "X (um)")
            ax.set_ylabel("Y (um)" if col == 0 else "Depth (um)")
        p = np.asarray(metrics["mass_fraction"])
        axes[row, 3].bar(cfg["z_um"], p, width=7, color="#427ca2")
        axes[row, 3].set(
            xlabel="Depth (um)",
            ylabel="Fraction of total mass",
            xlim=(5, 105),
            title=f"{sample} | native axial mass",
        )
        axes[row, 3].grid(alpha=0.2)

        fig, layer_axes = plt.subplots(2, 5, figsize=(16, 6.6), constrained_layout=True)
        for index, ax in enumerate(layer_axes.ravel()):
            ax.imshow(
                normalized[index],
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=xy_extent,
                interpolation="nearest",
            )
            ax.set_title(f"z = {cfg['z_um'][index]} um | mass {100*p[index]:.1f}%")
            ax.set(xlabel="X (um)", ylabel="Y (um)")
        fig.suptitle(
            f"{sample}: {descriptions[sample]}\nOne whole-volume divisor; linear display [0, 1]; no per-layer normalization"
        )
        fig.savefig(folder / "previews" / "layers_normalized.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6.7), constrained_layout=True)
        ax.set_facecolor("black")
        ax.imshow(
            depth_rgba(fine, np.asarray(cfg["fine_z_um"])),
            extent=xy_extent,
            interpolation="nearest",
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(10, 100), cmap="turbo"),
            ax=ax,
            label="Depth of peak density (um)",
        )
        ax.set(
            xlabel="X (um)",
            ylabel="Y (um)",
            title=f"{sample} | depth-colored XY view\nColor = fine-geometry depth; opacity = normalized density",
        )
        fig.savefig(folder / "previews" / "depth_colored_xy.png", dpi=160)
        plt.close(fig)
        records.append(
            {
                "sample_id": sample,
                "display_divisor": divisor,
                "display_gamma": 1,
                "per_layer_normalization": False,
                "metrics": metrics,
            }
        )
        if sample == "T03":
            groups = read_json(folder / "geometry.json")["geometry"]
            fig, zoom_axes = plt.subplots(
                3, 2, figsize=(11, 10), constrained_layout=True
            )
            for width, ax in zip((10, 8, 6, 4, 3, 2), zoom_axes.ravel()):
                selected = [item for item in groups if item["width_px"] == width]
                boxes = np.array(
                    [item["bbox_xy_one_based"] for item in selected], dtype=int
                )
                left, top = boxes[:, :2].min(axis=0) - 1
                right, bottom = boxes[:, 2:].max(axis=0)
                margin = 3
                crop = xy[
                    top - margin : bottom + margin, left - margin : right + margin
                ]
                ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"Width = gap = {width} px = {width*pitch:.3f} um")
                ax.set_axis_off()
            fig.suptitle(
                "T03 | six size groups, display-only local zooms\nSame volume normalization; panels enlarged independently, compare the width labels"
            )
            fig.savefig(folder / "previews" / "resolution_groups_zoom.png", dpi=170)
            plt.close(fig)
    overview.suptitle(
        "NEW GROUND TRUTH PREVIEWS ONLY | not network reconstructions\nEach object normalized by its own native-volume maximum; all layers/projections share that divisor",
        fontsize=14,
    )
    overview.savefig(figures / "overview_normalized.png", dpi=150)
    plt.close(overview)
    # Compact inline preview; rendered from the same scientific figure data,
    # not retouched after export or generated by an image model.
    image_paths = sorted(root.glob("*/previews/*.png")) + [
        figures / "overview_normalized.png"
    ]
    if any(sha256(root / path) != digest for path, digest in before.items()):
        raise RuntimeError("Truth content changed during plotting")
    write_json(
        root / "display_manifest.json",
        {
            "mode": "whole-volume normalization; linear grayscale; no clipping or per-layer rescaling",
            "truth_files_unchanged": True,
            "truth_sha256": before,
            "objects": records,
            "images": [str(p.relative_to(root)) for p in image_paths],
            "fine_geometry_note": "1 um geometry quadrature is not optical depth resolution",
        },
    )
    audit = read_json(root / "source_audit.json")
    preservation = read_json(root / "source_preservation.json")
    lines = [
        "# 新版数据真值预览与迁移核对",
        "",
        "> 本轮只有 GT：不是网络输出、散斑或 RL 重建。形态尚待用户确认。",
        "",
        "## 新版划分（草案，未切换训练）",
        "",
        "| 集合 | 对象 | 数量 |",
        "|---|---|---:|",
        "| 训练 | P01–P11（旧 T01 → P11；P07、P09 回归训练） | 11 |",
        "| 验证 | V01、V02、V03 | 3 |",
        "| 测试 | T02、T03、T04 | 3 |",
        "",
        "本轮旧 T01 文件夹和内部编号均未修改；migration_plan.csv 仅记录迁移方案。旧数据、历史配置及基线均保留。",
        "",
        "## 清晰预览",
        "",
        "![归一化总览](previews/overview_normalized.png)",
        "",
        "每个对象以整个原生真值体的最大值归一化，切片与三个投影共用同一系数，线性显示 0–1。没有逐层归一化、阈值清背景或改写浮点真值。",
        "",
        "三维图使用真实物理轴比例；深度着色图来自 1 μm 几何采样，不代表 1 μm 的光学轴向分辨率。",
        "",
        "| 对象 | 几何验收 | 有质量的原生深度 / μm |",
        "|---|---|---|",
    ]
    for record in records:
        sample, metrics = record["sample_id"], record["metrics"]
        text = {
            "T03": "36 根条纹；12 个横/竖三线组；2–10 px 线宽",
            "T04": "15 节点、22 边、8 个网孔；连通且非共面",
            "V03": "60 颗独立实心球；与 P08 不复用位置",
        }[sample]
        depths = np.atleast_1d(metrics["occupied_z_um"])
        lines.append(
            f"| {sample} | {text} | {'、'.join(f'{float(z):g}' for z in depths)} |"
        )
    for sample in NEW_OBJECTS:
        lines += [
            "",
            f"### {sample}",
            "",
            f"![单层归一化]({sample}/previews/layers_normalized.png)",
            "",
            f"![三维双视角]({sample}/previews/view3d_physical.png)",
            "",
            f"![深度着色]({sample}/previews/depth_colored_xy.png)",
        ]
        if sample == "T03":
            lines += [
                "",
                "局部图仅为读图放大，面板放大倍率不同；分辨率大小以标注的像素和微米线宽为准。",
                "",
                "![六档线组](T03/previews/resolution_groups_zoom.png)",
            ]
    lines += [
        "",
        "## 核验及后续边界",
        "",
        f"- 历史源目录：`{audit['source_root']}`。逐项检查了 14 个对象清单列出的文件存在性与大小。",
        f"- 发现 {len(audit['stale_sample_dir_references'])} 处对象路径引用仍指向旧目录；训练配置也尚未切换。本轮只记录，不静默改写。",
        f"- 生成前后核对 {preservation['tree_file_count']} 个源文件/链接的大小、mtime 和类型，元数据不变；另核对 {preservation['selected_hash_count']} 个关键文件的 SHA-256，内容不变。未声称对全部历史帧进行完整哈希复验。",
        "- MATLAB 新旧几何测试和完整 Python unittest 日志位于 logs/；本轮 CUDA_VISIBLE_DEVICES 为空，未启动 GPU 计算。",
        "- 后续唯一允许使用物理 0 号 A40；忙则等待，不共用、不转用其他卡。绑定前须核对型号、UUID、显存和计算进程。",
        "- 等用户确认三个形态后，再迁移旧 T01 为 P11，同步 MAT 内部编号、路径与清单；保持浮点数组、100 帧与原子集不变。",
        "- 后续新对象沿用 100 帧、10 个 10/90 子集及 RL3；本轮不生成、不训练。有 Set 的 sqrt 方案继续作为网络基线。",
        "- 新旧划分的指标不能直接混算；预览检查只用于对象有效性，不据此调整网络来迎合测试集。",
        "",
    ]
    (root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    cards = "\n".join(
        f'<section><h2>{html.escape(str(p.relative_to(root)))}</h2><a href="{html.escape(str(p.relative_to(root)), quote=True)}"><img loading="lazy" src="{html.escape(str(p.relative_to(root)), quote=True)}"></a></section>'
        for p in image_paths
    )
    (root / "gallery.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V3 真值预览</title><style>body{margin:24px auto;max-width:1600px;font:16px sans-serif;background:#eee}img{max-width:100%;height:auto}section{background:white;padding:18px;margin:24px 0}h2{font-size:18px}</style><h1>三个新对象的 CPU 真值预览</h1><p>待形态确认；没有生成散斑、RL 或网络输出。只允许后续使用物理 0 号 A40。</p>'
        + cards
        + "</html>",
        encoding="utf-8",
    )
    print(f"Saved {len(image_paths)} PNGs and review report: {root / 'report_zh.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
