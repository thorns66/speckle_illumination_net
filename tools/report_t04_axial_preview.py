"""Display true local axial structure without mixing unrelated XY regions."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import tifffile

from tools.report_dataset_v3_preview import load_truth
from tools.run_dataset_v3_preview import read_json, sha256, write_json
from tools.run_t04_axial_preview import REVISION
from utils.display_normalization import normalize_display_volume

FAMILY_NAMES = {
    "shallow_equal": "A: shallow, 1:1",
    "deep_equal": "B: deep, 1:1",
    "weak_deep": "C: weak deep layer, 1:0.5",
    "single_control": "D: single layer",
}


def local_roi(
    volume: np.ndarray, region: dict, pitch: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("Expected a ZYX volume")
    bounds = region["roi_bounds_xy_um"]
    x, y = np.arange(volume.shape[2]) * pitch, np.arange(volume.shape[1]) * pitch
    ix = np.flatnonzero((x >= bounds[0]) & (x <= bounds[2]))
    iy = np.flatnonzero((y >= bounds[1]) & (y <= bounds[3]))
    if not len(ix) or not len(iy):
        raise ValueError("Empty local ROI")
    return volume[:, iy[:, None], ix], x[ix], y[iy]


def label(region: dict) -> str:
    depths = np.atleast_1d(region["z_um"])
    if len(depths) == 1:
        return f"{region['region_id']} | single z={depths[0]:g} um"
    return (
        f"{region['region_id']} | z={depths[0]:g}/{depths[1]:g} um\n"
        f"center dz={depths[1]-depths[0]:g} um"
    )


def render(root: Path) -> None:
    preview = read_json(root / "preview_report.json")
    if (
        not preview["preview_complete"]
        or preview["morphology_approved"]
        or preview["dataset_complete"]
    ):
        raise ValueError("Only unapproved truth previews can be reported")
    folder = root / "T04"
    figures = folder / "previews"
    cfg = read_json(folder / "config.json")
    regions = read_json(folder / "geometry.json")["geometry"]
    metrics = read_json(folder / "truth_metrics.json")
    source_names = (
        "truth.mat",
        "ground_truth_float.tif",
        "ground_truth_fine_float.tif",
        "config.json",
        "geometry.json",
        "truth_metrics.json",
    )
    before = {name: sha256(folder / name) for name in source_names}
    g, fine = load_truth(folder)
    with h5py.File(folder / "truth.mat", "r") as handle:
        owner = "".join(chr(int(v)) for v in handle["cfg"]["sample_id"][()].reshape(-1))
    if owner != "T04" or cfg["geometry_revision"] != REVISION:
        raise ValueError("Wrong MAT owner or geometry revision")
    for file, expected in (
        ("ground_truth_float.tif", g),
        ("ground_truth_fine_float.tif", fine),
    ):
        if not np.array_equal(tifffile.imread(folder / file), expected):
            raise ValueError(f"Float TIFF/MAT mismatch: {file}")
    norm, divisor = normalize_display_volume(g)
    fine_norm, fine_divisor = normalize_display_volume(fine)
    pitch = cfg["object_pixel_pitch_um"]
    z = np.asarray(cfg["z_um"])
    ncols = len(cfg["axial_board"]["separations_um"])
    if len(regions) != 4 * ncols or metrics["region_count"] != len(regions):
        raise ValueError("Region layout and geometry metrics disagree")
    xy_extent = [-pitch / 2, 259.5 * pitch, 259.5 * pitch, -pitch / 2]

    fig, ax = plt.subplots(figsize=(8.5, 8.8), constrained_layout=True)
    ax.imshow(
        norm.max(axis=0),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=xy_extent,
        interpolation="nearest",
    )
    for region in regions:
        x0, y0, x1, y1 = region["roi_bounds_xy_um"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#37cacc",
                linewidth=0.8,
            )
        )
        ax.text(
            region["center_xy_um"][0],
            y0 - 4,
            region["region_id"],
            ha="center",
            va="bottom",
            color="#37cacc",
            fontsize=12,
        )
    ax.set(
        xlabel="X (um)",
        ylabel="Y (um)",
        title=f"T04 | XY map of {len(regions)} independent test regions\nTwo layers overlap exactly in XY; boxes and labels are display annotations",
    )
    fig.savefig(figures / "region_map_xy.png", dpi=155)
    plt.close(fig)

    # The primary overview uses LOCAL XZ projections, never whole-field XZ.
    for value, mode, z_limits in (
        (norm, "native", (5, 105)),
        (fine_norm, "fine", (5, 105)),
    ):
        fig, axes = plt.subplots(
            4, ncols, figsize=(3.4 * ncols, 15.8), constrained_layout=True
        )
        for region, ax in zip(regions, axes.ravel()):
            roi, xs, _ = local_roi(value, region, pitch)
            extent = [
                xs[0] - pitch / 2 - region["center_xy_um"][0],
                xs[-1] + pitch / 2 - region["center_xy_um"][0],
                z_limits[1],
                z_limits[0],
            ]
            ax.imshow(
                roi.max(axis=1),
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=extent,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set(
                title=label(region),
                xlabel="Local X (um)",
                ylabel="Depth Z (um)",
                yticks=[20, 40, 60, 80, 100],
            )
            if region["column"] == 1:
                ax.text(
                    -0.8,
                    0.5,
                    FAMILY_NAMES[region["family"]],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
        grid_label = (
            "10 um native slabs: dz=10 pairs touch; NO sampled valley between layers"
            if mode == "native"
            else "1 um fine-geometry quadrature (NOT optical resolution)"
        )
        fig.suptitle(
            f"T04: overlapping axial line pairs + single-layer controls\n{grid_label} | shared whole-volume normalization; no per-region or per-layer scaling",
            fontsize=13,
        )
        fig.savefig(figures / f"local_xz_{mode}.png", dpi=145)
        plt.close(fig)

    local_profiles = []
    for region in regions:
        roi, _, _ = local_roi(g, region, pitch)
        local_profiles.append(roi.sum(axis=(1, 2), dtype=np.float64))
    local_profiles = np.asarray(local_profiles)
    profile_divisor = float(local_profiles.max())
    fig, axes = plt.subplots(
        4,
        ncols,
        figsize=(4.2 * ncols, 11.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    profile_rows = []
    for region, profile, ax in zip(regions, local_profiles, axes.ravel()):
        relative = profile / profile_divisor
        container = ax.stem(z, relative, basefmt=" ")
        plt.setp(container.markerline, markersize=5, color="#165f91")
        plt.setp(container.stemlines, linewidth=2, color="#165f91")
        ax.set(
            title=label(region),
            xlim=(5, 105),
            ylim=(-0.025, 1.08),
            xticks=[20, 40, 60, 80, 100],
        )
        ax.grid(alpha=0.2)
        if region["row"] == 4:
            ax.set_xlabel("Depth Z (um)")
        if region["column"] == 1:
            ax.set_ylabel("Relative native-slab mass")
        for depth, mass, display in zip(z, profile, relative):
            profile_rows.append(
                {
                    "region_id": region["region_id"],
                    "z_um": float(depth),
                    "native_slab_mass": float(mass),
                    "display_relative_mass": float(display),
                }
            )
    fig.suptitle(
        f"T04 | local axial truth profiles, no interpolation; dz=10 has NO interior sample\nOne common mass divisor across all {len(regions)} regions; weak-layer ratio remains 0.5",
        fontsize=14,
    )
    fig.savefig(figures / "local_z_profiles.png", dpi=155)
    plt.close(fig)
    with (root / "local_axial_profiles.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(
            norm[k],
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=xy_extent,
            interpolation="nearest",
        )
        ax.set(title=f"z = {z[k]:g} um", xlabel="X (um)", ylabel="Y (um)")
    fig.suptitle(
        "T04 | native truth layers, shared linear display [0,1]\nNo per-layer normalization; no reconstruction has been run"
    )
    fig.savefig(figures / "layers_normalized.png", dpi=150)
    plt.close(fig)

    # A rotated 3D line diagram exposes both depths without hiding the weaker
    # plane. This is explicitly a geometry diagram, not a reconstruction.
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    for region in regions:
        x, y = region["center_xy_um"]
        for depth, amplitude in zip(
            np.atleast_1d(region["z_um"]), np.atleast_1d(region["amplitudes"])
        ):
            ax.plot(
                [x - 20, x + 20],
                [y, y],
                [depth, depth],
                linewidth=4,
                color=plt.get_cmap("turbo")((depth - 10) / 90),
                alpha=float(amplitude),
            )
    ax.set(
        xlim=(0, 292),
        ylim=(0, 292),
        zlim=(105, 5),
        xlabel="X (um)",
        ylabel="Y (um)",
        zlabel="Depth Z (um)",
    )
    ax.set_box_aspect((292, 292, 100))
    ax.view_init(elev=24, azim=-55)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(10, 100), cmap="turbo"),
        ax=ax,
        shrink=0.65,
        label="Depth (um)",
    )
    ax.set_title(
        f"T04 | geometry diagram, all {metrics['line_count']} centerlines\nColor = depth; opacity = amplitude; line thickness is schematic"
    )
    fig.savefig(figures / "geometry_3d_labeled.png", dpi=145)
    plt.close(fig)

    if any(sha256(folder / name) != digest for name, digest in before.items()):
        raise RuntimeError("Truth modified during scientific plotting")
    images = sorted(figures.glob("*.png"))
    for path in images:
        with Image.open(path) as im:
            im.verify()
    write_json(
        root / "artifact_verification.json",
        {
            "mat_owner_correct": True,
            "float_tiff_mat_exact": True,
            "truth_hashes_unchanged": True,
            "truth_sha256": before,
            "valid_png_count": len(images),
            "native_shape_zyx": list(g.shape),
            "fine_shape_zyx": list(fine.shape),
            "native_display_divisor": divisor,
            "fine_display_divisor": fine_divisor,
            "local_profile_shared_divisor": profile_divisor,
            "local_profile_rows": len(profile_rows),
            "separations_um": metrics["separations_um"],
            "adjacent_native_pair_count": metrics["adjacent_native_pair_count"],
            "native_valley_interpolation": False,
            "display_gamma": 1,
            "per_region_normalization": False,
            "images": [str(p.relative_to(root)) for p in images],
        },
    )
    tests = read_json(root / "test_summary.json")
    lines = [
        "# T04 修订：轴向双层短线真值预览",
        "",
        "> 只有真值，没有散斑、RL 或网络输出。等待形态确认；本轮使用 CPU。",
        "",
        "## 样本定义",
        "",
        "16 个独立区域，共 12 对双层短线和 4 个单层对照，总计 28 条短线。每条名义长度 40 μm、宽度 6 μm，XY 按像素面积积分；Z 为 σ=1 μm、3σ 截断的高斯薄层。两层 XY 图案完全重合。",
        "",
        "| 分组 | 第 1 列 | 第 2 列 | 第 3 列 | 第 4 列 |",
        "|---|---|---|---|---|",
        "| A：浅层 1:1 | 20/30 μm | 20/40 μm | 20/50 μm | 20/60 μm |",
        "| B：深层 1:1 | 50/60 μm | 50/70 μm | 50/80 μm | 50/90 μm |",
        "| C：深层较弱 1:0.5 | 30/40 μm | 30/50 μm | 30/60 μm | 30/70 μm |",
        "| D：单层对照 | 30 μm | 50 μm | 70 μm | 90 μm |",
        "",
        "双层的四列中心间距分别为 10、20、30、40 μm（不是薄层边缘空隙）。单层用来检查轴向展宽、拖尾和凭空出现的假层；保留原来的 30/50/70 μm 并补充 90 μm 对照。几何参数固定，不根据网络结果挑选。",
        "",
        "10 μm 档分别为 A1=20/30、B1=50/60、C1=30/40 μm。细几何中均为分开的两层；但原生 Z 间隔为 10 μm，经 slab 积分后两层相邻接触，中间没有谷值采样点。这是相邻层挑战，不可用平滑/插值制造双峰，也不能据此仅靠原生曲线的峰谷比宣称实现 10 μm 光学分辨率。可以比较局部深度质量分配、轴向误差、弱层恢复与层外泄漏，并结合较大间距和单层对照评估轴向性能。",
        "",
        "## 主要读图方式",
        "",
        "不要用整幅 XZ/YZ 投影判别双层：不同 XY 区域会混在一起。以下侧视图和曲线均来自各自固定 ROI。",
        "",
        "![局部原生侧视](T04/previews/local_xz_native.png)",
        "",
        "![局部深度曲线](T04/previews/local_z_profiles.png)",
        "",
        "全部原生切片/局部侧视图使用同一个原生体最大值，线性显示；强弱层仍保持 1:0.5。细几何侧视单独使用全细体最大值。局部曲线只除以全体 16 个区域共同的最大层质量，不逐区归一化，也不插值造双峰。",
        "",
        "![XY 区域地图](T04/previews/region_map_xy.png)",
        "",
        "![原生单层](T04/previews/layers_normalized.png)",
        "",
        "![立体几何图](T04/previews/geometry_3d_labeled.png)",
        "",
        "[细几何局部 XZ](T04/previews/local_xz_fine.png) · [真实三维等值面双视角](T04/previews/view3d_physical.png)",
        "",
        "1 μm 网格只用于几何积分；实际原生输出仍是 10 μm 间隔。细几何薄层不能证明系统具备 1 μm 光学轴向分辨率，真值中两峰分开也不是网络已经能分开的证据。",
        "",
        "## 核验和保护",
        "",
        f"- 16 区域/12 对/4 单层/28 短线全部通过几何检查；四组弱层质量比为 0.5。细粗质量相对误差 {metrics['fine_to_coarse_mass_relative_error']:.3g}。",
        f"- 细几何有 {metrics['fine_component_count']} 个独立连通体；原生体有 {metrics['native_component_count']} 个，因为 3 对 10 μm 双层各自相邻接触。逐 ROI 同时检查两层位置、无中间采样和细几何的真实分离，不把原生接触误判成样本生成失败。",
        f"- MATLAB：{tests['matlab']['passed']}/{tests['matlab']['total']} 项通过。Python unittest：{tests['python']['passed']} 项通过、{tests['python']['skipped']} 项跳过、0 项失败。",
        "- 跳过原因逐项保存在 test_summary.json。历史数据目录更名导致的旧路径集成测试、以及缺少外部 fixture 的测试，不在本轮静默改写。",
        f"- 10 页与 100 页浮点 TIFF 均与 MAT 对应数组逐元素一致；6 个真值/配置文件绘图前后 SHA-256 不变；{len(images)} 张 PNG 已验证可解码。",
        "- T03、V03、被替换的网状 T04 和不含 10 μm 的上一版双层板都原样保留。本轮仅新增 T04，不改训练数据，不执行 T01→P11 迁移。关联清单见 replacement_manifest.json，新版划分草案仍标记不可训练。",
        "- 启动器在绘图后核对旧预览全部文件内容哈希、历史数据文件元数据和有 Set 基线配置；最终结果见 source_preservation.json。未声称对全部历史训练帧进行完整内容哈希。",
        "- 本轮 CUDA_VISIBLE_DEVICES 为空，所有 GPU 隐藏。后续如获批准进行样本仿真，仅允许空闲的物理 0 号 A40；忙则等待，不共用、不转卡。",
        "- 代码快照、指纹和测试日志随日期目录保存；无正式数据完成清单、无仿真批准文件。",
        "",
    ]
    (root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    ordered = [
        figures / name
        for name in (
            "local_xz_native.png",
            "local_z_profiles.png",
            "region_map_xy.png",
            "layers_normalized.png",
            "geometry_3d_labeled.png",
        )
    ]
    ordered += [p for p in images if p not in ordered]
    cards = "\n".join(
        f'<section><h2>{html.escape(p.name)}</h2><a href="{p.relative_to(root).as_posix()}"><img loading="lazy" src="{p.relative_to(root).as_posix()}"></a></section>'
        for p in ordered
    )
    (root / "gallery.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T04 轴向真值预览</title><style>body{max-width:1400px;margin:20px auto;background:#eee;font:16px sans-serif}section{background:white;margin:20px 0;padding:16px}img{max-width:100%;height:auto}h2{font-size:18px}</style><h1>T04：10/20/30/40 μm 双层短线与单层对照</h1><p>只有真值，未进行重建。A/B/C 为双层组，D 为单层组；所有区域共同归一化。第一列为 10 μm 中心间距，原生层相邻接触、无中间谷值采样；细几何分开不代表光学分辨率。</p>'
        + cards
        + "</html>",
        encoding="utf-8",
    )
    print(
        f"Verified {len(regions)} regions, {len(profile_rows)} profile rows, {len(images)} PNGs; truth unchanged"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
