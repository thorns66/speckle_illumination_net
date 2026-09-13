"""Render the code-verified Mean-anchor architecture using vector primitives."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiment_paths import next_experiment_path

AUDIT = Path("/workspace/xyx/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(AUDIT))
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = {
    "mean": "#267C79", "mean_light": "#E9F3F0",
    "var": "#78649D", "var_light": "#F0EBF7",
    "set": "#AF7C33", "set_light": "#FCF2E3",
    "ink": "#253442", "muted": "#61717C", "line": "#AAB6BD",
    "neutral": "#F2F5F7", "white": "#FFFFFF",
}


def text(ax, x, y, value, *, size=7.0, color="ink", ha="center", weight="normal", **kw):
    return ax.text(x, y, value, fontsize=size, color=COLORS.get(color, color),
                   ha=ha, va="center", fontweight=weight, linespacing=1.45, **kw)


def box(ax, x, y, w, h, *, edge="line", fill="white", lw=0.7, radius=1.1):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=COLORS.get(fill, fill), edgecolor=COLORS.get(edge, edge),
        linewidth=lw, zorder=1)
    ax.add_patch(patch)
    return patch


def arrow(ax, points, *, color="muted", lw=0.8, dashed=False):
    points = np.asarray(points)
    if len(points) > 2:
        ax.plot(points[:-1, 0], points[:-1, 1], color=COLORS.get(color, color),
                linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-", zorder=0)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
        mutation_scale=6, linewidth=lw, color=COLORS.get(color, color),
        linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=0))


def cuboid(ax, x, y, w, h, depth, family):
    edge = COLORS[family]
    pale = COLORS[f"{family}_light"]
    ax.add_patch(Polygon([(x, y+h), (x+depth, y+h+depth),
                          (x+w+depth, y+h+depth), (x+w, y+h)],
                         facecolor=pale, edgecolor=edge, lw=0.6))
    ax.add_patch(Polygon([(x+w, y), (x+w+depth, y+depth),
                          (x+w+depth, y+h+depth), (x+w, y+h)],
                         facecolor=pale, edgecolor=edge, lw=0.6))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=pale, edgecolor=edge, lw=0.7))
    for position in np.linspace(x+w*.25, x+w*.75, 3):
        ax.plot([position, position], [y+.9, y+h-.9], lw=.4, color=edge, alpha=.4)


def panel(ax, letter, title, height):
    ax.set_xlim(0, 177)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_label(letter)
    text(ax, 0, height-2.5, letter, size=10, weight="bold", ha="left")
    text(ax, 6, height-2.5, title, size=8.7, weight="bold", ha="left")


def feature_panel(ax):
    panel(ax, "a", "Three-branch feature extraction and multiscale residual decoding", 77)
    text(ax, 36, 66.5, "Physics-derived inputs", size=6.7, color="muted")
    text(ax, 78, 66.5, "Feature pyramids", size=6.7, color="muted")
    text(ax, 118, 66.5, "Gated fusion", size=6.7, color="muted")
    text(ax, 148, 66.5, "3D decoder", size=6.7, color="muted")

    # Abstract frame icon; deliberately contains no simulated image data.
    for shift in (3.0, 1.5, 0.0):
        ax.add_patch(Rectangle((1+shift, 34+shift), 10, 12,
            facecolor=COLORS["neutral"], edgecolor=COLORS["muted"], lw=.65))
    text(ax, 8, 28, "10 input\nlight-field frames", size=6.6)
    text(ax, 8, 18.5, r"$I_1,\ldots,I_{10}$", size=8)
    arrow(ax, [(14, 41), (18, 41), (18, 56), (22, 56)], color="mean")
    arrow(ax, [(14, 41), (18, 41), (18, 37), (22, 37)], color="var")
    arrow(ax, [(14, 41), (18, 41), (18, 15), (22, 15)], color="set")

    for y, family, label, formula in (
        (50, "mean", "Mean-RL3", r"$g_m=\mathrm{RL3}_{H}(\mu_{10})$"),
        (31, "var", "Taylor-RL3-sqrt", r"$g_v=\sqrt{\mathrm{RL3}(v_{10})}$"),
        (9, "set", "Centered frames", r"$R_i=I_i-\mu_{10}$"),
    ):
        box(ax, 22, y, 29, 12, edge=family, fill=f"{family}_light")
        text(ax, 36.5, y+8.5, label, size=7, weight="bold", color=family)
        text(ax, 36.5, y+3.5, formula, size=8)
    text(ax, 36.5, 46.6, "Mean reconstruction anchor → b", size=6.0, color="mean")
    for y, family in ((56,"mean"), (37,"var")):
        arrow(ax, [(51,y), (56,y)], color=family)
        for j, (w,h) in enumerate(((5,9),(6.5,7),(8,5))):
            x = 58 + j*14
            cuboid(ax, x, y-h/2, w, h, 1.4, family)
            text(ax, x+w/2, y-7, str((16,32,64)[j]), size=6.4, color=family)
            if j < 2:
                arrow(ax, [(x+w+1.8,y), (x+13,y)], color=family, lw=.65)
        text(ax, 78, y+8.2, "RMS norm  ·  3D CNN", size=6.2, color=family)
        arrow(ax, [(97,y), (105,y)], color=family)

    box(ax, 56, 7, 42, 17, edge="set", fill="set_light")
    text(ax, 77, 20.5, "Shared 2D CNN  ·  8 / 16 / 32", size=6.4, color="set", weight="bold")
    text(ax, 77, 15.4, "Feature mean + std over frames", size=6.2)
    text(ax, 77, 10.3, "Depth broadcast + z  →  3D CNN", size=6.2)
    arrow(ax, [(51,15),(56,15)], color="set")
    arrow(ax, [(98,15),(105,15)], color="set")
    text(ax, 77, 3.4, "Permutation invariant", size=6.2, color="set")

    box(ax, 105, 9, 25, 53, fill="neutral")
    text(ax, 117.5, 56, "Mean + Taylor", size=6.5, weight="bold")
    text(ax, 117.5, 51.5, "Concat → 1×1×1", size=6.3)
    text(ax, 117.5, 42.5, "+", size=12, color="set")
    text(ax, 117.5, 35.7, "Gated Set", size=6.5, color="set", weight="bold")
    text(ax, 117.5, 30.7, "projection", size=6.5, color="set")
    text(ax, 117.5, 21, r"$\alpha_s G_s\odot W_s S_s$", size=8)
    text(ax, 117.5, 13.3, "At all 3 scales", size=6.2, color="muted")

    # Deep-to-shallow decoder with explicit same-scale feature skips.
    for y, label in ((54,"64"),(37,"32"),(20,"16")):
        box(ax, 138, y-4, 20, 8, fill="neutral")
        text(ax, 148, y, f"3D block · {label}", size=6.3)
        arrow(ax, [(130,y),(138,y)])
    for y in (54,37):
        arrow(ax, [(148,y-4),(148,y-13)])
        text(ax, 156, y-8.5, "↑XY", size=5.9, color="muted", ha="left")
    box(ax, 138, 7, 20, 6, fill="neutral")
    text(ax, 148, 10, "1×1×1 head", size=6.3)
    arrow(ax, [(148,16),(148,13)])
    arrow(ax, [(158,10),(162,10),(162,20),(165,20)])
    box(ax, 165, 14, 12, 12, edge="mean", fill="mean_light")
    text(ax, 171, 20, r"$r_\theta$", size=11, color="mean")
    text(ax, 170, 9.5, "Signed\nresidual → b", size=6.2)
    text(ax, 131, 3.4, "XY: 1, ½, ¼    |    Z unchanged (10 planes)", size=6.2, color="muted")


def reconstruction_panel(ax):
    panel(ax, "b", "Mean-anchored reconstruction with input-derived intensity calibration", 37)
    box(ax, 0, 8, 30, 18, edge="mean", fill="mean_light", lw=1)
    text(ax, 15, 21.3, "Mean anchor", size=7.1, color="mean", weight="bold")
    text(ax, 15, 14.1, r"$A=\beta_m\,g_m$", size=10, color="mean")
    arrow(ax, [(30,17),(39,17)], color="mean", lw=1.5)
    box(ax, 39, 8, 40, 18, edge="mean", fill="mean_light", lw=1)
    text(ax, 59, 21.3, "Nonnegative correction", size=7, weight="bold")
    text(ax, 59, 14.1, r"$u=P(A,r_\theta)$", size=10)
    text(ax, 59, 29.3, r"$r_\theta$ from decoder", size=8)
    arrow(ax, [(59,27.5),(59,26)])
    arrow(ax, [(79,17),(86,17)], color="mean", lw=1.5)
    box(ax, 86, 8, 31, 18, edge="mean", fill="mean_light", lw=1)
    text(ax, 101.5, 21.3, "Unit-mass shape", size=7, weight="bold")
    text(ax, 101.5, 14.1, r"$q=u\,/\,\sum u$", size=10)
    arrow(ax, [(117,17),(124,17)], color="mean", lw=1.5)
    box(ax, 124, 8, 30, 18, edge="mean", fill="mean_light", lw=1)
    text(ax, 139, 21.3, "E3 calibration", size=7, weight="bold")
    text(ax, 139, 14.1, r"$a=a_0(1+0.2\tanh\gamma)$", size=8)
    text(ax, 139, 29.3, r"$\mu_{10}$ and $H(q)$", size=8)
    arrow(ax, [(139,27.5),(139,26)])
    arrow(ax, [(154,17),(161,17)], color="mean", lw=1.5)
    box(ax, 161, 8, 16, 18, edge="mean", fill="mean", lw=1)
    text(ax, 169, 21.3, "Output", size=7, color="white", weight="bold")
    text(ax, 169, 14.1, r"$\hat g=aq$", size=10, color="white")
    text(ax, 0, 3, "Analytic β from input mean", size=6.3, color="mean", ha="left")
    text(ax, 61, 3, "Zero residual preserves anchor", size=6.3, color="muted")
    text(ax, 140, 3, "Full-volume calibration; no per-plane scaling", size=6.1, color="muted")


def training_panel(ax):
    panel(ax, "c", "Training-only physical supervision from 90 complementary frames", 44)
    box(ax, 0, 12, 29, 19, fill="neutral")
    text(ax, 14.5, 26, "90 target frames", size=7.0, weight="bold")
    text(ax, 14.5, 19, r"$\mu_{90},\;v_{90}$", size=10)
    text(ax, 14.5, 14.2, "Mean / unbiased variance", size=5.8)
    arrow(ax, [(14.5,12),(14.5,9),(93.5,9),(93.5,22),(97,22)], dashed=True)
    box(ax, 36, 12, 54, 19, fill="neutral")
    text(ax, 63, 26, "Forward light-field physics", size=7, weight="bold")
    text(ax, 63, 20, r"$H(q)$   and   $H^2(q^2)$", size=10)
    text(ax, 63, 14.3, "Fixed PSF · normalized moment matching", size=6.0)
    text(ax, 63, 35.1, r"$q,\,a$ from b", size=8)
    arrow(ax, [(63,33.3),(63,31)], dashed=True)
    arrow(ax, [(90,22),(97,22)], dashed=True)
    box(ax, 97, 12, 80, 19, fill="neutral")
    text(ax, 137, 26, "Self-supervised objective", size=7, weight="bold")
    text(ax, 137, 19.8,
         r"$L=L_{\mathrm{gain}}+L_{\mathrm{var}}+c_tL_{\mathrm{mean}}+\lambda_{\mathrm{TV}}L_{\mathrm{TV}}$", size=8.7)
    text(ax, 137, 14.2, "Mean-shape gradient budget ≤ 100% of variance", size=6.0)
    text(ax, 0, 6.2, "Input / target frames disjoint", size=6.4, color="muted", ha="left")
    text(ax, 65, 6.2, "No GT required for training", size=6.4, color="muted")
    text(ax, 135, 6.2, "50-step ramp · TV weight 10⁻⁵ · axial factor 0.5", size=6.1, color="muted")
    text(ax, 0, 0.8, "H² denotes the element-wise squared PSF operator. Solid arrows: inference; dashed arrows: training supervision.", size=6.1, color="muted", ha="left")


CAPTION = """# Mean 主分支网络架构图

图中表示当前 Mean anchor + Taylor/Mean/Set + E3 + mean≤100% 方案的实际结构，不包含实验性能结论。

## 中文读图

**a** 输入10帧光场图像。输入均值经 Mean-RL3 得到 g_m；无偏方差经平方PSF的 Taylor-RL3 后开根号得到 g_v。两路体数据分别进行整个体的 RMS 特征归一化，进入三尺度3D编码器，通道16/32/64。Set分支读取去均值后的10帧，经过共享2D CNN（8/16/32），在帧维度对特征求均值和标准差，再拼接并投影；按深度广播、拼接z/100坐标后通过3D卷积提升到体特征。Set不是Transformer。所有尺度只改变XY，保持10个深度平面。

每尺度融合为 F_s = C_s([V_s,M_s]) + α_s G_s ⊙ W_s S_s；其中 α_s 是可学习sigmoid标量，G_s是由物理特征与Set特征拼接后预测的sigmoid空间门。三尺度融合结果通过深至浅的3D解码器，使用同尺度跳跃拼接、XY上采样和1×1×1输出头，预测有正有负的残差rθ。

**b** Mean主分支落实在 A=β_m g_m。β₀=max(〈H(g_m),μ10〉/(‖H(g_m)‖²+ε), ε)，β_m=β₀[1+0.2tanh(b)]。特征归一化不改变这个物理强度基础。正值映射逐体素为：r≥0时P(A,r)=A+r；r<0时P(A,r)=A exp[r/max(A,ε)]。因此P(A,0)=A，不等价于ReLU(A+r)。对u=P(A,rθ)做整个体的总质量归一化q=u/Σu，再由输入10帧均值计算 a₀=max(〈H(q),μ10〉/max(‖H(q)‖²,ε),0)，a=a₀[1+0.2tanh(γ)]，最终ĝ=aq。校准的解析a₀使用stop-gradient的q，亮度与结构梯度分开。零残差保留Mean结构，但E3可调整最终亮度。

**c** 其余90帧的均值和无偏方差仅用于训练目标，不输入网络推理。平方PSF记为H²，指PSF核逐元素平方后形成的算子，不是H连续作用两次。归一化方差预测来自H²(q²)（当前噪声参数为0）；mean结构项来自归一化H(q)。亮度项比较aH(stopgrad(q))与μ90。总损失包括亮度、归一化对数方差、动态加权mean结构项和TV。mean结构项系数c_t=min[1,b_t‖∇q Lvar‖/(‖∇q Lmean‖+ε)]，b_t=min(t/50,1)，因此其形状梯度预算不超过方差项的100%。TV权重10⁻⁵，轴向系数0.5。示意箭头表示监督数据流；精确stop-gradient规则见本说明。

## Figure legend

**Mean-anchored, physics-guided light-field reconstruction.** **a,** Ten light-field frames provide a Mean-RL3 volume, a square-root Taylor-RL3 volume and a set of centered frames. Separate three-dimensional encoders extract Mean and Taylor features at three lateral scales. A shared two-dimensional frame encoder, permutation-invariant feature mean/standard-deviation pooling and depth-conditioned lifting provide Set features. Gated multiscale fusion and a residual decoder predict a signed volumetric correction. The depth axis is preserved. **b,** The scaled Mean-RL3 volume forms the reconstruction anchor. An anchor-preserving nonnegative map incorporates the signed correction. Whole-volume mass normalization separates shape from intensity, and E3 calibrates the final intensity using only the ten-frame input mean. **c,** Complementary frames provide training-only moment targets through fixed-PSF forward models. Training combines intensity, normalized variance, gradient-budgeted mean-shape and total-variation losses. The squared-PSF operator acts on the squared reconstruction shape. All icons are abstract vector schematics; no reconstruction results or performance claims are shown.

## 图形与使用说明

- 图宽183 mm，英文主图；PDF嵌入字体，SVG保留可编辑文字；PNG/TIFF为600 dpi。
- 采用绿色Mean、紫色Taylor、金色Set，Mean重建路径加粗；三个面板分别解释特征、重建、训练。
- 本图可用于论文排版和继续编辑；具体期刊尺寸与投稿要求需在确定目标期刊后核对。
- 生成只使用CPU，没有读取训练权重或运行网络推理，没有修改训练代码和进程。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or next_experiment_path(ROOT / "outputs", "mean_anchor_architecture")
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "outputs/v5_sim_real_no_p12_mean_anchor_e3_mean100_800_20260911_run01/mean_anchor_e3_mean100.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["reconstruction_anchor"] == "mean_rl3"
    assert config["model"]["set_encoder_type"] == "mean_std"
    assert all(config["ablation"][key] for key in ("use_set_branch", "use_mean_branch", "use_variance_branch", "use_gate"))
    assert config["model"].get("coarse_application") == "legacy_anchor_positive"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.20472441, 6.65354331))
    grid = fig.add_gridspec(3, 1, height_ratios=[77,37,44], hspace=.05,
                           left=3/183, right=180/183, bottom=3/169, top=166/169)
    axes = [fig.add_subplot(grid[i]) for i in range(3)]
    feature_panel(axes[0])
    reconstruction_panel(axes[1])
    training_panel(axes[2])
    fig.canvas.draw()
    basename = output / "mean_anchor_architecture"
    require_matplotlib_panel_alignment(fig, json_out=str(basename)+".alignment.json",
        overlay_svg=str(basename)+".alignment.svg", strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(output / "mean_anchor_architecture.svg")
    fig.savefig(output / "mean_anchor_architecture.pdf")
    fig.savefig(output / "mean_anchor_architecture.png", dpi=600)
    fig.savefig(output / "mean_anchor_architecture.tiff", dpi=600,
                pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(output / "preview.png", dpi=300)
    plt.close(fig)
    (output / "README_zh.md").write_text(CAPTION, encoding="utf-8")
    sources = ["models/configurable_anchor_lfm_net.py", "models/variance_anchored_lfm_net.py",
        "models/gated_fusion.py", "models/set_encoder.py", "models/encoder3d.py",
        "models/decoder3d.py", "models/blocks.py", "tools/v5_mixed_real_anchor_experiment.py",
        "tools/v3_compare_experiment.py", "tools/v5_mixed_dataset.py",
        "tools/draw_mean_anchor_architecture.py"]
    metadata = {
        "created":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "figure_dimensions_mm":[183,169], "raster_dpi":600, "backend":"matplotlib",
        "scientific_content":"Code-verified architecture; no empirical data panels",
        "config":str(config_path), "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_hashes":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        "reconstruction_anchor":"mean_rl3", "set_encoder":"mean_std",
        "inference_input_frames":10, "training_target_frames":90,
        "cpu_only":True,
    }
    (output / "provenance.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    (output / "draw_mean_anchor_architecture.py").write_text(Path(__file__).read_text())
    print(output, flush=True)


if __name__ == "__main__":
    main()
