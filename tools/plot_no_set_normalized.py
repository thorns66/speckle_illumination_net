"""Redraw saved Set/no-Set results clearly on CPU; preserve all source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from tools.report_no_set_ablation import (
    BASELINE,
    OBJECTS,
    load_prediction,
    load_truth,
    plot_object,
    read_metrics,
)
from utils.display_normalization import normalize_display_volume
from utils.experiment_paths import next_experiment_path


ROOT = Path(__file__).resolve().parents[1]


def read_display_run(path: Path) -> dict:
    complete = json.loads((path / "training_complete.json").read_text())
    contract = json.loads((path / "run_contract.json").read_text())
    rows = read_metrics(path / "test_metrics.csv")
    expected = {
        (sample, subset) for sample in OBJECTS["test"] for subset in range(1, 11)
    }
    if (
        not complete["complete"]
        or len(rows) != 30
        or {(row["sample_id"], row["subset_index"]) for row in rows} != expected
        or any(
            row["step"] != complete["best_step"] or row["split"] != "test"
            for row in rows
        )
    ):
        raise ValueError("Expected one complete frozen 30-item test evaluation")
    return {"path": path, "complete": complete, "contract": contract, "test": rows}


def source_hashes(runs: tuple[dict, dict]) -> dict:
    result = {}
    for run in runs:
        paths = [
            run["path"] / name
            for name in (
                "checkpoint_best.pt",
                "checkpoint_last.pt",
                "training_metrics.csv",
                "validation_metrics.csv",
                "test_metrics.csv",
                "training_complete.json",
            )
        ]
        for row in run["test"]:
            paths.append(
                run["path"]
                / "test"
                / f"step_{row['step']:06d}"
                / row["sample_id"]
                / f"subset_{row['subset_index']:02d}"
                / "reconstruction.npy"
            )
        for path in paths:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def plot_overview(
    reference: dict, candidate: dict, destination: Path, gamma: float
) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)
    for row_index, sample in enumerate(OBJECTS["test"]):
        volumes = [load_truth(Path(reference["contract"]["dataset_root"]), sample)]
        for run in (reference, candidate):
            row = next(
                item
                for item in run["test"]
                if item["sample_id"] == sample and item["subset_index"] == 1
            )
            volumes.append(load_prediction(run, row))
        for column, volume in enumerate(volumes):
            value, _ = normalize_display_volume(volume)
            axis = axes[row_index, column]
            plotted = axis.imshow(value.max(0), cmap="magma", norm=norm)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    ("Ground truth", "With Set branch", "Without Set branch")[column],
                    fontsize=16,
                )
            if column == 0:
                axis.set_ylabel(f"{sample} / subset 01", fontsize=15)
    fig.colorbar(
        plotted,
        ax=axes,
        shrink=0.65,
        label=f"Normalized intensity; common display gamma={gamma:g}",
    )
    fig.suptitle(
        "XY maximum-intensity projections\nEach WHOLE volume normalized to its own maximum; not absolute brightness",
        fontsize=16,
    )
    name = (
        "overview_normalized_linear.png"
        if gamma == 1
        else "overview_normalized_gamma0p5.png"
    )
    fig.savefig(destination / name, dpi=180)
    plt.close(fig)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    reference = read_display_run(args.baseline.resolve())
    candidate = read_display_run(args.run.resolve())
    if (
        reference["contract"]["dataset_fingerprint"]
        != candidate["contract"]["dataset_fingerprint"]
    ):
        raise ValueError("Cannot compare different frozen datasets")
    before = source_hashes((reference, candidate))
    destination = next_experiment_path(candidate["path"], "normalized_comparison")
    destination.mkdir(exist_ok=False)
    print("OUTPUT_DIR=" + str(destination), flush=True)
    display = []
    # Produce the three representative comparisons first, then all remaining subsets.
    for subset in range(1, 11):
        for sample in OBJECTS["test"]:
            display.append(
                plot_object(
                    reference,
                    candidate,
                    sample,
                    destination,
                    aligned=False,
                    normalized=True,
                    gamma=0.5,
                    subset_index=subset,
                )
            )
            if subset == 1:
                display.append(
                    plot_object(
                        reference,
                        candidate,
                        sample,
                        destination,
                        aligned=False,
                        normalized=True,
                        gamma=1.0,
                        subset_index=subset,
                    )
                )
        print(f"Rendered normalized comparisons for subset {subset:02d}/10", flush=True)
    overviews = [
        plot_overview(reference, candidate, destination, gamma) for gamma in (0.5, 1.0)
    ]
    after = source_hashes((reference, candidate))
    if before != after:
        raise RuntimeError(
            "A source checkpoint, metric file, or reconstruction changed while plotting"
        )
    lines = [
        "# Set 分支消融：归一化清晰对比",
        "",
        "原图暗是显示色标造成的：GT 峰值远高于网络输出，共同原始强度上限把重建细节压暗。这里只重新绘图，不重新训练或推理。",
        "",
        "## 显示规则",
        "",
        "- 每个方法、每个子集的整个三维重建体除以自身最大值，归一化到 0–1；GT 也采用相同规则。",
        "- 一个完整三维体只用一个缩放系数，XY/XZ/YZ 共用同一归一化；不逐层或逐投影重新拉伸。",
        "- 清晰版统一采用显示 gamma=0.5，线性版采用 gamma=1；不裁掉背景、不做阈值分割、不锐化。",
        "- gamma 只改变颜色映射，并不是修改 Taylor 的 raw/sqrt 输入，也不改变原始重建。",
        "- 归一化图用于比较结构、模糊和深度扩散，不能比较方法间的绝对亮度；深度曲线仍由原始重建计算。",
        "- 所有原始重建、权重和指标文件的 SHA-256 在绘图前后完全一致；原评价结果不变。",
        "",
        "## 三个测试对象总览（固定 subset_01）",
        "",
        f"![清晰总览]({overviews[0]})",
        "",
        f"[线性归一化总览]({overviews[1]})",
        "",
    ]
    for sample in OBJECTS["test"]:
        lines.extend(
            [
                f"## {sample}：XY / XZ / YZ 与深度曲线",
                "",
                f"![清晰对比](test_{sample}_subset01_normalized_gamma0p5.png)",
                "",
                f"[查看线性归一化版](test_{sample}_subset01_normalized_linear.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 全部 30 个测试子集",
            "",
            "下表为统一 gamma=0.5 的归一化图，无最优子集挑选。",
            "",
            "|子集|P07|T01|T02|",
            "|---|---|---|---|",
        ]
    )
    for subset in range(1, 11):
        links = [
            f"[{sample}](test_{sample}_subset{subset:02d}_normalized_gamma0p5.png)"
            for sample in OBJECTS["test"]
        ]
        lines.append("|" + "|".join([f"{subset:02d}", *links]) + "|")
    lines.extend(
        [
            "",
            "[打开可缩放完整图集](gallery.html)",
            "",
            "[原始定量对比报告](../comparison_with_set_baseline/report_zh.md)",
            "",
        ]
    )
    (destination / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for record in display:
        title = f"{record['sample_id']} subset {record['subset_index']:02d} / gamma {record['display_gamma']:g}"
        cards.append(
            f'<section><h2>{html.escape(title)}</h2><a href="{record["file"]}"><img loading="lazy" src="{record["file"]}"></a></section>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Set 分支归一化对比</title><style>body{font:16px system-ui;margin:24px;max-width:1500px}img{width:100%;height:auto}section{margin:32px 0}p{line-height:1.6}</style><h1>Set 分支消融：归一化清晰对比</h1><p>每个完整重建体除以自身最大值；共同显示 gamma，不逐层归一化。只能比较结构，不能比较绝对亮度。点击图片查看原尺寸。</p>'
        + "".join(cards)
    )
    (destination / "gallery.html").write_text(page, encoding="utf-8")
    source_files = [
        Path(__file__).resolve(),
        ROOT / "tools/report_no_set_ablation.py",
        ROOT / "utils/display_normalization.py",
    ]
    source_dir = destination / "plot_sources"
    source_dir.mkdir()
    for source in source_files:
        shutil.copy2(source, source_dir / source.name)
    record = {
        "created_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "normalization": "independent whole-volume maximum; one divisor shared by all depths and projections",
        "gamma": "color mapping only; primary=0.5, linear reference=1",
        "cpu_only": True,
        "source_artifacts_unchanged": True,
        "source_sha256": before,
        "display_settings": display,
        "overviews": overviews,
        "plot_source_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in source_files
        },
    }
    (destination / "display_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "comparison_figures": len(display),
                "overview_figures": len(overviews),
                "source_artifacts_unchanged": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
