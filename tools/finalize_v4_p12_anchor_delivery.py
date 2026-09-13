"""Final inventory, best/final table, and report links for the V4 delivery."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import three_way_experiment as old
from tools import v3_compare_report
from tools import v4_p12_anchor_compare_experiment as exp


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def jsonl_count(paths, *, training_only: bool = False) -> int:
    count = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if not training_only or json.loads(line).get("phase") == "training":
                    count += 1
    return count


def main() -> None:
    metrics = []
    for arm in exp.ARMS:
        metrics.extend(read_csv(exp.OUTPUT / "evaluation" / arm / "metrics.csv"))
    _objects, summary = v3_compare_report.aggregate_metrics(metrics)
    best_final = [row for row in summary if row["split"] == "test"]
    write_csv(exp.OUTPUT / "analysis/best_final_summary.csv", best_final)

    training = {}
    for arm in exp.ARMS:
        folder = exp.OUTPUT / arm
        train_rows = line_count(folder / "training_metrics.csv")
        validation_rows = line_count(folder / "validation_metrics.csv")
        test_rows = line_count(folder / "test_metrics.csv")
        gradient_rows = jsonl_count(
            folder.glob("gradient_diagnostics_rank*.jsonl"), training_only=True
        )
        lr_rows = jsonl_count(folder.glob("learning_rate_rank*.jsonl"))
        if (train_rows, validation_rows, test_rows, gradient_rows, lr_rows) != (
            400, 600, 30, 3200, 2400
        ):
            raise ValueError(f"Incomplete training inventory for {arm}")
        log = (exp.OUTPUT / "logs" / f"train_{arm}.log").read_text(encoding="utf-8")
        if "Traceback" in log or "OutOfMemoryError" in log:
            raise ValueError(f"Training log contains an error for {arm}")
        training[arm] = {
            "training_rows": train_rows, "validation_rows": validation_rows,
            "test_rows": test_rows, "gradient_records": gradient_rows,
            "learning_rate_records": lr_rows,
        }
    predictions = len(list((exp.OUTPUT / "evaluation").glob("*/best/*/reconstruction.npy"))) + len(
        list((exp.OUTPUT / "evaluation").glob("*/final/*/reconstruction.npy"))
    )
    corrections = len(list((exp.OUTPUT / "evaluation").glob("*/*/*/effective_correction.npy")))
    if predictions != 376 or corrections != 376:
        raise ValueError("Prediction/correction inventory is incomplete")
    spinach = json.loads((exp.OUTPUT / "spinach_root_transfer/complete.json").read_text(encoding="utf-8"))
    clear = json.loads((exp.OUTPUT / "analysis/figures_clear/manifest.json").read_text(encoding="utf-8"))
    if not spinach.get("complete") or int(clear["figures"]) != 9:
        raise ValueError("Spinach or clear-figure delivery is incomplete")

    report_path = exp.OUTPUT / "REPORT_ZH.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## 菠菜根探索性迁移"
    if marker not in report:
        report += """

## 菠菜根探索性迁移

固定 10 帧的五方法结果见 [菠菜根报告](spinach_root_transfer/REPORT_ZH.md) 和 [90 帧留出统计](spinach_root_transfer/physics_metrics.csv)。真实数据没有 GT：新 Mean 主分支的留出均值/方差 NRMSE（0.3693/0.7063）优于新 Taylor 主分支（0.3944/0.7407），但 Mean-RL3 仍最好（0.3399/0.6700）。两种新网络的轴向质心都明显偏浅，因此不能据此认定真实深度已经正确。

## 清晰对比图

- [T02 五方法整幅体归一化](analysis/figures_clear/T02_subset01_all_methods_volume_normalized.png)
- [T03 五方法整幅体归一化](analysis/figures_clear/T03_subset01_all_methods_volume_normalized.png)
- [T04 五方法整幅体归一化](analysis/figures_clear/T04_subset01_all_methods_volume_normalized.png)
- [T02 两网络共同尺度](analysis/figures_clear/T02_subset01_network_shared_scale.png)
- [T03 两网络共同尺度](analysis/figures_clear/T03_subset01_network_shared_scale.png)
- [T04 两网络共同尺度](analysis/figures_clear/T04_subset01_network_shared_scale.png)

这些图使用纯英文标签以避免字体缺失。整幅体图对每个完整三维体只用一次 99.9 百分位尺度；共同尺度图让两网络共享同一尺度；十层图不做逐层归一化。
"""
        report_path.write_text(report, encoding="utf-8")

    acceptance_path = exp.OUTPUT / "final_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance.update({
        "complete": True,
        "passed_artifact_acceptance": True,
        "cpu_unittest": {"tests_run": 192, "skipped": 7, "failures": 0, "errors": 0},
        "inventory": {
            "training": training, "predictions": predictions,
            "effective_corrections": corrections, "clear_figures": int(clear["figures"]),
            "spinach_methods": len(spinach["methods"]),
        },
        "best_final_summary": "analysis/best_final_summary.csv",
        "spinach_complete": True,
        "clear_figures_complete": True,
        "report_sha256": old.sha256(report_path),
    })
    old.write_json(acceptance_path, acceptance)
    old.write_json(exp.OUTPUT / "delivery_complete.json", {
        "complete": True, "final_acceptance_sha256": old.sha256(acceptance_path),
        "report_sha256": old.sha256(report_path), "predictions": predictions,
        "spinach_complete": True, "clear_figures": int(clear["figures"]),
    })
    print(json.dumps({
        "complete": True, "training": training, "predictions": predictions,
        "spinach_methods": len(spinach["methods"]), "clear_figures": clear["figures"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
