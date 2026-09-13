"""Final compatibility layer for V4 local-method names and world-size metadata."""
from __future__ import annotations

import json
import math

import numpy as np

from tools import v4_p12_anchor_analysis as analysis


_structure_row = analysis.evaluation._structure_row
_training_audit = analysis._training_audit


def structure_row_with_legacy_alias(*args, **kwargs):
    row = _structure_row(*args, **kwargs)
    row["gt_nrmse"] = row["gt_raw_nrmse"]
    return row


def corrected_training_audit():
    result = _training_audit()
    for arm in analysis.exp.ARMS:
        contract = json.loads(
            (analysis.exp.OUTPUT / arm / "run_contract.json").read_text(encoding="utf-8")
        )
        result[arm]["world_size"] = int(contract["world_size"])
    return result


def corrected_local_summary(tables):
    result = []
    source_names = {
        "mean_rl3": "mean_rl3",
        "taylor_rl3_sqrt": "taylor_rl3",
        **{arm: f"{arm}_final" for arm in analysis.exp.ARMS},
    }

    def finite_mean(rows, key):
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    def truth(value):
        return value is True or str(value).lower() in ("true", "1")

    def rate(rows, key):
        return float(np.mean([truth(row[key]) for row in rows])) if rows else float("nan")

    for method in analysis.METHODS:
        source = source_names[method]
        t02 = [
            row for row in tables["t02_tubes"]
            if row["method"] == source and float(row["threshold_fraction"]) == .1
        ]
        tubes = [row for row in t02 if str(row["tube"]).isdigit()]
        gaps = [row for row in t02 if str(row["tube"]).startswith("gap_")]
        t03 = [
            row for row in tables["t03_lines"]
            if row["method"] == source and float(row["threshold_fraction"]) == .1
        ]
        t04 = [
            row for row in tables["t04_axial"]
            if row["method"] == source and float(row["threshold_fraction"]) == .1
        ]
        pairs = [row for row in t04 if truth(row.get("separation_applicable"))]
        ten = []
        for row in t04:
            try:
                separation = float(row.get("separation_um"))
            except (TypeError, ValueError):
                continue
            if separation == 10:
                ten.append(row)
        result.append({
            "method": method,
            "source_method": source,
            "t02_centerline_localized_fraction": finite_mean(tubes, "centerline_localized_fraction"),
            "t02_false_break_rate": rate(tubes, "false_break"),
            "t02_gap_bridge_rate": rate(gaps, "bridged"),
            "t03_all_three_localized_rate": rate(t03, "all_three_localized"),
            "t03_separated_rate": rate(t03, "separated"),
            "t03_false_break_rate": float(np.mean([
                int(float(row["false_break_count"])) > 0 for row in t03
            ])) if t03 else float("nan"),
            "t04_20_40um_separated_rate": rate(pairs, "separated"),
            "t04_20_40um_local_w1_um": finite_mean(pairs, "local_depth_w1_um"),
            "t04_10um_localized_rate": rate(ten, "all_layers_localized"),
            "t04_10um_expected_energy_fraction": finite_mean(
                ten, "expected_layer_energy_fraction"
            ),
        })
    return result


analysis.evaluation._structure_row = structure_row_with_legacy_alias
analysis.local_summary = corrected_local_summary
analysis._training_audit = corrected_training_audit


if __name__ == "__main__":
    print(json.dumps(analysis.build_report(), indent=2, ensure_ascii=False))
