"""Compatibility entry point for the completed V4 anchor analysis."""
from __future__ import annotations

import json

from tools import v4_p12_anchor_analysis as analysis


_structure_row = analysis.evaluation._structure_row


def structure_row_with_legacy_alias(*args, **kwargs):
    row = _structure_row(*args, **kwargs)
    row["gt_nrmse"] = row["gt_raw_nrmse"]
    return row


analysis.evaluation._structure_row = structure_row_with_legacy_alias


if __name__ == "__main__":
    print(json.dumps(analysis.build_report(), indent=2, ensure_ascii=False))
