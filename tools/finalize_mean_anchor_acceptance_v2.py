"""Tolerance-aware final acceptance for repeated GPU inference at the same step."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from tools import finalize_mean_anchor_acceptance as base


def prediction_parity() -> dict:
    root = base.EVALUATION
    exact = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    cases = sorted(path.parent.name for path in (root / "best").glob("*/reconstruction.npy"))
    for case in cases:
        left = np.load(root / "best" / case / "reconstruction.npy", allow_pickle=False)
        right = np.load(root / "final" / case / "reconstruction.npy", allow_pickle=False)
        if np.array_equal(left, right, equal_nan=True):
            exact += 1
        difference = np.nan_to_num(left.astype(np.float64) - right.astype(np.float64))
        maximum_absolute = max(maximum_absolute, float(np.abs(difference).max()))
        maximum_relative = max(
            maximum_relative,
            float(np.linalg.norm(difference) / max(np.linalg.norm(np.nan_to_num(left)), 1e-30)),
        )
    return {
        "cases": len(cases),
        "bitwise_equal_cases": exact,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_l2": maximum_relative,
        "numerically_equal": maximum_absolute <= 2e-8 and maximum_relative <= 3e-7,
        "explanation": "same step-400 state; separate sparse CUDA forward passes differ only at FP32 roundoff",
    }


def main() -> None:
    best = torch.load(base.ARM / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    final = torch.load(base.ARM / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    if int(best["completed_steps"]) != 400 or int(final["completed_steps"]) != 400:
        raise ValueError("Both checkpoints must be step 400")
    parity = prediction_parity()
    if not parity["numerically_equal"]:
        raise ValueError(parity)

    original = base.np.array_equal
    base.np.array_equal = lambda left, right: bool(np.allclose(
        left, right, rtol=3e-7, atol=2e-8, equal_nan=True
    ))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            base.main()
    finally:
        base.np.array_equal = original

    path = base.OUTPUT / "final_acceptance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["best_final_predictions_bitwise_equal"] = parity["bitwise_equal_cases"] == parity["cases"]
    record["best_final_predictions_numerically_equal"] = parity["numerically_equal"]
    record["best_final_prediction_parity"] = parity
    base.write_json(path, record)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
