"""Run Mean-anchor training/evaluation against the frozen 17-object V3 view."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools import mean_anchor_experiment as exp
from tools import v3_compare_experiment as base


def select_data() -> Path:
    value = os.environ.get("MEAN_ANCHOR_DATA")
    if not value:
        raise ValueError("MEAN_ANCHOR_DATA is required")
    path = Path(value).resolve()
    if not (path / "dataset_splits.json").is_file():
        raise FileNotFoundError(path)
    exp.DATA = path
    base.DATA = path
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "infer", "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    select_data()
    if args.stage == "train":
        exp.run_train(exp.OUTPUT / f"{exp.ARM}.yaml", resume=args.resume)
    elif args.stage == "infer":
        from tools import v3_compare_evaluation as generic
        previous = generic.exp
        generic.exp = exp
        try:
            generic.evaluate_arm(exp.ARM, resume=args.resume)
        finally:
            generic.exp = previous
        from tools.mean_anchor_analysis import augment_evaluation
        augment_evaluation()
    else:
        from tools.mean_anchor_analysis import build_report
        build_report()


if __name__ == "__main__":
    main()

