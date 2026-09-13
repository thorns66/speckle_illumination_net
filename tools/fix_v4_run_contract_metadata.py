"""Correct the stale display-only train-object list in completed V4 run contracts."""
from __future__ import annotations

import json
from pathlib import Path

from tools import three_way_experiment as old
from tools import v4_p12_anchor_compare_experiment as exp


EXPECTED_FINGERPRINT = "5ff731378a5d7a81028da7427bc25b91564c4c87a390694cd8e8ff57b4e9ce50"


def main() -> None:
    preflight = json.loads((exp.OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    if preflight["dataset_fingerprint"] != EXPECTED_FINGERPRINT:
        raise ValueError("Unexpected V4 preflight fingerprint")
    if preflight["split_counts"] != {"train": 120, "validation": 30, "test": 30}:
        raise ValueError("Unexpected V4 split counts")
    corrected = []
    for arm in exp.ARMS:
        folder = exp.OUTPUT / arm
        complete = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
        if int(complete["completed_steps"]) != 400:
            raise ValueError(f"{arm} did not finish training")
        log = (exp.OUTPUT / "logs" / f"train_{arm}.log").read_text(encoding="utf-8")
        if "train=120 val=30 test=30" not in log:
            raise ValueError(f"{arm} training log does not prove the V4 split sizes")
        path = folder / "run_contract.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["dataset_fingerprint"] != EXPECTED_FINGERPRINT or not record.get("p12_in_training"):
            raise ValueError(f"{arm} lacks the independent P12/fingerprint evidence")
        if record["train_objects"] != [f"P{i:02d}" for i in range(1, 12)]:
            raise ValueError(f"{arm} stale metadata does not match the known writer bug")
        record["train_objects"] = [f"P{i:02d}" for i in range(1, 13)]
        record["metadata_correction"] = {
            "reason": "base V3 JSON wrapper overwrote the V4 display list after training",
            "training_data_unchanged": True,
            "evidence": [
                "V4 dataset fingerprint",
                "preflight train split count 120",
                "training log train=120",
                "existing p12_in_training=true",
            ],
            "tool": str(Path(__file__).resolve()),
        }
        old.write_json(path, record)
        corrected.append(str(path))
    old.write_json(exp.OUTPUT / "run_contract_metadata_correction.json", {
        "complete": True, "training_data_unchanged": True, "corrected": corrected,
        "dataset_fingerprint": EXPECTED_FINGERPRINT,
    })
    print(json.dumps({"complete": True, "corrected": corrected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
