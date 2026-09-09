"""Run the frozen sqrt/no-Set ablation on idle GPUs with dated output names."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml

from datasets.matlab_multivolume_dataset import load_dataset_index
from train_dataset import _gpu_inventory
from train_volume import _model_from_config
from training.multivolume_trainer import (
    _configure_trainable_parameters,
    _psf_cache_spec,
    _validate_config,
)
from utils.experiment_paths import next_experiment_path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs/multivolume_n10_no_mean_run01"
CONFIG = ROOT / "configs/multivolume_n10_no_set.yaml"


def normalized_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["experiment"].pop("name", None)
    result["experiment"].pop("output_dir", None)
    result["data"].setdefault("var_feature_representation", "sqrt")
    result["ablation"].pop("use_set_branch")
    return result


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_protocol(config_path: Path, baseline: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    reference = yaml.safe_load((baseline / "config_used.yaml").read_text())
    completed = json.loads((baseline / "training_complete.json").read_text())
    contract = json.loads((baseline / "run_contract.json").read_text())
    _validate_config(config)
    if (
        config["ablation"]["use_set_branch"]
        or not reference["ablation"]["use_set_branch"]
    ):
        raise ValueError(
            "Comparison requires a no-Set candidate and a with-Set reference"
        )
    if config["data"].get("var_feature_representation", "sqrt") != "sqrt":
        raise ValueError("This ablation requires sqrt features")
    if normalized_config(config) != normalized_config(reference):
        raise ValueError(
            "Candidate differs from reference beyond Set and experiment identity"
        )
    if not completed["complete"] or completed["completed_steps"] != 200:
        raise ValueError("Reference is not a completed 200-step experiment")
    indexed, fingerprint = load_dataset_index(ROOT / config["data"]["root"])
    if fingerprint != contract["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the completed reference")
    spec = _psf_cache_spec(config, config_path)
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    expected_cache = (
        ROOT / config["runtime"]["psf_cache_dir"] / f"selected_H_{digest}.npy"
    )
    if (
        expected_cache.resolve() != Path(contract["selected_psf_cache"]).resolve()
        or not expected_cache.is_file()
    ):
        raise ValueError("Selected PSF cache differs from the reference")
    torch.manual_seed(int(config["experiment"]["seed"]))
    full = _model_from_config(reference)
    initial_hash = state_hash(full)
    initial_rng = torch.get_rng_state().clone()
    torch.manual_seed(int(config["experiment"]["seed"]))
    candidate = _model_from_config(config)
    parameters = _configure_trainable_parameters(candidate)
    if initial_hash != state_hash(candidate) or not torch.equal(
        initial_rng, torch.get_rng_state()
    ):
        raise ValueError("Ablation changed initialization or RNG consumption")
    return {
        "verified_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "baseline": str(baseline),
        "baseline_best_step": completed["best_step"],
        "dataset_fingerprint": fingerprint,
        "split_items": {split: len(items) for split, items in indexed.items()},
        "selected_psf_cache": str(expected_cache),
        "psf_source_signature": spec,
        "only_effective_config_difference": "ablation.use_set_branch: true -> false",
        "initial_state_sha256": initial_hash,
        "initial_state_and_rng_identical": True,
        "parameter_contract": parameters,
        "reference_world_size": contract["world_size"],
        "selection_rule": "object-macro validation weighted_var_loss + weighted_tv_loss; no GT selection",
    }


def idle_gpus() -> tuple[list[int], list[dict]]:
    inventory = _gpu_inventory()
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    occupied = {
        line.split(",")[0].strip() for line in apps.stdout.splitlines() if line.strip()
    }
    free = [
        int(item["index"])
        for item in inventory
        if item["uuid"] not in occupied
        and int(item["memory_used_mib"]) <= 1024
        and int(item["utilization_percent"]) <= 10
    ]
    return free, inventory


def snapshot_sources(destination: Path, config_path: Path) -> dict:
    files = [
        ROOT / "train_dataset.py",
        ROOT / "train_volume.py",
        ROOT / "infer_dataset.py",
        config_path,
    ]
    for directory in ("models", "training", "datasets", "losses", "physics", "utils"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    files.extend([Path(__file__).resolve(), ROOT / "tests/test_no_set_ablation.py"])
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = destination / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="CPU-only checks; no output writes or GPUs",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = verify_protocol(config_path, args.baseline.resolve())
    if args.check_only:
        print(json.dumps(protocol, indent=2, ensure_ascii=False))
        return
    previous = None
    while True:
        free, inventory = idle_gpus()
        if free:
            break
        state = [(item["index"], item["memory_used_mib"]) for item in inventory]
        if state != previous:
            print(
                "Waiting for idle GPUs; no sharing. Inventory: " + str(state),
                flush=True,
            )
            previous = state
        time.sleep(30)
    config = yaml.safe_load(config_path.read_text())
    destination = next_experiment_path(ROOT / "outputs", config["experiment"]["name"])
    log_path = destination.parent / f"{destination.name}.launcher.log"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_dataset.py"),
        "--config",
        str(config_path),
        "--gpus",
        ",".join(map(str, free)),
        "--output-dir",
        str(destination),
    ]
    protocol.update(
        command=command,
        launch_inventory=inventory,
        selected_gpus=free,
        output_dir=str(destination),
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"OUTPUT_DIR={destination}\nIDLE_GPUS={free}\nLOG={log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(protocol, ensure_ascii=False) + "\n")
        log.flush()
        child = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            # The training launcher owns creation of the fresh output directory.
            while (
                not (destination / "gpu_selection.json").is_file()
                and child.poll() is None
            ):
                time.sleep(0.2)
            if (destination / "gpu_selection.json").is_file():
                protocol["source_sha256"] = snapshot_sources(destination, config_path)
                protocol["git_head"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip()
                (destination / "comparison_protocol.json").write_text(
                    json.dumps(protocol, indent=2, ensure_ascii=False)
                )
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                child.wait()
            raise
    print(f"TRAINING_EXIT_CODE={code}\nOUTPUT_DIR={destination}", flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
