"""Run the dated P12 Taylor-anchor versus Mean-anchor comparison end to end."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import torch
import yaml

from datasets.matlab_multivolume_dataset import load_dataset_index
from tools import three_way_experiment as old
from tools import v4_p12_anchor_compare_experiment as exp


OUTPUT = exp.OUTPUT
ARMS = exp.ARMS
EXPECTED_FINGERPRINT = "5ff731378a5d7a81028da7427bc25b91564c4c87a390694cd8e8ff57b4e9ce50"
MIN_FREE_MIB_TRAIN = 15 * 1024
MIN_FREE_MIB_INFER = 14 * 1024
GPU_POOL = tuple(range(6))


def write_json(path: Path, value: Any) -> None:
    old.write_json(path, value)


def config(arm: str) -> dict[str, Any]:
    source = yaml.safe_load(
        (ROOT / "outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100.yaml")
        .read_text(encoding="utf-8")
    )
    value = copy.deepcopy(source)
    anchor = exp.ANCHORS[arm]
    value["experiment"].update(
        name=f"v4_p12_{arm}", seed=20260901, output_dir=str(OUTPUT / arm)
    )
    value["data"].update(
        root=str(exp.DATA), cache_dir=str(OUTPUT / "data_cache"), precompute_cache=False
    )
    value["model"]["reconstruction_anchor"] = anchor
    value["optimization"].update(
        max_steps=400, global_batch_size=8, micro_batch_per_gpu=1,
        lr_network=1e-3, lr_beta=1e-4, validate_every=20, checkpoint_every=20,
    )
    value["runtime"].update(amp=False, tensorboard=True)
    value["loss"].update(
        lambda_mean=0.0, lambda_var=1.0, lambda_tv=1e-5, lambda_tv_z=0.5
    )
    value["three_way"].update(
        kind="e3", lr_gain=1e-4,
        selection_rule=(
            "normalized mean + normalized variance + TV object macro; "
            "bounded mean-shape term excluded"
        ),
    )
    value["v3_compare"].update(
        kind=arm,
        shared_initial_state=str(exp.SHARED_INITIAL_STATE),
        common_seed=20260901,
        from_scratch=True,
        shape_gradient_budget=1.0,
        shape_gradient_eps=1e-12,
        shape_coefficient_cap=1.0,
        ramp_steps=50,
        gradient_limit_location="per-sample normalized reconstruction q",
        shape_mean_definition="SmoothL1(H(q)/mean(H(q)), mu90/mean(mu90))",
        lr_drop_after_steps=200,
        lr_network_after_drop=1e-4,
        lr_scalar_after_drop=1e-5,
        numerical_precision="full FP32; AMP and TF32 disabled",
        best_selection_uses_gt=False,
        beta_anchor=anchor,
        beta_cache_key=anchor,
        p12_training_extension=True,
    )
    return value


def source_paths() -> list[Path]:
    return [
        ROOT / "models/configurable_anchor_lfm_net.py",
        ROOT / "models/variance_anchored_lfm_net.py",
        ROOT / "tools/v4_p12_anchor_compare_experiment.py",
        ROOT / "tools/v4_p12_anchor_checks.py",
        Path(__file__).resolve(),
        ROOT / "tools/v3_compare_experiment.py",
        ROOT / "tools/v3_compare_evaluation.py",
        ROOT / "tools/v3_compare_local_audit.py",
        ROOT / "training/multivolume_trainer.py",
        ROOT / "training/global_batch_schedule.py",
        ROOT / "datasets/matlab_multivolume_dataset.py",
        ROOT / "utils/dataset_splits.py",
        ROOT / "losses/self_supervised_losses.py",
        ROOT / "physics/lfm_operator.py",
        ROOT / "physics/psf_loader.py",
        ROOT / "tools/three_way_experiment.py",
        ROOT / "tools/priority_validation_analysis.py",
        ROOT / "train_volume.py",
    ]


def _dataset_contract() -> tuple[dict[str, list], str, dict[str, Any]]:
    indexed, fingerprint = load_dataset_index(exp.DATA)
    counts = {split: len(items) for split, items in indexed.items()}
    if counts != {"train": 120, "validation": 30, "test": 30}:
        raise ValueError(f"Unexpected V4 split counts: {counts}")
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"V4 dataset fingerprint changed: {fingerprint}")
    owners = {
        split: sorted({item.sample_id for item in items}) for split, items in indexed.items()
    }
    if owners["train"] != [f"P{i:02d}" for i in range(1, 13)]:
        raise ValueError(f"P12 is not in the authoritative training split: {owners['train']}")
    audit = json.loads((exp.DATA / "final_hash_audit.json").read_text(encoding="utf-8"))
    if not audit.get("complete") or audit.get("errors") or int(audit["immutable_artifacts"]) != 7320:
        raise ValueError("V4 final artifact audit is not complete")
    if int(audit["sample_artifact_counts"].get("P12", 0)) != 394:
        raise ValueError("P12 artifact inventory changed")
    return indexed, fingerprint, {"counts": counts, "owners": owners, "audit": audit}


def prepare(*, resume: bool) -> dict[str, Any]:
    marker = OUTPUT / "preflight.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"{OUTPUT} exists; use --resume")
        record = json.loads(marker.read_text(encoding="utf-8"))
        for path, digest in record["training_source_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen training source changed: {path}")
        for path, digest in record["config_hashes"].items():
            if old.sha256(path) != digest:
                raise ValueError(f"Frozen config changed: {path}")
        _indexed, fingerprint, _dataset = _dataset_contract()
        if fingerprint != record["dataset_fingerprint"]:
            raise ValueError("Dataset fingerprint changed after preflight")
        return record

    _indexed, fingerprint, dataset = _dataset_contract()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    config_hashes = {}
    initial_hashes = {}
    schedule_signatures = {}
    from training.global_batch_schedule import FixedGlobalBatchScheduler
    for arm in ARMS:
        current = config(arm)
        exp.validate_config(current)
        config_path = OUTPUT / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
        config_hashes[str(config_path.resolve())] = old.sha256(config_path)
        model = exp.build_model(current, initial=True)
        common = {
            name: value for name, value in model.state_dict().items()
            if name != "mean_gain_gamma"
        }
        initial_hashes[arm] = old.state_hash(common)
        scheduler = FixedGlobalBatchScheduler(120, 8, 20260901)
        schedule_signatures[arm] = old.sha256_bytes(
            json.dumps([scheduler.next_batch() for _ in range(400)]).encode("utf-8")
        )
        del model, common
    payload = torch.load(exp.SHARED_INITIAL_STATE, map_location="cpu", weights_only=False)
    if set(initial_hashes.values()) != {payload["common_state_sha256"]}:
        raise ValueError("The two arms do not reproduce the shared initialization")
    if len(set(schedule_signatures.values())) != 1:
        raise ValueError("The two arms do not use the same 400-step sample schedule")
    del payload

    sources = source_paths()
    source_hashes = {str(path.resolve()): old.sha256(path) for path in sources}
    for path in sources:
        destination = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    psf = Path(config(ARMS[0])["psf"]["H_path"])
    references = [
        exp.SHARED_INITIAL_STATE,
        ROOT / "outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_last.pt",
        ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/mean_anchor_e3_mean100/checkpoint_last.pt",
    ]
    record = {
        "complete": True,
        "plan": "P12-extended from-scratch Taylor-anchor versus Mean-anchor E3+mean<=100%",
        "output": str(OUTPUT),
        "prepared_unix": time.time(),
        "dataset_root": str(exp.DATA),
        "dataset_fingerprint": fingerprint,
        "dataset_contract": dataset,
        "split_counts": dataset["counts"],
        "config_hashes": config_hashes,
        "training_source_hashes": source_hashes,
        "reference_hashes": {str(path.resolve()): old.sha256(path) for path in references},
        "shared_initial_state": str(exp.SHARED_INITIAL_STATE),
        "shared_initial_file_sha256": old.sha256(exp.SHARED_INITIAL_STATE),
        "shared_initial_common_state_sha256": next(iter(initial_hashes.values())),
        "arm_common_initial_hashes": initial_hashes,
        "sample_schedule_sha256": next(iter(schedule_signatures.values())),
        "psf_path": str(psf),
        "psf_sha256": old.sha256(psf),
        "anchors": exp.ANCHORS,
        "set_branch": True,
        "taylor_feature_representation": "sqrt",
        "max_steps_per_arm": 400,
        "global_batch_size": 8,
        "validation_interval": 20,
        "lr_schedule": {
            "steps_1_200": {"network": 1e-3, "beta_gamma": 1e-4},
            "steps_201_400": {"network": 1e-4, "beta_gamma": 1e-5},
        },
        "gpu_policy": {
            "pool": list(GPU_POOL),
            "idle_cards_preferred": True,
            "sharing_allowed_if_free_memory_mib_at_least": MIN_FREE_MIB_TRAIN,
            "arms_run_sequentially": True,
        },
        "no_training_gt": True,
    }
    write_json(marker, record)
    print(json.dumps({
        "preflight": "complete", "output": str(OUTPUT),
        "dataset_fingerprint": fingerprint, "train_items": 120,
        "shared_initial_state": record["shared_initial_common_state_sha256"],
    }, ensure_ascii=False), flush=True)
    return record


def inventory() -> dict[int, dict[str, Any]]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    by_gpu: dict[str, list[int]] = {}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid = [field.strip() for field in line.split(",")]
            by_gpu.setdefault(uuid, []).append(int(pid))
    result = {}
    for line in rows.splitlines():
        index, uuid, name, total, used, utilization = [field.strip() for field in line.split(",")]
        result[int(index)] = {
            "uuid": uuid, "name": name, "total": int(total), "memory": int(used),
            "free": int(total) - int(used), "utilization": int(utilization),
            "pids": by_gpu.get(uuid, []), "busy": uuid in by_gpu,
        }
    return result


def eligible_gpus(*, minimum_free_mib: int, required: int | None = None) -> list[int]:
    current = inventory()
    candidates = [
        index for index in GPU_POOL
        if index in current and "A40" in current[index]["name"].upper()
        and current[index]["free"] >= minimum_free_mib
    ]
    candidates.sort(key=lambda index: (
        current[index]["busy"], current[index]["utilization"], current[index]["memory"]
    ))
    needed = 1 if required is None else required
    if len(candidates) < needed:
        raise RuntimeError(
            f"Need {needed} A40 GPUs with at least {minimum_free_mib} MiB free; "
            f"eligible={candidates}, inventory={current}"
        )
    return candidates if required is None else candidates[:required]


def training_gpus(arm: str, *, resume: bool) -> list[int]:
    checkpoint = OUTPUT / arm / "checkpoint_last.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires --resume: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        world_size = int(payload["world_size"])
        del payload
        return eligible_gpus(minimum_free_mib=MIN_FREE_MIB_TRAIN, required=world_size)
    return eligible_gpus(minimum_free_mib=MIN_FREE_MIB_TRAIN)[:6]


def launch(worker: str, cards: list[int], *, arm: str | None = None, resume: bool = False):
    if not cards:
        raise ValueError("At least one GPU is required")
    current = inventory()
    threshold = MIN_FREE_MIB_TRAIN if worker in ("train", "gpu-check") else MIN_FREE_MIB_INFER
    for index in cards:
        if current[index]["free"] < threshold:
            raise RuntimeError(f"GPU {index} no longer has enough memory: {current[index]}")
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(current[index]["uuid"] for index in cards),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SPECKLE_PHYSICAL_GPUS=",".join(map(str, cards)),
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tools"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        MPLCONFIGDIR=str(OUTPUT / "mpl_cache"),
        V4_ANCHOR_OUTPUT=str(OUTPUT),
    )
    arguments = [str(Path(__file__).resolve()), "--worker", worker]
    if arm:
        arguments += ["--experiment", arm]
    if resume:
        arguments.append("--resume")
    command = [sys.executable, *arguments]
    if worker == "train" and len(cards) > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(cards)}", *arguments,
        ]
    log = OUTPUT / "logs" / f"{worker}_{arm or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
    record_path = OUTPUT / "jobs" / f"{worker}_{arm or 'all'}.json"
    write_json(record_path, {
        "pid": process.pid, "worker": worker, "experiment": arm, "gpus": cards,
        "gpu_inventory_at_launch": {str(index): current[index] for index in cards},
        "command": command, "started_unix": time.time(), "status": "running",
        "log": str(log),
    })
    process.record_path = record_path
    print(json.dumps({
        "started": worker, "experiment": arm, "pid": process.pid,
        "gpus": cards, "shared": [index for index in cards if current[index]["busy"]],
        "log": str(log),
    }, ensure_ascii=False), flush=True)
    return process


def wait(process) -> None:
    code = process.wait()
    record = json.loads(process.record_path.read_text(encoding="utf-8"))
    record.update(
        exit_code=code, status="complete" if code == 0 else "failed", finished_unix=time.time()
    )
    write_json(process.record_path, record)
    if code:
        raise RuntimeError(f"Worker failed; inspect {record['log']}")


def worker(name: str, arm: str | None, resume: bool) -> None:
    if name == "gpu-check":
        from tools.v4_p12_anchor_checks import gpu_checks
        gpu_checks()
    elif name == "train":
        if arm is None:
            raise ValueError("Training worker requires an arm")
        exp.run_train(OUTPUT / f"{arm}.yaml", resume=resume)
    elif name == "infer":
        if arm is None:
            raise ValueError("Inference worker requires an arm")
        from tools import v3_compare_evaluation as evaluation
        previous = evaluation.exp
        evaluation.exp = exp
        try:
            evaluation.evaluate_arm(arm, resume=resume)
        finally:
            evaluation.exp = previous
    else:
        raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "preflight", "train", "infer", "report"), default="all"
    )
    parser.add_argument("--experiment", choices=("all", *ARMS), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("gpu-check", "train", "infer"))
    args = parser.parse_args()
    selected = ARMS if args.experiment == "all" else (args.experiment,)
    if args.worker:
        worker(args.worker, None if args.experiment == "all" else args.experiment, args.resume)
        return
    if args.stage in ("all", "preflight", "train"):
        prepare(resume=args.resume)
    if args.stage in ("all", "preflight", "train") and not (
        args.resume and (OUTPUT / "gpu_checks.json").exists()
    ):
        wait(launch(
            "gpu-check", eligible_gpus(minimum_free_mib=MIN_FREE_MIB_TRAIN, required=1)
        ))
    if args.stage == "preflight":
        return
    if args.stage in ("all", "train"):
        if not json.loads((OUTPUT / "gpu_checks.json").read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError("GPU preflight did not pass")
        for arm in selected:
            if args.resume and (OUTPUT / arm / "training_complete.json").exists():
                continue
            cards = training_gpus(arm, resume=args.resume)
            wait(launch(
                "train", cards, arm=arm,
                resume=args.resume and (OUTPUT / arm / "checkpoint_last.pt").exists(),
            ))
    if args.stage in ("all", "infer"):
        pending = [
            arm for arm in selected
            if not (args.resume and (OUTPUT / "evaluation" / arm / "complete.json").exists())
        ]
        while pending:
            cards = eligible_gpus(minimum_free_mib=MIN_FREE_MIB_INFER)
            batch = pending[:len(cards)]
            pending = pending[len(batch):]
            jobs = [
                launch("infer", [card], arm=arm, resume=args.resume)
                for card, arm in zip(cards, batch)
            ]
            for job in jobs:
                wait(job)
    if args.stage in ("all", "report"):
        from tools.v4_p12_anchor_analysis import build_report
        print(json.dumps(build_report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
