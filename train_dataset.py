from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _gpu_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        index, uuid, name, memory, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return rows


def _select_gpus(specification: str, inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    if specification.strip().lower() == "all":
        return inventory
    try:
        indices = [int(value.strip()) for value in specification.split(",")]
    except ValueError as exception:
        raise ValueError("--gpus must be 'all' or comma-separated nvidia-smi indices") from exception
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--gpus must contain at least one unique GPU index")
    by_index = {int(item["index"]): item for item in inventory}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"GPU indices do not exist: {missing}")
    return [by_index[index] for index in indices]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared multi-volume 10-frame/90-target training launcher"
    )
    parser.add_argument("--config", default="configs/multivolume_n10_no_mean.yaml")
    parser.add_argument("--gpus", default="all", help="nvidia-smi indices, e.g. 0,2,5, or all")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--phase-chunk-size", type=int)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-validation-items", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--limit-test-items", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _worker_arguments(args: argparse.Namespace) -> list[str]:
    values = ["--worker", "--config", str(Path(args.config).expanduser().resolve())]
    for name in (
        "output_dir",
        "resume",
        "max_steps",
        "global_batch_size",
        "validate_every",
        "phase_chunk_size",
        "limit_validation_items",
        "limit_test_items",
    ):
        value = getattr(args, name)
        if value is not None:
            values.extend(("--" + name.replace("_", "-"), str(value)))
    return values


def _run_worker(args: argparse.Namespace) -> None:
    if not torch_cuda_environment_ready():
        raise RuntimeError("Worker was not launched with SPECKLE_PHYSICAL_GPUS")
    from training.multivolume_trainer import run_training

    run_training(args)


def torch_cuda_environment_ready() -> bool:
    return bool(os.environ.get("SPECKLE_PHYSICAL_GPUS"))


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        _run_worker(args)
        return
    if args.resume and args.output_dir:
        raise ValueError("--resume already determines output directory; omit --output-dir")
    if not args.resume and not args.output_dir:
        raise ValueError("--output-dir is required for a new training run")
    inventory = _gpu_inventory()
    selected = _select_gpus(args.gpus, inventory)
    busy = [
        item
        for item in selected
        if int(item["memory_used_mib"]) > 1024 or int(item["utilization_percent"]) > 10
    ]
    if busy and not args.allow_busy_gpus:
        raise RuntimeError(
            "Selected GPUs are busy: "
            + ", ".join(
                f"{item['index']} ({item['memory_used_mib']} MiB, {item['utilization_percent']}%)"
                for item in busy
            )
        )
    if args.resume:
        destination = Path(args.resume).expanduser().resolve().parent
    else:
        destination = Path(args.output_dir).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {destination}; choose a new directory or --resume"
            )
    selection = {
        "requested": args.gpus,
        "physical_indices": [item["index"] for item in selected],
        "uuid_order": [item["uuid"] for item in selected],
        "inventory_at_launch": inventory,
    }
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "gpu_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item["uuid"]) for item in selected)
    environment["SPECKLE_PHYSICAL_GPUS"] = ",".join(str(item["index"]) for item in selected)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # This six-A40 host has two PCIe/NUMA islands and no NVLink. Direct NCCL
    # P2P initialization hangs across the host bridges; host-staged collectives
    # are stable and communication is small relative to the LFM physics work.
    environment.setdefault("NCCL_P2P_DISABLE", "1")
    environment.setdefault("NCCL_IB_DISABLE", "1")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    child_arguments = _worker_arguments(args)
    if len(selected) == 1:
        command = [sys.executable, str(Path(__file__).resolve()), *child_arguments]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(selected)}",
            str(Path(__file__).resolve()),
            *child_arguments,
        ]
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
