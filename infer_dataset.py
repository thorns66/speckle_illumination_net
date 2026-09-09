from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference from one frozen ten-frame MATLAB subset"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--subset", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _inventory() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows: list[dict[str, object]] = []
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
    return rows


def _worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from datasets.matlab_multivolume_dataset import load_inference_input
    from train_volume import _model_from_config
    from training.multivolume_trainer import (
        CHECKPOINT_FORMAT,
        _analytic_beta0,
        _load_operator,
        _prepare_psf_cache,
    )
    from utils.io import save_volume_tiff

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a shared multi-volume checkpoint")
    config = checkpoint["config"]
    model = _model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    synthetic_config_path = PROJECT_ROOT / "configs" / "checkpoint_config.yaml"
    psf_cache = _prepare_psf_cache(config, synthetic_config_path, 0, 1)
    operator = _load_operator(
        config, synthetic_config_path, device, selected_h_cache=psf_cache
    )

    var_feature_representation = config["data"].get(
        "var_feature_representation", "sqrt"
    )
    raw = load_inference_input(
        args.sample_dir,
        args.subset,
        var_feature_representation=var_feature_representation,
    )
    tensor_names = {
        "f_var",
        "f_var_feature",
        "g_mean",
        "input_mean",
        "residual_frames",
        "z_values_um",
    }
    item = {
        name: torch.from_numpy(np.asarray(value)).float().unsqueeze(0).to(device)
        if name in tensor_names
        else value
        for name, value in raw.items()
    }
    with torch.inference_mode():
        beta0 = _analytic_beta0(operator, item["f_var"], item["input_mean"])
        output = model(
            item["f_var"],
            item["g_mean"],
            item["residual_frames"],
            item["z_values_um"],
            var_feature_volume=item["f_var_feature"],
            beta0=beta0,
        )
    reconstruction = output.reconstruction[0, 0].float().cpu().numpy()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "reconstruction.npy", reconstruction)
    save_volume_tiff(destination / "reconstruction.tif", reconstruction)
    np.save(destination / "input_f_var.npy", raw["f_var"][0])
    np.save(destination / "input_f_var_feature.npy", raw["f_var_feature"][0])
    np.save(destination / "input_g_mean.npy", raw["g_mean"][0])
    record = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": checkpoint["format"],
        "checkpoint_best_step": int(checkpoint["best_step"]),
        "sample_dir": str(Path(args.sample_dir).expanduser().resolve()),
        "sample_id": str(raw["sample_id"]),
        "subset_index": int(raw["subset_index"]),
        "input_frame_indices_one_based": np.asarray(raw["input_indices"]).tolist(),
        "target_or_ground_truth_read": False,
        "var_feature_representation": var_feature_representation,
        "physical_gpu": os.environ["SPECKLE_PHYSICAL_GPUS"],
        "beta0": float(beta0.item()),
        "beta": float(output.beta.item()),
        "z_values_um": np.asarray(raw["z_values_um"]).tolist(),
    }
    (destination / "inference_contract.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        if not os.environ.get("SPECKLE_PHYSICAL_GPUS"):
            raise RuntimeError("Inference worker must be launched by infer_dataset.py")
        _worker(args)
        return

    inventory = _inventory()
    matches = [item for item in inventory if int(item["index"]) == args.gpu]
    if not matches:
        raise ValueError(f"Physical GPU {args.gpu} does not exist")
    selected = matches[0]
    if (
        int(selected["memory_used_mib"]) > 1024
        or int(selected["utilization_percent"]) > 10
    ) and not args.allow_busy_gpu:
        raise RuntimeError(
            f"GPU {args.gpu} is busy: {selected['memory_used_mib']} MiB, "
            f"{selected['utilization_percent']}%"
        )
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(selected["uuid"])
    environment["SPECKLE_PHYSICAL_GPUS"] = str(args.gpu)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--checkpoint",
        str(Path(args.checkpoint).expanduser().resolve()),
        "--sample-dir",
        str(Path(args.sample_dir).expanduser().resolve()),
        "--subset",
        str(args.subset),
        "--gpu",
        str(args.gpu),
        "--output-dir",
        str(destination),
    ]
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
