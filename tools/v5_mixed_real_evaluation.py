"""Evaluate V5 step400/final600/best on simulation test and real train fields."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import training.multivolume_trainer as trainer
from losses.self_supervised_losses import TaylorH2VarianceModel
from tools import v3_compare_evaluation as legacy_evaluation
from tools import v5_mixed_real_anchor_experiment as exp
from tools.mixed_resolution_lfm import MixedResolutionLFM
from tools.v5_mixed_dataset import MixedExperimentDataset, RealFieldDataset
from utils.io import save_volume_tiff


ROLES = ("step400", "final600", "best")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def checkpoint_paths(arm: str) -> dict[str, Path]:
    folder = exp.OUTPUT / arm
    return {
        "step400": folder / "checkpoint_step_000400.pt",
        "final600": folder / "checkpoint_step_000600.pt",
        "best": folder / "checkpoint_best.pt",
    }


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = exp.build_model(config, initial=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return payload, config, model


def to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return trainer._to_device(item, device)


def simulation_operator(device: torch.device) -> MixedResolutionLFM:
    return MixedResolutionLFM(exp.SPARSE_CACHE, device, full_h=None, phase_chunk_size=32)


def real_operator(device: torch.device, config: dict[str, Any]) -> MixedResolutionLFM:
    contract = json.loads(
        (exp.ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/mean_anchor_e3_mean100/run_contract.json")
        .read_text(encoding="utf-8")
    )
    selected = np.load(contract["selected_psf_cache"], mmap_mode="c", allow_pickle=False)
    h = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
    return MixedResolutionLFM(
        exp.SPARSE_CACHE, device, full_h=h,
        phase_chunk_size=int(config["runtime"]["operator_phase_chunk_size"]),
        real_shape=(1029, 1421),
        load_sparse_simulation=False,
    )


def evaluate_simulation(arm: str, device: torch.device) -> list[dict[str, Any]]:
    output_root = exp.OUTPUT / "comparison/simulation" / arm
    rows: list[dict[str, Any]] = []
    dataset = MixedExperimentDataset(
        exp.SIMULATION_DATA, "test", real_root=exp.REAL_DATA,
        cache_dir=exp.OUTPUT / "data_cache", var_feature_representation="sqrt",
    )
    operator = simulation_operator(device)
    for role, path in checkpoint_paths(arm).items():
        payload, config, model = load_model(path, device)
        variance_model = TaylorH2VarianceModel(operator, **config["noise"])
        beta_cache: dict[Any, float] = {}
        for index in range(len(dataset)):
            item = to_device(dataset[index], device)
            with torch.inference_mode():
                beta0 = exp.cached_beta0(beta_cache, item, operator, config)
                prediction, _ = exp.forward(model, item, operator, beta0, config)
                losses = exp.loss(prediction, item, operator, variance_model, config)
            volume = prediction.reconstruction[0, 0].cpu().numpy().astype(np.float32)
            truth = item["ground_truth"][0, 0].cpu().numpy()
            metrics = legacy_evaluation._structure_row(volume, truth)
            row = {
                "arm": arm, "checkpoint_role": role,
                "checkpoint_step": int(payload["completed_steps"]),
                "sample_id": item["sample_id"], "subset_index": int(item["subset_index"]),
                "beta0": float(beta0.item()), "beta": float(prediction.beta.item()),
                **losses.scalar_metrics(), **metrics,
            }
            rows.append(row)
            destination = output_root / role / f"{item['sample_id']}_subset_{int(item['subset_index']):02d}"
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "reconstruction.npy", volume)
            np.save(destination / "physical_anchor.npy", prediction._physical_anchor[0, 0].cpu().numpy().astype(np.float32))
            np.save(destination / "effective_correction.npy", prediction._effective_correction[0, 0].cpu().numpy().astype(np.float32))
            save_volume_tiff(destination / "reconstruction.tif", volume)
            write_json(destination / "complete.json", {"complete": True, **row})
        del model, variance_model, payload
        torch.cuda.empty_cache()
    write_csv(output_root / "metrics.csv", rows)
    return rows


def evaluate_real(arm: str, device: torch.device) -> list[dict[str, Any]]:
    output_root = exp.OUTPUT / "comparison/real_training_fields" / arm
    rows: list[dict[str, Any]] = []
    dataset = RealFieldDataset(exp.REAL_DATA)
    first_payload, first_config, first_model = load_model(checkpoint_paths(arm)["step400"], device)
    operator = real_operator(device, first_config)
    del first_payload, first_model
    torch.cuda.empty_cache()
    beta_cache: dict[Any, float] = {}
    for role, path in checkpoint_paths(arm).items():
        payload, config, model = load_model(path, device)
        variance_model = TaylorH2VarianceModel(operator, **config["noise"])
        for index in range(len(dataset)):
            item = to_device(dataset[index], device)
            with torch.inference_mode():
                beta0 = exp.cached_beta0(beta_cache, item, operator, config)
                prediction, _ = exp.forward(model, item, operator, beta0, config)
                losses = exp.loss(prediction, item, operator, variance_model, config)
            volume = prediction.reconstruction[0, 0].cpu().numpy().astype(np.float32)
            anchor = prediction._physical_anchor[0, 0].cpu().numpy().astype(np.float32)
            correction = prediction._effective_correction[0, 0].cpu().numpy().astype(np.float32)
            row = {
                "arm": arm, "checkpoint_role": role,
                "checkpoint_step": int(payload["completed_steps"]),
                "sample_id": item["sample_id"], "field_id": item["field_id"],
                "subset_index": int(item["subset_index"]), "result_role": "training_field_diagnostic_no_gt",
                "beta0": float(beta0.item()), "beta": float(prediction.beta.item()),
                "total_mass": float(volume.sum(dtype=np.float64)),
                "correction_to_anchor_l2": float(np.linalg.norm(correction.astype(np.float64)) / max(np.linalg.norm(anchor.astype(np.float64)), 1e-30)),
                **losses.scalar_metrics(),
            }
            rows.append(row)
            destination = output_root / role / f"real_{item['field_id']}_subset_{int(item['subset_index']):02d}"
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "reconstruction.npy", volume)
            np.save(destination / "physical_anchor.npy", anchor)
            np.save(destination / "effective_correction.npy", correction)
            save_volume_tiff(destination / "reconstruction.tif", volume)
            write_json(destination / "complete.json", {"complete": True, **row})
        del model, variance_model, payload
        torch.cuda.empty_cache()
    write_csv(output_root / "metrics.csv", rows)
    return rows


def normalized(volume: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(volume, np.float64), 0)
    return value / max(value.sum(), 1e-30)


def stability_rows(domain: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    objects = ("T02", "T03", "T04") if domain == "simulation" else ("45", "55")
    for arm in exp.ARMS:
        for role in ROLES:
            for owner in objects:
                base = exp.OUTPUT / "comparison" / (
                    "simulation" if domain == "simulation" else "real_training_fields"
                ) / arm / role
                values = []
                for subset in range(1, 11):
                    name = f"{owner}_subset_{subset:02d}" if domain == "simulation" else f"real_{owner}_subset_{subset:02d}"
                    values.append(np.load(base / name / "reconstruction.npy", allow_pickle=False))
                q = np.stack([normalized(value) for value in values])
                center = q.mean(0)
                dispersion = float(np.sqrt(np.mean(np.sum((q - center) ** 2, axis=(1, 2, 3)))) / max(np.linalg.norm(center), 1e-30))
                profiles = q.sum((2, 3))
                pairwise = [float(np.abs(np.cumsum(profiles[a]) - np.cumsum(profiles[b])).sum() * 10)
                            for a, b in itertools.combinations(range(10), 2)]
                masses = np.asarray([np.maximum(value, 0).sum(dtype=np.float64) for value in values])
                result.append({
                    "domain": domain, "sample_id": owner, "arm": arm, "checkpoint_role": role,
                    "shape_relative_dispersion": dispersion,
                    "pairwise_axial_w1_um": float(np.mean(pairwise)),
                    "mass_coefficient_of_variation": float(masses.std(ddof=1) / max(masses.mean(), 1e-30)),
                    "subsets": 10, "pair_count": 45,
                })
            selected = [row for row in result if row["domain"] == domain and row["arm"] == arm and row["checkpoint_role"] == role]
            result.append({
                "domain": domain, "sample_id": "object_macro", "arm": arm, "checkpoint_role": role,
                **{name: float(np.mean([row[name] for row in selected])) for name in (
                    "shape_relative_dispersion", "pairwise_axial_w1_um", "mass_coefficient_of_variation"
                )},
                "subsets": 10 * len(objects), "pair_count": 45 * len(objects),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("all", *exp.ARMS), default="all")
    parser.add_argument("--domain", choices=("all", "simulation", "real"), default="all")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    exp.configure_precision()
    arms = exp.ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        if args.domain in ("all", "simulation"):
            evaluate_simulation(arm, device)
        if args.domain in ("all", "real"):
            evaluate_real(arm, device)
    if args.arm == "all" and args.domain == "all":
        rows = stability_rows("simulation") + stability_rows("real")
        write_csv(exp.OUTPUT / "comparison/stability.csv", rows)
        write_json(exp.OUTPUT / "comparison/evaluation_complete.json", {
            "complete": True, "new_network_predictions": 300,
            "simulation": 180, "real_training_field_diagnostics": 120,
            "checkpoint_roles": list(ROLES), "real_has_gt": False,
        })


if __name__ == "__main__":
    main()
