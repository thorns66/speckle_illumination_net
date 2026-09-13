"""Intensity-preserving real-data preparation and frozen six-method comparison.

No training. Field IDs are acquisition identifiers, never depth labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiment_paths import next_experiment_path

RAW = ROOT / "data/菠菜根/原始图"
MATLAB = "/workspace/xyx/MATLAB/R2023b/bin/matlab"
PYTHON = "/workspace/xyx/.conda/envs/speckle_net/bin/python"
PSF = ROOT / "psf/NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat"
FIXED_INDICES = [9, 10, 27, 44, 54, 74, 77, 83, 85, 92]
CHECKPOINTS = {
    "before_p12_taylor": ROOT / "outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_last.pt",
    "before_p12_mean": ROOT / "outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/mean_anchor_e3_mean100/checkpoint_last.pt",
    "after_p12_taylor": ROOT / "outputs/v4_p12_anchor_compare_e3_mean100_400_20260909_run01/taylor_anchor_e3_mean100/checkpoint_last.pt",
    "after_p12_mean": ROOT / "outputs/v4_p12_anchor_compare_e3_mean100_400_20260909_run01/mean_anchor_e3_mean100/checkpoint_last.pt",
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(folder, prefix):
    indexed = {}
    for path in folder.glob("*.bmp"):
        match = re.fullmatch(rf"Z{prefix}_image_(\d+)(?:\(\d+\))?\.bmp", path.name)
        if match is None:
            raise ValueError(f"Unexpected raw filename: {path}")
        number = int(match.group(1))
        if number in indexed:
            raise ValueError(f"Duplicate frame number: {number}")
        indexed[number] = path
    if set(indexed) != set(range(100)):
        raise ValueError(f"Expected exactly raw indices 0..99 in {folder}")
    return [indexed[i] for i in range(100)]


def initialize():
    output = next_experiment_path(ROOT / "outputs", "spinach_real_reprocess")
    dataset = next_experiment_path(ROOT / "data", "spinach_real_preserved")
    output.mkdir(parents=True)
    dataset.mkdir(parents=True)
    fields = []
    for field_id, prefix, split in [("45", "45", "train"), ("55", "50", "test")]:
        paths = source_files(RAW / field_id, prefix)
        fields.append({"field_id": field_id, "source_filename_prefix": "Z" + prefix,
                       "previous_alias": prefix, "split": split, "depth_ground_truth_um": None,
                       "source_files": [str(p) for p in paths],
                       "source_sha256": {str(p): sha(p) for p in paths},
                       "gray_dir": str(dataset / field_id / "grayscale_float"),
                       "rectified_dir": str(dataset / field_id / "rectified_float")})
    calibration = dict(xCenter=669.68, yCenter=504.36, rx=48.285, ry=-1.5,
                       dx=1.005, dy=48.12, Nnum=49, XcutLeft=1, XcutRight=0, YcutUp=0, YcutDown=1)
    numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", (RAW / "1.txt").read_text().splitlines()[0])]
    assert numbers == [calibration[k] for k in ["xCenter", "yCenter", "rx", "ry", "dx", "dy"]]
    cfg = {"output": str(output), "dataset": str(dataset), "fields": fields,
           "calibration": calibration, "psf_path": str(PSF), "psf_sha256": sha(PSF),
           "z_um": list(range(10, 101, 10)), "iterations": 3,
           "input_indices": FIXED_INDICES, "holdout_indices": sorted(set(range(1, 101)) - set(FIXED_INDICES)),
           "checkpoints": {name: {"path": str(path), "sha256": sha(path)} for name, path in CHECKPOINTS.items()},
           "legacy_first_frame": str(ROOT / "data/菠菜根/50_001.tif"),
           "input_policy": "uint8 RGB /255 once, rgb2gray, supplied geometry, float32; no per-frame normalization",
           "frame_index_mapping": "processed index 1..100 = raw filename image index 0..99 plus one; suffix ignored only after uniqueness check",
           "depth_policy": "field 45/55 and source Z50 are not depth; fixed checkpoint grid 10..100 um; no depth GT",
           "scope": "prepare both 100-frame fields; freeze one 10/90 subset for this comparison; no training or modification of historical simulation splits",
           "source_hashes": {str(RAW / name): sha(RAW / name) for name in ["1.txt", "ImageRectification2.m", "preprocessing2.m"]}}
    save(output / "manifest.json", cfg)
    save(dataset / "dataset_splits.json", {"train": ["45"], "validation": [], "test": ["55"],
                                           "no_gt": True, "ids_are_not_depth": True,
                                           "source_aliases": {"55": "Z50; previously called field 50"},
                                           "training_started": False})
    print(output, flush=True)
    return output


def matlab_call(output, label, expression, uuid=""):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, MATLAB_PREFDIR=str(output / "matlab_preferences" / label))
    paths = [ROOT / "matlab_code" / name for name in ["real_data", "pilot_dataset", "Util", "Solver"]]
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    code = "addpath(" + ",".join(map(quote, paths)) + ");" + expression
    with (output / (label + ".log")).open("a") as log:
        result = subprocess.run([MATLAB, "-singleCompThread", "-softwareopengl", "-batch", code],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"MATLAB {label} failed; see {output / (label + '.log')}")


def statistics(frames, indices):
    selected = frames[np.asarray(indices) - 1]
    return (selected.mean(0, dtype=np.float64).astype(np.float32),
            selected.var(0, ddof=1, dtype=np.float64).astype(np.float32))


def prepare(output):
    cfg = read(output / "manifest.json")
    if not all((Path(f["rectified_dir"]) / "preprocessing_record.json").exists() for f in cfg["fields"]):
        matlab_call(output, "preprocess", f"spinach_preserve_intensity('{output / 'manifest.json'}');")
    # Losslessly remove LZW packaging; never touch the raw BMP files.
    from PIL import Image
    transcoded = 0
    for field in cfg["fields"]:
        for key in ["gray_dir", "rectified_dir"]:
            for path in Path(field[key]).glob("frame_*.tif"):
                with tifffile.TiffFile(path) as handle:
                    compressed = int(handle.pages[0].compression) != 1
                if compressed:
                    with Image.open(path) as handle:
                        value = np.asarray(handle).copy()
                    assert value.dtype == np.float32 and np.isfinite(value).all()
                    temporary = path.with_suffix(".uncompressed.tmp.tif")
                    tifffile.imwrite(temporary, value, photometric="minisblack")
                    assert np.array_equal(value, tifffile.imread(temporary))
                    temporary.replace(path)
                    transcoded += 1
    save(output / "tiff_packaging.json", {"lossless_float32_transcodes": transcoded, "pixel_values_unchanged": True})
    for field in cfg["fields"]:
        folder = output / field["field_id"]
        folder.mkdir(exist_ok=True)
        files = [Path(field["rectified_dir"]) / f"frame_{i:03d}.tif" for i in range(1, 101)]
        frames = np.stack([tifffile.imread(p) for p in files])
        if frames.dtype != np.float32 or frames.shape != (100, 1029, 1421):
            raise ValueError(f"Unexpected processed shape/type: {frames.shape}, {frames.dtype}")
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("Invalid float frames")
        im, iv = statistics(frames, cfg["input_indices"])
        hm, hv = statistics(frames, cfg["holdout_indices"])
        values = dict(mean=im, variance=iv, holdout_mean=hm, holdout_variance=hv)
        stats_dir = Path(cfg["dataset"]) / field["field_id"] / "subset_01"
        stats_dir.mkdir(exist_ok=True)
        for name, value in values.items():
            tifffile.imwrite(stats_dir / (name + ".tif"), value, photometric="minisblack")
        actual = {"field_id": field["field_id"], "split": field["split"],
                  "source_filename_prefix": field["source_filename_prefix"],
                  "image_shape_yx": list(frames.shape[1:]), "z_um": cfg["z_um"], "iterations": 3,
                  "input_indices": cfg["input_indices"], "holdout_indices": cfg["holdout_indices"],
                  "selected_files": [str(files[i - 1]) for i in cfg["input_indices"]],
                  "holdout_files": [str(files[i - 1]) for i in cfg["holdout_indices"]],
                  "input_policy": cfg["input_policy"], "psf_path": cfg["psf_path"],
                  "psf_sha256": cfg["psf_sha256"],
                  "mean_tiff": str(stats_dir / "mean.tif"), "variance_tiff": str(stats_dir / "variance.tif"),
                  "holdout_mean_tiff": str(stats_dir / "holdout_mean.tif"),
                  "holdout_variance_tiff": str(stats_dir / "holdout_variance.tif"),
                  "output_mat": {mode: str(stats_dir / (mode + "_rl3.mat")) for mode in ["mean", "taylor"]},
                  "rectified_hashes": {str(p): sha(p) for p in files}, "gpu_uuid": {}}
        save(folder / "manifest.json", actual)
        save(stats_dir / "manifest.json", actual)
        record = read(Path(field["rectified_dir"]) / "preprocessing_record.json")
        if field["field_id"] == "55":
            replay = record["records"][0]
            if replay["legacy_replay_relative_l2"] > .02:
                raise RuntimeError("Legacy replay does not establish the old field50 source mapping")
        print(f"PREPARED field={field['field_id']} split={field['split']} shape={frames.shape}", flush=True)
        del frames
    save(output / "preparation_complete.json", {"complete": True, "train": ["45"], "test": ["55"], "processed_frames": 200, "training_started": False})


def inventory():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return [dict(index=int(a), uuid=b.strip(), free_mib=int(c), utilization=int(d))
            for a, b, c, d in (line.split(",") for line in result.splitlines())]


def network(manifest):
    import torch
    from scipy.io import savemat
    from models.configurable_anchor_lfm_net import model_from_config
    from tools.spinach_root_network_v2 import execute, mat_volume
    cfg = read(manifest)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == cfg["gpu_uuid"]["network"]
    assert sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["completed_steps"] == 400
    config = checkpoint["config"]
    anchor_kind = config["model"].get("reconstruction_anchor", "taylor_sqrt")
    assert anchor_kind == cfg["anchor_kind"]
    model = model_from_config(config)
    model.register_parameter("mean_gain_gamma", torch.nn.Parameter(torch.zeros(())))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    mean = mat_volume(cfg["output_mat"]["mean"], "reconstruction_raw")
    taylor = mat_volume(cfg["output_mat"]["taylor"], "reconstruction_sqrt")
    source = cfg["output_mat"]["mean" if anchor_kind == "mean_rl3" else "taylor"]
    with h5py.File(source, "r") as handle:
        projection = np.asarray(handle["predicted_mean"], dtype=np.float32).T.copy()
    frames = np.stack([tifffile.imread(path) for path in cfg["selected_files"]])
    assert frames.dtype == np.float32
    mu = frames.mean(0, dtype=np.float64).astype(np.float32)
    assert np.array_equal(mu, tifffile.imread(cfg["mean_tiff"]))
    beta0 = float(np.sum(projection.astype(np.float64) * mu) / (np.sum(projection.astype(np.float64) ** 2) + 1e-8))
    assert np.isfinite(beta0) and beta0 > 0
    base, mode = execute(model, taylor, mean, frames - mu, np.asarray(cfg["z_um"], np.float32), beta0, device)
    assert np.isfinite(base).all() and (base >= 0).all()
    savemat(cfg["network_base_mat"], {"base_reconstruction": base.transpose(1, 2, 0)}, do_compression=False)
    beta = beta0 * (1 + float(config["model"]["beta_range"]) * math.tanh(float(model.raw_beta.detach().cpu())))
    anchor = (mean if anchor_kind == "mean_rl3" else taylor) * beta
    correction_ratio = float(np.linalg.norm((base - anchor).astype(np.float64)) / max(np.linalg.norm(anchor.astype(np.float64)), 1e-30))
    cfg["e3_gain_factor"] = 1 + float(config["three_way"]["gain_bound"]) * math.tanh(float(model.mean_gain_gamma.detach().cpu()))
    save(manifest, cfg)
    save(cfg["network_record"], {"complete": True, "beta0": beta0, "beta": beta,
                                "correction_to_anchor_l2": correction_ratio, "anchor_kind": anchor_kind,
                                "checkpoint_step": 400, "inference_target_access": False,
                                "e3_gain_factor": cfg["e3_gain_factor"], **mode})


def job(output, field_id, kind, gpu):
    cfg = read(output / field_id / "manifest.json")
    cfg["gpu_uuid"] = dict(mean=gpu["uuid"], taylor=gpu["uuid"], network=gpu["uuid"], gain=gpu["uuid"])
    directory = output / field_id / kind
    directory.mkdir(exist_ok=True)
    manifest = directory / "manifest.json"
    if kind in ["mean", "taylor"]:
        save(manifest, cfg)
        # Original solver retained; only its provenance string is changed below.
        expression = (f"spinach_root_rl_worker_v2('{manifest}','{kind}');"
                      f"p='{cfg['output_mat'][kind]}';s=load(p);s.input_policy='{cfg['input_policy']}';save(p,'-struct','s','-v7.3');")
        matlab_call(directory, "rl", expression, gpu["uuid"])
    else:
        checkpoint = CHECKPOINTS[kind]
        cfg.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), checkpoint_step=400,
                   checkpoint_role="final", anchor_kind="mean_rl3" if kind.endswith("mean") else "taylor_sqrt",
                   network_base_mat=str(directory / "network_base.mat"),
                   network_final_mat=str(directory / "reconstruction.mat"), network_record=str(directory / "network_record.json"))
        save(manifest, cfg)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONPATH=str(ROOT),
                   MPLCONFIGDIR=str(directory / "mpl_cache"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with (directory / "network.log").open("a") as log:
            result = subprocess.run([PYTHON, str(Path(__file__).resolve()), "network", str(manifest)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Network failed: {directory}")
        matlab_call(directory, "gain", f"spinach_root_gain_worker_v2('{manifest}');", gpu["uuid"])
    save(directory / "complete.json", {"complete": True, "field_id": field_id, "method": kind, "gpu_index": gpu["index"]})


def run(output):
    from concurrent.futures import ThreadPoolExecutor
    cfg = read(output / "manifest.json")
    assert (output / "preparation_complete.json").exists()
    assert sha(PSF) == cfg["psf_sha256"]
    for value in cfg["checkpoints"].values():
        assert sha(value["path"]) == value["sha256"]
    queue = [(field, mode) for field in ["55", "45"] for mode in ["mean", "taylor"]]
    queue += [(field, method) for field in ["55", "45"] for method in CHECKPOINTS]
    queue = [(f, m) for f, m in queue if not (output / f / m / "complete.json").exists()]
    active = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while queue or active:
            for future in list(active):
                if future.done():
                    info = active.pop(future)
                    future.result()
                    print(f"DONE {info}", flush=True)
            used = {info["gpu"]["index"] for info in active.values()}
            cards = sorted([g for g in inventory() if g["index"] in range(6) and g["index"] not in used and g["free_mib"] >= 30000],
                           key=lambda g: (g["utilization"] > 5, -g["free_mib"]))
            for card in cards:
                if len(active) >= 2:
                    break
                eligible = next(((f, m) for f, m in queue if m in ["mean", "taylor"] or all((output / f / mode / "complete.json").exists() for mode in ["mean", "taylor"])), None)
                if eligible is None:
                    break
                field, method = eligible
                queue.remove(eligible)
                future = pool.submit(job, output, field, method, card)
                active[future] = {"field": field, "method": method, "gpu": card}
                print(f"START {active[future]}", flush=True)
            save(output / "progress.json", {"pending": queue, "active": list(active.values()), "updated_unix": time.time(), "supervisor_pid": os.getpid()})
            if queue or active:
                time.sleep(5)
    save(output / "inference_complete.json", {"complete": True, "fields": ["45", "55"], "rl_volumes": 4, "network_volumes": 8, "gpu_workers_exited": True, "final_gpu_inventory": inventory()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["init", "prepare", "run", "network"])
    parser.add_argument("path", type=Path, nargs="?")
    args = parser.parse_args()
    if args.stage == "init":
        initialize()
    else:
        {"prepare": prepare, "run": run, "network": network}[args.stage](args.path.resolve())


if __name__ == "__main__":
    main()
