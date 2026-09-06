"""Build one reusable dense population Cs artifact for the linear simulator."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from physics.finite_phase_cs import dense_phase_covariance, finite_phase_transfer
from physics.speckle_oracle import SpeckleGeneratorConfig
from tools.audit_finite_speckle_stationarity import exact_moments


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", default="outputs/linear_float_oracle_50um/exact_population_cs")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if (output/"covariance_raw.npy").exists() or (output/"report.json").exists():
        raise FileExistsError("An artifact or partial artifact already exists; do not overwrite")
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    config = SpeckleGeneratorConfig()
    side, count = config.sampling, config.sampling**2
    required_bytes = 2*count*count*4 + 2*2**30
    free, total = torch.cuda.mem_get_info(device)
    if free < required_bytes:
        raise RuntimeError(f"Need {required_bytes/2**30:.2f} GiB free, have {free/2**30:.2f}; do not compete with other tasks")
    started = time.perf_counter()
    print(json.dumps({"stage":"start","matrix_shape":[count,count],"disk_bytes":count*count*4,
                      "estimated_workspace_gib":required_bytes/2**30}),flush=True)
    transfer = finite_phase_transfer(config, device=device, dtype=torch.float32, row_chunk=256)
    torch.cuda.synchronize(device)
    print(json.dumps({"stage":"field_transfer_ready","elapsed_s":time.perf_counter()-started}),flush=True)
    covariance, mean = dense_phase_covariance(transfer,row_chunk=512,consume_transfer=True,source_chunk=1024)
    del transfer
    # Only arithmetic symmetrization. No eigenvalue clipping, added nugget,
    # center normalization, or total-sum normalization changes the generator.
    chunk = 4096
    for start in range(0,count,chunk):
        for other in range(start,count,chunk):
            left = covariance[start:start+chunk,other:other+chunk]
            right = covariance[other:other+chunk,start:start+chunk]
            averaged = .5*(left+right.T)
            left.copy_(averaged)
            if other != start: right.copy_(averaged.T)
    torch.cuda.synchronize(device)
    print(json.dumps({"stage":"population_covariance_ready","elapsed_s":time.perf_counter()-started}),flush=True)
    refmean, refvar, _ = exact_moments(config)
    mean_error = float((mean.cpu().double().reshape(side,side)-refmean).norm()/refmean.norm())
    diagonal_error = float((covariance.diag().cpu().double().reshape(side,side)-refvar).norm()/refvar.norm())
    lag_checks = {}
    grid = torch.arange(count,device=device).reshape(side,side)
    for dy,dx in ((1,0),(0,1),(17,49),(49,49),(129,0),(259,259)):
        _, reference, _ = exact_moments(config,(dy,dx))
        ref = reference[:side-dy,:side-dx].flatten()
        indices = grid[:side-dy,:side-dx].flatten()
        actual = covariance[indices,indices+dy*side+dx].cpu().double()
        absolute = float((actual-ref).abs().max())
        lag_checks[f"{dy},{dx}"] = {"relative_l2":float((actual-ref).norm()/ref.norm().clamp_min(1e-30)),
                                       "maximum_error_over_max_variance":absolute/float(refvar.max())}
    rng = torch.Generator(device=device).manual_seed(20260991)
    probes = torch.randn((32,count),generator=rng,device=device)/count**.5
    quadratic = ((probes@covariance)*probes).sum(1)
    indices = torch.randperm(count,generator=rng,device=device)[:512]
    principal = covariance[indices[:,None],indices[None]].double()
    minimum_principal_eigenvalue = float(torch.linalg.eigvalsh(principal).min())
    minimum_quadratic = float(quadratic.min())
    for start in range(0,count,1024):
        if not torch.isfinite(covariance[start:start+1024]).all():
            raise FloatingPointError("Nonfinite covariance")
    maximum_lag_error = max(item["maximum_error_over_max_variance"] for item in lag_checks.values())
    if max(mean_error,diagonal_error,maximum_lag_error)>2e-5 or min(minimum_quadratic,minimum_principal_eigenvalue)<-1e-8:
        raise AssertionError((mean_error,diagonal_error,maximum_lag_error,minimum_quadratic,minimum_principal_eigenvalue))
    peak = torch.cuda.max_memory_allocated(device)
    if peak>42*2**30: raise RuntimeError("Build exceeded 42 GiB allocation budget")
    output.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"stage":"validated_saving","elapsed_s":time.perf_counter()-started,
                      "mean_relative_error":mean_error,"variance_relative_error":diagonal_error,
                      "peak_gpu_gib":peak/2**30}),flush=True)
    matrix_cpu = covariance.cpu().numpy()
    np.save(output/"covariance_raw.npy",matrix_cpu,allow_pickle=False)
    np.save(output/"mean_raw.npy",mean.cpu().numpy().reshape(side,side),allow_pickle=False)
    report = {"complete":True,"arguments":vars(args),"object_shape":[side,side],
              "generator":config.to_dict(),"uses_random_illumination_realizations":False,
              "formula":"Cs_raw = (B B^T)^2 - (B^2)(B^2)^T, elementwise powers; mean_raw=(B^2)1",
              "units":"Raw intensity covariance. Runtime divides by fixed system mean squared; no scene fit.",
              "scope":"Only finite linear random-phase simulator, before any per-realization normalization. Not a real-system calibration or a cross-depth covariance model.",
              "arithmetic":"Float32 GEMM, source chunks 1024 accumulated in Float64, TF32 disabled; Float64 impulse; arithmetic symmetrization only; no PSD projection/nugget",
              "mean_relative_l2_error":mean_error,"variance_relative_l2_error":diagonal_error,
              "lag_checks":lag_checks,"random_quadratic_minimum":minimum_quadratic,
              "principal_512_minimum_eigenvalue":minimum_principal_eigenvalue,
              "psd_caution":"Formula is mathematically PSD. Random directions and one principal block check rounding only; full numerical eigenspectrum not computed.",
              "matrix_bytes":int(matrix_cpu.nbytes),"peak_gpu_memory_gib":peak/2**30,
              "elapsed_s":time.perf_counter()-started}
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
