#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/projected_covariance_likelihood"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
run_one() {
  local gpu="$1" label="$2" ridge="$3"
  mkdir -p "${root}/${label}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${python_bin}" -u tools/diagnose_projected_covariance_likelihood.py \
    --device "cuda:${gpu}" --output "${root}/${label}" --steps 200 --save-every 20 \
    --dimension 1024 --basis-sigma 4 --ridge "$ridge" --bound 0.5 \
    --loss nll --target oracle >"${root}/${label}/run.log" 2>&1
}
run_one 0 oracle_nll_d1024_j1e4 1e-4 & p0=$!
run_one 1 oracle_nll_d1024_j1e5 1e-5 & p1=$!
printf 'GPU0=%s GPU1=%s\n' "$p0" "$p1"
status=0
for pid in "$p0" "$p1"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
