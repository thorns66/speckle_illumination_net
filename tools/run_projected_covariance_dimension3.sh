#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/projected_covariance_likelihood"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
run_one() {
  local gpu="$1" label="$2" dimension="$3" loss="$4"
  mkdir -p "${root}/${label}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${python_bin}" -u tools/diagnose_projected_covariance_likelihood.py \
    --device "cuda:${gpu}" --output "${root}/${label}" --steps 200 --save-every 20 \
    --dimension "$dimension" --basis-sigma 4 --ridge 1e-3 --bound 0.5 \
    --loss "$loss" --target oracle >"${root}/${label}/run.log" 2>&1
}
run_one 0 oracle_nll_d512_r0 512 nll & p0=$!
run_one 1 oracle_nll_d1024_r0 1024 nll & p1=$!
run_one 3 oracle_mse_d1024_r0 1024 mse & p3=$!
printf 'GPU0=%s GPU1=%s GPU3=%s\n' "$p0" "$p1" "$p3"
status=0
for pid in "$p0" "$p1" "$p3"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
