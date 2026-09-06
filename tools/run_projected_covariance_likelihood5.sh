#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/projected_covariance_likelihood"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
run_one() {
  local gpu="$1" label="$2"; shift 2
  mkdir -p "${root}/${label}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${python_bin}" -u tools/diagnose_projected_covariance_likelihood.py \
    --device "cuda:${gpu}" --output "${root}/${label}" --steps 200 --save-every 20 \
    --dimension 128 --basis-sigma 4 --ridge 1e-3 --bound 0.5 "$@" \
    >"${root}/${label}/run.log" 2>&1
}
run_one 0 n800_mse_d128 --loss mse & p0=$!
run_one 1 n800_nll_d128_r0 --loss nll & p1=$!
run_one 3 n800_nll_d128_reg001 --loss nll --regularizer-gradient-ratio 0.01 & p3=$!
run_one 4 n800_nll_d128_reg010 --loss nll --regularizer-gradient-ratio 0.10 & p4=$!
run_one 5 oracle_nll_d128_r0 --loss nll --target oracle & p5=$!
printf 'GPU0=%s GPU1=%s GPU3=%s GPU4=%s GPU5=%s\n' "$p0" "$p1" "$p3" "$p4" "$p5"
status=0
for pid in "$p0" "$p1" "$p3" "$p4" "$p5"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
