#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/oracle_sensor_bandwidth"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
run_one() {
  local gpu="$1" name="$2" sigma="$3" bound="$4"
  mkdir -p "${root}/${name}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u tools/diagnose_covariance_oracle_provenance.py \
    --device "cuda:${gpu}" --output "${root}/${name}" --cs-type stationary_analytic \
    --probe-sigma "$sigma" --bound "$bound" >"${root}/${name}/run.log" 2>&1
}
run_one 0 sigma0_b0p5 0 0.5 & p0=$!
run_one 1 sigma0p5_b0p5 0.5 0.5 & p1=$!
run_one 2 sigma1_b0p5 1 0.5 & p2=$!
run_one 3 sigma2_b0p5 2 0.5 & p3=$!
run_one 4 sigma0_b2 0 2 & p4=$!
run_one 5 sigma4_b2 4 2 & p5=$!
printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
