#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/oracle_provenance"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
run_one() {
  local gpu="$1" name="$2"; shift 2
  mkdir -p "${root}/${name}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u tools/diagnose_covariance_oracle_provenance.py \
    --device "cuda:${gpu}" --output "${root}/${name}" "$@" >"${root}/${name}/run.log" 2>&1
}
run_one 0 stationary_empirical --cs-type stationary_empirical & p0=$!
run_one 1 stationary_analytic --cs-type stationary_analytic & p1=$!
run_one 2 finite128_matched --cs-type finite --patterns 128 & p2=$!
run_one 3 finite1024_matched --cs-type finite --patterns 1024 & p3=$!
run_one 4 finite8192_matched --cs-type finite --patterns 8192 & p4=$!
run_one 5 finite1024_independent --cs-type finite --patterns 1024 --target-link independent & p5=$!
printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
