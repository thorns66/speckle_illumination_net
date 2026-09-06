#!/usr/bin/env bash
set -euo pipefail
result_root="outputs/linear_float_oracle_50um/cs_information"
run_one() {
  local gpu="$1" label="$2" norm="$3" bound="$4"
  mkdir -p "${result_root}/${label}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /workspace/xyx/.conda/envs/speckle_net/bin/python -u tools/diagnose_cs_vector_refresh.py \
    --device "cuda:${gpu}" --output "${result_root}/${label}" \
    --normalization "${norm}" --bound "${bound}" --steps 200 \
    >"${result_root}/${label}/run.log" 2>&1
}
run_one 3 n800_vector_refresh_per_b0p5 per_probe 0.5 & pid3=$!
run_one 4 n800_vector_refresh_global_b0p5 global 0.5 & pid4=$!
run_one 5 n800_vector_refresh_global_b2 global 2 & pid5=$!
printf 'GPU3=%s GPU4=%s GPU5=%s\n' "$pid3" "$pid4" "$pid5"
status=0
for pid in "$pid3" "$pid4" "$pid5"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
