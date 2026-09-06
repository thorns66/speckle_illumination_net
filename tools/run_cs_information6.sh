#!/usr/bin/env bash
set -euo pipefail

result_root="outputs/linear_float_oracle_50um/cs_information"
dataset_root="outputs/linear_float_oracle_50um/frame_count_nested"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"

run_one() {
  local gpu="$1" label="$2" count="$3"
  shift 3
  mkdir -p "${result_root}/${label}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${python_bin}" -u tools/diagnose_cs_information.py \
    --device "cuda:${gpu}" --dataset "${dataset_root}/n${count}" \
    --output "${result_root}/${label}" --steps 200 --save-every 20 \
    --seed 20260901 --learning-rate 0.01 "$@" \
    >"${result_root}/${label}/run.log" 2>&1
}

run_one 0 n800_self16_b0p5 800 --mode self --object-probes 16 & pid0=$!
run_one 1 n800_ustat16_b0p5 800 --mode ustat --object-probes 16 & pid1=$!
run_one 2 n800_ustat64_b0p5 800 --mode ustat --object-probes 64 & pid2=$!
run_one 3 n800_vector_b0p5 800 --mode vector --bound 0.5 & pid3=$!
run_one 4 n800_vector_b2 800 --mode vector --bound 2 & pid4=$!
run_one 5 n400_vector_b0p5 400 --mode vector --bound 0.5 & pid5=$!
printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5"

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
