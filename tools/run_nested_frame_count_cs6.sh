#!/usr/bin/env bash
set -euo pipefail

root="outputs/linear_float_oracle_50um"
dataset_root="${root}/frame_count_nested"
result_root="${root}/frame_count_results"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"

run_one() {
  local gpu="$1"
  local count="$2"
  local bound="$3"
  local label="n${count}_b${bound/./p}_s0p5"
  local output="${result_root}/${label}"
  mkdir -p "${output}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=. \
    "${python_bin}" -u /tmp/train_smoothed_cs_variance.py \
    --device "cuda:${gpu}" \
    --dataset "${dataset_root}/n${count}" \
    --output-dir "${output}" \
    --bound "${bound}" \
    --smoothing-sigma 0.5 \
    --steps 200 \
    --learning-rate 0.01 \
    --train-probes 16 \
    --holdout-probes 512 \
    --probe-batch-size 16 \
    --mask-threshold 0.005 \
    --save-every 20 \
    --seed 20260901 \
    >"${output}/run.log" 2>&1
}

mkdir -p "${result_root}"
run_one 0 200 0.5 & pid0=$!
run_one 1 200 2.0 & pid1=$!
run_one 2 400 0.5 & pid2=$!
run_one 3 400 2.0 & pid3=$!
run_one 4 800 0.5 & pid4=$!
run_one 5 800 2.0 & pid5=$!

printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' \
  "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5"

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
