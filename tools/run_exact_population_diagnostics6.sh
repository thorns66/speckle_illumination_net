#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/exact_population_diagnostics"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
mkdir -p "$root"
audit_one() {
  local gpu="$1" n="$2" width="$3" dataset="$4"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u tools/audit_oracle_candidate_empirical_transfer.py \
    --device "cuda:${gpu}" --dataset "$dataset" --cs-model finite_phase --window-sigma "$width" \
    --output "$root/n${n}_w${width}.json" >"$root/n${n}_w${width}.log" 2>&1
}
oracle_one() {
  local gpu="$1" name="$2" bound="$3"
  mkdir -p "$root/$name"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u tools/diagnose_covariance_oracle_provenance.py \
    --device "cuda:${gpu}" --output "$root/$name" --cs-type finite_phase \
    --probe-sigma 0 --bound "$bound" >"$root/$name/run.log" 2>&1
}
audit_one 0 100 0 outputs/linear_float_oracle_50um/dataset & p0=$!
audit_one 1 800 0 outputs/linear_float_oracle_50um/frame_count_nested/n800 & p1=$!
audit_one 2 100 16 outputs/linear_float_oracle_50um/dataset & p2=$!
audit_one 3 800 16 outputs/linear_float_oracle_50um/frame_count_nested/n800 & p3=$!
oracle_one 4 oracle_b0p5 0.5 & p4=$!
oracle_one 5 oracle_b2 2 & p5=$!
printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
