#!/usr/bin/env bash
set -euo pipefail
root="outputs/linear_float_oracle_50um/localized_covariance_audit"
python_bin="/workspace/xyx/.conda/envs/speckle_net/bin/python"
mkdir -p "$root"
run_one() {
  local gpu="$1" n="$2" width="$3" dataset="$4"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u tools/audit_oracle_candidate_empirical_transfer.py \
    --device "cuda:${gpu}" --dataset "$dataset" --window-sigma "$width" \
    --output "$root/n${n}_w${width}.json" >"$root/n${n}_w${width}.log" 2>&1
}
run_one 0 100 16 outputs/linear_float_oracle_50um/dataset & p0=$!
run_one 1 100 32 outputs/linear_float_oracle_50um/dataset & p1=$!
run_one 2 100 64 outputs/linear_float_oracle_50um/dataset & p2=$!
run_one 3 800 16 outputs/linear_float_oracle_50um/frame_count_nested/n800 & p3=$!
run_one 4 800 32 outputs/linear_float_oracle_50um/frame_count_nested/n800 & p4=$!
run_one 5 800 64 outputs/linear_float_oracle_50um/frame_count_nested/n800 & p5=$!
printf 'GPU0=%s GPU1=%s GPU2=%s GPU3=%s GPU4=%s GPU5=%s\n' "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
