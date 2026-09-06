#!/usr/bin/env bash
set -euo pipefail

# Five independent 50/50 frame splits. The caller must first verify that every
# GPU in GPU_LIST is idle. Example override: GPU_LIST="0 2 3 4 5".
read -r -a gpus <<< "${GPU_LIST:-1 2 3 4 5}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  printf '%s\n' "GPU_LIST must contain exactly five GPU indices" >&2
  exit 2
fi

root="outputs/fixed_depth_lateral_50um_h2_crossfit5"
common=(
  --config configs/depth50_n100_no_mean_loss.yaml
  --mode h2
  --anchor f_var
  --bound 0.5
  --curvature-gradient-ratio 0.03
  --steps 200
  --learning-rate 0.01
  --train-frame-fraction 0.5
  --save-every 5
  --phase-chunk-size 16
  --seed 20260901
)

run_one() {
  local gpu="$1"
  local fold="$2"
  local split_seed="$3"
  local label
  label=$(printf 'fold%02d_seed%d' "${fold}" "${split_seed}")
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
    conda run --no-capture-output -n speckle_net \
    python -u tools/diagnose_fixed_depth_lateral.py \
    --device "cuda:${gpu}" \
    --frame-split-seed "${split_seed}" \
    --output-dir "${root}/${label}" \
    "${common[@]}" \
    >"${root}/${label}.log" 2>&1
}

mkdir -p "${root}"
pids=()
for fold in 0 1 2 3 4; do
  split_seed=$((20260906 + fold))
  run_one "${gpus[fold]}" "${fold}" "${split_seed}" &
  pids+=("$!")
  printf 'fold=%d gpu=%s split_seed=%d pid=%s\n' \
    "${fold}" "${gpus[fold]}" "${split_seed}" "${pids[fold]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
