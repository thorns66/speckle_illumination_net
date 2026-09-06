#!/usr/bin/env bash
set -euo pipefail

root="outputs/fixed_depth_lateral_50um_crossfit"
common=(
  --config configs/depth50_n100_no_mean_loss.yaml
  --anchor f_var
  --steps 200
  --learning-rate 0.01
  --train-frame-fraction 0.5
  --frame-split-seed 20260906
  --train-probes 16
  --probes-per-step 4
  --holdout-probes 16
  --probe-sigma 16
  --train-patterns 1024
  --holdout-patterns 2048
  --phase-chunk-size 16
  --save-every 5
)

run_one() {
  local gpu="$1"
  local label="$2"
  shift 2
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
    conda run --no-capture-output -n speckle_net \
    python -u tools/diagnose_fixed_depth_lateral.py \
    --device "cuda:${gpu}" \
    --output-dir "${root}/${label}" \
    "${common[@]}" "$@" \
    >"${root}/${label}.log" 2>&1
}

mkdir -p "${root}"
run_one 1 xh1_h2_b0p25_curv001 --mode h2 --bound 0.25 \
  --curvature-gradient-ratio 0.01 &
pid1=$!
run_one 2 xh2_h2_b0p5_curv003 --mode h2 --bound 0.5 \
  --curvature-gradient-ratio 0.03 &
pid2=$!
run_one 3 xc1_cov_b0p25_curv001 --mode covariance --bound 0.25 \
  --curvature-gradient-ratio 0.01 &
pid3=$!
run_one 4 xc2_cov_b0p5_curv003 --mode covariance --bound 0.5 \
  --curvature-gradient-ratio 0.03 &
pid4=$!
run_one 5 xj1_joint_b0p5_r01_curv003 --mode joint --bound 0.5 \
  --covariance-gradient-ratio 0.1 --curvature-gradient-ratio 0.03 &
pid5=$!

printf '%s\n' "GPU1=${pid1}" "GPU2=${pid2}" "GPU3=${pid3}" "GPU4=${pid4}" "GPU5=${pid5}"
wait "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"
