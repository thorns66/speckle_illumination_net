#!/usr/bin/env bash
set -euo pipefail
CM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:-serenet}"; GPU="${2:-3}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$GPU"
unset PYTHONPATH PYTHONHOME
exec "$CM_DIR/.envs/$METHOD/bin/python" -B "$CM_DIR/tests/preflight.py" --method "$METHOD" "${@:3}"
