#!/usr/bin/env bash
set -euo pipefail
CM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="$1"; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
ARGS=()
while (( $# )); do
  case "$1" in
    --output|--resume|--checkpoint|--predictions)
      ARGS+=("$1" "$(realpath -m -- "$2")"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
cd "$CM_DIR"
exec "$CM_DIR/.envs/$METHOD/bin/python" -B -m adapters.run "${ARGS[@]}" --method "$METHOD"
