#!/usr/bin/env bash
set -euo pipefail
CM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
for method in serenet vcdnet; do
  prefix="$CM_DIR/.envs/$method"
  if [[ ! -x "$prefix/bin/python" ]]; then
    if [[ -f "$CM_DIR/environments/${method}.conda-explicit.txt" ]]; then
      conda create -y --copy --prefix "$prefix" --file "$CM_DIR/environments/${method}.conda-explicit.txt"
    else
      conda create -y --copy --prefix "$prefix" python=3.11 pip
    fi
    "$prefix/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    if [[ -f "$CM_DIR/environments/${method}.lock.txt" ]]; then
      "$prefix/bin/python" -m pip install -r "$CM_DIR/environments/${method}.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu121
    else
      "$prefix/bin/python" -m pip install -r "$CM_DIR/environments/common.txt"
    fi
  fi
  "$prefix/bin/python" -m pip check
  "$prefix/bin/python" -m pip freeze > "$CM_DIR/environments/${method}.lock.txt"
  conda list --explicit --prefix "$prefix" > "$CM_DIR/environments/${method}.conda-explicit.txt"
done
