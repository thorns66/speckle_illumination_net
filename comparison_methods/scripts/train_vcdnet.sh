#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "$0")/launch.sh" vcdnet train "$@"
