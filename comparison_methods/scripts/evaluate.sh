#!/usr/bin/env bash
set -euo pipefail
METHOD="$1"; shift
exec bash "$(dirname -- "$0")/launch.sh" "$METHOD" evaluate "$@"
