#!/usr/bin/env bash
# Wrapper for cron/manual use: runs the AIFS forecast pipeline with the project venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "error: $ROOT/.venv is missing; run ./setup.sh first (see INSTRUCTION.MD)" >&2
    exit 1
fi
mkdir -p "$ROOT/forecasts"
exec "$ROOT/.venv/bin/python" "$ROOT/pipeline/forecast.py" "$@" 2>&1 | tee -a "$ROOT/forecasts/pipeline.log"
