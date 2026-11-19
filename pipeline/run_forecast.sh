#!/usr/bin/env bash
# Wrapper for cron/manual use: runs the AIFS forecast pipeline with the project venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/forecasts"
exec "$ROOT/.venv/bin/python" "$ROOT/pipeline/forecast.py" "$@" 2>&1 | tee -a "$ROOT/forecasts/pipeline.log"
