#!/usr/bin/env bash
# Run the AIFS forecast for the latest cycle and render weather cards (see pipeline/predict_now.py -h).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "error: $ROOT/.venv is missing; run ./setup.sh first (see INSTRUCTION.MD)" >&2
    exit 1
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/mpl}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
exec "$ROOT/.venv/bin/python" "$ROOT/pipeline/predict_now.py" "$@"
