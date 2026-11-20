#!/usr/bin/env bash
# One-time environment setup: anemoi-core clone, .venv with the pinned packages, and
# libgfortran if the system lacks it. Safe to re-run; finished steps are skipped.
# The model checkpoint is not fetched here; the first forecast run downloads it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ANEMOI_CORE_URL="https://github.com/ecmwf/anemoi-core.git"
ANEMOI_CORE_TAG="models-0.9.3"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

usage() {
    cat <<EOF
Usage: ./setup.sh [--cpu] [--torch-index URL]

  --cpu              Install the CPU-only torch wheel (no CUDA)
  --torch-index URL  PyTorch wheel index (default: $TORCH_INDEX)

Caches go to .cache/ in the repo unless UV_CACHE_DIR / CONDA_PKGS_DIRS are set.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        --torch-index) TORCH_INDEX="${2:?--torch-index needs a URL}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# $HOME has a small quota on some machines: keep package caches in the repo by default.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"

command -v git >/dev/null || die "git not found"
command -v uv >/dev/null || die "uv not found; install it from https://docs.astral.sh/uv/"

# 1. anemoi-core (anemoi-models is installed from it, editable)
if [[ ! -d anemoi-core/.git ]]; then
    log "cloning anemoi-core at $ANEMOI_CORE_TAG"
    git clone --branch "$ANEMOI_CORE_TAG" "$ANEMOI_CORE_URL" anemoi-core
else
    head_tag="$(git -C anemoi-core describe --tags --exact-match 2>/dev/null || true)"
    if [[ "$head_tag" != "$ANEMOI_CORE_TAG" ]]; then
        die "anemoi-core/ is not at $ANEMOI_CORE_TAG (HEAD: ${head_tag:-untagged});" \
            "run: git -C anemoi-core checkout $ANEMOI_CORE_TAG"
    fi
    log "anemoi-core already at $ANEMOI_CORE_TAG"
fi

# 2. Python 3.12 environment with the versions the checkpoint pins
#    (models/aifs-single-2.0/pyproject.toml). Don't bump these to fix things; see AGENTS.md.
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    [[ -e .venv ]] && die ".venv/ exists but has no bin/python; remove it and re-run"
    log "creating .venv (Python 3.12)"
    uv venv -p 3.12 .venv
else
    log ".venv already exists"
fi

# torch/torchvision come from the PyTorch index and are repeated below: otherwise the
# second resolve pulls the newest torchvision from PyPI, which upgrades torch.
TORCH_PINS=(torch==2.7.0 torchvision==0.22.0)
log "installing ${TORCH_PINS[*]} from $TORCH_INDEX"
uv pip install --python "$PY" "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"

log "installing anemoi stack and pipeline dependencies"
uv pip install --python "$PY" "${TORCH_PINS[@]}" \
    -e anemoi-core/models anemoi-graphs==0.6.4 anemoi-training==0.6.3 \
    "anemoi-inference[huggingface]==0.8.3" "anemoi-plugins-ecmwf-inference[opendata]==0.2.1" \
    anemoi-transform==0.1.16.post2 anemoi-utils==0.4.35.post3 torch-geometric==2.6.1 \
    earthkit-regrid==0.5.1 ecmwf-opendata==0.3.29 "earthkit-data<1" "earthkit-plots<1" \
    cartopy matplotlib scipy pyyaml pytest

# 3. libgfortran.so.5 (needed by mir's atlas binding). pipeline/forecast.py puts
#    .sysdeps/runtime on LD_LIBRARY_PATH, so no sudo is needed.
ldconfig_bin="$(command -v ldconfig || echo /sbin/ldconfig)"
if "$ldconfig_bin" -p 2>/dev/null | grep -q 'libgfortran\.so\.5'; then
    log "libgfortran.so.5 found on the system"
elif [[ -e .sysdeps/runtime/libgfortran.so.5 ]]; then
    log "libgfortran.so.5 already in .sysdeps/runtime"
else
    command -v conda >/dev/null || die "libgfortran.so.5 missing and conda not found;" \
        "install libgfortran5 from your package manager or conda-forge"
    log "installing libgfortran5 from conda-forge into .sysdeps"
    [[ -d .sysdeps/conda-meta ]] || conda create -y -p .sysdeps -c conda-forge --override-channels libgfortran5
    mkdir -p .sysdeps/runtime
    ln -sf ../lib/libgfortran.so.5 ../lib/libquadmath.so.0 .sysdeps/runtime/
fi

log "checking imports"
"$PY" -W ignore -c "import torch, anemoi.models, anemoi.inference, earthkit.data, cartopy; \
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

log "done. Next: ./predict_now.sh  (the first run downloads the ~1 GB model)"
