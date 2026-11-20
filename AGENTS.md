# Agents

Canonical instructions for every AI coding agent in this project. Write
guidance **here**, and keep it to what an agent would otherwise get wrong: it
is loaded into every session.

## Project

Runs ECMWF's AIFS Single v2.0 from the latest ECMWF open-data cycle
(`pipeline/forecast.py`) and renders regional weather cards
(`pipeline/predict_now.py`). User-facing docs are in `INSTRUCTION.MD`.

## Running

- Setup: `./setup.sh [--cpu]` (anemoi-core, `.venv`, libgfortran; safe to re-run).
- Full run: `./predict_now.sh` (latest cycle, 240 h, Europe). Use `--skip-forecast` to
  re-render cards only.
- Forecast only: `pipeline/run_forecast.sh [--lead-time 12h] [--device cuda]`.
- Always use `.venv/bin/python`. System Python is 3.10 and lacks the dependencies.
- AIFS ENS (`pipeline/ensemble.py --model ens`) runs in `.venv-ens` (`./setup.sh --ens`), which
  pins newer anemoi packages than `.venv`. Don't install those pins into `.venv`.

## Conventions

- Package versions are pinned to what the checkpoint was published with (see
  `models/aifs-single-2.0/pyproject.toml`). Do not upgrade anemoi-* packages to fix
  something. Fix it at runtime in `pipeline/compat/` instead.
- The checkpoint pickles `flash_attn.flash_attn_interface.flash_attn_func`.
  `pipeline/compat/flash_attn/` must stay API-compatible with that function.
- Accumulated GRIB fields (tp, cp, sf, ro, ssrd, strd) are accumulated from the forecast
  start and have no step-0 field.
- Keep caches out of `$HOME`, which has a small quota. Use `.cache/` in the repo and
  `UV_CACHE_DIR` on a large disk.

## Boundaries

- Never commit `models/`, `anemoi-core/`, `forecasts*/`, `.sysdeps/` or `.cache/`. GRIB
  files are ~2 GB.
- Don't start long CPU forecasts; they take hours. Test on a GPU with `--lead-time 12h`.
- Ask before changing the cron setup or deleting forecast output.

## Verify

- `.venv/bin/python -m pytest -q tests` passes (no GPU or forecast data needed).
- `pipeline/run_forecast.sh --lead-time 12h --device cuda --output-root forecasts-test`
  ends with `done:` and writes `DONE`.
- `./predict_now.sh --skip-forecast --output-root forecasts-test` reports
  `rendered N cards` with no failures.
- Look at the PNGs, not just the exit code.
