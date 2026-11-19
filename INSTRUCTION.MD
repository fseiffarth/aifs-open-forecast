# AIFS forecast pipeline: how to run it

This pipeline runs **ECMWF AIFS Single v2.0** (`ecmwf/aifs-single-2.0`, operational since
12 May 2026) from the **newest available ECMWF open-data analysis**. Every run forecasts
from "now", meaning the latest published 00/06/12/18 UTC cycle. `predict_now.sh` then turns
the forecast into weather cards (Germany by default).

All commands below are run from the repository root.

## What is where

| Path | Content | In git? |
|------|---------|---------|
| `pipeline/forecast.py` | The forecast pipeline (latest cycle → GRIB) | yes |
| `pipeline/run_forecast.sh` | Wrapper that uses the venv and appends to `forecasts/pipeline.log` | yes |
| `pipeline/predict_now.py`, `predict_now.sh` | Forecast + weather cards | yes |
| `pipeline/compat/` | Compatibility fixes, see [Compatibility fixes](#compatibility-fixes) | yes |
| `models/aifs-single-2.0/` | Model checkpoint and config from Hugging Face | no (downloaded on first run) |
| `anemoi-core/` | Clone of `ecmwf/anemoi-core` at tag `models-0.9.3` | no (clone, see Setup) |
| `.venv/` | Python 3.12 environment | no |
| `.sysdeps/` | `libgfortran.so.5` from conda-forge (only if the system lacks it) | no |
| `.cache/` | All caches (earthkit, Hugging Face, cartopy, tmp) | no |
| `forecasts/<YYYYMMDDHH>/` | `aifs-single-2.0_<YYYYMMDDHH>.grib`, `config.yaml`, `run.log`, `DONE`, `cards/` | no |
| `forecasts/latest` | Symlink to the newest finished forecast | no |

## Setup (once per machine or checkout)

```bash
# 1. Model (≈ 1 GB, CC BY 4.0): optional. The first forecast run downloads it into
#    models/aifs-single-2.0/ and verifies the checksum. To fetch it ahead of time:
uvx --from huggingface_hub hf download ecmwf/aifs-single-2.0 --local-dir models/aifs-single-2.0
sha256sum models/aifs-single-2.0/aifs-single-mse-2.0.ckpt
# expected: 4e23ef40157b9e99b5228e5dcffb0d0d5a789f019ffcc89033ee9caae5d7ff6b

# 2. anemoi-core (anemoi-models is installed from it, editable)
git clone https://github.com/ecmwf/anemoi-core.git
git -C anemoi-core checkout models-0.9.3

# 3. Python 3.12 environment with the versions the checkpoint pins
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python \
  torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  -e anemoi-core/models anemoi-graphs==0.6.4 anemoi-training==0.6.3 \
  "anemoi-inference[huggingface]==0.8.3" "anemoi-plugins-ecmwf-inference[opendata]==0.2.1" \
  anemoi-transform==0.1.16.post2 anemoi-utils==0.4.35.post3 torch-geometric==2.6.1 \
  earthkit-regrid==0.5.1 ecmwf-opendata==0.3.29 "earthkit-data<1" "earthkit-plots<1" \
  cartopy matplotlib scipy pyyaml

# 4. Only if `ldconfig -p | grep libgfortran.so.5` prints nothing (no sudo needed):
conda create -y -p .sysdeps -c conda-forge --override-channels libgfortran5
mkdir -p .sysdeps/runtime
ln -sf ../lib/libgfortran.so.5 ../lib/libquadmath.so.0 .sysdeps/runtime/
```

flash-attn is optional. See the hardware notes below.

If your home directory has a small quota, set `UV_CACHE_DIR` (and `CONDA_PKGS_DIRS`) to a
directory on a larger disk before steps 3 and 4. The pipeline keeps its own caches in `.cache/`.

## Run a forecast now

```bash
pipeline/run_forecast.sh                      # latest cycle, 10 days (240h), device auto
pipeline/run_forecast.sh --lead-time 48h      # shorter forecast
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--lead-time` | `240h` | Forecast length (`12h`, `5d`, `240h`, …). The model steps 6 h at a time |
| `--device` | `auto` | `cpu`, `cuda` or `auto`. `auto` uses the GPU if it has ≥ 20 GB free; otherwise CPU |
| `--date` | latest | Explicit initial time `YYYYMMDDHH`. Open data only keeps the last few days |
| `--source` | `ecmwf` | Open-data mirror: `ecmwf`, `aws`, `azure` or `google` |
| `--force` | off | Re-run a cycle that is already marked `DONE` |
| `--threads` | all cores | CPU threads for torch |
| `--output-root` | `forecasts/` | Where to write results |

What one run does:

1. It asks open data for the newest cycle whose atmosphere **and** wave fields are both
   published. A cycle normally appears 7–9 h after its nominal time.
2. If that cycle already has `DONE`, it exits immediately.
3. It writes `config.yaml` (the model's `inference.yaml` plus date, lead time, device and
   absolute paths).
4. It runs anemoi-inference, which downloads the initial conditions (t and t−6h), regrids
   them to N320, rolls the model forward and writes GRIB.

A lock file (`forecasts/.lock`) stops two runs from overlapping.

## Predict now & weather cards

`predict_now.sh` runs the forecast for the latest cycle and then renders weather cards for a
region. It goes through `pipeline/forecast.py`, so a finished cycle is reused and the lock
still applies.

```bash
./predict_now.sh                                 # latest cycle, 10 days (240 h), Germany
./predict_now.sh --lead-time 48h                 # shorter forecast
./predict_now.sh --skip-forecast                 # only render the newest finished cycle
./predict_now.sh --grib forecasts/2026111900/aifs-single-2.0_2026111900.grib   # render one file
./predict_now.sh --skip-forecast --region europe --workers 16
```

Options:

- `--region`: a named bounding box from the `REGIONS` dict in `pipeline/predict_now.py`.
  Add your own with a bbox, time zone and cities.
- `--lead-time`: default 240h (10 days).
- `--device`, `--force`, `--date`, `--source`, `--output-root`, `--workers`.
- `--only`: render a subset, using GRIB codes such as `2t,tp,t_850hPa`.

- `--ensemble {auto,off}`: `auto` (default) adds ensemble bands, probabilities and the
  `5-ensemble_*` cards when `forecasts/<cycle>/ensemble/<region>/` has at least 3 members;
  `off` renders the deterministic forecast only. The number of members used is logged and
  shown on the cards.
- `--ensemble-members N`: after the forecast, run `pipeline/ensemble.py --members N` (default
  0 = do not run it). `N` must be even; 50 gives 51 members.

Output goes to `forecasts/<cycle>/cards/<region>/`. Open `index.html` there. File names
sort in the same order as the gallery:

| File | Content |
|------|---------|
| `0-overview.png` | Per-city forecast (temperature, rain, cloud cover, wind; local time) |
| `1-surface_<name>.png` | Surface and soil fields, e.g. `1-surface_2m-temperature.png`, `1-surface_10m-wind-speed.png`, `1-surface_cloud-cover-low.png`, `1-surface_solar-radiation-down.png` |
| `1-surface_<name>_amounts.png` / `_accumulated-total.png` | Accumulations, e.g. `1-surface_precipitation_amounts.png` (6-hourly maps for short forecasts, daily totals for long ones), `1-surface_snowfall_accumulated-total.png` |
| `2-upper-air_<name>_<llll>hPa.png` | Pressure levels, e.g. `2-upper-air_temperature_0850hPa.png`, `2-upper-air_geopotential-height_0500hPa.png` |
| `3-waves_<name>.png` | Ocean waves, e.g. `3-waves_wave-height.png`, `3-waves_wave-height-10-12s.png` |
| `4-constant_<name>.png` | Constant fields: `elevation`, `land-sea-mask`, `sea-depth`, … |
| `5-ensemble_<name>.png` | Ensemble mean and spread (or probabilities for precipitation), only with an ensemble |

What the cards contain:

- Map frames are evenly spaced. The stride is the smallest of 6, 12, 24, 48 h … that fits in
  12 frames. A 10-day forecast therefore shows one map per day at 00 UTC. Accumulations are
  summed over the stride, giving daily totals, so no rain is lost between frames.

- Each card has a small-multiples map strip with a shared colour bar, coastlines, borders and
  state borders. Below it is a time series for the region's cities, using the nearest grid point.
- Units are converted to °C, hPa, mm, m/s (with km/h), and gpm.
- Radiation is shown as mean W/m² per interval.
- Fields with no data in the region (e.g. waves over land) are listed as skipped in `index.html`.
- Natural Earth shapes are downloaded on first use into `.cache/cartopy`.

## Ensemble forecast (forecast confidence)

AIFS Single is deterministic. `pipeline/ensemble.py` estimates forecast uncertainty by running
the same model 51 times from slightly different initial conditions: member 0 is the unperturbed
control, members 1-50 add smooth random perturbations (about 500 km scale) to the atmospheric
fields, in +/- pairs. Amplitudes are about typical analysis errors (see the script header).

```bash
.venv/bin/python pipeline/ensemble.py                  # latest cycle, 50 + 1 members, 240 h, Germany
.venv/bin/python pipeline/ensemble.py --members 10     # fewer members (must be even)
./predict_now.sh --ensemble-members 50                 # forecast, then ensemble, then cards
```

- Runtime: about 50 s per member for 10 days on the RTX PRO 6000, so about 45 min for all 51.
  Do not run it while another forecast uses the GPU. A stopped run resumes and skips members
  that are already stored; `--force` starts again.
- Storage: only the region is kept, `forecasts/<cycle>/ensemble/<region>/member_NNN.npz`
  (about 19 MB per member for Germany, roughly 1 GB in total), plus `meta.json` and `DONE`.
  The full global output of 51 members would be about 117 GB.
- Card integration is automatic: `predict_now.sh` (with `--ensemble auto`) uses every member
  that is stored, even from a run in progress (at least 3), so cards can be re-rendered while the
  ensemble is still running. Use `--skip-forecast --date <cycle>` to render a specific cycle.
- Be careful with the result. This is a simple initial-condition ensemble and not ECMWF's ENS
  (no singular vectors, no ensemble of data assimilations, no model-error perturbations).
  Probabilities are not calibrated.
- One check against ECMWF's AIFS ENS (open data, cycle 2026-11-19 06Z, 50 perturbed members on
  each side, Germany box). The numbers are the ratio of our spread to ECMWF's spread. Spread is
  the area mean of the per-grid-point standard deviation over the members. A value of 1.00 means
  the same spread.

  | day | 2 m temp. | 10 m wind | MSL pressure | 850 hPa temp. | 500 hPa height |
  |----:|------:|------:|------:|------:|------:|
  |  1 | 0.59 | 0.64 | 1.37 | 0.81 | 1.51 |
  |  2 | 0.91 | 0.89 | 0.94 | 1.10 | 1.21 |
  |  3 | 0.69 | 0.73 | 1.08 | 1.07 | 1.39 |
  |  4 | 1.07 | 1.06 | 1.11 | 1.06 | 1.09 |
  |  5 | 0.89 | 0.89 | 1.06 | 1.00 | 1.10 |
  |  6 | 1.03 | 0.89 | 1.06 | 1.09 | 1.19 |
  |  7 | 0.92 | 0.91 | 1.05 | 1.16 | 1.10 |
  |  8 | 0.91 | 1.10 | 1.14 | 0.96 | 1.22 |
  |  9 | 1.01 | 0.98 | 0.98 | 1.00 | 1.12 |
  | 10 | 1.12 | 0.91 | 0.91 | 1.05 | 1.01 |

  Absolute values at day 10, ECMWF vs ours: 2 m temperature 3.4 / 3.8 K, MSL pressure
  10.6 / 9.6 hPa, 500 hPa height 128 / 130 gpm.

  - From day 4 on, the spread is within about 10-20 % of ECMWF's, for every variable.
  - In the first 3 days, the near-surface spread (2 m temperature, 10 m wind) is too small,
    by 30-40 % on day 1. The 500 hPa height spread is too large, by up to 50 %. The upper-air
    perturbations are a bit too strong, and the surface ones too weak. ECMWF's ensemble also
    perturbs the land surface and the model itself, which this ensemble does not.
  - In this case the deterministic forecast (the control) was about 6 K colder than both ensemble
    means at day 10 (ECMWF 4.75 °C, ours 4.57 °C). A control at the edge of the band can be real.
  - This is a single case. Spread agreement does not show that the ensemble is reliable, which
    would need verification against observations over many cycles.

## Keep it running automatically

New cycles appear four times a day. Schedule the wrapper hourly with cron. Each call either
finds a new cycle and forecasts it, or exits in seconds:

```bash
crontab -e
# add (use the absolute path of your checkout):
15 * * * * /path/to/aifs/predict_now.sh >/dev/null 2>&1
```

Progress and errors go to `forecasts/pipeline.log`. Details for each run are in
`forecasts/<cycle>/run.log`.

## Use the output

The output is GRIB on the model's native N320 reduced Gaussian grid (542,080 points), written
every 6 h from step 0. Accumulated fields (precipitation, snowfall, runoff, radiation) are
accumulated from the start of the forecast and have no step-0 field.

```bash
.venv/bin/python - <<'EOF'
import glob, earthkit.data as ekd
ds = ekd.from_source("file", glob.glob("forecasts/latest/*.grib")[0])
print(ds.ls(keys=["param", "level", "step"]).head(20))
t2m = ds.sel(param="2t")
print(t2m[-1].datetime(), t2m[-1].to_numpy().mean() - 273.15, "°C global mean")
EOF
```

## Hardware and measured timings

| Machine | Device | 12 h test | 10-day forecast |
|---------|--------|-----------|-----------------|
| RTX PRO 6000 Blackwell (96 GB, sm_120) | `cuda` | 1 min 34 s end to end | **3 min 23 s** for `predict_now.sh` (default 240 h): forecast 2 min 41 s (≈ 1 min download/regrid/model load, ≈ 50 s for 40 steps at ≈ 1 s each after a ≈ 10 s warm-up, ≈ 50 s writing 2.3 GB of GRIB), then 42 s rendering 131 cards |
| 2× RTX 2080 (8 GB) + 16-core CPU | `cpu` | did not finish the first step in > 1 h | not practical |

- **Use a GPU with ≥ 24 GB.** On CPU, the processor's sliding-window attention runs under
  mixed-precision autocast and took about 400 s per layer in profiling, so a single 6 h step
  takes hours.
- **flash-attn:** the checkpoint pickles a direct reference to
  `flash_attn.flash_attn_interface.flash_attn_func`, so the module must be importable even to
  load the model. If the real flash-attn isn't installed, or the GPU is older than Ampere,
  the pipeline puts `pipeline/compat/` on `PYTHONPATH`. That directory has a blockwise
  sliding-window attention in plain PyTorch, which matches reference SDPA to ~5e-7. The
  timings above use it. To use the real flash-attn on Ampere, Ada or Hopper, install the wheel
  listed in `models/aifs-single-2.0/pyproject.toml`.
- **Memory knobs:** `ANEMOI_INFERENCE_NUM_CHUNKS` (default 16 here) chunks the graph mappers.
  `ATTN_SHIM_QUERY_CHUNK` (default 2048) sets how many queries the attention shim processes at once.

## Compatibility fixes

The model's own version pins do not run the open-data CLI path as published. The pipeline
works around this without modifying installed packages:

1. **Orography filter keys** (`pipeline/compat/run_anemoi_inference.py`).
   `anemoi-plugins-ecmwf-inference` 0.2.1 uses the newer `orography`/`geopotential` filter keys
   but passes `orog=`. The pinned `anemoi-transform` 0.1.16.post2 reads `orog`/`z`. Unpatched,
   this fails with `KeyError: 'z'`. The launcher supplies both key sets.
2. **`libgfortran.so.5`.** `mir` (used for regridding) preloads every library of atlas,
   including a Fortran binding that needs libgfortran. The pipeline adds `.sysdeps/runtime`
   to `LD_LIBRARY_PATH` when that directory exists.
3. **flash-attn shim** (`pipeline/compat/flash_attn/`), described above.

## Troubleshooting

- **`No requests` or HTTP 404 from open data:** the newest cycle is only partly published.
  Re-run later, or pass `--date` for the previous cycle.
- **Rate limited (500-connection cap) on the ECMWF portal:** use `--source aws` (or `azure` or
  `google`).
- **Run failed:** read `forecasts/<cycle>/run.log`. The next cron call retries automatically,
  because `DONE` is only written on success.
- **Python traceback at the very end of `run.log`, ending in `OSError: [Errno 39] Directory
  not empty: '.cache/tmp/tmp…'`:** this comes from temp-dir cleanup at exit and is harmless.
  Check for `DONE`.

## Licences and attribution

- AIFS Single v2.0 weights: © ECMWF, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- ECMWF open data, the forecast input: CC BY 4.0. Forecasts and cards derived from them
  must credit ECMWF.
- anemoi-core and anemoi-inference: Apache 2.0.
