# AIFS forecast pipeline: how to run it

This pipeline runs **ECMWF AIFS Single v2.0** (`ecmwf/aifs-single-2.0`, operational since
12 May 2026) from the **newest available ECMWF open-data analysis**. Every run forecasts
from "now", meaning the latest published 00/06/12/18 UTC cycle. `predict_now.sh` then turns
the forecast into weather cards (Europe by default).

All commands below are run from the repository root.

## What is where

| Path | Content | In git? |
|------|---------|---------|
| `pipeline/forecast.py` | The forecast pipeline (latest cycle → GRIB) | yes |
| `pipeline/run_forecast.sh` | Wrapper that uses the venv and appends to `forecasts/pipeline.log` | yes |
| `pipeline/predict_now.py`, `predict_now.sh` | Forecast + weather cards | yes |
| `pipeline/compat/` | Compatibility fixes, see [Compatibility fixes](#compatibility-fixes) | yes |
| `models/aifs-single-2.0/` | Model checkpoint and config from Hugging Face | no (downloaded on first run) |
| `setup.sh` | One-time environment setup (anemoi-core, `.venv`, libgfortran) | yes |
| `anemoi-core/` | Clone of `ecmwf/anemoi-core` at tag `models-0.9.3` | no (`setup.sh` clones it) |
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

# 2. anemoi-core clone, Python 3.12 venv with the pinned packages, libgfortran if missing
./setup.sh            # CUDA 12.8 torch wheel
./setup.sh --cpu      # CPU-only torch wheel
./setup.sh --ens      # also .venv-ens for the AIFS ENS v2.0 ensemble (see below)
```

`setup.sh` is safe to re-run: it skips the clone and venv creation if they already exist,
re-applies the pinned package versions, and installs libgfortran only if `ldconfig` doesn't
find it. Read the script for the exact commands. It pins `torchvision==0.22.0` together with
`torch==2.7.0`; without that pin the resolver pulls the newest torchvision and upgrades torch.

flash-attn is optional. See the hardware notes below.

`setup.sh` keeps the uv and conda package caches in `.cache/`. Set `UV_CACHE_DIR` /
`CONDA_PKGS_DIRS` to use a different directory. `predict_now.sh` and `pipeline/run_forecast.sh`
stop with a message pointing to `setup.sh` if `.venv/` is missing.

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
./predict_now.sh                                 # latest cycle, 10 days (240 h), Europe
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
  0 = do not run it). 50 gives 51 members.
- `--ensemble-model {both,ens,single}`: which ensemble to run and show (default `both`). The
  cards show AIFS ENS when it exists, otherwise the perturbation ensemble. With both, the
  `6-compare_*` cards are added. See the ensemble section below.

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
| `6-compare_<name>.png` | AIFS ENS vs AIFS Single + perturbations per city, and spread against lead time; only when both ensembles exist |

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

There are two ways to get an ensemble, and by default the pipeline runs both and compares them:

| | AIFS ENS v2.0 (`--model ens`) | AIFS Single + perturbations (`--model single`) |
|---|---|---|
| Model | [`ecmwf/aifs-ens-2.0`](https://huggingface.co/ecmwf/aifs-ens-2.0), trained for ensembles (CRPS loss, ~2.6 GB) | AIFS Single v2.0, the deterministic model |
| Members | ECMWF's own 50 perturbed initial states (open data `enfo`/`waef`) + member 0 from the analysis | our smooth random perturbations of the analysis, in +/- pairs |
| Model uncertainty | built-in noise, seeded per member | none |
| Output | `forecasts/<cycle>/ensemble-aifs-ens/<region>/` | `forecasts/<cycle>/ensemble/<region>/` |
| 51 members, 10 days, Europe, RTX PRO 6000 | 80 min (≈ 92 s per member, ≈ 25 s of it downloading and regridding the member's input) | 58 min |
| 51 members, 10 days, Europe, RTX 3090 | ≈ 97 min without restarts (≈ 114 s per member, GPU memory ≈ 12.3 GB) | 1 h 47 min (≈ 125 s per member) |
| Spread vs ECMWF's AIFS ENS | within about 5 % (it is the same system) | 10-25 % too small from day 3 |

**Use AIFS ENS** for forecasts. It is what ECMWF runs operationally. The perturbation ensemble
is kept as a transparent baseline and for the comparison cards.

```bash
./setup.sh --ens                                              # once: .venv-ens (newer anemoi pins)
./predict_now.sh --ensemble-members 50                        # forecast, both ensembles, cards
./predict_now.sh --ensemble-members 50 --ensemble-model ens   # AIFS ENS only (80 min instead of 140)
.venv/bin/python pipeline/ensemble.py --model ens --members 50   # ensemble only
```

AIFS ENS pins newer packages (anemoi-models 0.11.2, anemoi-graphs 0.8.1) than AIFS Single, so
it runs in its own `.venv-ens`; `ensemble.py` starts the worker in the right venv. The first run
downloads the checkpoint into `models/aifs-ens-2.0/` and checks its sha256. ECMWF publishes the
ensemble initial states with the same delay as the deterministic analysis, so the same cycles work.
`--members` is 1-50 for AIFS ENS (ECMWF has 50 perturbed states). The next member's input is
downloaded while the current one runs.

Open-data downloads sometimes hang without an error (connection stuck, no timeout). If the
worker prints nothing for 10 min, `ensemble.py` kills it and starts it again, up to 3 times;
stored members are kept. The 3090 run below needed this twice.

**RTX 3090 vs RTX PRO 6000 (AIFS ENS, cycle 2026-11-19 12Z, Europe, same seed).** Unlike the
perturbation ensemble, the members do not agree to rounding level, probably because the model
noise is drawn on the GPU and the two cards do not turn the same seed into the same numbers. The mean absolute
difference per member in 2 m temperature is 0.07 K at day 1, 0.33 K at day 5 and 1.3 K at
day 10, about half the spread by then. The ensemble statistics agree: spread of 2 m temperature
0.57 / 1.19 / 2.73 K (3090) vs 0.57 / 1.18 / 2.76 K (RTX PRO 6000) at days 1 / 5 / 10, of
500 hPa geopotential 47 / 378 / 894 vs 48 / 379 / 911 m²/s², and of sea-level pressure within
1 % up to day 5 and 7 % at day 10 (801 vs 862 Pa). Treat the two runs as two samples of the same
ensemble, not as copies.

**Comparison for cycle 2026-11-19 12Z.** Both ensembles were run for Europe and compared with
ECMWF's own AIFS ENS forecast from open data (50 perturbed members each). The numbers are the
ratio of the spread to ECMWF's spread (area mean of the grid-point standard deviation, Europe
box -12..35° E, 32..62° N). Each cell is *AIFS ENS (ours) / AIFS Single + perturbations*.

| day | 2 m temp. | 10 m wind | MSL pressure | 850 hPa temp. | 500 hPa height |
|----:|------:|------:|------:|------:|------:|
|  1 | 1.06 / 1.04 | 1.09 / 0.67 | 1.04 / 1.26 | 1.04 / 0.98 | 1.04 / 1.29 |
|  2 | 1.08 / 0.92 | 1.08 / 0.62 | 1.03 / 1.02 | 1.07 / 0.86 | 1.03 / 1.03 |
|  3 | 1.07 / 0.83 | 1.06 / 0.70 | 1.04 / 0.93 | 1.08 / 0.77 | 1.01 / 0.90 |
|  4 | 1.05 / 0.79 | 1.01 / 0.70 | 0.97 / 0.81 | 1.00 / 0.74 | 0.96 / 0.79 |
|  5 | 1.01 / 0.76 | 0.99 / 0.80 | 0.99 / 0.85 | 0.97 / 0.77 | 0.98 / 0.82 |
|  6 | 1.02 / 0.81 | 1.00 / 0.82 | 1.01 / 0.87 | 0.98 / 0.77 | 0.97 / 0.82 |
|  7 | 1.04 / 0.87 | 1.02 / 0.93 | 1.00 / 0.98 | 0.96 / 0.83 | 1.03 / 0.95 |
|  8 | 1.07 / 0.96 | 1.02 / 0.97 | 0.97 / 0.94 | 0.96 / 0.85 | 1.00 / 0.97 |
|  9 | 1.04 / 0.93 | 1.00 / 0.93 | 0.98 / 0.86 | 1.00 / 0.87 | 0.99 / 0.91 |
| 10 | 1.02 / 0.92 | 1.01 / 0.88 | 1.06 / 0.86 | 0.99 / 0.92 | 1.00 / 0.92 |

- Our AIFS ENS run reproduces ECMWF's spread within about 5 % on every day (day 10, 2 m
  temperature: 2.82 K vs 2.76 K). This checks the AIFS ENS integration (member inputs, noise,
  regridding). It is not bit-identical to ECMWF: the noise seeds differ.
- The perturbation ensemble is 10-25 % too narrow from day 3 to day 10 for every variable.
  Near the surface (10 m wind) it is 30-40 % too narrow in the first 4 days. Raising the
  perturbation amplitudes helps the upper air but barely changes this: AIFS Single was trained
  to minimise the mean squared error and damps small perturbations near the surface.
- Over Germany alone the picture is the same (2 m temperature 0.64-0.89 for days 2-9).
- `6-compare_spread.png` shows the same comparison for each cycle, without ECMWF's data, and
  the `6-compare_<variable>.png` cards show both ensembles per city.

### AIFS Single + perturbations

AIFS Single is deterministic. `pipeline/ensemble.py` estimates forecast uncertainty by running
the same model 51 times from slightly different initial conditions: member 0 is the unperturbed
control, members 1-50 add smooth random perturbations (about 500 km scale) to the atmospheric
fields, in +/- pairs. Amplitudes are about typical analysis errors (see the script header).

```bash
.venv/bin/python pipeline/ensemble.py                  # latest cycle, 50 + 1 members, 240 h, Europe
.venv/bin/python pipeline/ensemble.py --members 10     # fewer members (must be even)
./predict_now.sh --ensemble-members 50                 # forecast, then ensemble, then cards
```

- Runtime: about 65 s per member for 10 days over Europe on the RTX PRO 6000, so about 58 min
  for all 51.
  Do not run it while another forecast uses the GPU. A stopped run resumes and skips members
  that are already stored; `--force` starts again.
- Storage: only the region is kept, `forecasts/<cycle>/ensemble/<region>/member_NNN.npz`
  (about 19 MB per member for Germany, roughly 1 GB in total), plus `meta.json` and `DONE`.
  Europe (11,617 vs 1,020 grid points) takes about 220 MB per member, 11 GB per cycle and per
  ensemble, so 22 GB with both. The model run itself is the same for any region.
  The full global output of 51 members would be about 117 GB.
- Card integration is automatic: `predict_now.sh` (with `--ensemble auto`) uses every member
  that is stored, even from a run in progress (at least 3), so cards can be re-rendered while the
  ensemble is still running. Use `--skip-forecast --date <cycle>` to render a specific cycle.
- Be careful with the result. This is a simple initial-condition ensemble and not ECMWF's ENS
  (no singular vectors, no ensemble of data assimilations, no model-error perturbations).
  Probabilities are not calibrated.
- An earlier check, made with the first (untuned) perturbation amplitudes, against ECMWF's AIFS ENS (open data, cycle 2026-11-19 06Z, 50 perturbed members on
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
| RTX 3090 (24 GB, sm_86) | `cuda` | 1 min 11 s end to end | **4 min 06 s** for `predict_now.sh` (default 240 h): forecast 3 min 18 s (36 s download/regrid, 4 s model load, 2 min 26 s for 40 steps at ≈ 2 s model time each plus GRIB writing, no noticeable warm-up), then 45 s rendering 131 Germany cards (41 s for Europe) |
| 2× RTX 2080 (8 GB) + 16-core CPU | `cpu` | did not finish the first step in > 1 h | not practical |

Both GPU rows use the PyTorch attention shim (no flash-attn installed). The RTX 3090 timings
were measured on 2026-11-19, cycle 12Z.

- **RTX 3090 vs RTX PRO 6000:** each 6 h step takes about 2 s of model time instead of
  about 1 s. End to end, `predict_now.sh` takes 4 min 06 s instead of 3 min 23 s, because
  download, GRIB writing and rendering don't depend much on the GPU. A rerun of cycle
  2026-11-19 00Z on the 3090 matched the RTX PRO 6000 output to rounding level after 12 h
  (mean |diff|: 2 m temperature 0.02 K, MSL 1.5 Pa, z500 0.9 m²/s²).
- **GPU memory:** with the defaults below, `nvidia-smi` showed at most **7.4 GB** in use on
  the RTX 3090 during the 240 h run (about 7.3 GB throughout; this includes the CUDA context
  and PyTorch's cached memory). The memory is set by the chunking, not by the GPU, so 16 GB
  cards should be enough. They are untested, and `--device auto` only picks a GPU with
  ≥ 20 GB free, so pass `--device cuda` on smaller cards. 24 GB leaves headroom for other
  processes on the card.
- **On CPU**, the processor's sliding-window attention runs under
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
