#!/usr/bin/env python3
"""Ensemble forecast with AIFS Single v2.0 by perturbing the initial conditions.

AIFS Single is a deterministic model: the same input always gives the same forecast. To
estimate forecast uncertainty, this script runs it many times from slightly different
initial conditions (an "initial-condition ensemble"):

  member 0          control, unperturbed (identical to forecast.py)
  members 1 .. N    control + perturbation, in +/- pairs (member 2k-1 = +p_k, 2k = -p_k),
                    as in ECMWF's ENS. Default N = 50, i.e. 51 members like ENS / AIFS ENS.

Perturbations are smooth random fields (~500 km correlation length) on the prognostic
atmospheric variables, with amplitudes of roughly typical analysis errors (see AMPLITUDES).
The same perturbation is added to both input times (t-6 h and t). Soil, snow and ocean-wave
fields are not perturbed. This is a simple, transparent scheme; ECMWF's operational ENS uses
singular vectors plus an ensemble of data assimilations. In one check against ECMWF's AIFS ENS
(cycle 2026-11-19 06Z, Germany box), the spread was similar from about day 5 on (2 m temperature
3.0 K vs 2.8 K at day 10) and the ensemble means agreed within 0.2 K; the first days and other
cases are not verified, and the probabilities are not calibrated.

With --model ens, the members come from ECMWF's AIFS ENS v2.0 instead (models/aifs-ens-2.0,
run in .venv-ens because it pins newer anemoi packages). This is how ECMWF makes its AIFS
ensemble: member k starts from ECMWF's own perturbed initial state k (open-data streams
enfo/waef; member 0 from the unperturbed analysis, oper/wave) and the model adds its own noise, which is trained for
ensemble forecasting (CRPS loss). Nothing is perturbed here; the model noise is seeded per
member, so a rerun gives the same members up to GPU non-determinism.

The open-data input is downloaded and regridded once (AIFS ENS: once per member). All members
run in one process with the model loaded once. Only the region of interest is stored (51
global GRIBs would be ~117 GB): forecasts/<cycle>/ensemble/<region>/member_NNN.npz (AIFS ENS:
ensemble-aifs-ens/<region>/) plus meta.json and DONE. An interrupted run resumes and skips
members that are already stored.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

G = 9.80665
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
PAD = 1.0  # degrees around the region bbox, as in predict_now.load_data

# Perturbation standard deviations (roughly typical analysis-error magnitudes).
# Geopotential height in gpm (converted to m^2/s^2), temperature in K, wind in m/s,
# humidity as a relative fraction. Tuned once against ECMWF's AIFS ENS (cycle 2026-11-19 06Z,
# Germany, see INSTRUCTION.MD): the first version had too much 500 hPa / MSL spread and too
# little near-surface spread on days 1-3.
Z_GPM = {1000: 4, 925: 4, 850: 4, 700: 3.5, 600: 3.5, 500: 3.5, 400: 4, 300: 5, 250: 5, 200: 5,
         150: 5, 100: 5, 50: 8, 10: 12}
T_K = {1000: 1.2, 925: 1.0, 850: 0.8, 700: 0.5, 600: 0.4, 500: 0.4, 400: 0.4, 300: 0.4,
       250: 0.5, 200: 0.5, 150: 0.5, 100: 0.5, 50: 0.8, 10: 0.8}
UV_MS = {1000: 1.8, 925: 1.6, 850: 1.3, 700: 1.0, 600: 0.8, 500: 0.8, 400: 0.8, 300: 1.0,
         250: 1.0, 200: 1.0, 150: 1.0, 100: 0.8, 50: 0.8, 10: 0.8}
Q_REL = 0.10
SFC = {"msl": 50.0, "sp": 50.0,  # Pa, same pattern as z_1000 (4 gpm ~ 0.5 hPa)
       "2t": 1.4, "skt": 1.4, "2d": 1.4,  # K, pattern of t_1000
       "stl1": 1.0, "stl2": 0.5,  # K, soil temperature, pattern of t_1000
       "10u": 1.8, "10v": 1.8,  # m/s, patterns of u_1000 / v_1000
       "tcw": 0.05,  # relative, pattern of q_850
       "swvl1": 0.10, "swvl2": 0.10}  # relative, soil moisture, pattern of q_1000
SFC_SMALL = 0.7  # variance share of small-scale noise in near-surface perturbations
VERTICAL_CORR = 0.5  # share of variance common to all levels of a variable

AMPLITUDES = dict(z_gpm=Z_GPM, t_K=T_K, uv_ms=UV_MS, q_rel=Q_REL, surface=SFC, vertical_corr=VERTICAL_CORR)


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Perturbations
# --------------------------------------------------------------------------------------
class NoiseMaker:
    """Smooth, unit-variance random fields on the model grid.

    White noise on a `res`-degree lat-lon grid, Gaussian-smoothed with `sigma` cells and
    periodic longitude, normalised to unit variance and interpolated (cubic) to the model grid
    points. The default (2.5 deg, sigma 2) gives a correlation length of about 500 km.
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray, res: float = 2.5, sigma: float = 2.0):
        self.res, self.sigma = res, sigma
        self.nlat, self.nlon = int(round(180 / res)) + 1, int(round(360 / res))
        self.pad = int(3 * sigma) + 3
        i = (90.0 - lat) / res
        j = np.mod(lon, 360.0) / res + self.pad
        self.coords = np.vstack([i, j])

    def field(self, rng: np.random.Generator) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, map_coordinates

        w = rng.standard_normal((self.nlat, self.nlon))
        w = gaussian_filter(w, sigma=self.sigma, mode=("nearest", "wrap"))
        w /= w.std()
        w = np.concatenate([w[:, -self.pad:], w, w[:, :self.pad]], axis=1)
        out = map_coordinates(w, self.coords, order=3, mode="nearest")
        return (out / out.std()).astype(np.float32)


def make_perturbation(noise: NoiseMaker, rng: np.random.Generator, names: set[str],
                      small: NoiseMaker | None = None) -> dict[str, np.ndarray]:
    """Additive perturbations (name -> 1-D array) for one +/- pair, before the sign is applied.

    Relative perturbations (q, tcw) are returned as fractions, flagged by a 'rel:' prefix.
    """
    a, b = np.sqrt(VERTICAL_CORR), np.sqrt(1 - VERTICAL_CORR)
    out: dict[str, np.ndarray] = {}
    common = {fam: noise.field(rng) for fam in ("z", "t", "u", "v", "q")}
    level_fields: dict[tuple[str, int], np.ndarray] = {}
    for fam in ("z", "t", "u", "v", "q"):
        for lev in LEVELS:
            level_fields[(fam, lev)] = a * common[fam] + b * noise.field(rng)
    for lev in LEVELS:
        if f"z_{lev}" in names:
            out[f"z_{lev}"] = level_fields[("z", lev)] * Z_GPM[lev] * G
        if f"t_{lev}" in names:
            out[f"t_{lev}"] = level_fields[("t", lev)] * T_K[lev]
        for fam in ("u", "v"):
            if f"{fam}_{lev}" in names:
                out[f"{fam}_{lev}"] = level_fields[(fam, lev)] * UV_MS[lev]
        if f"q_{lev}" in names:
            out[f"rel:q_{lev}"] = level_fields[("q", lev)] * Q_REL
    pattern = {"msl": ("z", 1000), "sp": ("z", 1000), "2t": ("t", 1000), "skt": ("t", 1000),
               "2d": ("t", 1000), "stl1": ("t", 1000), "stl2": ("t", 1000),
               "10u": ("u", 1000), "10v": ("v", 1000), "tcw": ("q", 850),
               "swvl1": ("q", 1000), "swvl2": ("q", 1000)}
    # Near-surface fields: part of the large-scale low-level pattern plus independent
    # small-scale noise (~150 km), since near-surface analysis errors are mostly small-scale.
    small = small or noise
    local = {fam: small.field(rng) for fam in ("t", "u", "v", "q")}
    for name, (fam, lev) in pattern.items():
        if name in names:
            base = level_fields[(fam, lev)]
            if name not in ("msl", "sp", "tcw"):  # keep these consistent with the levels above
                base = np.sqrt(1 - SFC_SMALL) * base + np.sqrt(SFC_SMALL) * local[fam]
            key = f"rel:{name}" if name in ("tcw", "swvl1", "swvl2") else name
            out[key] = base * SFC[name]
    return out


def apply_perturbation(fields: dict, pert: dict, sign: float) -> dict:
    new = dict(fields)
    for key, p in pert.items():
        if key.startswith("rel:"):
            name = key[4:]
            new[name] = np.maximum(fields[name] * (1.0 + sign * p[None, :]), 0.0)
        else:
            new[key] = fields[key] + sign * p[None, :]
    if "2d" in new and "2t" in new:  # dew point cannot exceed temperature
        new["2d"] = np.minimum(new["2d"], new["2t"])
    return new


# --------------------------------------------------------------------------------------
# AIFS ENS: initial conditions of ECMWF ensemble member k
# --------------------------------------------------------------------------------------
CONSTANT_PARAMS = {"lsm", "z", "sdor", "slor"}  # time-invariant surface fields
ENS_MEMBER = {"number": 0}


def install_member_patch() -> None:
    """Make the open-data input fetch the initial state of ECMWF ensemble member ENS_MEMBER.

    The AIFS ENS checkpoint asks for streams enfo/waef with type=fc. At step 0, open data has
    only the perturbed members there (type=pf, number=1..50), no control. The unperturbed
    state is the deterministic analysis (streams oper/wave), which is what ECMWF's AIFS ENS
    notebook uses without a member number, so member 0 comes from there. Constant fields are
    always taken from the deterministic stream, as in that notebook.
    """
    from anemoi.plugins.ecmwf.inference.opendata import opendata

    original = opendata.retrieve
    deterministic = {"enfo": "oper", "waef": "wave"}

    def retrieve(requests, grid, area, patch=None, **kwargs):
        out = []
        for r in requests:
            r = dict(r)
            params = r.get("param")
            params = set(params) if isinstance(params, (list, tuple)) else {params}
            n = ENS_MEMBER["number"]
            if r.get("levtype") == "sfc" and params <= CONSTANT_PARAMS:
                r.update(stream="oper", type="fc")
            elif r.get("stream") in deterministic:
                if n == 0:
                    r.update(stream=deterministic[r["stream"]], type="fc")
                else:
                    r.update(type="pf", number=n)
            out.append(r)
        return original(out, grid, area, patch=patch, **kwargs)

    opendata.retrieve = retrieve


# --------------------------------------------------------------------------------------
# Worker: runs inside the anemoi-inference environment
# --------------------------------------------------------------------------------------
def worker(args) -> int:
    sys.path.append(str(HERE / "compat"))
    import run_anemoi_inference  # noqa: F401  (applies the orography-filter compatibility patch)
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners import create_runner
    from anemoi.utils.dates import as_timedelta

    ens_model = args.model == "ens"
    if ens_model:
        import torch

        install_member_patch()
    out_dir = Path(args.out_dir)
    lo0, lo1, la0, la1 = json.loads(args.bbox)
    config = RunConfiguration.load(args.config, [])
    runner = create_runner(config)
    lead = as_timedelta(config.lead_time)
    runner.lead_time = lead
    runner.time_step = runner.checkpoint.timestep
    date = config.date

    t0 = time.time()
    prog = runner.create_prognostics_input().create_input_state(date=date)
    const = runner.create_constant_coupled_forcings_input().create_input_state(date=date)
    forc = runner.create_dynamic_forcings_input().create_input_state(date=date)
    base = runner._combine_states(prog, const, forc)
    runner.input_state_hook(const)
    log(f"input state ready in {time.time() - t0:.0f}s: {len(base['fields'])} fields")

    lat = np.asarray(base["latitudes"])
    lon = np.asarray(base["longitudes"])
    lon180 = np.where(lon > 180, lon - 360, lon)
    sel = np.where((lon180 >= lo0 - PAD) & (lon180 <= lo1 + PAD) & (lat >= la0 - PAD) & (lat <= la1 + PAD))[0]
    log(f"storing {sel.size} grid points of the region")

    noise = NoiseMaker(lat, lon)
    small = NoiseMaker(lat, lon, res=1.0, sigma=1.5)
    prognostic = set(runner.checkpoint.prognostic_variables)
    names = prognostic & set(base["fields"])
    pert_cache: dict[int, dict] = {}

    todo = [m for m in range(args.members + 1) if not (out_dir / f"member_{m:03d}.npz").exists()]
    if ens_model:
        # Each AIFS ENS member has its own initial state (~100 MB download + regridding, ~25 s).
        # Fetch the next member's state in a background thread while the GPU runs this one.
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=1)

        def load_member(m: int) -> dict:
            if m == 0:  # control: the unperturbed analysis, loaded above
                return base
            ENS_MEMBER["number"] = m  # only this thread fetches input, so this is not racy
            p = runner.create_prognostics_input().create_input_state(date=date)
            return runner._combine_states(p, const, forc)

        pending = {todo[0]: pool.submit(load_member, todo[0])} if todo else {}

    for i, m in enumerate(todo):
        path = out_dir / f"member_{m:03d}.npz"
        tm = time.time()
        state = dict(base)
        if ens_model:
            state = dict(pending.pop(m).result())
            if i + 1 < len(todo):
                pending[todo[i + 1]] = pool.submit(load_member, todo[i + 1])
            state["fields"] = dict(state["fields"])
            torch.manual_seed(args.seed * 1000 + m)  # model noise, reproducible per member
            torch.cuda.manual_seed_all(args.seed * 1000 + m)
        elif m == 0:
            state["fields"] = dict(base["fields"])
        else:
            k, sign = (m + 1) // 2, (1.0 if m % 2 == 1 else -1.0)
            if k not in pert_cache:
                rng = np.random.default_rng([args.seed, int(f"{date:%Y%m%d%H}"), k])
                pert_cache.clear()
                pert_cache[k] = make_perturbation(noise, rng, names, small)
            state["fields"] = apply_perturbation(base["fields"], pert_cache[k], sign)
        post = runner.create_post_processors()  # fresh accumulators for every member
        steps, var_names, data = [], None, []
        for out in runner.run(input_state=state, lead_time=lead):
            for p in post:
                out = p.process(out)
            if var_names is None:
                var_names = sorted(out["fields"])
            steps.append(int((out["date"] - date).total_seconds() // 3600))
            data.append(np.stack([np.asarray(out["fields"][v])[sel] for v in var_names]).astype(np.float32))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, values=np.stack(data), steps=np.array(steps), names=np.array(var_names),
                 lat=lat[sel], lon=lon180[sel], member=m)
        tmp.rename(path)
        log(f"member {m:3d}/{args.members} done in {time.time() - tm:.0f}s")
    return 0


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
# anemoi logs every forecast step (seconds apart), so a silent worker is stuck, typically in an
# open-data download whose connection the server dropped (seen with AIFS ENS: no timeout).
STALL_S = 600
RESTARTS = 3


def run_worker(cmd: list[str], out_dir: Path, env: dict) -> tuple[int, bool]:
    """Run the worker, copying its output to run.log. Returns (exit code, killed for stalling)."""
    import threading

    last = [time.time()]
    with open(out_dir / "run.log", "a") as lf:
        proc = subprocess.Popen(cmd, cwd=out_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)

        def pump():
            for line in proc.stdout:
                last[0] = time.time()
                lf.write(line)
                lf.flush()
                if line.startswith("[") and ("member" in line or "input state" in line or "storing" in line):
                    print(line, end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        stalled = False
        while proc.poll() is None:
            if time.time() - last[0] > STALL_S:
                stalled = True
                os.killpg(proc.pid, 9)
                break
            time.sleep(5)
        rc = proc.wait()
        reader.join(timeout=10)
    return rc, stalled
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="single", choices=["single", "ens"],
                   help="single: AIFS Single with our perturbed initial conditions (default); "
                        "ens: AIFS ENS v2.0 from ECMWF's ensemble initial conditions")
    p.add_argument("--members", type=int, default=50,
                   help="number of perturbed members (default 50; plus the control = 51)")
    p.add_argument("--lead-time", default="240h")
    p.add_argument("--region", default="europe")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--date", help="initial time YYYYMMDDHH instead of the latest cycle")
    p.add_argument("--source", default="ecmwf", choices=["ecmwf", "aws", "azure", "google"])
    p.add_argument("--output-root", default=str(HERE.parent / "forecasts"))
    p.add_argument("--seed", type=int, default=1, help="random seed (with the cycle date)")
    p.add_argument("--force", action="store_true", help="discard stored members and start again")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    p.add_argument("--out-dir", help=argparse.SUPPRESS)
    p.add_argument("--bbox", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.model == "single" and args.members % 2:
        p.error("--members must be even (perturbations come in +/- pairs)")
    if args.model == "ens" and not 1 <= args.members <= 50:
        p.error("--members must be 1-50 with --model ens (ECMWF has 50 perturbed members)")
    if args.worker:
        return worker(args)

    import yaml

    import forecast
    from ens_stats import ensemble_dir
    from predict_now import REGIONS

    region = REGIONS[args.region]
    date = dt.datetime.strptime(args.date, "%Y%m%d%H") if args.date else forecast.latest_cycle(args.source)
    out_dir = ensemble_dir(Path(args.output_root).resolve() / f"{date:%Y%m%d%H}", args.region, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = open(out_dir / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another ensemble run for this cycle/region is in progress; exiting")
        return 0
    meta_path, done = out_dir / "meta.json", out_dir / "DONE"
    meta = dict(cycle=f"{date:%Y%m%d%H}", members=args.members + 1, perturbed=args.members,
                lead_time=args.lead_time, region=args.region, bbox=region["bbox"], seed=args.seed, model=args.model)
    if args.model == "ens":
        meta.update(method="AIFS ENS v2.0 from ECMWF ensemble initial conditions (member 0: unperturbed analysis), "
                           "model noise seeded per member", amplitudes=None)
    else:
        meta.update(method="initial-condition perturbations, smooth random fields, +/- pairs", amplitudes=AMPLITUDES)
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if args.force or (old and {k: old.get(k) for k in ("lead_time", "seed", "amplitudes")}
                      != {k: json.loads(json.dumps(meta))[k] for k in ("lead_time", "seed", "amplitudes")}):
        for f in out_dir.glob("member_*.npz"):
            f.unlink()
        done.unlink(missing_ok=True)
    if done.exists() and len(list(out_dir.glob("member_*.npz"))) >= args.members + 1:
        log(f"ensemble for {date:%Y-%m-%d %HZ} ({args.region}) already complete: {out_dir}")
        return 0
    meta_path.write_text(json.dumps(meta, indent=1))

    python = forecast.MODELS[args.model]["venv"] / "bin" / "python"
    if not python.exists():
        log(f"error: {python} is missing; run ./setup.sh" + (" --ens" if args.model == "ens" else ""))
        return 1
    forecast.ensure_model(args.model)
    device = forecast.resolve_device(args.device)
    cfg = forecast.build_config(date, args.lead_time, device, out_dir, args.model)
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = forecast.runtime_env(device)
    log(f"ensemble from {date:%Y-%m-%d %HZ}: {args.members + 1} members, {args.lead_time}, device {device}, "
        f"model {args.model}, region {args.region} -> {out_dir}")
    t0 = time.time()
    cmd = [str(python), str(Path(__file__).resolve()), "--worker", "--config", str(cfg_path),
           "--out-dir", str(out_dir), "--bbox", json.dumps(list(region["bbox"])),
           "--members", str(args.members), "--seed", str(args.seed), "--model", args.model]
    for attempt in range(1, RESTARTS + 2):
        rc, stalled = run_worker(cmd, out_dir, env)
        if rc == 0:
            break
        if not stalled or attempt > RESTARTS:
            log(f"FAILED (exit {rc}); see {out_dir / 'run.log'}")
            return rc or 1
        log(f"worker printed nothing for {STALL_S // 60} min (hung open-data download?); "
            f"restarting it, finished members are kept ({attempt}/{RESTARTS})")
    done.write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    log(f"ensemble done in {(time.time() - t0) / 60:.1f} min: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
