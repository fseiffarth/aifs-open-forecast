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
fields are not perturbed. This is a simple, transparent scheme. ECMWF's operational ENS uses
singular vectors plus an ensemble of data assimilations, so expect this ensemble to be
under-dispersive, i.e. too confident, especially in the first days.

The open-data input is downloaded and regridded once. All members run in one process with
the model loaded once. Only the region of interest is stored (51 global GRIBs would be
~117 GB): forecasts/<cycle>/ensemble/<region>/member_NNN.npz plus meta.json and DONE.
An interrupted run resumes and skips members that are already stored.
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
# humidity as a relative fraction.
Z_GPM = {1000: 6, 925: 6, 850: 6, 700: 7, 600: 8, 500: 9, 400: 10, 300: 12, 250: 13, 200: 13,
         150: 13, 100: 12, 50: 15, 10: 20}
T_K = {1000: 0.5, 925: 0.5, 850: 0.5, 700: 0.5, 600: 0.4, 500: 0.4, 400: 0.4, 300: 0.5,
       250: 0.6, 200: 0.6, 150: 0.6, 100: 0.6, 50: 1.0, 10: 1.0}
UV_MS = {1000: 1.0, 925: 1.0, 850: 1.0, 700: 1.0, 600: 1.5, 500: 1.5, 400: 1.5, 300: 2.0,
         250: 2.0, 200: 2.0, 150: 2.0, 100: 1.5, 50: 1.5, 10: 1.5}
Q_REL = 0.10
SFC = {"msl": 75.0, "sp": 75.0,  # Pa, same pattern as z_1000 (6 gpm ~ 0.75 hPa)
       "2t": 0.8, "skt": 0.8, "2d": 0.8,  # K, pattern of t_1000
       "10u": 0.8, "10v": 0.8,  # m/s, patterns of u_1000 / v_1000
       "tcw": 0.05}  # relative, pattern of q_850
VERTICAL_CORR = 0.7  # share of variance common to all levels of a variable

AMPLITUDES = dict(z_gpm=Z_GPM, t_K=T_K, uv_ms=UV_MS, q_rel=Q_REL, surface=SFC, vertical_corr=VERTICAL_CORR)


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Perturbations
# --------------------------------------------------------------------------------------
class NoiseMaker:
    """Smooth, unit-variance random fields on the model grid.

    White noise on a 2.5 deg lat-lon grid, Gaussian-smoothed (sigma 2 cells, i.e. ~500 km)
    with periodic longitude, normalised to unit variance and interpolated (cubic) to the
    model grid points.
    """

    RES = 2.5

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.nlat, self.nlon = int(180 / self.RES) + 1, int(360 / self.RES)
        self.pad = 6
        i = (90.0 - lat) / self.RES
        j = np.mod(lon, 360.0) / self.RES + self.pad
        self.coords = np.vstack([i, j])

    def field(self, rng: np.random.Generator) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, map_coordinates

        w = rng.standard_normal((self.nlat, self.nlon))
        w = gaussian_filter(w, sigma=2.0, mode=("nearest", "wrap"))
        w /= w.std()
        w = np.concatenate([w[:, -self.pad:], w, w[:, :self.pad]], axis=1)
        out = map_coordinates(w, self.coords, order=3, mode="nearest")
        return (out / out.std()).astype(np.float32)


def make_perturbation(noise: NoiseMaker, rng: np.random.Generator, names: set[str]) -> dict[str, np.ndarray]:
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
               "2d": ("t", 1000), "10u": ("u", 1000), "10v": ("v", 1000), "tcw": ("q", 850)}
    for name, (fam, lev) in pattern.items():
        if name in names:
            key = f"rel:{name}" if name == "tcw" else name
            out[key] = level_fields[(fam, lev)] * SFC[name]
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
# Worker: runs inside the anemoi-inference environment
# --------------------------------------------------------------------------------------
def worker(args) -> int:
    sys.path.append(str(HERE / "compat"))
    import run_anemoi_inference  # noqa: F401  (applies the orography-filter compatibility patch)
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners import create_runner
    from anemoi.utils.dates import as_timedelta

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
    prognostic = set(runner.checkpoint.prognostic_variables)
    names = prognostic & set(base["fields"])
    pert_cache: dict[int, dict] = {}

    for m in range(args.members + 1):
        path = out_dir / f"member_{m:03d}.npz"
        if path.exists():
            continue
        tm = time.time()
        state = dict(base)
        if m == 0:
            state["fields"] = dict(base["fields"])
        else:
            k, sign = (m + 1) // 2, (1.0 if m % 2 == 1 else -1.0)
            if k not in pert_cache:
                rng = np.random.default_rng([args.seed, int(f"{date:%Y%m%d%H}"), k])
                pert_cache.clear()
                pert_cache[k] = make_perturbation(noise, rng, names)
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
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", type=int, default=50,
                   help="number of perturbed members (default 50; plus the control = 51)")
    p.add_argument("--lead-time", default="240h")
    p.add_argument("--region", default="germany")
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
    if args.members % 2:
        p.error("--members must be even (perturbations come in +/- pairs)")
    if args.worker:
        return worker(args)

    import yaml

    import forecast
    from predict_now import REGIONS

    region = REGIONS[args.region]
    date = dt.datetime.strptime(args.date, "%Y%m%d%H") if args.date else forecast.latest_cycle(args.source)
    out_dir = Path(args.output_root).resolve() / f"{date:%Y%m%d%H}" / "ensemble" / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = open(out_dir / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another ensemble run for this cycle/region is in progress; exiting")
        return 0
    meta_path, done = out_dir / "meta.json", out_dir / "DONE"
    meta = dict(cycle=f"{date:%Y%m%d%H}", members=args.members + 1, perturbed=args.members,
                lead_time=args.lead_time, region=args.region, bbox=region["bbox"], seed=args.seed,
                method="initial-condition perturbations, smooth random fields, +/- pairs", amplitudes=AMPLITUDES)
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

    forecast.ensure_model()
    device = forecast.resolve_device(args.device)
    cfg = forecast.build_config(date, args.lead_time, device, out_dir)
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = forecast.runtime_env(device)
    log(f"ensemble from {date:%Y-%m-%d %HZ}: {args.members + 1} members, {args.lead_time}, device {device}, "
        f"region {args.region} -> {out_dir}")
    t0 = time.time()
    with open(out_dir / "run.log", "a") as lf:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(cfg_path),
               "--out-dir", str(out_dir), "--bbox", json.dumps(list(region["bbox"])),
               "--members", str(args.members), "--seed", str(args.seed)]
        proc = subprocess.Popen(cmd, cwd=out_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            lf.write(line)
            if line.startswith("[") and ("member" in line or "input state" in line or "storing" in line):
                print(line, end="", flush=True)
        rc = proc.wait()
    if rc != 0:
        log(f"FAILED (exit {rc}); see {out_dir / 'run.log'}")
        return rc
    done.write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    log(f"ensemble done in {(time.time() - t0) / 60:.1f} min: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
