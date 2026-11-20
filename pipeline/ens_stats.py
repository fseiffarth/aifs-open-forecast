"""Statistics helpers for the initial-condition ensemble (see ensemble.py).

Pure numpy/scipy, no plotting: loading the stored members, matching them to the GRIB grid
points, mapping GRIB parameter names to anemoi names, percentiles, per-period amounts and
exceedance probabilities. predict_now.py uses these for the bands and probability maps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_MEMBERS = 3
POINT_TOL = 1e-3  # degrees; ensemble points must coincide with the GRIB points

# GRIB short names that anemoi calls differently (levtype pl is handled by anemoi_name).
GRIB_TO_ANEMOI = {"fscov": "snowc"}


def anemoi_name(param: str, levtype: str, level) -> str:
    """Anemoi variable name of a GRIB field: ('t','pl',850) -> 't_850', ('fscov','sfc',None) -> 'snowc'."""
    if levtype == "pl":
        return f"{param}_{int(level)}"
    return GRIB_TO_ANEMOI.get(param, param)


ENSEMBLE_SUBDIR = {"single": "ensemble", "ens": "ensemble-aifs-ens"}


def ensemble_dir(cycle_dir: Path, region: str, model: str = "single") -> Path:
    """Where ensemble.py stores the members: <cycle>/ensemble/<region> (AIFS Single with
    perturbed initial conditions) or <cycle>/ensemble-aifs-ens/<region> (AIFS ENS)."""
    return Path(cycle_dir) / ENSEMBLE_SUBDIR[model] / region


def member_files(ens_dir: Path) -> list[Path]:
    """Finished member files (half-written '.tmp.npz' files are ignored)."""
    return sorted(p for p in Path(ens_dir).glob("member_*.npz") if ".tmp" not in p.name)


def match_points(ens_lat, ens_lon, lat, lon, tol: float = POINT_TOL):
    """Index into the ensemble points for every (lat, lon) target point, matched by coordinates.

    Returns None if some target point has no ensemble point within `tol` degrees.
    """
    from scipy.spatial import cKDTree

    d, idx = cKDTree(np.c_[ens_lon, ens_lat]).query(np.c_[lon, lat])
    if not np.all(d <= tol):
        return None
    return idx


def load_ensemble(ens_dir, lat, lon, log=print):
    """Load all stored members onto the given grid points (e.g. the GRIB region points).

    Returns a dict (vals (members, steps, variables, points) float32, steps, names, index,
    members, expected, dir) or None when there is no usable ensemble.
    """
    ens_dir = Path(ens_dir)
    files = member_files(ens_dir)
    if not files:
        return None
    meta = {}
    try:
        meta = json.loads((ens_dir / "meta.json").read_text())
    except Exception:
        pass
    vals, ids, steps, names, perm = None, [], None, None, None
    for f in files:
        try:
            with np.load(f) as d:
                s, n = d["steps"], [str(x) for x in d["names"]]
                if vals is None:
                    perm = match_points(d["lat"], d["lon"], lat, lon)
                    if perm is None:
                        log(f"ensemble ignored: its grid points do not match the GRIB region points ({ens_dir})")
                        return None
                    steps, names = s, n
                    vals = np.empty((len(files), len(s), len(n), len(lat)), np.float32)
                elif not (np.array_equal(s, steps) and n == names):
                    log(f"ensemble: skipping {f.name} (different steps or variables)")
                    continue
                vals[len(ids)] = d["values"][:, :, perm]
                ids.append(int(d["member"]) if "member" in d else len(ids))
        except Exception as exc:  # truncated/corrupt file: skip it
            log(f"ensemble: skipping {f.name} ({type(exc).__name__}: {exc})")
    if len(ids) < MIN_MEMBERS:
        log(f"ensemble ignored: only {len(ids)} usable member(s), need at least {MIN_MEMBERS}")
        return None
    return dict(vals=vals[:len(ids)], steps=np.asarray(steps), names=names,
                index={n: i for i, n in enumerate(names)}, members=ids,
                expected=int(meta.get("members", len(ids))), dir=ens_dir)


def align_steps(block, steps, target_steps, zero_start: bool = False):
    """Pick the given forecast steps from (members, steps, points) data.

    Missing steps become NaN, except step 0 of accumulations (`zero_start`), which is 0 by definition.
    """
    pos = {int(s): i for i, s in enumerate(steps)}
    out = np.full((block.shape[0], len(target_steps), block.shape[2]), np.nan, np.float32)
    for j, t in enumerate(target_steps):
        if int(t) in pos:
            out[:, j] = block[:, pos[int(t)]]
        elif int(t) == 0 and zero_start:
            out[:, j] = 0.0
    return out


def percentiles(a, qs=(10, 90), axis: int = 0):
    """Percentiles over the member axis. NaN (no data) propagates instead of being skipped."""
    return np.percentile(a, qs, axis=axis)


def exceedance_prob(a, threshold: float, axis: int = 0):
    """Fraction of members with value >= threshold (NaN where any member has no data)."""
    p = (np.asarray(a) >= threshold).mean(axis=axis)
    return np.where(np.isnan(a).any(axis=axis), np.nan, p)


def period_sums(per_step, st0, st1, hours: int = 24):
    """Sum per-interval amounts (members, intervals, points) over consecutive `hours` windows.

    Only complete windows are returned: (sums (members, windows, points), window starts, window ends).
    The amounts must already be per interval (per-member differences of the accumulation).
    """
    st0, st1 = np.asarray(st0), np.asarray(st1)
    starts = np.arange(st0[0], st1[-1] - hours + 1, hours)
    sums, a_, b_ = [], [], []
    for a in starts:
        m = (st0 >= a) & (st1 <= a + hours)
        if m.any() and int((st1[m] - st0[m]).sum()) == hours:
            sums.append(per_step[:, m].sum(axis=1))
            a_.append(a)
            b_.append(a + hours)
    if not sums:
        return np.zeros((per_step.shape[0], 0, per_step.shape[2])), np.array([], int), np.array([], int)
    return np.stack(sums, axis=1), np.array(a_), np.array(b_)


def band_stats(a):
    """Mean, std and 10/90 % percentiles over axis 0 of (members, steps, points), as float32."""
    p10, p90 = percentiles(a, (10, 90))
    return dict(p10=p10.astype(np.float32), p90=p90.astype(np.float32),
                mean=a.mean(axis=0, dtype=np.float64).astype(np.float32),
                std=a.std(axis=0, dtype=np.float64).astype(np.float32))
