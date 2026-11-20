#!/usr/bin/env python3
"""Forecast "now" with AIFS and render weather cards for a region (default: Europe).

Steps:
  1. run pipeline/forecast.py for the latest ECMWF open-data cycle (skipped if that cycle is
     already finished, or with --skip-forecast / --grib),
  2. read the GRIB output, subset every field to the region once,
  3. write PNG "weather cards" plus index.html to forecasts/<cycle>/cards/<region>/:
       0-overview.png                          classic per-city forecast card
       1-surface_<name>.png                    surface/soil fields, e.g. 1-surface_2m-temperature.png
       1-surface_<name>_amounts.png / _accumulated-total.png   accumulations, e.g. 1-surface_precipitation_amounts.png
       2-upper-air_<name>_<llll>hPa.png        pressure levels, e.g. 2-upper-air_temperature_0850hPa.png
       3-waves_<name>.png                      ocean wave fields
       4-constant_<name>.png                   constant fields (orography, land-sea mask, ...)

Examples:
  predict_now.sh                              # latest cycle, 10-day (240 h) forecast, Europe
  predict_now.sh --lead-time 120h --device cpu
  predict_now.sh --skip-forecast              # only re-render the newest finished cycle
  predict_now.sh --grib some.grib --region europe
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import argparse
import datetime as dt
import html
import math
import multiprocessing as mp
import subprocess
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ens_stats  # noqa: E402

# --------------------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------------------
REGIONS = {
    "germany": dict(
        title="Germany",
        bbox=(5.5, 15.5, 47.0, 55.2),  # lon_min, lon_max, lat_min, lat_max (plot extent)
        tz="Europe/Berlin",
        cities={
            "Berlin": (52.52, 13.405), "Hamburg": (53.55, 9.99), "München": (48.137, 11.575),
            "Köln": (50.94, 6.96), "Frankfurt": (50.11, 8.68), "Stuttgart": (48.78, 9.18),
        },
    ),
    "europe": dict(
        title="Europe",
        bbox=(-12.0, 32.0, 35.0, 62.0),
        tz="Europe/Berlin",
        cities={
            "London": (51.51, -0.13), "Paris": (48.86, 2.35), "Madrid": (40.42, -3.70),
            "Rome": (41.90, 12.50), "Berlin": (52.52, 13.405), "Warsaw": (52.23, 21.01),
        },
    ),
}
CITY_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GRID_RES = 0.1  # degrees, regular plotting grid
G = 9.80665
MAX_FRAMES = 12

WAVE_PARAMS = {"swh", "mwd", "mwp", "mp2", "pp1d", "cdww", "mdww", "mpww", "shww", "shts", "mdts", "mpts", "wmb"}
SURFACE_ORDER = ["2t", "2d", "skt", "10si", "10u", "10v", "100si", "100u", "100v", "msl", "sp", "tp", "cp", "sf",
                 "tcc", "lcc", "mcc", "hcc", "tcw", "ssrd", "strd", "ro", "sd", "snowc", "stl1", "stl2",
                 "swvl1", "swvl2"]

# --------------------------------------------------------------------------------------
# Variable specifications
# --------------------------------------------------------------------------------------
# style: temp (diverging around 0 degC), div (symmetric diverging), seq (sequential), precip
# (discrete levels), cyclic (0-360), plain.  kind: inst | acc (accumulated since start) |
# flux (accumulated energy -> mean W/m2).


def S(name, unit, scale=1.0, offset=0.0, style="seq", cmap="viridis", kind="inst", **kw):
    return dict(name=name, unit=unit, scale=scale, offset=offset, style=style, cmap=cmap, kind=kind, **kw)


SPECS = {
    "2t": S("2 m temperature", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "2d": S("2 m dew point", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "skt": S("Skin temperature", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "stl1": S("Soil temperature, layer 1 (0-7 cm)", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "stl2": S("Soil temperature, layer 2 (7-28 cm)", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "swvl1": S("Soil moisture, layer 1 (0-7 cm)", "m³/m³", cmap="YlGnBu", vmin0=True, floor=0.05),
    "swvl2": S("Soil moisture, layer 2 (7-28 cm)", "m³/m³", cmap="YlGnBu", vmin0=True, floor=0.05),
    "10u": S("10 m wind, eastward component (u)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "10v": S("10 m wind, northward component (v)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "100u": S("100 m wind, eastward component (u)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "100v": S("100 m wind, northward component (v)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "10si": S("10 m wind speed", "m/s", cmap="PuBuGn", vmin0=True, kmh=True, floor=5),
    "100si": S("100 m wind speed", "m/s", cmap="PuBuGn", vmin0=True, kmh=True, floor=5),
    "msl": S("Mean sea-level pressure", "hPa", scale=0.01, cmap="Spectral_r", contour=4),
    "sp": S("Surface pressure", "hPa", scale=0.01, cmap="Spectral_r", contour=10),
    "tcw": S("Total column water (water vapour)", "kg/m² (mm)", cmap="GnBu", vmin0=True, floor=5),
    "tcc": S("Total cloud cover", "%", cmap="Greys", fixed=(0, 100)),
    "lcc": S("Low cloud cover", "%", cmap="Greys", fixed=(0, 100)),
    "mcc": S("Medium cloud cover", "%", cmap="Greys", fixed=(0, 100)),
    "hcc": S("High cloud cover", "%", cmap="Greys", fixed=(0, 100)),
    "tp": S("Total precipitation", "mm", scale=1000, style="precip", cmap="Blues", kind="acc"),
    "cp": S("Convective precipitation", "mm", scale=1000, style="precip", cmap="Purples", kind="acc"),
    "sf": S("Snowfall (water equivalent)", "mm", scale=1000, style="precip", cmap="PuBu", kind="acc"),
    "ro": S("Runoff", "mm", scale=1000, style="precip", cmap="YlGnBu", kind="acc"),
    "ssrd": S("Surface solar radiation downwards", "W/m²", cmap="YlOrRd", kind="flux", vmin0=True, floor=20),
    "strd": S("Surface thermal radiation downwards", "W/m²", cmap="magma", kind="flux", vmin0=True, floor=20),
    "sd": S("Snow depth (water equivalent)", "mm", scale=1000, cmap="Blues", vmin0=True, floor=2),
    "snowc": S("Snow cover", "%", cmap="Blues", fixed=(0, 100)),
    # waves
    "swh": S("Significant wave height", "m", cmap="YlGnBu", vmin0=True, floor=1),
    "mwp": S("Mean wave period", "s", cmap="viridis", vmin0=True, floor=4),
    "mp2": S("Mean zero-crossing wave period", "s", cmap="viridis", vmin0=True, floor=4),
    "pp1d": S("Peak wave period", "s", cmap="viridis", vmin0=True, floor=4),
    "mwd": S("Mean wave direction (coming from)", "°", style="cyclic", cmap="twilight", fixed=(0, 360)),
    "cdww": S("Drag coefficient with waves", "-", cmap="Purples"),
    # constants
    "z": S("Surface elevation (orography)", "m", scale=1 / G, cmap="terrain", style="plain"),
    "lsm": S("Land-sea mask (1 = land)", "0-1", cmap="Greens", fixed=(0, 1)),
    "sdor": S("Std. dev. of sub-grid orography", "m", cmap="YlOrBr", vmin0=True),
    "slor": S("Slope of sub-grid orography", "-", cmap="YlOrBr", vmin0=True),
    "wmb": S("Model bathymetry (sea depth)", "m", cmap="Blues", vmin0=True),
}
PL_SPECS = {
    "t": S("Temperature", "°C", offset=-273.15, style="temp", cmap="RdYlBu_r"),
    "z": S("Geopotential height", "gpm", scale=1 / G, cmap="Spectral_r", contour=-1),
    "u": S("Wind, eastward component (u)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "v": S("Wind, northward component (v)", "m/s", style="div", cmap="RdBu_r", kmh=True),
    "w": S("Vertical velocity (positive = sinking)", "Pa/s", style="div", cmap="PuOr"),
    "q": S("Specific humidity", "g/kg", scale=1000, cmap="GnBu", vmin0=True),
    "r": S("Relative humidity", "%", cmap="BrBG", fixed=(0, 100)),
}


def spec_for(param: str, levtype: str, mname: str, munits: str) -> dict:
    sp = (PL_SPECS if levtype == "pl" else SPECS).get(param)
    if sp is not None:
        return dict(sp)
    u = (munits or "").strip()
    if u.startswith("(0 - 1)"):
        return S(mname or param, "%", scale=100, cmap="Greys", fixed=(0, 100))
    low = (mname or "").lower()
    cm = "YlGnBu" if "wave" in low else "viridis"
    return S(mname or param, u or "1", cmap=cm)


def is_wave(param: str, mname: str) -> bool:
    # Decide by GRIB code, not by name: e.g. ssrd is "surface short-wave radiation" but not an ocean-wave field.
    import re

    return param in WAVE_PARAMS or re.fullmatch(r"h\d{4}", param) is not None  # h1012 ... h2530 wave-height bands


# --------------------------------------------------------------------------------------
# Global context shared with forked workers
# --------------------------------------------------------------------------------------
CTX: dict = {}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_data(grib: Path, region: dict):
    """Read the GRIB once; return (entries, geometry info). Fields are subset to the region."""
    import earthkit.data as ekd
    from scipy.spatial import Delaunay, cKDTree
    import scipy.sparse as sps

    ds = ekd.from_source("file", str(grib))
    first = ds[0]
    ll = first.to_latlon(flatten=True)
    lat, lon = np.asarray(ll["lat"]), np.asarray(ll["lon"])
    lon = np.where(lon > 180, lon - 360, lon)
    npts = lat.size
    lo0, lo1, la0, la1 = region["bbox"]
    pad = 1.0
    sel = np.where((lon >= lo0 - pad) & (lon <= lo1 + pad) & (lat >= la0 - pad) & (lat <= la1 + pad))[0]
    slon, slat = lon[sel], lat[sel]
    log(f"grid: {npts} points, {sel.size} in padded region")

    base = None
    entries: dict = {}
    for f in ds:
        m = f.metadata
        if f.to_latlon is None:
            continue
        param = m("param")
        levtype = m("levtype", default="sfc")
        level = int(m("levelist", default=0) or 0) if levtype == "pl" else None
        step = int(m("step"))
        if base is None:
            base = m("base_datetime")
        vals = f.to_numpy(flatten=True)
        if vals.size != npts:
            continue
        vals = vals[sel].astype(np.float32)
        vals[np.abs(vals) > 1e20] = np.nan
        key = (param, levtype, level)
        e = entries.setdefault(key, dict(param=param, levtype=levtype, level=level, steps={},
                                         mname=m("name", default=param), munits=m("units", default="")))
        e["steps"][step] = vals
        if str(m("stepRange", default="")).startswith("0-"):
            e["from_start"] = True  # accumulated since forecast start (stepRange 0-N)
    for e in entries.values():
        # Accumulations have no step-0 field; the accumulation at t=0 is zero by definition.
        # Add it so the first interval (0 -> first step) is not lost when differencing.
        if e.pop("from_start", False) and 0 not in e["steps"]:
            e["steps"][0] = np.zeros_like(next(iter(e["steps"].values())))
        st = sorted(e["steps"])
        e["step_list"] = np.array(st)
        e["vals"] = np.stack([e["steps"][s] for s in st])
        del e["steps"]

    # derived wind speeds
    for h in ("10", "100"):
        ku, kv = (f"{h}u", "sfc", None), (f"{h}v", "sfc", None)
        if ku in entries and kv in entries and np.array_equal(entries[ku]["step_list"], entries[kv]["step_list"]):
            entries[(f"{h}si", "sfc", None)] = dict(
                param=f"{h}si", levtype="sfc", level=None, mname=f"{h} metre wind speed", munits="m s**-1",
                step_list=entries[ku]["step_list"], vals=np.hypot(entries[ku]["vals"], entries[kv]["vals"]),
                uv=(ku, kv))

    # interpolation operator (barycentric weights on a Delaunay triangulation, built once)
    gx = np.arange(lo0, lo1 + 1e-6, GRID_RES)
    gy = np.arange(la0, la1 + 1e-6, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    tri = Delaunay(np.c_[slon, slat])
    q = np.c_[GX.ravel(), GY.ravel()]
    simp = tri.find_simplex(q)
    ok = simp >= 0
    T = tri.transform[simp[ok]]
    b = np.einsum("nij,nj->ni", T[:, :2, :], q[ok] - T[:, 2, :])
    w = np.c_[b, 1 - b.sum(axis=1)]
    rows = np.repeat(np.where(ok)[0], 3)
    cols = tri.simplices[simp[ok]].ravel()
    W = sps.csr_matrix((w.ravel(), (rows, cols)), shape=(q.shape[0], slon.size))

    # nearest grid point of each city
    tree = cKDTree(np.c_[slon * math.cos(math.radians(51)), slat])
    cities = {}
    for name, (clat, clon) in region["cities"].items():
        d, i = tree.query([clon * math.cos(math.radians(51)), clat])
        cities[name] = dict(idx=int(i), lat=clat, lon=clon, glat=float(slat[i]), glon=float(slon[i]))
    geo = dict(W=W, gx=gx, gy=gy, shape=GX.shape, cities=cities, lat=slat, lon=slon)
    return entries, geo, base


def to_grid(vals_pts: np.ndarray, W, shape) -> np.ndarray:
    """Interpolate point values (n,) onto the regular grid, NaN-aware (sea-only fields)."""
    ok = np.isfinite(vals_pts)
    if ok.all():
        num = W @ vals_pts
        den = np.asarray(W.sum(axis=1)).ravel()
    else:
        num = W @ np.where(ok, vals_pts, 0.0)
        den = W @ ok.astype(np.float64)
    out = np.full(num.shape, np.nan)
    good = den > 0.5
    out[good] = num[good] / den[good]
    return out.reshape(shape)


def load_lines(bbox):
    """Coast/border/state line segments clipped to the region (Natural Earth, cached)."""
    import cartopy

    cartopy.config["data_dir"] = str(ROOT / ".cache" / "cartopy")
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    import shapely

    lo0, lo1, la0, la1 = bbox
    clip = box(lo0 - 1, la0 - 1, lo1 + 1, la1 + 1)
    layers = {}
    sources = [("coast", "physical", "coastline"), ("border", "cultural", "admin_0_boundary_lines_land")]
    # State/province borders only help on country-sized views; on Europe they bury the field.
    if lo1 - lo0 < 20:
        sources.append(("state", "cultural", "admin_1_states_provinces_lines"))
    for name, cat, ds_name in sources:
        segs = []
        for res in ("10m", "50m"):
            try:
                path = shpreader.natural_earth(resolution=res, category=cat, name=ds_name)
                for g in shpreader.Reader(path).geometries():
                    if g is None or not g.intersects(clip):
                        continue
                    for part in shapely.get_parts(g.intersection(clip)):
                        if part.geom_type == "LineString" and len(part.coords) > 1:
                            segs.append(np.asarray(part.coords))
                        elif part.geom_type == "Polygon":
                            segs.append(np.asarray(part.exterior.coords))
                break
            except Exception as exc:  # offline / missing: try coarser, else skip
                log(f"natural earth {ds_name} {res} unavailable ({type(exc).__name__}: {exc})")
        layers[name] = segs
    return layers


# --------------------------------------------------------------------------------------
# Helpers for plotting
# --------------------------------------------------------------------------------------
def nice_limits(lo, hi, nbins=8):
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    t = ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return float(t[0]), float(t[-1]), t


PRECIP_CANDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
                750, 1000]


def make_norm(spec, frames_data):
    """Return (norm, cmap, ticks, extend) shared by all frames."""
    cmap = plt.get_cmap(spec["cmap"]).copy()
    allv = np.concatenate([f[np.isfinite(f)].ravel() for f in frames_data]) if frames_data else np.array([0.0])
    if allv.size == 0:
        allv = np.array([0.0])
    lo, hi = float(allv.min()), float(allv.max())
    style = spec["style"]
    if style == "precip":
        cands = [c for c in PRECIP_CANDS if c <= hi]
        above = [c for c in PRECIP_CANDS if c > hi]
        lev = (cands[-9:] if cands else []) + (above[:1] if above else [])
        if len(lev) < 3:
            lev = PRECIP_CANDS[2:6]
        cmap = plt.get_cmap(spec["cmap"], len(lev)).copy()
        cmap.set_under("#ffffff00")
        norm = mcolors.BoundaryNorm(lev, cmap.N, extend="max")
        return norm, cmap, lev, "max"
    if "fixed" in spec:
        a, b = spec["fixed"]
        t = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(a, b)
        return mcolors.Normalize(a, b), cmap, t, "neither"
    if style == "div":
        m = max(abs(lo), abs(hi), 1e-9)
        m = max(m, spec.get("floor", 0))
        _, m2, t = nice_limits(-m, m)
        m2 = max(abs(t[0]), abs(t[-1]))
        t = ticker.MaxNLocator(nbins=8, symmetric=True, steps=[1, 2, 2.5, 5, 10]).tick_values(-m2, m2)
        return mcolors.Normalize(-m2, m2), cmap, t, "neither"
    if spec.get("vmin0"):
        lo = 0.0
        hi = max(hi, spec.get("floor", 0.0))
    a, b, t = nice_limits(lo, hi)
    if style == "temp" and a < 0 < b:
        return mcolors.TwoSlopeNorm(0, a, b), cmap, t, "neither"
    return mcolors.Normalize(a, b), cmap, t, "neither"


def frame_plan(arr, st0, st1, kind, variant):
    """Choose map frames on a regular time grid.

    Picks the smallest stride (6, 12, 24, 48, ... h) that yields at most MAX_FRAMES frames, so
    frames are evenly spaced and (for strides >= 24 h) always at the same time of day.
    Instantaneous fields and running totals are sampled at the stride; per-interval
    accumulations and fluxes are aggregated over it (e.g. 24 h precipitation totals), so no
    rain falls "between" frames. Returns (frame values, start steps, end steps, source indices or None).
    """
    n = arr.shape[0]
    if n <= MAX_FRAMES:
        return arr, st0, st1, list(range(n))
    base = int(np.min(st1 - st0)) if kind != "inst" else int(np.min(np.diff(st1)))
    base = max(base, 1)
    strides = [m * base for m in range(1, 1000) if (24 % (m * base) == 0 or (m * base) % 24 == 0)]
    for S in strides:
        if kind == "inst" or variant == "cumul":
            sel = [i for i in range(n) if st1[i] % S == 0]
            if 0 < len(sel) <= MAX_FRAMES:
                return arr[sel], st0[sel], st1[sel], sel
        else:
            starts = np.arange(st0[0], st1[-1] - S + 1, S)
            if 0 < len(starts) <= MAX_FRAMES:
                vals, a_, b_ = [], [], []
                for a in starts:
                    m = (st0 >= a) & (st1 <= a + S)
                    if kind == "flux":
                        wts = (st1[m] - st0[m]).astype(float)
                        vals.append(np.tensordot(wts / wts.sum(), arr[m], axes=1))
                    else:
                        vals.append(arr[m].sum(axis=0))
                    a_.append(a)
                    b_.append(a + S)
                return np.stack(vals), np.array(a_), np.array(b_), None
    sel = sample_idx(n, MAX_FRAMES)  # fallback, should not happen
    return arr[sel], st0[sel], st1[sel], sel


def sample_idx(n, k):
    if n <= k:
        return list(range(n))
    return sorted(set(np.round(np.linspace(0, n - 1, k)).astype(int).tolist()))


def fmt_valid(t: dt.datetime) -> str:
    return f"{t:%a %d %b %H} UTC"


GROUP_PREFIX = {"Surface": "1-surface", "Pressure levels": "2-upper-air", "Waves": "3-waves", "Constants": "4-constant",
                "Ensemble": "5-ensemble"}


def slugify(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.replace("%", "pct")).strip("-").lower()
    return text or "field"


# Short, readable file names per GRIB code (fallback: slug of the long GRIB name).
FILE_NAMES = {
    "2t": "2m-temperature", "2d": "2m-dewpoint", "skt": "skin-temperature",
    "10u": "10m-wind-u", "10v": "10m-wind-v", "10si": "10m-wind-speed",
    "100u": "100m-wind-u", "100v": "100m-wind-v", "100si": "100m-wind-speed",
    "msl": "mean-sea-level-pressure", "sp": "surface-pressure",
    "tp": "precipitation", "cp": "convective-precipitation", "sf": "snowfall", "ro": "runoff",
    "tcc": "cloud-cover-total", "lcc": "cloud-cover-low", "mcc": "cloud-cover-medium", "hcc": "cloud-cover-high",
    "tcw": "total-column-water", "ssrd": "solar-radiation-down", "strd": "thermal-radiation-down",
    "sd": "snow-depth", "snowc": "snow-cover", "fscov": "snow-cover",
    "stl1": "soil-temperature-0-7cm", "stl2": "soil-temperature-7-28cm",
    "swvl1": "soil-moisture-0-7cm", "swvl2": "soil-moisture-7-28cm",
    # pressure levels
    "z": "geopotential-height", "t": "temperature", "q": "specific-humidity",
    "u": "wind-u", "v": "wind-v", "w": "vertical-velocity",
    # waves
    "swh": "wave-height", "mwd": "wave-direction", "mwp": "wave-period", "cdww": "wave-drag-coefficient",
    "h1012": "wave-height-10-12s", "h1214": "wave-height-12-14s", "h1417": "wave-height-14-17s",
    "h1721": "wave-height-17-21s", "h2125": "wave-height-21-25s", "h2530": "wave-height-25-30s",
    # constants
    "lsm": "land-sea-mask", "wmb": "sea-depth", "sdor": "orography-roughness", "slor": "orography-slope",
}
CONST_FILE_NAMES = {"z": "elevation"}


def card_filename(group: str, name: str, level, variant, kind: str, param: str = "") -> str:
    """Readable, sortable card file name, e.g. '2-upper-air_temperature_0850hPa.png'."""
    short = (CONST_FILE_NAMES if group == "Constants" else {}).get(param) or FILE_NAMES.get(param)
    parts = [GROUP_PREFIX[group], short or slugify(name)[:40].rstrip("-")]
    if level:
        parts.append(f"{int(level):04d}hPa")
    if kind == "acc" and group != "Constants":
        parts.append("accumulated-total" if variant == "cumul" else "amounts")
    return "_".join(parts) + ".png"


def card_path(key, variant=None):
    param, levtype, level = key
    stem = f"{param}_{level}hPa" if levtype == "pl" else param
    if variant == "cumul":
        stem += "_total"
    return stem


# --------------------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------------------
def draw_lines(ax, lines):
    for name, color, lw, ls in (("coast", "#1a1a1a", 0.7, "-"), ("border", "#333333", 0.8, "-"),
                                ("state", "#555555", 0.45, (0, (4, 2)))):
        segs = lines.get(name)
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, zorder=4))


def setup_map(ax, bbox, lines, cities, labels=True):
    lo0, lo1, la0, la1 = bbox
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    ax.set_aspect(1 / math.cos(math.radians((la0 + la1) / 2)), adjustable="box")
    ax.set_facecolor("#e9eef2")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#666")
    draw_lines(ax, lines)
    for (name, c), col in zip(cities.items(), CITY_COLORS):
        ax.plot(c["lon"], c["lat"], "o", ms=3.2, mfc=col, mec="white", mew=0.6, zorder=6)
        if labels:
            ax.annotate(name, (c["lon"], c["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                        zorder=7, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def render_card(task):
    """Render one weather card. task = (key, variant). Returns (task, filename, error)."""
    key, variant = task
    try:
        if key[0] == "ens":
            return task, *_render_ens_card(key[1])
        return _render_card(key, variant)
    except Exception as exc:  # keep going; report at the end
        import traceback

        return task, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


def _series(entry, spec, variant):
    """Convert raw stored values -> (values (n,pts), start_steps, end_steps, label)."""
    v = entry["vals"].astype(np.float64)
    steps = entry["step_list"]
    kind = spec["kind"]
    if kind == "inst":
        return v * spec["scale"] + spec["offset"], steps, steps
    hrs = np.diff(steps)[:, None].astype(float)
    if variant == "cumul":
        return (v[1:] - v[:1]) * spec["scale"], steps[:1].repeat(len(steps) - 1), steps[1:]
    d = np.maximum(np.diff(v, axis=0), 0)
    if kind == "flux":
        return d / (hrs * 3600.0) * (spec["scale"] or 1.0), steps[:-1], steps[1:]
    return d * spec["scale"], steps[:-1], steps[1:]


def resolve_spec(entry):
    """Plot specification of a loaded GRIB entry (shared by the cards and the ensemble code)."""
    spec = spec_for(entry["param"], entry["levtype"], entry["mname"], entry["munits"])
    if entry["levtype"] == "pl":
        spec = dict(PL_SPECS.get(entry["param"], spec))
        if entry["param"] not in PL_SPECS:
            spec = spec_for(entry["param"], "sfc", entry["mname"], entry["munits"])
    if entry["param"] in ("10si", "100si"):
        spec = dict(SPECS[entry["param"]])
    if entry["param"] == "z" and entry["levtype"] != "pl":
        spec = dict(SPECS["z"])
    return spec


# --------------------------------------------------------------------------------------
# Ensemble statistics (members stored by ensemble.py; CTX["ens"] holds the raw members)
# --------------------------------------------------------------------------------------
def ens_block(ens, entries, entry, zero_start=False):
    """Raw member values of a GRIB entry as (members, entry steps, points), or None if not in the ensemble.

    Matched by anemoi name; steps are aligned to the entry's steps (the ensemble has no step 0:
    NaN there, or 0 for accumulations). Derived wind speeds are computed per member from u and v.
    """
    uv = entry.get("uv")
    if uv:
        u, v = (ens_block(ens, entries, entries[k]) for k in uv)
        return None if u is None or v is None else np.hypot(u, v)
    i = ens["index"].get(ens_stats.anemoi_name(entry["param"], entry["levtype"], entry["level"]))
    if i is None:
        return None
    return ens_stats.align_steps(ens["vals"][:, :, i, :], ens["steps"], entry["step_list"], zero_start)


def ens_series(ens, entries, key, spec, variant):
    """Per-member version of `_series`: (values (members, n, points), start steps, end steps) or None.

    The per-period amounts / mean fluxes are formed for every member first; any percentile is
    taken afterwards (never percentiles of accumulations that are differenced later).
    """
    entry = entries[key]
    blk = ens_block(ens, entries, entry, zero_start=spec["kind"] in ("acc", "flux"))
    if blk is None:
        return None
    nm, ns, npt = blk.shape
    flat = np.moveaxis(blk, 0, 1).reshape(ns, nm * npt)
    out, s0, s1 = _series(dict(vals=flat, step_list=entry["step_list"]), spec, variant)
    return out.reshape(out.shape[0], nm, npt).swapaxes(0, 1).astype(np.float32), s0, s1


def ens_band(ens, entries, key, variant):
    """10/90 % percentiles, mean and std of one card's time series, same shape as the card's series."""
    spec = resolve_spec(entries[key])
    r = ens_series(ens, entries, key, spec, variant)
    return None if r is None else ens_stats.band_stats(r[0])


def add_band(ax, times, lo, hi, ser, color, alpha=0.18):
    """Translucent 10-90 % ensemble band; where the ensemble has no value (step 0) it collapses to the line."""
    lo = np.where(np.isfinite(lo), lo, ser)
    hi = np.where(np.isfinite(hi), hi, ser)
    ax.fill_between(times, lo, hi, color=color, alpha=alpha, lw=0, zorder=1)


ENS_CARDS = [
    dict(id="2t", key=("2t", "sfc", None), name="2 m temperature", file="2m-temperature"),
    dict(id="msl", key=("msl", "sfc", None), name="Mean sea-level pressure", file="mean-sea-level-pressure"),
    dict(id="10si", key=("10si", "sfc", None), name="10 m wind speed", file="10m-wind-speed"),
    dict(id="tp", key=("tp", "sfc", None), name="Precipitation, probability of daily amounts",
         file="precipitation-probability", precip=True),
    dict(id="t850", key=("t", "pl", 850), name="Temperature at 850 hPa", file="temperature_0850hPa"),
    dict(id="z500", key=("z", "pl", 500), name="Geopotential height at 500 hPa", file="geopotential-height_0500hPa"),
]


def build_ens_cards(ens, entries, log=lambda m: None):
    """Ensemble-card data for the key variables, computed once (main process) and shared via CTX."""
    out = {}
    for cd in ENS_CARDS:
        key = cd["key"]
        if key not in entries:
            continue
        spec = resolve_spec(entries[key])
        r = ens_series(ens, entries, key, spec, None)
        if r is None:
            log(f"ensemble card {cd['id']}: variable not in the ensemble, skipped")
            continue
        arr, s0, s1 = r
        P = dict(cd, spec=spec, unit=spec["unit"])
        if cd.get("precip"):
            sums, a, b = ens_stats.period_sums(arr, s0, s1, 24)
            if sums.shape[1] == 0:
                log("ensemble card tp: lead time below 24 h, skipped")
                continue
            st = ens_stats.band_stats(sums)
            P.update(rows=[dict(kind="prob", thr=t, data=ens_stats.exceedance_prob(sums, t) * 100) for t in (1, 10)],
                     f0=a, f1=b, ts=dict(st, steps=b), ts_unit="mm per 24 h (ensemble mean)")
        else:
            valid = np.isfinite(arr).any(axis=(0, 2))
            if valid.sum() < 2:
                continue
            arr, s0, s1 = arr[:, valid], s0[valid], s1[valid]
            st = ens_stats.band_stats(arr)
            fm, f0, f1, fidx = frame_plan(st["mean"], s0, s1, "inst", None)
            P.update(rows=[dict(kind="mean", data=fm), dict(kind="spread", data=st["std"][fidx])],
                     f0=f0, f1=f1, ts=dict(st, steps=s1), ts_unit=f"{spec['unit']} (ensemble mean)")
        out[cd["id"]] = P
    return out


def _render_card(key, variant):
    c = CTX
    entry = c["entries"][key]
    spec = resolve_spec(entry)
    constant = c["constant"](key)
    region, geo = c["region"], c["geo"]
    bbox = region["bbox"]
    init: dt.datetime = c["init"]
    W, shape = geo["W"], geo["shape"]

    if constant:
        arr = entry["vals"][:1].astype(np.float64) * spec["scale"] + spec["offset"]
        st0, st1 = np.array([0]), np.array([0])
    else:
        arr, st0, st1 = _series(entry, spec, variant)
    if arr.shape[0] == 0 or not np.isfinite(arr).any():
        return (key, variant), None, "no valid data in region"

    cities = geo["cities"]
    # frames
    if constant:
        f_arr, f0, f1, fidx = arr, st0, st1, [0]
    else:
        f_arr, f0, f1, fidx = frame_plan(arr, st0, st1, spec["kind"], variant)
    frames = [to_grid(f_arr[j], W, shape) for j in range(f_arr.shape[0])]
    norm, cmap, ticks, extend = make_norm(spec, frames)
    cmap = cmap.copy()
    cmap.set_bad("#00000000")

    name = spec["name"]
    if entry["levtype"] == "pl":
        name = f"{name} at {entry['level']} hPa"
    unit = spec["unit"]
    kind = spec["kind"]
    per_interval = kind != "inst" and variant != "cumul" and not constant
    interval = int(np.median(st1 - st0)) if per_interval else 0  # time-series resolution
    finterval = int(np.median(f1 - f0)) if per_interval else 0  # map-frame aggregation
    ts_unit_lbl = unit
    if kind == "acc" and variant == "cumul":
        name += " - accumulated total"
        unit_lbl = ts_unit_lbl = f"{unit} since initial time"
    elif kind == "acc":
        name += " - daily amounts" if finterval == 24 else f" - {finterval}-hourly amounts"
        unit_lbl = f"{unit} per {finterval} h"
        ts_unit_lbl = f"{unit} per {interval} h"
    elif kind == "flux":
        unit_lbl = f"{unit} (mean over {finterval} h)"
        ts_unit_lbl = f"{unit} (mean over {interval} h)"
    else:
        unit_lbl = unit

    n = len(frames)
    rows = 1 if n <= 6 else 2
    ncols = math.ceil(n / rows)
    FW, FH = 16.0, 11.5
    fig = plt.figure(figsize=(FW, FH), dpi=100, facecolor="white")
    title_h = 1.0
    ts_h = 2.0 if not constant else 0.0  # time-series axes height (inches)
    cbar_y = (0.5 + ts_h + 0.9) if not constant else 1.3  # colorbar axes bottom (inches from page bottom)
    maps_bottom = cbar_y + 0.65 + (0.3 if spec.get("kmh") else 0)
    maps_h = FH - title_h - 0.5 - maps_bottom
    # --- title
    fig.text(0.02, 1 - 0.35 / FH, name, fontsize=23, fontweight="bold", va="center", ha="left", color="#14213d")
    last_valid = init + dt.timedelta(hours=int(st1.max()))
    if constant:
        sub = f"Constant field (time-independent)  ·  {unit_lbl}  ·  {region['title']}"
    else:
        sub = (f"Initial time {init:%a %Y-%m-%d %H} UTC   ·   valid {fmt_valid(init + dt.timedelta(hours=int(st1.min())))}"
               f" to {fmt_valid(last_valid)}   ·   {unit_lbl}")
    fig.text(0.02, 1 - 0.78 / FH, sub, fontsize=11.5, va="center", ha="left", color="#444")
    fig.text(0.98, 1 - 0.35 / FH, "AIFS Single v2.0", fontsize=11, ha="right", va="center", color="#888")

    # --- maps
    mw = min(0.96, max(0.35, ncols * 0.2)) if not constant else 0.5
    left = (1 - mw) / 2
    top = 1 - (title_h + 0.5) / FH
    bottom = top - maps_h / FH
    # each map spans two half-columns so a short last row can be centred
    gs = fig.add_gridspec(rows, 2 * ncols if not constant else 1, left=left, right=left + mw, top=top,
                          bottom=bottom, wspace=0.08, hspace=0.18)
    last_row_off = ncols * rows - n  # half-columns to shift the last row by
    quiv = entry.get("uv")
    im = None
    for j in range(n):
        r, c0 = divmod(j, ncols)
        if constant:
            ax = fig.add_subplot(gs[0, 0])
        else:
            c0 = 2 * c0 + (last_row_off if r == rows - 1 else 0)
            ax = fig.add_subplot(gs[r, c0:c0 + 2])
        g = frames[j]
        gm = np.ma.masked_invalid(g)
        if spec["style"] == "precip":
            gm = np.ma.masked_less(gm, norm.boundaries[0])
        im = ax.imshow(gm, origin="lower", extent=(geo["gx"][0], geo["gx"][-1], geo["gy"][0], geo["gy"][-1]),
                       cmap=cmap, norm=norm, interpolation="bilinear",
                       zorder=2)
        if spec.get("contour"):
            cstep = spec["contour"]
            fin = g[np.isfinite(g)]
            if fin.size:
                if cstep > 0:
                    lv = np.arange(math.floor(fin.min() / cstep) * cstep, fin.max() + cstep, cstep)
                else:
                    lv = ticker.MaxNLocator(nbins=14).tick_values(fin.min(), fin.max())
                if len(lv) > 1:
                    ax.contour(geo["gx"], geo["gy"], g, levels=lv, colors="k", linewidths=0.35, alpha=0.6, zorder=3)
        if quiv and fidx is not None:
            ug = to_grid(c["entries"][quiv[0]]["vals"][fidx[j]], W, shape)
            vg = to_grid(c["entries"][quiv[1]]["vals"][fidx[j]], W, shape)
            sk = max(1, int(round(0.7 / GRID_RES)) if bbox[1] - bbox[0] < 20 else int(round(2.0 / GRID_RES)))
            ax.quiver(geo["gx"][::sk], geo["gy"][::sk], ug[::sk, ::sk], vg[::sk, ::sk], color="k", alpha=0.6,
                      pivot="mid", width=0.005, zorder=5)
        setup_map(ax, bbox, c["lines"], cities)
        if constant:
            ttl = "Fixed field"
        elif kind == "inst":
            v = init + dt.timedelta(hours=int(f1[j]))
            ttl = f"{fmt_valid(v)}\n+{int(f1[j])} h"
        elif variant == "cumul":
            v = init + dt.timedelta(hours=int(f1[j]))
            ttl = f"until {fmt_valid(v)}\n0 to +{int(f1[j])} h"
        else:
            a = init + dt.timedelta(hours=int(f0[j]))
            b = init + dt.timedelta(hours=int(f1[j]))
            ttl = (f"{a:%d %b %H}–{b:%H} UTC" if (b - a) < dt.timedelta(hours=24)
                   else f"{a:%d %b %H}–{b:%d %b %H} UTC") + f"\n+{int(f0[j])} to +{int(f1[j])} h"
        ax.set_title(ttl, fontsize=9.5 if ncols <= 5 else 8.5, pad=3, linespacing=1.15)

    # --- colorbar
    cax = fig.add_axes([0.25, cbar_y / FH, 0.5, 0.17 / FH])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend, ticks=list(ticks) if spec["style"] != "precip" else None)
    if spec["style"] == "precip":
        cb.set_ticks(list(norm.boundaries))
        cb.ax.set_xticklabels([f"{b:g}" for b in norm.boundaries])
    cb.set_label(unit_lbl, fontsize=11)
    cb.ax.tick_params(labelsize=9.5)
    if spec.get("kmh"):
        sec = cb.ax.secondary_xaxis("top", functions=(lambda x: x * 3.6, lambda x: x / 3.6))
        sec.set_xlabel("km/h", fontsize=9.5, labelpad=2)
        sec.tick_params(labelsize=9)

    # --- city panel
    if constant:
        tx = fig.add_axes([0.72, 0.3, 0.25, 0.4])
        tx.axis("off")
        tx.text(0, 1, "Nearest grid point values", fontsize=13, fontweight="bold", va="top")
        for k, (cn, cc) in enumerate(cities.items()):
            val = arr[0, cc["idx"]]
            tx.text(0, 0.86 - 0.12 * k, f"{cn}:", fontsize=12, color=CITY_COLORS[k], va="top", fontweight="bold")
            tx.text(0.45, 0.86 - 0.12 * k, "n/a (no data)" if not np.isfinite(val) else f"{val:.3g} {unit}",
                    fontsize=12, va="top")
    else:
        tx_times = [init + dt.timedelta(hours=int(s)) for s in st1]
        band = c["ens_stats"].get((key, variant)) if c.get("ens") else None
        ax = fig.add_axes([0.065, 0.75 / FH, 0.80, ts_h / FH])
        drawn = 0
        for (cn, cc), col in zip(cities.items(), CITY_COLORS):
            ser = arr[:, cc["idx"]]
            if np.isfinite(ser).any():
                if band is not None and band["p10"].shape == arr.shape:
                    add_band(ax, tx_times, band["p10"][:, cc["idx"]], band["p90"][:, cc["idx"]], ser, col)
                ax.plot(tx_times, ser, "-o", color=col, lw=1.8, ms=3.5, label=cn, zorder=3)
                drawn += 1
        if not drawn:  # e.g. sea-only fields: no city is at sea -> regional statistics instead
            with np.errstate(all="ignore"):
                ax.plot(tx_times, np.nanmax(arr, axis=1), "-o", color="#d62728", lw=1.8, ms=3.5, label="Region maximum")
                ax.plot(tx_times, np.nanmean(arr, axis=1), "-o", color="#1f77b4", lw=1.8, ms=3.5, label="Region mean")
            ax.set_title("No city grid point has data (sea-only field): regional max and mean over all valid points",
                         fontsize=11, loc="left", color="#333")
        ax.grid(alpha=0.3)
        ax.set_ylabel(ts_unit_lbl, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H UTC"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        ax.tick_params(labelsize=9)
        if drawn:
            ax.set_title("Nearest grid point to each city", fontsize=11, loc="left", color="#333")
            if band is not None and band["p10"].shape == arr.shape:
                ax.set_title(f"shading: 10–90 % ensemble range ({c['ens']['n']} members)", fontsize=9.5,
                             loc="right", color="#666")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if spec.get("kmh"):
            sec = ax.secondary_yaxis("right", functions=(lambda x: x * 3.6, lambda x: x / 3.6))
            sec.set_ylabel("km/h", fontsize=10)
            ax.legend(loc="center left", bbox_to_anchor=(1.07, 0.5), frameon=False, fontsize=10)
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.text(0.98, 0.012, f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC  ·  grid: N320 (~31 km), "
             f"nearest point / bilinear interpolation", fontsize=8, ha="right", color="#999")
    fn = c["fnames"][(key, variant)]
    fig.savefig(c["outdir"] / fn, dpi=100)
    plt.close(fig)
    return (key, variant), fn, None


def _render_ens_card(cid):
    """Ensemble card: row 1 = ensemble mean (or probability), row 2 = spread (or second threshold), plus city panel."""
    c = CTX
    P = c["ens_cards"][cid]
    geo, region, init, lines = c["geo"], c["region"], c["init"], c["lines"]
    bbox, W, shape, cities = region["bbox"], geo["W"], geo["shape"], geo["cities"]
    spec, rows, f0, f1, ts = P["spec"], P["rows"], P["f0"], P["f1"], P["ts"]
    nm, unit, precip = c["ens"]["n"], P["unit"], bool(P.get("precip"))
    n = len(f1)
    cmap_b = lambda cm: plt.get_cmap(cm).copy()  # noqa: E731

    grids = [[to_grid(r["data"][j], W, shape) for j in range(n)] for r in rows]
    if precip:
        norm = mcolors.Normalize(0, 100)
        cm = cmap_b("YlGnBu")
        scales = [dict(norm=norm, cmap=cm, ticks=np.arange(0, 101, 20), extend="neither",
                       label="probability (%)")] * 2
        row_lbl = [f"P(daily ≥ {r['thr']} mm), %" for r in rows]
        single_bar = True
    else:
        norm, cm, ticks, ext = make_norm(spec, grids[0])
        fin = np.concatenate([g[np.isfinite(g)] for g in grids[1]]) if n else np.array([0.0])
        hi = float(np.percentile(fin, 99.5)) if fin.size else 1.0
        sa, sb, st_ticks = nice_limits(0, max(hi, 1e-6), nbins=6)
        scales = [dict(norm=norm, cmap=cm, ticks=ticks, extend=ext, label=f"ensemble mean ({unit})"),
                  dict(norm=mcolors.Normalize(sa, sb), cmap=cmap_b("magma_r"), ticks=st_ticks, extend="max",
                       label=f"ensemble spread: standard deviation ({unit})")]
        row_lbl = ["ensemble mean", "spread (std. dev.)"]
        single_bar = False

    FW, FH = 16.0, 11.5
    fig = plt.figure(figsize=(FW, FH), dpi=100, facecolor="white")
    title_h, ts_h = 1.0, 2.0
    cbar_y = 0.5 + ts_h + 1.15
    maps_bottom = cbar_y + 0.65 + (0.3 if spec.get("kmh") and not precip else 0)
    maps_h = FH - title_h - 0.5 - maps_bottom
    fig.text(0.02, 1 - 0.35 / FH, f"Ensemble: {P['name']}", fontsize=23, fontweight="bold", va="center",
             color="#14213d")
    sub = (f"Initial time {init:%a %Y-%m-%d %H} UTC   ·   valid {fmt_valid(init + dt.timedelta(hours=int(f1.min())))} to "
           f"{fmt_valid(init + dt.timedelta(hours=int(f1.max())))}   ·   {nm}-member ensemble "
           f"({c['ens']['label']})   ·   {region['title']}")
    fig.text(0.02, 1 - 0.78 / FH, sub, fontsize=11.5, va="center", color="#444")
    fig.text(0.98, 1 - 0.35 / FH, c["ens"]["model_name"], fontsize=11, ha="right", va="center", color="#888")

    mw = min(0.96, max(0.35, n * 0.095))
    left = (1 - mw) / 2
    top = 1 - (title_h + 0.5) / FH
    gs = fig.add_gridspec(2, n, left=left, right=left + mw, top=top, bottom=top - maps_h / FH, wspace=0.06,
                          hspace=0.12)
    ims = [None, None]
    for r in range(2):
        sc = scales[r]
        for j in range(n):
            ax = fig.add_subplot(gs[r, j])
            gm = np.ma.masked_invalid(grids[r][j])
            if precip:
                gm = np.ma.masked_less(gm, 1.0)
            ims[r] = ax.imshow(gm, origin="lower", extent=(geo["gx"][0], geo["gx"][-1], geo["gy"][0], geo["gy"][-1]),
                               cmap=sc["cmap"], norm=sc["norm"], interpolation="bilinear", zorder=2)
            if r == 0 and not precip and spec.get("contour"):
                g, cstep = grids[0][j], spec["contour"]
                fin = g[np.isfinite(g)]
                if fin.size:
                    lv = (np.arange(math.floor(fin.min() / cstep) * cstep, fin.max() + cstep, cstep) if cstep > 0
                          else ticker.MaxNLocator(nbins=14).tick_values(fin.min(), fin.max()))
                    if len(lv) > 1:
                        ax.contour(geo["gx"], geo["gy"], g, levels=lv, colors="k", linewidths=0.35, alpha=0.6, zorder=3)
            setup_map(ax, bbox, lines, cities, labels=(r == 0 and j == 0))
            if j == 0:
                ax.set_ylabel(row_lbl[r], fontsize=10.5, labelpad=6)
            if r == 0:
                a = init + dt.timedelta(hours=int(f0[j]))
                b = init + dt.timedelta(hours=int(f1[j]))
                ttl = (f"{a:%d %b %H} UTC –\n{b:%d %b %H} UTC\n+{int(f0[j])} to +{int(f1[j])} h" if precip
                       else f"{fmt_valid(b)}\n+{int(f1[j])} h")
                ax.set_title(ttl, fontsize=8.5 if n > 5 else 9.5, pad=3, linespacing=1.15)
    # colourbars
    if single_bar:
        pos = [[0.25, cbar_y / FH, 0.5, 0.17 / FH]]
    else:
        pos = [[0.07, cbar_y / FH, 0.38, 0.17 / FH], [0.55, cbar_y / FH, 0.38, 0.17 / FH]]
    for k, ps in enumerate(pos):
        sc = scales[k]
        cb = fig.colorbar(ims[k], cax=fig.add_axes(ps), orientation="horizontal", extend=sc["extend"],
                          ticks=list(sc["ticks"]))
        cb.set_label(sc["label"], fontsize=11)
        cb.ax.tick_params(labelsize=9.5)
        if spec.get("kmh") and not precip:
            sec = cb.ax.secondary_xaxis("top", functions=(lambda x: x * 3.6, lambda x: x / 3.6))
            sec.set_xlabel("km/h", fontsize=9.5, labelpad=2)
            sec.tick_params(labelsize=9)

    # city panel: ensemble mean, 10-90 % band
    ax = fig.add_axes([0.065, 0.75 / FH, 0.80, ts_h / FH])
    tx = [init + dt.timedelta(hours=int(s)) for s in ts["steps"]]
    for (cn, cc), col in zip(cities.items(), CITY_COLORS):
        m = ts["mean"][:, cc["idx"]]
        if np.isfinite(m).any():
            add_band(ax, tx, ts["p10"][:, cc["idx"]], ts["p90"][:, cc["idx"]], m, col, 0.2)
            ax.plot(tx, m, "-o", color=col, lw=1.8, ms=3.5, label=cn, zorder=3)
    ax.grid(alpha=0.3)
    ax.set_ylabel(P["ts_unit"], fontsize=10)
    ax.set_title("Nearest grid point to each city: ensemble mean", fontsize=11, loc="left", color="#333")
    ax.set_title(f"shading: 10–90 % ensemble range ({nm} members)", fontsize=9.5, loc="right", color="#666")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H UTC"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if spec.get("kmh") and not precip:
        sec = ax.secondary_yaxis("right", functions=(lambda x: x * 3.6, lambda x: x / 3.6))
        sec.set_ylabel("km/h", fontsize=10)
        ax.legend(loc="center left", bbox_to_anchor=(1.07, 0.5), frameon=False, fontsize=10)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.text(0.98, 0.012, f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC  ·  {c['ens']['note']}, "
             f"probabilities not calibrated  ·  grid: N320 (~31 km)", fontsize=8, ha="right",
             color="#999")
    fn = c["fnames"][(("ens", cid), None)]
    fig.savefig(c["outdir"] / fn, dpi=100)
    plt.close(fig)
    return fn, None


# --------------------------------------------------------------------------------------
# Overview card
# --------------------------------------------------------------------------------------
def render_overview():
    c = CTX
    E, geo, region, init = c["entries"], c["geo"], c["region"], c["init"]
    tz = ZoneInfo(region["tz"])
    need = lambda k: E.get(k)  # noqa: E731
    t2, tp, tcc = need(("2t", "sfc", None)), need(("tp", "sfc", None)), need(("tcc", "sfc", None))
    u10, v10 = need(("10u", "sfc", None)), need(("10v", "sfc", None))
    if t2 is None:
        return None
    cities = geo["cities"]
    steps = t2["step_list"]
    times = [(init.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=int(s))).astimezone(tz) for s in steps]
    xs = mdates.date2num(times)
    ens = c.get("ens")
    fig = plt.figure(figsize=(16, 10.6 if ens else 10), dpi=100, facecolor="white")
    fig.text(0.02, 0.965, f"Weather forecast {region['title']}", fontsize=24, fontweight="bold", color="#14213d",
             va="center")
    sub = (f"AIFS Single v2.0 forecast started {init:%a %d %b %Y, %H} UTC  ·  local time "
           f"({region['tz']})  ·  nearest model grid point to each city")
    if ens:
        sub += f"  ·  {ens['n']}-member ensemble ({ens['label']})"
    fig.text(0.02, 0.928, sub, fontsize=11.5, color="#444", va="center")
    outer = fig.add_gridspec(2, 3, left=0.045, right=0.96, top=0.86, bottom=0.075 if ens else 0.05, wspace=0.28, hspace=0.45)
    ncity = len(cities)
    # shared axes for all cities so the panels can be compared at a glance
    idxs = [cc["idx"] for cc in cities.values()]
    all_t = t2["vals"][:, idxs] - 273.15
    has_rain = tp is not None and len(tp["step_list"]) > 1
    # ensemble statistics at the city points (members first, percentiles afterwards)
    et = er = ec = None
    if ens:
        b = ens_block(ens, E, t2)
        if b is not None:
            tb = b[:, :, idxs] - 273.15
            p10, p90 = ens_stats.percentiles(tb)
            et = dict(p10=p10, p90=p90, mn=tb.min(axis=0), mx=tb.max(axis=0))
        if has_rain:
            r = ens_series(ens, E, ("tp", "sfc", None), SPECS["tp"], None)
            if r is not None:
                per, s0, s1 = r[0][:, :, idxs], r[1], r[2]
                p10, p90 = ens_stats.percentiles(per)
                er = dict(p10=p10, p90=p90)
                sums, wa, wb = ens_stats.period_sums(per, s0, s1, 24)
                if sums.shape[1]:
                    ec = dict(p=ens_stats.exceedance_prob(sums, 1.0) * 100, a=wa, b=wb)
    lo_t, hi_t = all_t.min(), all_t.max()
    if et is not None:
        lo_t, hi_t = min(lo_t, np.nanmin(et["mn"])), max(hi_t, np.nanmax(et["mx"]))
    t_lo, t_hi = np.floor(lo_t) - 2, np.ceil(hi_t) + 2.5
    rain_hi = max(2.5, (np.maximum(np.diff(tp["vals"][:, idxs], axis=0), 0) * 1000).max() * 2.2) if has_rain else 0
    if er is not None:
        rain_hi = max(rain_hi, float(np.nanmax(er["p90"])) * 2.2)
    for k, (cn, cc) in enumerate(cities.items()):
        if ec is not None:
            sub = outer[k // 3, k % 3].subgridspec(4, 1, height_ratios=[0.9, 4.2, 0.8, 1.5], hspace=0.08)
            axc, axm, axr, axw = (fig.add_subplot(sub[j]) for j in range(4))
        else:
            sub = outer[k // 3, k % 3].subgridspec(3, 1, height_ratios=[0.9, 4.2, 1.5], hspace=0.08)
            axc, axm, axw = (fig.add_subplot(sub[j]) for j in range(3))
            axr = None
        axes = [a for a in (axc, axm, axr, axw) if a is not None]
        i = cc["idx"]
        temp = t2["vals"][:, i] - 273.15
        axm.plot(xs, temp, color="#d62728", lw=2.4, marker="o", ms=4, zorder=5)
        if et is not None:
            for lo, hi, al in ((et["mn"], et["mx"], 0.10), (et["p10"], et["p90"], 0.24)):
                add_band(axm, xs, lo[:, k], hi[:, k], temp, "#d62728", al)
        else:
            axm.fill_between(xs, temp, t_lo, color="#d62728", alpha=0.08)
        axm.set_ylim(t_lo, t_hi)
        axm.set_ylabel("°C", color="#d62728", fontsize=10)
        axm.tick_params(axis="y", colors="#d62728", labelsize=9)
        axm.grid(alpha=0.25)
        head = f"max {temp.max():.0f}°C / min {temp.min():.0f}°C"
        if has_rain:
            acc = np.maximum(np.diff(tp["vals"][:, i]), 0) * 1000
            xe = xs[1:]
            half = np.median(np.diff(xs)) * 0.42
            axp = axm.twinx()
            axp.bar(xe - np.median(np.diff(xs)) / 2, acc, width=half * 2, color="#1f77b4", alpha=0.75, zorder=2)
            if er is not None and er["p10"].shape[0] == len(acc):
                axp.vlines(xe - np.median(np.diff(xs)) / 2, er["p10"][:, k], er["p90"][:, k], color="#0b2e6b", lw=1.1,
                           zorder=3)
            axp.set_ylim(0, rain_hi)
            axp.set_ylabel("rain mm / 6 h", color="#1f77b4", fontsize=9)
            axp.tick_params(axis="y", colors="#1f77b4", labelsize=9)
            axm.set_zorder(axp.get_zorder() + 1)
            axm.patch.set_visible(False)
            head += f"  ·  rain {acc.sum():.1f} mm"
        axc.set_title(cn, loc="left", fontsize=15, fontweight="bold", color="#14213d", pad=18)
        axc.text(0, 1.08, head, transform=axc.transAxes, fontsize=9.5, color="#444", va="bottom")
        # cloud strip
        axc.set_ylim(0, 1)
        if tcc is not None:
            cl = np.clip(tcc["vals"][:, i], 0, 1)
            w = np.median(np.diff(xs))
            xcl = mdates.date2num([(init.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=int(s))).astimezone(tz)
                                   for s in tcc["step_list"]])  # tcc may lack step 0: align by valid time
            for x, cv in zip(xcl, cl):
                axc.bar(x, 1, width=w, color=plt.cm.Greys(0.12 + 0.7 * cv), edgecolor="white",
                        lw=0.5 if len(xcl) <= 16 else 0)
                if len(xcl) <= 16:  # numbers only fit for short forecasts; shading alone otherwise
                    axc.text(x, 0.5, f"{cv * 100:.0f}", ha="center", va="center", fontsize=7,
                             color="white" if cv > 0.45 else "#333")
        axc.set_ylabel("cloud %", fontsize=8, rotation=0, ha="right", va="center")
        axc.set_yticks([])
        # chance of rain: P(24 h amount >= 1 mm) in the ensemble
        if axr is not None:
            axr.set_ylim(0, 1)
            for pr, wa, wb in zip(ec["p"][:, k], ec["a"], ec["b"]):
                xc = mdates.date2num((init.replace(tzinfo=dt.timezone.utc)
                                      + dt.timedelta(hours=float(wa + wb) / 2)).astimezone(tz))
                axr.bar(xc, 1, width=(wb - wa) / 24 * 0.96, color=plt.cm.Blues(0.05 + 0.75 * pr / 100), lw=0)
                axr.text(xc, 0.5, f"{pr:.0f}", ha="center", va="center", fontsize=7.5,
                         color="white" if pr > 55 else "#223")
            axr.set_ylabel("rain ≥ 1 mm\nper 24 h, %", fontsize=7.5, rotation=0, ha="right", va="center")
            axr.set_yticks([])
        # wind strip
        axw.set_ylim(-1, 1)
        if u10 is not None and v10 is not None:
            u, v = u10["vals"][:, i], v10["vals"][:, i]
            # long forecasts: one arrow + number every k steps (~12 per panel) so labels don't overlap
            k = max(1, int(np.ceil(len(xs) / 12)))
            xw, u, v = xs[::k], u[::k], v[::k]
            sp = np.hypot(u, v)
            axw.quiver(xw, np.full_like(xw, 0.35), u / np.maximum(sp, 1e-6), v / np.maximum(sp, 1e-6), pivot="mid",
                       angles="uv", scale_units="height", scale=4, width=0.006, color="#2a4d69", zorder=3)
            for x, s in zip(xw, sp):
                axw.text(x, -0.55, f"{s * 3.6:.0f}", ha="center", va="center", fontsize=7.5, color="#2a4d69")
        axw.set_ylabel("wind km/h", fontsize=8, rotation=0, ha="right", va="center")
        axw.set_yticks([])
        for a in axes:
            a.set_xlim(xs[0] - 0.12, xs[-1] + 0.12)
            for s in ("top",):
                a.spines[s].set_visible(False)
            # local midnights
        for a in axes:
            a.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12 if len(xs) > 12 else 6), tz=tz) if len(xs) <= 20
                                      else mdates.DayLocator(interval=2 if len(xs) > 28 else 1, tz=tz))
            a.xaxis.set_major_formatter(mdates.DateFormatter("%a %H", tz=tz) if len(xs) <= 20
                                        else mdates.DateFormatter("%a %d", tz=tz))
        axc.tick_params(labelbottom=False, bottom=False)
        axm.tick_params(labelbottom=False)
        if axr is not None:
            axr.tick_params(labelbottom=False, bottom=False)
        axw.tick_params(labelsize=8)
        for lab in axw.get_xticklabels():
            lab.set_rotation(0)
        d0 = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
        d = d0 + dt.timedelta(days=1)
        while d < times[-1]:
            for a in axes:
                a.axvline(mdates.date2num(d), color="#999", lw=0.8, ls=":")
            d += dt.timedelta(days=1)
    note = ("Red line: air temperature (2 m)  ·  blue bars: rain per 6 h  ·  grey strip: total cloud cover %  ·  "
            "arrows: wind direction (10 m), number = km/h")
    if ens:
        fig.text(0.98, 0.027, "Dark red band: ensemble 10–90 %, light band: ensemble min–max  ·  dark whiskers on the bars: "
                 "10–90 % of the 6 h rain  ·  bottom row: ensemble chance of ≥ 1 mm rain in each 24 h from the initial time (%)",
                 fontsize=9, ha="right", color="#777")
    fig.text(0.98, 0.012, note, fontsize=9, ha="right", color="#777")
    fn = "0-overview.png"
    fig.savefig(c["outdir"] / fn, dpi=100)
    plt.close(fig)
    return fn


# --------------------------------------------------------------------------------------
# Index page
# --------------------------------------------------------------------------------------
def write_index(outdir, region, init, items, skipped, lead):
    groups = ["Overview", "Ensemble", "Comparison", "Surface", "Pressure levels", "Waves", "Constants"]
    h = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIFS weather cards {html.escape(region['title'])} {init:%Y-%m-%d %H}Z</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#14213d}}
header{{padding:20px 28px;background:#14213d;color:#fff}} header h1{{margin:0;font-size:26px}} header p{{margin:6px 0 0;opacity:.8}}
main{{padding:8px 28px 40px}} h2{{margin:28px 0 10px;border-bottom:2px solid #ccd;padding-bottom:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 4px #0002;overflow:hidden;text-decoration:none;color:inherit;display:block}}
.card img{{width:100%;display:block}} .card div{{padding:8px 10px;font-size:14px}} .card small{{color:#667}}
</style></head><body><header><h1>AIFS weather cards: {html.escape(region['title'])}</h1>
<p>Initial time {init:%A %d %B %Y, %H} UTC &middot; lead time up to {lead} h &middot; {len(items)} cards</p></header><main>"""]
    for g in groups:
        its = [i for i in items if i["group"] == g]
        if not its:
            continue
        h.append(f"<h2>{g} ({len(its)})</h2><div class='grid'>")
        for i in its:
            h.append(f"<a class='card' href='{i['file']}'><img loading='lazy' src='{i['file']}' alt=''>"
                     f"<div><b>{html.escape(i['title'])}</b><br><small>{html.escape(i['unit'])}</small></div></a>")
        h.append("</div>")
    if skipped:
        h.append("<h2>Skipped</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in skipped) + "</ul>")
    h.append("</main></body></html>")
    (outdir / "index.html").write_text("\n".join(h))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def find_cycle(out_root: Path, date: str | None) -> Path | None:
    if date:
        d = out_root / date
        return d if d.exists() else None
    link = out_root / "latest"
    if link.exists():
        return link.resolve()
    done = sorted(p.parent for p in out_root.glob("*/DONE"))
    return done[-1] if done else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="europe", choices=sorted(REGIONS), help="named region (default europe)")
    p.add_argument("--lead-time", default="240h", help="forecast length passed to forecast.py, e.g. 48h, 5d, 240h "
                   "(default 240h = 10 days; about 3.5 min on a large GPU)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--force", action="store_true", help="re-run the forecast even if the cycle is already done")
    p.add_argument("--date", help="cycle YYYYMMDDHH instead of the latest")
    p.add_argument("--source", default="ecmwf", choices=["ecmwf", "aws", "azure", "google"])
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--output-root", default=str(ROOT / "forecasts"))
    p.add_argument("--skip-forecast", action="store_true", help="do not run the forecast; render the newest finished cycle")
    p.add_argument("--grib", help="render this GRIB file (implies --skip-forecast); cards go next to it")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel card renderers")
    p.add_argument("--ensemble", default="auto", choices=["auto", "off"],
                   help="auto: add ensemble bands/probabilities when forecasts/<cycle>/ensemble/<region>/ has members "
                   "(default); off: render the deterministic forecast only")
    p.add_argument("--ensemble-members", type=int, default=0, metavar="N",
                   help="after the forecast, run pipeline/ensemble.py with N perturbed members (50 = 51 members, "
                   "about 1 h for 10 days on a large GPU). Default 0: do not run it, but use stored members")
    p.add_argument("--ensemble-model", default="both", choices=["both", "ens", "single"],
                   help="which ensemble to run and show: ens = AIFS ENS v2.0 (ensemble-aifs-ens/<region>/); "
                   "single = AIFS Single with our perturbed initial conditions (ensemble/<region>/); both (default) = "
                   "run both, show AIFS ENS on the cards (or the other one if only that exists), and add "
                   "comparison cards (6-compare_*.png) when both exist")
    p.add_argument("--only", help="comma-separated params to render (e.g. 2t,tp,t_850), for debugging")
    args = p.parse_args()
    out_root = Path(args.output_root)

    if args.grib:
        grib = Path(args.grib).resolve()
        cycle_dir = grib.parent
    else:
        if not args.skip_forecast:
            cmd = [sys.executable, str(Path(__file__).parent / "forecast.py"), "--lead-time", args.lead_time,
                   "--device", args.device, "--source", args.source, "--threads", str(args.threads),
                   "--output-root", str(out_root)]
            if args.date:
                cmd += ["--date", args.date]
            if args.force:
                cmd.append("--force")
            log("running forecast: " + " ".join(cmd[1:]))
            rc = subprocess.call(cmd)
            if rc != 0:
                log(f"forecast failed (exit {rc}); not rendering")
                return rc
        run_models = ["ens", "single"] if args.ensemble_model == "both" else [args.ensemble_model]
        for model in run_models if args.ensemble_members > 0 else []:
            cmd = [sys.executable, str(Path(__file__).parent / "ensemble.py"), "--members", str(args.ensemble_members),
                   "--lead-time", args.lead_time, "--region", args.region, "--device", args.device,
                   "--source", args.source, "--output-root", str(out_root), "--model", model]
            if args.date:
                cmd += ["--date", args.date]
            if args.force:
                cmd.append("--force")
            log("running ensemble: " + " ".join(cmd[1:]))
            rc = subprocess.call(cmd)
            if rc != 0:
                log(f"ensemble ({model}) failed (exit {rc}); rendering with the members that exist")
        cycle_dir = find_cycle(out_root, args.date)
        if cycle_dir is None or not (cycle_dir / "DONE").exists():
            log("no finished forecast cycle found" + (" (another run may still be in progress)" if not args.skip_forecast else ""))
            return 1
        grib = next(iter(sorted(cycle_dir.glob("*.grib"))), None)
        if grib is None:
            log(f"no GRIB in {cycle_dir}")
            return 1

    region = REGIONS[args.region]
    outdir = cycle_dir / "cards" / args.region
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.only:  # full render: drop cards from older naming schemes; --only keeps the rest
        for old in outdir.glob("*.png"):
            old.unlink()
    t0 = time.time()
    log(f"reading {grib}")
    entries, geo, base = load_data(grib, region)
    init = base if isinstance(base, dt.datetime) else dt.datetime.fromisoformat(str(base))
    nsteps_all = max(len(e["step_list"]) for e in entries.values())
    log(f"{len(entries)} variables loaded in {time.time() - t0:.1f}s; init {init:%Y-%m-%d %H}Z, "
        f"steps up to +{max(int(e['step_list'].max()) for e in entries.values())} h")
    lines = load_lines(region["bbox"])

    ens, loaded = None, {}
    for model in (["ens", "single"] if args.ensemble_model == "both" else [args.ensemble_model]):
        ens_dir = ens_stats.ensemble_dir(cycle_dir, args.region, model)
        if args.ensemble != "auto" or not ens_dir.is_dir():
            continue
        te = time.time()
        e = ens_stats.load_ensemble(ens_dir, geo["lat"], geo["lon"], log)
        if not e:
            continue
        e["n"] = len(e["members"])
        if model == "ens":
            e.update(label="AIFS ENS v2.0", model_name="AIFS ENS v2.0", note="AIFS ENS v2.0 ensemble")
        else:
            e.update(label="initial-condition perturbations", model_name="AIFS Single v2.0",
                     note="simple initial-condition ensemble (not ECMWF ENS)")
        done = (e["dir"] / "DONE").exists()
        log(f"ensemble {model}: {e['n']} members used" + ("" if done else f" (run not finished: {e['expected']} planned)")
            + f", {len(e['names'])} variables, steps +{int(e['steps'][0])}..+{int(e['steps'][-1])} h, "
            f"loaded in {time.time() - te:.1f}s")
        loaded[model] = e
    if loaded:
        ens = loaded.get("ens") or loaded["single"]  # the cards show AIFS ENS when it exists
        log(f"cards show the {ens['label']} ensemble")
    CTX.update(entries=entries, geo=geo, region=region, init=init, outdir=outdir, lines=lines, ens=ens, ens_stats={},
               ens_cards={}, constant=lambda k: len(entries[k]["step_list"]) == 1 and nsteps_all > 1)

    tasks, meta, fnames, used_names = [], {}, {}, set()
    CTX["fnames"] = fnames
    only = set(args.only.split(",")) if args.only else None
    for key, e in entries.items():
        stem = card_path(key)
        if only and stem not in only and key[0] not in only:
            continue
        spec = spec_for(e["param"], e["levtype"], e["mname"], e["munits"])
        if e["levtype"] == "pl":
            spec = dict(PL_SPECS.get(e["param"], spec))
        if e["param"] == "z" and e["levtype"] != "pl":
            spec = dict(SPECS["z"])
        const = CTX["constant"](key)
        if const:
            group = "Constants"
        elif e["levtype"] == "pl":
            group = "Pressure levels"
        elif is_wave(e["param"], e["mname"]):
            group = "Waves"
        else:
            group = "Surface"
        variants = [None]
        if spec["kind"] == "acc" and not const:
            variants.append("cumul")
        for v in variants:
            title = spec["name"] + (f" at {e['level']} hPa" if e["levtype"] == "pl" else "")
            if v == "cumul":
                title += " (total)"
            elif spec["kind"] == "acc":
                title += " (per interval)"
            tasks.append((key, v))
            fname = card_filename(group, spec["name"], e["level"] if e["levtype"] == "pl" else None, v, spec["kind"],
                                  e["param"])
            if fname in used_names:  # two fields with the same readable name: disambiguate with the GRIB code
                fname = fname[:-4] + f"_{slugify(e['param'])}.png"
            used_names.add(fname)
            fnames[(key, v)] = fname
            meta[(key, v)] = dict(group=group, title=title, unit=spec["unit"], param=e["param"], level=e["level"] or 0)
    def sort_key(t):
        m = meta[t]
        pr = SURFACE_ORDER.index(m["param"]) if m["param"] in SURFACE_ORDER else 999
        return (["Surface", "Pressure levels", "Waves", "Constants", "Ensemble"].index(m["group"]), pr, m["param"], m["level"], t[1] or "")

    if ens:
        te = time.time()
        for t in tasks:  # bands for every card with ensemble data (computed once, shared with the workers)
            if meta[t]["group"] != "Constants":
                st = ens_band(ens, entries, t[0], t[1])
                if st is not None:
                    CTX["ens_stats"][t] = st
        log(f"ensemble bands for {len(CTX['ens_stats'])} of {len(tasks)} cards in {time.time() - te:.1f}s")
        missing = sorted({meta[t]["param"] + (f"_{meta[t]['level']}" if meta[t]["level"] else "") for t in tasks
                          if t not in CTX["ens_stats"] and meta[t]["group"] != "Constants"})
        if missing:
            log("no ensemble data for: " + ", ".join(missing))
        CTX["ens_cards"] = build_ens_cards(ens, entries, log)
        for cid, P in CTX["ens_cards"].items():
            if only and not ({"ens", "ensemble", cid} & only):
                continue
            t = (("ens", cid), None)
            tasks.append(t)
            fnames[t] = f"{GROUP_PREFIX['Ensemble']}_{P['file']}.png"
            meta[t] = dict(group="Ensemble", title=P["name"], level=0, param=cid,
                           unit="P(daily ≥ 1 mm, ≥ 10 mm)" if P.get("precip") else "mean and spread")
    tasks.sort(key=sort_key)

    t1 = time.time()
    items, skipped, failed = [], [], []
    ov = render_overview()
    if ov:
        items.append(dict(group="Overview", file=ov, title="Forecast overview", unit="temperature, rain, cloud, wind"))
    log(f"rendering {len(tasks)} cards with {args.workers} workers")
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(render_card, tasks, chunksize=1))
    else:
        results = [render_card(t) for t in tasks]
    res = {task: (fn, err) for task, fn, err in results}
    for t in tasks:
        fn, err = res[t]
        m = meta[t]
        if fn:
            items.append(dict(m, file=fn))
        elif err and err.startswith("no valid data"):
            skipped.append(f"{m['title']}: no valid data in region")
        else:
            failed.append((m["title"], err))
    if len(loaded) == 2 and not only:
        tc = time.time()
        import compare_cards

        cmp_items = compare_cards.render(loaded, geo, region, init, outdir, log)
        items.extend(cmp_items)
        log(f"comparison: {len(cmp_items)} cards in {time.time() - tc:.1f}s")
    gorder = ["Overview", "Ensemble", "Comparison", "Surface", "Pressure levels", "Waves", "Constants"]
    lead = max(int(e["step_list"].max()) for e in entries.values())
    write_index(outdir, region, init, items, skipped, lead)
    counts = {g: sum(1 for i in items if i["group"] == g) for g in gorder}
    log(f"rendered {len(items)} cards in {time.time() - t1:.1f}s (total {time.time() - t0:.1f}s): {counts}")
    for s in skipped:
        log("skipped: " + s)
    for t, e in failed:
        log(f"FAILED {t}: {e}")
    log(f"cards: {outdir}  (open {outdir / 'index.html'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
