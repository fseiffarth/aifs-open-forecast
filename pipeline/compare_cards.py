"""Comparison cards: AIFS Single with perturbed initial conditions vs AIFS ENS v2.0.

Rendered by predict_now.py when both ensembles exist for a cycle and region:

  6-compare_<variable>.png   one panel per city: 10-90 % band and median of each ensemble, plus
                             the deterministic AIFS Single forecast (the control member)
  6-compare_spread.png       spread (standard deviation over the members, mean over the region)
                             against lead time for both ensembles, and how far their means differ

Both ensembles must be loaded with ens_stats.load_ensemble onto the same points.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

G = 9.80665
COL = {"ens": "#1f5fa8", "single": "#e07b00", "det": "#222222"}
LABEL = {"ens": "AIFS ENS v2.0 (ECMWF initial states + model noise)",
         "single": "AIFS Single + our initial-condition perturbations"}
SHORT = {"ens": "AIFS ENS", "single": "Single + perturbations"}

VARS = [
    dict(id="2m-temperature", name="2 m temperature", unit="°C", get=lambda g: g("2t") - 273.15),
    dict(id="precipitation", name="Precipitation per 24 h", unit="mm", precip=True),
    dict(id="10m-wind-speed", name="10 m wind speed", unit="km/h", get=lambda g: np.hypot(g("10u"), g("10v")) * 3.6),
    dict(id="mean-sea-level-pressure", name="Mean sea-level pressure", unit="hPa", get=lambda g: g("msl") / 100),
    dict(id="temperature_0850hPa", name="Temperature at 850 hPa", unit="°C", get=lambda g: g("t_850") - 273.15),
    dict(id="geopotential-height_0500hPa", name="Geopotential height at 500 hPa", unit="gpm",
         get=lambda g: g("z_500") / G),
]


def _getter(ens, steps):
    pos = [int(np.where(ens["steps"] == s)[0][0]) for s in steps]

    def g(name):
        return ens["vals"][:, pos, ens["index"][name], :].astype(np.float64)  # (members, steps, points)
    return g


def _daily_precip(ens, steps):
    """Per-member 24 h totals (mm) from tp accumulated since the start: (members, days, points), day ends."""
    g = _getter(ens, steps)
    tp = g("tp") * 1000.0
    ends = [s for s in steps if s % 24 == 0]
    acc = {s: tp[:, steps.index(s)] for s in ends}
    acc[0] = np.zeros_like(tp[:, 0])
    days = [acc[e] - acc[e - 24] for e in ends if e - 24 in acc]
    return np.clip(np.stack(days, axis=1), 0, None), [e for e in ends if e - 24 in acc]


def _series(ens, var, steps):
    """(values (members, times, points), steps of the times) for one comparison variable."""
    if var.get("precip"):
        return _daily_precip(ens, steps)
    return var["get"](_getter(ens, steps)), list(steps)


def _control(ens):
    return ens["members"].index(0) if 0 in ens["members"] else None


def render(ens_by_model: dict, geo: dict, region: dict, init: dt.datetime, outdir, log=print) -> list[dict]:
    """Render the comparison cards. Returns index items (group, file, title, unit)."""
    if not {"ens", "single"} <= set(ens_by_model):
        return []
    steps = sorted(set(map(int, ens_by_model["ens"]["steps"])) & set(map(int, ens_by_model["single"]["steps"])))
    if len(steps) < 2:
        log("comparison: the two ensembles have no common forecast steps")
        return []
    tz = ZoneInfo(region["tz"])
    lat = np.asarray(geo["lat"])
    w = np.cos(np.radians(lat))
    n = {m: len(e["members"]) for m, e in ens_by_model.items()}
    sub = (f"Initial time {init:%a %Y-%m-%d %H} UTC   ·   {region['title']}   ·   "
           f"{SHORT['ens']}: {n['ens']} members   ·   {SHORT['single']}: {n['single']} members")
    items = []

    # ---- per-variable city cards -------------------------------------------------------
    data = {}
    for var in VARS:
        data[var["id"]] = {m: _series(e, var, steps) for m, e in ens_by_model.items()}
        fn = f"6-compare_{var['id']}.png"
        try:
            _city_card(var, data[var["id"]], ens_by_model, geo, region, init, tz, sub, outdir / fn)
            items.append(dict(group="Comparison", file=fn, title=var["name"], unit="AIFS ENS vs Single + perturbations"))
        except Exception as exc:  # one bad card must not stop the rest
            log(f"comparison card {fn} failed: {type(exc).__name__}: {exc}")

    # ---- spread against lead time ------------------------------------------------------
    fn = "6-compare_spread.png"
    try:
        _spread_card(data, w, sub, outdir / fn)
        items.append(dict(group="Comparison", file=fn, title="Ensemble spread against lead time",
                          unit="region mean of the standard deviation"))
    except Exception as exc:
        log(f"comparison card {fn} failed: {type(exc).__name__}: {exc}")
    return items


def _city_card(var, series, ens_by_model, geo, region, init, tz, sub, path):
    cities = geo["cities"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.12, wspace=0.08, hspace=0.25)
    fig.text(0.02, 0.955, f"{var['name']}: AIFS ENS vs AIFS Single + perturbations", fontsize=21, weight="bold",
             color="#14213d", va="center")
    fig.text(0.02, 0.905, sub, fontsize=11.5, color="#444", va="center")
    precip = var.get("precip")
    for ax, (cname, c) in zip(axes.ravel(), cities.items()):
        i = c["idx"]
        for k, m in enumerate(("single", "ens")):
            vals, st = series[m]
            t = [(init + dt.timedelta(hours=s)).astimezone(tz) if init.tzinfo else
                 (init.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=s)).astimezone(tz) for s in st]
            x = vals[:, :, i]
            p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=0)
            if precip:
                off = dt.timedelta(hours=-12 + (-3 if m == "single" else 3))
                tt = [ti + off for ti in t]
                ax.bar(tt, p50, width=0.24, color=COL[m], alpha=0.85, label=f"{SHORT[m]}: median, whisker 10-90 %")
                ax.errorbar(tt, p50, yerr=[p50 - p10, p90 - p50], fmt="none", ecolor=COL[m], elinewidth=1.2,
                            capsize=2, alpha=0.9)
            else:
                ax.fill_between(t, p10, p90, color=COL[m], alpha=0.22, lw=0, label=f"{SHORT[m]}: 10-90 %")
                ax.plot(t, p50, color=COL[m], lw=1.8, label=f"{SHORT[m]}: median")
        ci = _control(ens_by_model["single"])
        if ci is not None and not precip:
            vals, st = series["single"]
            t = [(init.replace(tzinfo=init.tzinfo or dt.timezone.utc) + dt.timedelta(hours=s)).astimezone(tz) for s in st]
            ax.plot(t, vals[ci, :, i], color=COL["det"], lw=1.3, ls="--", label="AIFS Single (deterministic)")
        ax.set_title(cname, fontsize=14, weight="bold", loc="left", color="#14213d")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=tz))
    for ax in axes[:, 0]:
        ax.set_ylabel(var["unit"], fontsize=11)
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(h), fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.985, 0.012, f"Nearest model grid point to each city · local time ({region['tz']}) · "
             f"{LABEL['ens']} vs {LABEL['single']}", fontsize=8.5, ha="right", color="#888")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _spread_card(data, w, sub, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.12, wspace=0.22, hspace=0.32)
    fig.text(0.02, 0.955, "Ensemble spread: AIFS ENS vs AIFS Single + perturbations", fontsize=21, weight="bold",
             color="#14213d", va="center")
    fig.text(0.02, 0.905, sub, fontsize=11.5, color="#444", va="center")
    for ax, var in zip(axes.ravel(), VARS):
        series = data[var["id"]]
        for m in ("single", "ens"):
            vals, st = series[m]
            spread = np.average(vals.std(axis=0, ddof=1), axis=1, weights=w)
            ax.plot(np.array(st) / 24, spread, "-o", ms=3, color=COL[m], lw=2, label=f"spread, {SHORT[m]}")
        (a, st), (b, _) = series["ens"], series["single"]
        rms = np.sqrt(np.average((a.mean(0) - b.mean(0)) ** 2, axis=1, weights=w))
        ax.plot(np.array(st) / 24, rms, ":", color="#555", lw=1.8, label="RMS difference of the two ensemble means")
        ax.set_title(f"{var['name']} ({var['unit']})", fontsize=12.5, weight="bold", loc="left", color="#14213d")
        ax.set_xlabel("lead time (days)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=9)
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.985, 0.008, "Spread: standard deviation over the members at each grid point, averaged over the region "
             "(cos-latitude weights). Precipitation: 24 h totals.", fontsize=8.5, ha="right", color="#888")
    fig.savefig(path, dpi=100)
    plt.close(fig)
