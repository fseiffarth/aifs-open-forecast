"""Comparison cards (AIFS ENS vs AIFS Single + perturbations) on synthetic ensembles."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import compare_cards  # noqa: E402

NAMES = ["10u", "10v", "2t", "msl", "t_850", "tp", "z_500"]
STEPS = np.arange(6, 73, 6)


def fake_ensemble(n_members, spread, rng):
    npts = 30
    vals = np.empty((n_members, len(STEPS), len(NAMES), npts), np.float32)
    base = {"10u": 3.0, "10v": -2.0, "2t": 280.0, "msl": 101300.0, "t_850": 270.0, "z_500": 55000.0}
    for j, n in enumerate(NAMES):
        if n == "tp":  # accumulated from the start, in m: 1 mm per 6 h for every member
            vals[:, :, j] = (STEPS / 6 * 0.001)[None, :, None]
        else:
            vals[:, :, j] = base[n] + spread * rng.standard_normal((n_members, len(STEPS), npts))
    return dict(vals=vals, steps=STEPS, names=NAMES, index={n: i for i, n in enumerate(NAMES)},
                members=list(range(n_members)))


def test_daily_precip_differences_accumulation():
    ens = fake_ensemble(3, 1.0, np.random.default_rng(0))
    daily, ends = compare_cards._daily_precip(ens, list(map(int, STEPS)))
    assert ends == [24, 48, 72]
    np.testing.assert_allclose(daily, 4.0, atol=1e-4)  # 4 intervals of 1 mm per day


def test_render_writes_cards(tmp_path):
    rng = np.random.default_rng(1)
    ens = {"ens": fake_ensemble(6, 2.0, rng), "single": fake_ensemble(5, 1.0, rng)}
    lat = np.linspace(47, 55, 30)
    geo = dict(lat=lat, cities={f"City {i}": dict(idx=i * 5) for i in range(6)})
    region = dict(title="Test", tz="Europe/Berlin")
    items = compare_cards.render(ens, geo, region, dt.datetime(2026, 11, 19, 12), tmp_path, log=print)
    files = sorted(i["file"] for i in items)
    assert len(files) == len(compare_cards.VARS) + 1
    assert "6-compare_spread.png" in files
    for f in files:
        assert (tmp_path / f).stat().st_size > 10_000


def test_render_needs_both_models(tmp_path):
    ens = {"ens": fake_ensemble(4, 1.0, np.random.default_rng(2))}
    assert compare_cards.render(ens, dict(lat=np.zeros(30), cities={}), dict(title="T", tz="UTC"),
                                dt.datetime(2026, 1, 1), tmp_path) == []
