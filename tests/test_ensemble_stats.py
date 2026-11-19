"""Ensemble statistics helpers and the initial-condition perturbation code (no GPU, no forecast data)."""

import numpy as np
import pytest

import ens_stats as es
import ensemble as ens_mod
import predict_now as pn


def test_anemoi_name_mapping():
    assert es.anemoi_name("t", "pl", 850) == "t_850"
    assert es.anemoi_name("z", "pl", 500.0) == "z_500"
    assert es.anemoi_name("2t", "sfc", None) == "2t"
    assert es.anemoi_name("fscov", "sfc", None) == "snowc"
    assert es.anemoi_name("stl1", "sol", None) == "stl1"


def test_percentiles_and_probability():
    a = np.arange(101, dtype=float)[:, None]  # 101 members, one point
    p10, p90 = es.percentiles(a, (10, 90))
    assert p10[0] == pytest.approx(10) and p90[0] == pytest.approx(90)
    assert es.exceedance_prob(a, 90)[0] == pytest.approx(11 / 101)  # members 90..100
    nan = a.copy()
    nan[3] = np.nan
    assert np.isnan(es.exceedance_prob(nan, 10)[0]) and np.isnan(es.percentiles(nan)[0][0])


def test_match_points_by_coordinates_not_order():
    lat, lon = np.array([50.0, 51.0, 52.0]), np.array([8.0, 9.0, 10.0])
    perm = [2, 0, 1]
    idx = es.match_points(lat[perm], lon[perm], lat, lon)
    np.testing.assert_array_equal(np.array(perm)[idx], np.arange(3))  # target i -> its own point
    assert es.match_points(lat + 0.5, lon, lat, lon) is None


def test_align_steps_adds_zero_for_accumulations_only():
    blk = np.ones((2, 2, 1), np.float32)  # steps 6, 12
    out = es.align_steps(blk, [6, 12], [0, 6, 12], zero_start=True)
    assert out[:, 0].tolist() == [[0.0], [0.0]]
    out = es.align_steps(blk, [6, 12], [0, 6, 12], zero_start=False)
    assert np.isnan(out[:, 0]).all() and (out[:, 1:] == 1).all()


def _members_acc(nm=5, nsteps=8):
    rng = np.random.default_rng(0)
    inc = rng.random((nm, nsteps, 2)) * 3.0  # mm per 6 h
    return inc, np.cumsum(inc, axis=1) / 1000.0  # accumulations in m from the start (steps 6, 12, ...)


def test_per_period_amounts_are_per_member_differences():
    inc, acc = _members_acc()
    steps = 6 * (np.arange(acc.shape[1]) + 1)
    ens = dict(vals=acc[:, :, None, :].astype(np.float32), steps=steps, index={"tp": 0})
    entry = dict(param="tp", levtype="sfc", level=None, step_list=np.r_[0, steps])
    entries = {("tp", "sfc", None): entry}
    spec = pn.SPECS["tp"]
    per, s0, s1 = pn.ens_series(ens, entries, ("tp", "sfc", None), spec, None)
    assert per.shape == (5, 8, 2) and list(s1) == list(steps)
    np.testing.assert_allclose(per, inc, rtol=1e-4, atol=1e-4)  # first interval is not lost (step 0 = 0)
    cum, _, _ = pn.ens_series(ens, entries, ("tp", "sfc", None), spec, "cumul")
    np.testing.assert_allclose(cum[:, -1], inc.sum(axis=1), rtol=1e-4)
    # percentile of per-period amounts differs from the difference of percentile accumulations
    p90_amount = es.percentiles(per, (90,))[0]
    p90_acc = es.percentiles(np.cumsum(inc, axis=1), (90,))[0]
    assert not np.allclose(p90_amount[1:], np.diff(p90_acc, axis=0), atol=1e-3)


def test_period_sums_complete_windows_only():
    per = np.ones((3, 10, 2))  # 10 intervals of 6 h = 60 h
    st1 = 6 * (np.arange(10) + 1)
    sums, a, b = es.period_sums(per, st1 - 6, st1, 24)
    assert sums.shape == (3, 2, 2) and list(a) == [0, 24] and list(b) == [24, 48]
    assert (sums == 4).all()
    none = es.period_sums(per[:, :3], st1[:3] - 6, st1[:3], 24)[0]
    assert none.shape[1] == 0


def test_derived_wind_speed_per_member_and_missing_step_zero():
    u = np.full((4, 2, 1, 1), 3.0, np.float32)
    v = np.full((4, 2, 1, 1), 4.0, np.float32)
    ens = dict(vals=np.concatenate([u, v], axis=2), steps=[6, 12], index={"10u": 0, "10v": 1})
    steps = np.array([0, 6, 12])
    ent = {("10u", "sfc", None): dict(param="10u", levtype="sfc", level=None, step_list=steps),
           ("10v", "sfc", None): dict(param="10v", levtype="sfc", level=None, step_list=steps)}
    si = dict(param="10si", levtype="sfc", level=None, step_list=steps, uv=(("10u", "sfc", None), ("10v", "sfc", None)))
    b = pn.ens_block(ens, ent, si)
    assert np.isnan(b[:, 0]).all() and np.allclose(b[:, 1:], 5.0)
    assert pn.ens_block(ens, ent, dict(param="zz", levtype="sfc", level=None, step_list=steps)) is None


def test_load_ensemble_skips_unfinished_and_matches_order(tmp_path):
    lat, lon = np.array([50.0, 51.0, 52.0]), np.array([8.0, 9.0, 10.0])
    names = np.array(["2t", "msl"])
    for m in range(4):
        vals = np.full((2, 2, 3), float(m), np.float32)
        vals[:, 0, :] += lat[::-1]  # value = latitude of the stored point (stored order is reversed)
        np.savez(tmp_path / f"member_{m:03d}.npz", values=vals, steps=np.array([6, 12]), names=names,
                 lat=lat[::-1], lon=lon[::-1], member=m)
    (tmp_path / "member_004.tmp.npz").write_bytes(b"half written")
    e = es.load_ensemble(tmp_path, lat, lon)
    assert e["members"] == [0, 1, 2, 3] and e["vals"].shape == (4, 2, 2, 3)
    np.testing.assert_allclose(e["vals"][1, 0, 0], 1.0 + lat)  # reordered back to the target order
    assert es.load_ensemble(tmp_path, lat + 5, lon) is None
    (tmp_path / "member_003.npz").unlink()
    (tmp_path / "member_002.npz").unlink()
    assert es.load_ensemble(tmp_path, lat, lon) is None  # fewer than 3 members


# ------------------------------------------------------------------ perturbations
@pytest.fixture(scope="module")
def noise():
    lat = np.linspace(-85, 85, 2000)
    lon = np.linspace(-180, 180, 2000)
    return ens_mod.NoiseMaker(lat, lon)


def test_noise_has_unit_variance(noise):
    f = noise.field(np.random.default_rng(1))
    assert f.std() == pytest.approx(1.0, abs=1e-3) and abs(f.mean()) < 0.3


def test_perturbation_is_symmetric_and_q_stays_nonnegative(noise):
    names = {"2t", "msl", "z_500", "t_850", "q_850", "tcw"}
    pert = ens_mod.make_perturbation(noise, np.random.default_rng(2), names)
    assert set(pert) == {"2t", "msl", "z_500", "t_850", "rel:q_850", "rel:tcw"}
    n = noise.coords.shape[1]
    fields = {k: np.full((1, n), 280.0 if k not in ("q_850", "tcw") else 1e-3) for k in names}
    fields["z_500"] = fields["z_500"] * 180
    plus = ens_mod.apply_perturbation(fields, pert, +1.0)
    minus = ens_mod.apply_perturbation(fields, pert, -1.0)
    for k in ("2t", "msl", "z_500", "t_850"):
        np.testing.assert_allclose(plus[k] + minus[k], 2 * fields[k], rtol=1e-6)
        assert not np.allclose(plus[k], fields[k])
    assert min(plus["q_850"].min(), minus["q_850"].min(), plus["tcw"].min(), minus["tcw"].min()) >= 0
    big = {k: -3.0 * v for k, v in pert.items()}  # even a huge negative relative change cannot make q negative
    assert ens_mod.apply_perturbation(fields, big, 1.0)["q_850"].min() >= 0
