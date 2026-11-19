"""Frame selection and file naming of the weather cards."""

import numpy as np
import pytest

import predict_now as pn

STEPS = np.arange(0, 241, 6)  # 10-day forecast in 6 h steps


def test_short_forecast_keeps_every_step():
    st = np.arange(0, 13, 6)
    arr = np.random.rand(len(st), 4)
    vals, s0, s1, idx = pn.frame_plan(arr, st, st, "inst", None)
    assert idx == [0, 1, 2] and (vals == arr).all()


def test_instantaneous_frames_are_daily_at_same_hour():
    arr = np.random.rand(len(STEPS), 4)
    vals, s0, s1, idx = pn.frame_plan(arr, STEPS, STEPS, "inst", None)
    assert list(s1) == list(range(0, 241, 24))
    assert len(vals) <= pn.MAX_FRAMES
    np.testing.assert_array_equal(vals, arr[idx])


def test_accumulations_are_summed_over_frame_interval():
    st0, st1 = STEPS[:-1], STEPS[1:]
    arr = np.ones((len(st0), 3))  # 1 mm every 6 h
    vals, f0, f1, idx = pn.frame_plan(arr, st0, st1, "acc", None)
    assert idx is None
    assert list(f0) == list(range(0, 240, 24)) and list(f1) == list(range(24, 241, 24))
    np.testing.assert_array_equal(vals, 4.0)  # no rain lost between frames
    assert vals.sum() == arr.sum()


def test_fluxes_are_time_weighted_means():
    st0, st1 = STEPS[:-1], STEPS[1:]
    arr = np.arange(len(st0), dtype=float)[:, None].repeat(2, axis=1)
    vals, f0, f1, _ = pn.frame_plan(arr, st0, st1, "flux", None)
    np.testing.assert_allclose(vals[0], arr[:4].mean(axis=0))


def test_running_totals_are_sampled_not_summed():
    st0, st1 = np.zeros(len(STEPS) - 1, int), STEPS[1:]
    arr = np.cumsum(np.ones((len(st1), 2)), axis=0)
    vals, f0, f1, idx = pn.frame_plan(arr, st0, st1, "acc", "cumul")
    assert all(s % 24 == 0 for s in f1)
    np.testing.assert_array_equal(vals, arr[idx])


@pytest.mark.parametrize("args, expected", [
    (("Surface", "2 m temperature", None, None, "inst", "2t"), "1-surface_2m-temperature.png"),
    (("Pressure levels", "Temperature", 850, None, "inst", "t"), "2-upper-air_temperature_0850hPa.png"),
    (("Surface", "Total precipitation", None, None, "acc", "tp"), "1-surface_precipitation_amounts.png"),
    (("Surface", "Total precipitation", None, "cumul", "acc", "tp"),
     "1-surface_precipitation_accumulated-total.png"),
    (("Constants", "Orography", None, None, "inst", "z"), "4-constant_elevation.png"),
])
def test_card_filename(args, expected):
    assert pn.card_filename(*args) == expected
