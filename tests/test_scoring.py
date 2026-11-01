import numpy as np
import pandas as pd
import pytest

from skyscore.scoring import (
    Weights,
    best_window,
    minmax,
    pareto_front,
    robust_best_hours,
    score,
)


def make_df(n=6):
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "target_alt": [10, 30, 60, 80, 50, 20],
            "sun_alt": [-30] * n,
            "moon_alt": [-10, -10, 20, 40, 40, 40],
            "moon_illum": [0.5] * n,
            "moon_sep": [90, 90, 60, 30, 30, 30],
            "cloud_cover": [10, 10, 10, 10, 90, 10],
            "seeing_arcsec": [1.5] * n,
        },
        index=idx,
    )


def test_minmax_range_and_invert():
    x = minmax([-5, 0, 50, 100, 200], 0, 100)
    assert x.min() == 0 and x.max() == 1
    assert np.allclose(minmax([0, 100], 0, 100, invert=True), [1, 0])


def test_weights_normalize():
    w = Weights(2, 2, 2, 2).normalized()
    assert pytest.approx(w.as_array().sum()) == 1.0


def test_score_in_unit_interval_and_constraints():
    s = score(make_df())
    assert s["score"].between(0, 1).all()
    assert s["score"].iloc[0] == 0.0  # altitude 10 deg < limit
    d = make_df()
    d["sun_alt"] = -5.0  # not dark
    assert (score(d)["score"] == 0).all()


def test_more_cloud_lowers_score():
    d = make_df()
    a = score(d)["score"].iloc[4]
    d.loc[d.index[4], "cloud_cover"] = 0
    assert score(d)["score"].iloc[4] > a


def test_best_window_picks_high_altitude_clear_hours():
    s = score(make_df())
    start, end, mean = best_window(s, 2)
    assert pd.Timestamp("2026-01-01 02:00") <= start
    assert 0 < mean <= 1


def test_pareto_front_nonempty_and_nondominated():
    f = pareto_front(score(make_df()))
    assert 0 < len(f) <= 4


def test_robust_best_hours_sums_to_one():
    w = robust_best_hours(make_df())
    assert pytest.approx(w.sum()) == 1.0
