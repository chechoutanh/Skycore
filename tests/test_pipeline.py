import pandas as pd
import pytest

pytest.importorskip("astropy")  # needs `pip install -r requirements.txt`

from skyscore.pipeline import build_table  # noqa: E402
from skyscore.seeing_model import SeeingModel, simulated_training_set


def test_pipeline_offline_runs():
    X, y = simulated_training_set(hours=24 * 30)
    model = SeeingModel(n_estimators=30).fit(X, y, label_source="test")
    df, meta = build_table(10.78, 106.70, "M42 (Orion Nebula)", pd.Timestamp("2026-01-15"), offline=True, model=model)
    assert len(df) == 24
    assert df["score"].between(0, 1).all()
    assert "Synthetic" in meta["weather_source"]
    assert df["observable"].any()


def test_night_grid_covers_local_night_anywhere():
    from skyscore.pipeline import night_grid

    for lon, local_midnight_utc in [(106.7, "2026-01-15 17:00"), (-75.0, "2026-01-16 05:00")]:
        assert pd.Timestamp(local_midnight_utc) in night_grid(pd.Timestamp("2026-01-15"), lon)


def test_sky_table_far_future_works_offline():
    from skyscore.astro import sky_table

    t = sky_table(10.78, 106.70, "M31 (Andromeda Galaxy)", pd.date_range("2028-06-01", periods=2, freq="h"))
    assert t.notna().all().all()


def test_sun_is_a_daytime_target():
    X, y = simulated_training_set(hours=24 * 30)
    model = SeeingModel(n_estimators=20).fit(X, y)
    df, _ = build_table(10.78, 106.70, "Sun", pd.Timestamp("2026-01-15"), offline=True, model=model)
    obs = df[df["observable"]]
    assert len(obs) > 0 and (obs["sun_alt"] > 15).all()
    assert (df["g_moon"] == 1.0).all()
    # The grid covers the chosen day (local midnight to midnight), not the next morning.
    assert df.index[0] == pd.Timestamp("2026-01-14 17:00")


def test_every_target_has_geometry():
    from skyscore.astro import TARGETS, sky_table

    times = pd.date_range("2026-01-15 12:00", periods=2, freq="h")
    for name in TARGETS:
        t = sky_table(10.78, 106.70, name, times)
        assert t["target_alt"].between(-90, 90).all(), name
