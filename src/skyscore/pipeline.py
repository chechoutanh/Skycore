"""End-to-end pipeline: geometry + weather + seeing ML + scoring."""

from __future__ import annotations

import pandas as pd

from .astro import sky_table, sun_limit
from .scoring import Weights, score
from .seeing_model import SeeingModel, make_features, train_default
from .weather import get_forecast


def night_grid(date: pd.Timestamp, lon: float, hours: int = 24, start_hour: int = 12) -> pd.DatetimeIndex:
    """Hourly UTC grid from local solar ``start_hour`` of ``date``.

    The default (noon) covers that evening's night anywhere on Earth; 0 covers the daytime.
    """
    start_utc = pd.Timestamp(date).normalize() + pd.Timedelta(hours=round(start_hour - lon / 15))
    return pd.date_range(start_utc, periods=hours, freq="h")


def build_table(
    lat: float,
    lon: float,
    target: str,
    date: pd.Timestamp,
    weights: Weights | None = None,
    offline: bool = False,
    model: SeeingModel | None = None,
    hours: int = 24,
) -> tuple[pd.DataFrame, dict]:
    daytime = sun_limit(target) > 0  # the Sun: plan the day of `date`, not its night
    times = night_grid(date, lon, hours, start_hour=0 if daytime else 12)
    sky = sky_table(lat, lon, target, times)
    weather, source = get_forecast(lat, lon, times, offline=offline)
    model = model or train_default()
    feats = make_features(weather)
    df = sky.join(weather)
    df["seeing_arcsec"] = model.predict(feats)
    df = score(df, weights)
    meta = {
        "weather_source": source,
        "seeing_label_source": model.label_source,
        "seeing_cv": model.metrics_,
    }
    return df, meta
