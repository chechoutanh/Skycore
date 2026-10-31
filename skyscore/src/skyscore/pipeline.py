"""End-to-end pipeline: geometry + weather + seeing ML + scoring."""

from __future__ import annotations

import pandas as pd

from .astro import sky_table
from .scoring import Weights, score
from .seeing_model import SeeingModel, make_features, train_default
from .weather import get_forecast


def night_grid(date: pd.Timestamp, hours: int = 24) -> pd.DatetimeIndex:
    """Hourly UTC grid starting at 00:00 UTC of ``date``."""
    return pd.date_range(pd.Timestamp(date).normalize(), periods=hours, freq="h")


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
    times = night_grid(date, hours)
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
