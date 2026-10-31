"""Weather data: Open-Meteo forecast with a synthetic offline fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "cloud_cover",
    "relative_humidity_2m",
    "temperature_2m",
    "dew_point_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_speed_250hPa",  # jet-stream-level wind: main driver of seeing
]
COLUMNS = HOURLY_VARS


def fetch_open_meteo(lat: float, lon: float, days: int = 3, timeout: float = 10.0) -> pd.DataFrame:
    """Hourly forecast (UTC) from the free Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "GMT",
        "wind_speed_unit": "kmh",
    }
    r = requests.get(URL, params=params, timeout=timeout)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[COLUMNS].astype(float)


def synthetic_weather(
    lat: float,
    start: pd.Timestamp,
    hours: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Plausible synthetic hourly weather (AR(1) noise + diurnal cycle).

    Used (a) as an offline fallback for the demo and (b) to generate the
    training set for the seeing model. It is NOT real data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq="h")
    hod = (idx.hour.values + lat / 15.0) % 24  # rough local solar hour

    def ar1(phi: float, sigma: float) -> np.ndarray:
        x = np.zeros(hours)
        for i in range(1, hours):
            x[i] = phi * x[i - 1] + rng.normal(0, sigma)
        return x

    temp = 27 + 4 * np.sin((hod - 9) / 24 * 2 * np.pi) + ar1(0.97, 0.4)
    humidity = np.clip(75 - 12 * np.sin((hod - 9) / 24 * 2 * np.pi) + 8 * ar1(0.95, 0.5), 20, 100)
    dew = temp - (100 - humidity) / 5.0
    pressure = 1010 + 3 * ar1(0.995, 0.3)
    cloud = np.clip(50 + 30 * ar1(0.96, 0.9) + 8 * rng.normal(size=hours), 0, 100)
    wind10 = np.clip(8 + 4 * ar1(0.93, 0.8) + rng.gamma(2, 1, hours), 0, None)
    wind250 = np.clip(60 + 35 * ar1(0.98, 0.8) + rng.gamma(2, 5, hours), 5, None)

    return pd.DataFrame(
        {
            "cloud_cover": cloud,
            "relative_humidity_2m": humidity,
            "temperature_2m": temp,
            "dew_point_2m": dew,
            "pressure_msl": pressure,
            "wind_speed_10m": wind10,
            "wind_speed_250hPa": wind250,
        },
        index=idx,
    )


def get_forecast(
    lat: float,
    lon: float,
    times: pd.DatetimeIndex,
    offline: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Weather aligned to ``times``. Returns (frame, source_label)."""
    if not offline:
        try:
            raw = fetch_open_meteo(lat, lon, days=4)
            out = raw.reindex(times).interpolate(limit_direction="both")
            if out.notna().all().all():
                return out, "Open-Meteo (live forecast)"
        except Exception:  # network down, API change, etc.
            pass
    syn = synthetic_weather(lat, times[0] - pd.Timedelta(hours=6), len(times) + 12, seed=42)
    return syn.reindex(times), "Synthetic (offline demo data)"
