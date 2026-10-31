"""Seeing prediction model.

IMPORTANT (data honesty): free, long-term *measured* seeing archives for a
specific site are rare. This module therefore ships with a physics-inspired
*proxy* label generator (``proxy_seeing``) so the whole ML pipeline can be
developed, tested and demonstrated end-to-end. Models trained on proxy labels
only learn the proxy. To make the model scientifically meaningful, replace
the labels with real measurements, e.g. your club's own observing logs
(estimated seeing in arcsec from double-star splitting / FWHM of stars in
images), then call ``SeeingModel.fit(features, real_labels)``.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .weather import synthetic_weather

FEATURES = [
    "wind_speed_250hPa",
    "wind_speed_10m",
    "temp_change_3h",
    "pressure_change_3h",
    "dew_spread",
    "relative_humidity_2m",
    "cloud_cover",
]


def make_features(w: pd.DataFrame) -> pd.DataFrame:
    """Engineer model features from an hourly weather frame."""
    f = pd.DataFrame(index=w.index)
    f["wind_speed_250hPa"] = w["wind_speed_250hPa"]
    f["wind_speed_10m"] = w["wind_speed_10m"]
    f["temp_change_3h"] = w["temperature_2m"].diff(3).abs().bfill().fillna(0.0)
    f["pressure_change_3h"] = w["pressure_msl"].diff(3).abs().bfill().fillna(0.0)
    f["dew_spread"] = w["temperature_2m"] - w["dew_point_2m"]
    f["relative_humidity_2m"] = w["relative_humidity_2m"]
    f["cloud_cover"] = w["cloud_cover"]
    return f[FEATURES]


def proxy_seeing(f: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Heuristic seeing (arcsec): jet stream + ground wind + thermal change."""
    rng = np.random.default_rng(seed)
    s = (
        0.7
        + 0.012 * np.clip(f["wind_speed_250hPa"] - 40, 0, None)
        + 0.03 * f["wind_speed_10m"]
        + 0.25 * f["temp_change_3h"]
        + 0.15 * f["pressure_change_3h"]
        + 0.004 * np.clip(f["relative_humidity_2m"] - 60, 0, None)
    )
    s = s * rng.lognormal(0, 0.08, len(f))
    return pd.Series(np.clip(s, 0.5, 5.0), index=f.index, name="seeing_arcsec")


def simulated_training_set(hours: int = 24 * 240, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    w = synthetic_weather(lat=10.8, start=pd.Timestamp("2025-01-01"), hours=hours, seed=seed)
    X = make_features(w)
    y = proxy_seeing(X, seed=seed)
    return X, y


class SeeingModel:
    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        )
        self.metrics_: dict[str, float] = {}
        self.label_source = "unfitted"

    def fit(self, X: pd.DataFrame, y: pd.Series, label_source: str = "custom") -> "SeeingModel":
        """Fit with time-ordered CV (no shuffling, avoids look-ahead leakage)."""
        maes, r2s = [], []
        for tr, te in TimeSeriesSplit(n_splits=5).split(X):
            m = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=0
            ).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            maes.append(mean_absolute_error(y.iloc[te], p))
            r2s.append(r2_score(y.iloc[te], p))
        self.metrics_ = {"cv_mae_arcsec": float(np.mean(maes)), "cv_r2": float(np.mean(r2s))}
        self.model.fit(X, y)
        self.label_source = label_source
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[FEATURES])

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SeeingModel":
        return joblib.load(path)


def train_default() -> SeeingModel:
    X, y = simulated_training_set()
    return SeeingModel().fit(X, y, label_source="physics-inspired proxy (simulated)")
