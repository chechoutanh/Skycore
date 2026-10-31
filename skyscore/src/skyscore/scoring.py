"""Multi-objective scoring of observing hours.

Every criterion is first mapped to a "goodness" g_i in [0, 1] (1 = best), then

    S = sum_i w_i * g_i,   sum_i w_i = 1,

subject to hard constraints (astronomical night, minimum target altitude).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SUN_LIMIT_DEG = -18.0  # astronomical twilight
MIN_ALT_DEG = 15.0  # below this, airmass is too large
SEEING_BEST, SEEING_WORST = 0.8, 3.5  # arcsec, for normalization


@dataclass(frozen=True)
class Weights:
    altitude: float = 0.35
    moon: float = 0.20
    cloud: float = 0.30
    seeing: float = 0.15

    def normalized(self) -> "Weights":
        s = self.altitude + self.moon + self.cloud + self.seeing
        if s <= 0:
            raise ValueError("At least one weight must be positive.")
        return Weights(self.altitude / s, self.moon / s, self.cloud / s, self.seeing / s)

    def as_array(self) -> np.ndarray:
        w = self.normalized()
        return np.array([w.altitude, w.moon, w.cloud, w.seeing])


def minmax(x, lo: float, hi: float, invert: bool = False) -> np.ndarray:
    """Min-max normalization to [0, 1], clipped; ``invert`` flips the scale."""
    if hi == lo:
        raise ValueError("lo and hi must differ.")
    z = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - z if invert else z


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add goodness columns g_* and the ``observable`` mask.

    Required columns: target_alt, sun_alt, moon_alt, moon_illum, moon_sep,
    cloud_cover, seeing_arcsec.
    """
    out = df.copy()
    # 1/airmass-like term: sin(altitude), 0 below the horizon.
    out["g_altitude"] = np.clip(np.sin(np.deg2rad(out["target_alt"])), 0.0, 1.0)
    # Moon penalty: bright, high and close to the target is worst.
    moon_up = (out["moon_alt"] > 0).astype(float)
    proximity = np.exp(-out["moon_sep"] / 40.0)
    out["g_moon"] = 1.0 - out["moon_illum"] * moon_up * (0.4 + 0.6 * proximity)
    out["g_cloud"] = minmax(out["cloud_cover"], 0, 100, invert=True)
    out["g_seeing"] = minmax(out["seeing_arcsec"], SEEING_BEST, SEEING_WORST, invert=True)
    out["observable"] = (out["sun_alt"] < SUN_LIMIT_DEG) & (out["target_alt"] > MIN_ALT_DEG)
    return out


def score(df: pd.DataFrame, w: Weights | None = None) -> pd.DataFrame:
    """Return df with component columns and score S in [0, 1]."""
    w = (w or Weights()).normalized()
    out = add_components(df)
    s = (
        w.altitude * out["g_altitude"]
        + w.moon * out["g_moon"]
        + w.cloud * out["g_cloud"]
        + w.seeing * out["g_seeing"]
    )
    out["score"] = np.where(out["observable"], s, 0.0)
    return out


def best_window(df_scored: pd.DataFrame, hours: int = 3) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Contiguous ``hours``-long window with the highest mean score."""
    if len(df_scored) < hours:
        raise ValueError("Not enough rows for the requested window.")
    roll = df_scored["score"].rolling(hours).mean()
    end = roll.idxmax()
    start = end - pd.Timedelta(hours=hours - 1)
    return start, end, float(roll.max())


def pareto_front(df_scored: pd.DataFrame, cols=("g_altitude", "g_moon", "g_cloud", "g_seeing")) -> pd.DataFrame:
    """Non-dominated observable hours (all criteria maximized)."""
    d = df_scored[df_scored["observable"]]
    X = d[list(cols)].to_numpy()
    keep = np.ones(len(d), dtype=bool)
    for i in range(len(d)):
        dominated = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dominated.any():
            keep[i] = False
    return d[keep]


def robust_best_hours(df: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.Series:
    """Weight-sensitivity analysis.

    Draw weights from a flat Dirichlet, find the best hour for each draw and
    return how often each timestamp wins. A sharp peak means the recommendation
    does not depend on the exact weights chosen.
    """
    comp = add_components(df)
    G = comp[["g_altitude", "g_moon", "g_cloud", "g_seeing"]].to_numpy()
    obs = comp["observable"].to_numpy()
    if not obs.any():
        return pd.Series(0.0, index=df.index)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(4), size=n)  # (n, 4)
    S = W @ G.T  # (n, T)
    S[:, ~obs] = -1.0
    wins = np.bincount(S.argmax(axis=1), minlength=len(df)) / n
    return pd.Series(wins, index=df.index, name="win_frequency")
