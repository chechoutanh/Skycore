"""Evaluate SkyScore against real observing logs.

Inputs are two CSV files (see ``data/README.md``):
  events.csv        one row per observing event (site, date, target, UTC offset)
  observations.csv  one row per observed hour (measured seeing, cloud, rating)

For every logged hour we rebuild what SkyScore would have predicted *before*
the event (archived Open-Meteo forecast + seeing model + score), then compare
it with what was measured. Baselines are "climatology" (always predict the
average of the measurements), because measurements of the same night are not
available when the plan is made.

Usage:
    python -m skyscore.evaluation data/events.csv data/observations.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .astro import TARGETS, sky_table
from .scoring import score
from .seeing_model import SeeingModel, make_features, train_default
from .weather import get_forecast

# True when a higher number means better seeing.
SEEING_METHODS = {"fwhm_arcsec": False, "pickering_1_10": True, "scale_1_5": True}


def session_midpoint(start: str, end: str) -> tuple[str, float]:
    """('19:00', '21:00') -> ('20:00', 2.0): the logs store a session as its midpoint plus its length.

    An end at or before the start means the session ran past midnight.
    """
    a, b = (pd.Timedelta(f"{s.strip()}:00") for s in (start, end))
    if b <= a:
        b += pd.Timedelta(days=1)
    mid = (a + (b - a) / 2) % pd.Timedelta(days=1)
    minutes = int(mid.total_seconds() // 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}", (b - a).total_seconds() / 3600


def load_logs(events_path: str, observations_path: str) -> pd.DataFrame:
    """Join the two logs and convert local observing times to naive UTC."""
    ev = pd.read_csv(events_path)
    ob = pd.read_csv(observations_path)
    unknown = set(ob["seeing_method"].dropna()) - set(SEEING_METHODS)
    if unknown:
        raise ValueError(f"Unknown seeing_method {sorted(unknown)}; use one of {sorted(SEEING_METHODS)}")
    bad_targets = set(ev["target"]) - set(TARGETS)
    if bad_targets:
        raise ValueError(f"Unknown target {sorted(bad_targets)}; use one of {TARGETS}")
    df = ob.merge(ev, on="event_id", how="left", validate="many_to_one")
    if df["lat"].isna().any():
        missing = sorted(df.loc[df["lat"].isna(), "event_id"].unique())
        raise ValueError(f"observations.csv has event_id not in events.csv: {missing}")

    local = pd.to_datetime(df["date"]) + pd.to_timedelta(df["time"].astype(str) + ":00")
    # Hours after midnight belong to the night that started on `date`; Sun sessions are daytime, same day.
    local += pd.to_timedelta(((local.dt.hour < 12) & (df["target"] != "Sun")).astype(int), unit="D")
    df["mid_utc"] = local - pd.to_timedelta(df["utc_offset"], unit="h")
    df["hour_utc"] = df["mid_utc"].dt.round("h")
    if "duration_h" not in df:
        df["duration_h"] = np.nan
    return df


def add_predictions(df: pd.DataFrame, model: SeeingModel, offline: bool = False) -> pd.DataFrame:
    """Attach forecast cloud, predicted seeing and score S to every logged row.

    A row with ``duration_h`` summarizes a session centred on ``time``: it is compared with the
    forecast averaged over the hours inside that session. Without it, the nearest hour is used.
    """
    parts = []
    for _, g in df.groupby("event_id", sort=False):
        e = g.iloc[0]
        half = pd.to_timedelta(g["duration_h"].fillna(0) / 2, unit="h")
        start = (g["mid_utc"] - half).dt.ceil("h")
        end = (g["mid_utc"] + half).dt.floor("h")
        # Always include the nearest hour, so short or unset durations fall back to it.
        start = start.where(start <= g["hour_utc"], g["hour_utc"])
        end = end.where(end >= g["hour_utc"], g["hour_utc"])
        # Continuous grid with 3 h lead-in, because features use 3-hour differences.
        grid = pd.date_range(start.min() - pd.Timedelta(hours=3), end.max(), freq="h")
        weather, source = get_forecast(e["lat"], e["lon"], grid, offline=offline)
        t = sky_table(e["lat"], e["lon"], e["target"], grid).join(weather)
        t["seeing_pred"] = model.predict(make_features(weather))
        t = score(t.rename(columns={"seeing_pred": "seeing_arcsec"}))
        t = t[["cloud_cover", "seeing_arcsec", "score", "observable"]].rename(
            columns={"cloud_cover": "cloud_forecast", "seeing_arcsec": "seeing_pred"}
        )
        agg = {"cloud_forecast": "mean", "seeing_pred": "mean", "score": "mean", "observable": "any"}
        pred = pd.DataFrame([t.loc[s:e_].agg(agg) for s, e_ in zip(start, end)], index=g.index)
        parts.append(g.join(pred).assign(weather_source=source))
    return pd.concat(parts)


def evaluate(df: pd.DataFrame) -> dict:
    """Metrics on a frame returned by ``add_predictions``."""
    m: dict = {
        "events": int(df["event_id"].nunique()),
        "hours": int(len(df)),
        "weather_sources": sorted(df["weather_source"].unique()),
    }

    # Seeing: rank agreement for every scale; error in arcsec when FWHM was measured.
    for method, higher_better in SEEING_METHODS.items():
        d = df[df["seeing_method"] == method].dropna(subset=["seeing_value"])
        if len(d) < 3:
            continue
        badness = -d["seeing_value"] if higher_better else d["seeing_value"]
        m[f"seeing_{method}_spearman"] = float(d["seeing_pred"].corr(badness, method="spearman"))
        if method == "fwhm_arcsec":
            m["seeing_mae_model"] = float((d["seeing_pred"] - d["seeing_value"]).abs().mean())
            m["seeing_mae_climatology"] = float((d["seeing_value"] - d["seeing_value"].mean()).abs().mean())

    c = df.dropna(subset=["cloud_observed"])
    if len(c):
        m["cloud_mae_forecast"] = float((c["cloud_forecast"] - c["cloud_observed"]).abs().mean())
        m["cloud_mae_climatology"] = float((c["cloud_observed"] - c["cloud_observed"].mean()).abs().mean())

    # Recommendation: rating at SkyScore's top hour vs a random logged hour vs the best logged hour.
    r = df.dropna(subset=["quality_rating"])
    if len(r):
        per = r.groupby("event_id").apply(
            lambda g: pd.Series(
                {
                    "picked": g.loc[g["score"].idxmax(), "quality_rating"],
                    "random": g["quality_rating"].mean(),
                    "best": g["quality_rating"].max(),
                }
            ),
            include_groups=False,
        )
        m["rating_at_skyscore_hour"] = float(per["picked"].mean())
        m["rating_at_random_hour"] = float(per["random"].mean())
        m["rating_at_best_hour"] = float(per["best"].mean())
        m["top_hour_hit_rate"] = float((per["picked"] == per["best"]).mean())
    return m


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events")
    p.add_argument("observations")
    p.add_argument("--offline", action="store_true", help="synthetic weather (pipeline check only, not a real result)")
    p.add_argument("--out", default="data/eval_hours.csv", help="per-hour predictions vs measurements")
    a = p.parse_args(argv)

    df = add_predictions(load_logs(a.events, a.observations), train_default(), offline=a.offline)
    df.to_csv(a.out, index=False)
    for k, v in evaluate(df).items():
        print(f"{k:28s} {np.round(v, 3) if isinstance(v, float) else v}")
    print(f"\nPer-hour table written to {a.out}")


if __name__ == "__main__":
    main()
