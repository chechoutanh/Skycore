import pandas as pd
import pytest

pytest.importorskip("astropy")

from skyscore.evaluation import add_predictions, evaluate, load_logs  # noqa: E402
from skyscore.seeing_model import SeeingModel, simulated_training_set  # noqa: E402


@pytest.fixture
def logs(tmp_path):
    ev = tmp_path / "events.csv"
    ob = tmp_path / "observations.csv"
    ev.write_text(
        "event_id,date,club,site,lat,lon,utc_offset,target\n"
        "E01,2026-01-15,A,HCM,10.78,106.70,7,M42 (Orion Nebula)\n"
        "E02,2026-01-20,B,HCM,10.78,106.70,7,M42 (Orion Nebula)\n"
    )
    ob.write_text(
        "event_id,time,seeing_value,seeing_method,cloud_observed,quality_rating\n"
        "E01,20:00,2.1,fwhm_arcsec,10,4\n"
        "E01,22:00,2.8,fwhm_arcsec,40,3\n"
        "E01,01:00,1.9,fwhm_arcsec,5,5\n"
        "E02,21:00,3,scale_1_5,20,3\n"
        "E02,23:00,4,scale_1_5,0,4\n"
    )
    return str(ev), str(ob)


def test_local_time_to_utc_crosses_midnight(logs):
    df = load_logs(*logs)
    # 01:00 local (UTC+7) on the night of Jan 15 is Jan 15 18:00 UTC.
    assert df["hour_utc"].iloc[2] == pd.Timestamp("2026-01-15 18:00")
    assert df["hour_utc"].iloc[0] == pd.Timestamp("2026-01-15 13:00")


def test_evaluate_offline(logs):
    X, y = simulated_training_set(hours=24 * 30)
    model = SeeingModel(n_estimators=20).fit(X, y)
    df = add_predictions(load_logs(*logs), model, offline=True)
    m = evaluate(df)
    assert m["events"] == 2 and m["hours"] == 5
    assert df["seeing_pred"].notna().all() and df["score"].notna().all()
    assert {"seeing_mae_model", "cloud_mae_forecast", "rating_at_skyscore_hour"} <= m.keys()
    assert m["rating_at_random_hour"] <= m["rating_at_best_hour"]


def test_unknown_method_rejected(tmp_path, logs):
    ev, ob = logs
    pd.read_csv(ob).assign(seeing_method="good").to_csv(ob, index=False)
    with pytest.raises(ValueError, match="seeing_method"):
        load_logs(ev, ob)
