import pandas as pd

from skyscore.seeing_model import FEATURES, SeeingModel, make_features, simulated_training_set
from skyscore.weather import synthetic_weather


def test_features_shape_and_no_nan():
    w = synthetic_weather(10.8, pd.Timestamp("2026-01-01"), 72, seed=1)
    f = make_features(w)
    assert list(f.columns) == FEATURES
    assert not f.isna().any().any()


def test_model_learns_proxy_better_than_baseline():
    X, y = simulated_training_set(hours=24 * 60)
    m = SeeingModel(n_estimators=50).fit(X, y)
    assert m.metrics_["cv_r2"] > 0.5
    assert m.metrics_["cv_mae_arcsec"] < y.std()
    assert m.predict(X.head(5)).shape == (5,)
