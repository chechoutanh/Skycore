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
