import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_tool_module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "evaluate_probabilistic_reentry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_probabilistic_reentry",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation = load_tool_module()


def synthetic_prices(periods: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    trend = np.linspace(100.0, 140.0, periods)
    return pd.DataFrame(
        {
            "SPX": trend,
            "QQQ": trend * 1.01,
            "SEMIS": trend * 1.02,
            "CASH": np.linspace(100.0, 101.0, periods),
            "VIX": np.linspace(25.0, 18.0, periods),
            "VIX3M": np.linspace(26.0, 20.0, periods),
        },
        index=dates,
    )


def test_features_exclude_decision_day_prices() -> None:
    prices = synthetic_prices()
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("QQQ")] *= 10.0

    original_features, _, _ = evaluation.build_features_and_labels(prices)
    changed_features, _, _ = evaluation.build_features_and_labels(changed)

    assert np.allclose(
        original_features.iloc[-1],
        changed_features.iloc[-1],
        equal_nan=True,
    )


def test_walkforward_training_embargoes_unmatured_labels() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    features = pd.DataFrame(
        {
            column: np.linspace(-1.0, 1.0, len(dates))
            for column in evaluation.FEATURE_COLUMNS
        },
        index=dates,
    )
    label = pd.Series(
        np.arange(len(dates)) % 2,
        index=dates,
        name="future_growth_outperforms_cash",
        dtype=float,
    )
    prediction_date = dates[60]

    result = evaluation.walkforward_probabilities(
        features,
        label,
        pd.DatetimeIndex([prediction_date]),
        horizon_days=20,
        minimum_training_observations=20,
    )

    assert result.loc[prediction_date, "training_end_date"] == dates[40]
    assert result.loc[prediction_date, "training_count"] == 41


def test_probe_is_cash_funded_and_charges_round_trip_notional() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    weights = pd.DataFrame(
        {
            "QQQ": [0.0, 0.0],
            "SEMIS": [0.0, 0.0],
            "CASH": [1.0, 1.0],
        },
        index=dates,
    )
    daily = pd.DataFrame({"net_return": [0.0, 0.0]}, index=dates)
    returns = pd.DataFrame(
        {
            "QQQ": [0.01, 0.0],
            "SEMIS": [0.0, 0.0],
            "CASH": [0.0, 0.0],
        },
        index=dates,
    )
    predictions = pd.DataFrame(
        {
            "probability": [0.75, 0.50],
            "base_rate": [0.50, 0.50],
        },
        index=dates,
    )

    result = evaluation.simulate_probe(
        weights,
        daily,
        returns,
        predictions,
        cap=0.20,
        cost_bps=10.0,
    )

    assert np.isclose(result.iloc[0]["added_qqq_weight"], 0.10)
    assert np.isclose(result.iloc[0]["incremental_cost"], 0.0002)
    assert np.isclose(result.iloc[0]["net_return"], 0.0008)
    assert result.iloc[1]["added_qqq_weight"] == 0.0
    assert np.isclose(result.iloc[1]["incremental_cost"], 0.0002)
