import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_tool_module(name: str):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probability = load_tool_module("evaluate_probabilistic_reentry")
calibration = load_tool_module("evaluate_state_calibrated_reentry")


def test_state_calibration_uses_only_mature_low_exposure_rows() -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    raw = pd.DataFrame(
        {
            "probability": np.linspace(0.20, 0.80, len(dates)),
            "base_rate": 0.50,
        },
        index=dates,
    )
    label = pd.Series(
        np.arange(len(dates)) % 2,
        index=dates,
        name="future_growth_outperforms_cash",
        dtype=float,
    )
    weights = pd.DataFrame(
        {
            "QQQ": np.where(np.arange(len(dates)) < 80, 0.0, 0.30),
            "SEMIS": 0.0,
        },
        index=dates,
    )

    result = calibration.state_calibrated_probabilities(
        raw.iloc[[100]],
        label,
        weights,
        horizon_days=20,
        minimum_observations=20,
    )

    assert result.iloc[0]["training_count"] == 0

    result = calibration.state_calibrated_probabilities(
        raw,
        label,
        weights,
        horizon_days=20,
        minimum_observations=20,
    )

    row = result.loc[dates[100]]
    month = (dates[100].year, dates[100].month)
    month_start_position = next(
        position
        for position, date in enumerate(dates)
        if (date.year, date.month) == month
    )
    expected_count = min(month_start_position - 20 + 1, 80)
    assert row["training_count"] == expected_count
    assert row["training_end_date"] == dates[expected_count - 1]


def test_state_calibration_can_learn_an_inverse_probability_relation() -> None:
    dates = pd.bdate_range("2020-01-01", periods=140)
    raw_probability = np.tile(np.linspace(0.10, 0.90, 20), 7)
    raw = pd.DataFrame(
        {
            "probability": raw_probability,
            "base_rate": 0.50,
        },
        index=dates,
    )
    label = pd.Series(
        (raw_probability < 0.50).astype(float),
        index=dates,
        name="future_growth_outperforms_cash",
    )
    weights = pd.DataFrame(
        {"QQQ": 0.0, "SEMIS": 0.0},
        index=dates,
    )

    result = calibration.state_calibrated_probabilities(
        raw,
        label,
        weights,
        horizon_days=20,
        minimum_observations=60,
    )

    assert result["calibration_slope"].dropna().iloc[-1] < 0.0
