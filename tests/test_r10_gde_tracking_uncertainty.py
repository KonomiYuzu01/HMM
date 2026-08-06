import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_r10_gde_tracking_uncertainty.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_r10_gde_tracking_uncertainty",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_gde_implementation_residual_uses_90_90_less_80_formula():
    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {
            "GDE": [100.0, 102.0, 101.0],
            "SPY": [100.0, 101.0, 102.0],
            "GLD": [100.0, 101.0, 100.0],
            "BIL": [100.0, 100.1, 100.2],
        },
        index=dates,
    )
    returns = prices.pct_change(fill_method=None)
    expected = (
        returns["GDE"]
        - 0.90 * returns["SPY"]
        - 0.90 * returns["GLD"]
        + 0.80 * returns["BIL"]
    ).dropna()
    actual = MODULE.gde_implementation_residual(prices)
    assert np.allclose(actual, expected)


def test_zero_tracking_residual_preserves_deterministic_candidate():
    dates = pd.date_range("2024-01-01", periods=12)
    baseline_values = np.tile([0.0010, -0.0005], 6)
    baseline = pd.Series(baseline_values, index=dates)
    candidate = pd.DataFrame(
        {
            "net_return": baseline_values + 0.0001,
            "gde_target": np.full(len(dates), 0.05),
        },
        index=dates,
    )
    residual_dates = pd.date_range("2023-01-01", periods=6)
    residual = pd.Series(0.0, index=residual_dates)
    result = MODULE.bootstrap_tracking_uncertainty(
        baseline,
        candidate,
        residual,
        simulations=100,
        block_days=3,
    )
    assert result["probability_positive_cagr_delta"] == 1.0
    assert result["probability_sharpe_not_worse"] == 1.0
    assert result["probability_drawdown_within_18pct"] == 1.0
    assert result["probability_all_objectives"] == 1.0
