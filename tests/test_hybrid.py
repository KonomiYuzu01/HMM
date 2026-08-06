import numpy as np
import pandas as pd

from regime_strategy.backtest import BacktestResult
from regime_strategy.hybrid import (
    TrendSleeveResult,
    combine_sleeves,
    trend_target_weights,
)


def test_trend_target_uses_maximum_of_fast_and_slow_volatility() -> None:
    calm = np.tile([0.002, -0.001], (80, 1))
    shock = np.tile([[0.05, 0.04], [-0.03, -0.02]], (10, 1))
    growth = np.vstack([calm, shock])
    returns = pd.DataFrame(
        {
            "QQQ": growth[:, 0],
            "SEMIS": growth[:, 1],
            "CASH": np.zeros(100),
        }
    )
    target, trend_on, gross, sizing_volatility = trend_target_weights(
        returns,
        ["QQQ", "SEMIS", "CASH"],
        ["QQQ", "SEMIS"],
        2,
        60,
        20,
        60,
        0.20,
        1.75,
    )

    assert trend_on
    assert sizing_volatility > 0.20
    assert gross < 1.0
    assert np.isclose(target.sum(), 1.0)


def test_hybrid_outer_rebalance_charges_explicit_cost() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    hmm_daily = pd.DataFrame(
        {"net_return": [0.20, 0.0, 0.0], "cost": [0.0, 0.0, 0.0]},
        index=dates,
    )
    trend_daily = pd.DataFrame(
        {"net_return": [0.0, 0.0, 0.0], "cost": [0.0, 0.0, 0.0]},
        index=dates,
    )
    weights = pd.DataFrame(
        {"QQQ": [1.0, 1.0, 1.0], "CASH": [0.0, 0.0, 0.0]},
        index=dates,
    )
    hmm = BacktestResult(
        hmm_daily,
        weights,
        pd.DataFrame(index=dates),
        pd.DataFrame(index=dates),
        pd.DataFrame(index=dates),
    )
    trend = TrendSleeveResult(trend_daily, weights, pd.DataFrame(index=dates))
    result = combine_sleeves(hmm, trend, 0.5, 1, 0.0, 10.0)

    assert result.daily.iloc[1]["outer_turnover"] > 0.0
    assert result.daily.iloc[1]["outer_cost"] > 0.0
    assert np.isclose(result.sleeve_weights.iloc[1]["HMM"], 0.5)
