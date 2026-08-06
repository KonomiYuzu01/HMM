import numpy as np
import pandas as pd

from regime_strategy.backtest import BacktestResult
from regime_strategy.ensemble import (
    apply_model_disagreement_risk_control,
    combine_model_sleeves,
)


def test_model_disagreement_reliability_moves_growth_to_cash() -> None:
    target, multiplier, reallocated, disagreement = (
        apply_model_disagreement_risk_control(
            np.array([0.5, 0.5]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.5, 0.5]),
            ["QQQ", "CASH"],
            {"enabled": True, "assets": ["QQQ"], "destination": "CASH"},
        )
    )

    assert np.isclose(multiplier, 0.5)
    assert np.isclose(disagreement, 0.25)
    assert np.isclose(reallocated, 0.25)
    assert np.allclose(target, np.array([0.25, 0.75]))


def _member(name: str, returns: list[float], costs: list[float]) -> BacktestResult:
    dates = pd.bdate_range("2024-01-01", periods=len(returns))
    daily = pd.DataFrame(
        {
            "net_return": returns,
            "gross_return": returns,
            "financing_cost": np.zeros(len(returns)),
            "cost": costs,
            "turnover": np.zeros(len(returns)),
        },
        index=dates,
    )
    qqq = 1.0 if name == "risk" else 0.0
    weights = pd.DataFrame(
        {"QQQ": np.full(len(returns), qqq), "CASH": np.full(len(returns), 1.0 - qqq)},
        index=dates,
    )
    asset_returns = pd.DataFrame(
        {"QQQ": np.zeros(len(returns)), "CASH": np.zeros(len(returns))},
        index=dates,
    )
    return BacktestResult(
        daily,
        weights,
        pd.DataFrame(index=dates),
        pd.DataFrame(index=dates),
        asset_returns,
    )


def test_model_ensemble_preserves_equal_initial_capital_and_asset_budget() -> None:
    result = combine_model_sleeves(
        {
            "risk": _member("risk", [0.10, 0.0], [0.0, 0.0]),
            "cash": _member("cash", [0.0, 0.0], [0.0, 0.0]),
        },
        rebalance_every_days=21,
        no_trade_turnover=0.01,
        cost_bps_per_dollar_traded=7.5,
    )

    assert np.allclose(result.sleeve_weights.iloc[0], [0.5, 0.5])
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert np.isclose(result.daily.iloc[0]["net_return"], 0.05)


def test_model_ensemble_rebalance_charges_outer_cost() -> None:
    result = combine_model_sleeves(
        {
            "risk": _member("risk", [0.20, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "cash": _member("cash", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        },
        rebalance_every_days=1,
        no_trade_turnover=0.0,
        cost_bps_per_dollar_traded=10.0,
    )

    assert result.daily.iloc[1]["outer_turnover"] > 0.0
    assert result.daily.iloc[1]["outer_cost"] > 0.0
    assert np.allclose(result.sleeve_weights.iloc[1], [0.5, 0.5])


def test_netted_ensemble_does_not_trade_offsetting_member_orders() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    risk = _member("risk", [0.0, 0.0], [0.001, 0.001])
    cash = _member("cash", [0.0, 0.0], [0.001, 0.001])
    risk.weights = pd.DataFrame(
        {"QQQ": [1.0, 0.0], "CASH": [0.0, 1.0]},
        index=dates,
    )
    cash.weights = pd.DataFrame(
        {"QQQ": [0.0, 1.0], "CASH": [1.0, 0.0]},
        index=dates,
    )

    result = combine_model_sleeves(
        {"risk": risk, "cash": cash},
        rebalance_every_days=21,
        no_trade_turnover=0.0,
        cost_bps_per_dollar_traded=10.0,
        net_member_trades=True,
    )

    assert np.allclose(result.weights, [[0.5, 0.5], [0.5, 0.5]])
    assert result.daily.iloc[0]["turnover"] > 0.0
    assert np.isclose(result.daily.iloc[1]["turnover"], 0.0)
    assert np.isclose(result.daily.iloc[1]["cost"], 0.0)
    assert result.daily.iloc[1]["weighted_internal_cost"] > 0.0
