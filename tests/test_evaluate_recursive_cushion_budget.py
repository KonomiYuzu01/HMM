from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r10_gde_capital_efficiency import ASSETS
from tools.evaluate_recursive_cushion_budget import (
    CushionCandidate,
    RecursiveCushionPolicy,
    cap_total_non_cash,
)


def _weights(index: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=index, columns=ASSETS)
    result["QQQ"] = 0.40
    result["SEMIS"] = 0.40
    result["GOLD"] = 0.20
    return result


def test_cushion_cap_tightens_monotonically_as_equity_falls() -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    trend = pd.DataFrame({"both_above_sma": True}, index=index)
    weights = _weights(index)
    policy = RecursiveCushionPolicy(CushionCandidate("test"), trend, weights)
    caps: list[float] = []
    for date, equity in zip(index, [1.00, 0.95, 0.90, 0.85], strict=True):
        _, _, diagnostics = policy(date, weights.loc[date], equity, 1.00)
        caps.append(float(diagnostics["accepted_non_cash_cap"]))
    assert caps == sorted(caps, reverse=True)
    assert caps == pytest.approx([1.0, 0.8, 0.6, 0.2])


def test_bear_multiplier_is_more_defensive_than_bull_multiplier() -> None:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    weights = _weights(index)
    bull = RecursiveCushionPolicy(
        CushionCandidate("bull"),
        pd.DataFrame({"both_above_sma": True}, index=index),
        weights,
    )
    bear = RecursiveCushionPolicy(
        CushionCandidate("bear"),
        pd.DataFrame({"both_above_sma": False}, index=index),
        weights,
    )
    _, _, bull_diag = bull(index[0], weights.loc[index[0]], 0.90, 1.00)
    _, _, bear_diag = bear(index[0], weights.loc[index[0]], 0.90, 1.00)
    assert bear_diag["accepted_non_cash_cap"] < bull_diag["accepted_non_cash_cap"]


def test_total_non_cash_cap_preserves_weights_and_moves_residual_to_cash() -> None:
    index = pd.date_range("2024-01-02", periods=1, freq="B")
    target = _weights(index).iloc[0]
    adjusted = cap_total_non_cash(target, 0.40)
    assert adjusted.drop("CASH").sum() == pytest.approx(0.40)
    assert adjusted["CASH"] == pytest.approx(0.60)
    assert adjusted.sum() == pytest.approx(1.0)


def test_large_bull_multiplier_restores_full_budget_away_from_floor() -> None:
    index = pd.date_range("2024-01-02", periods=1, freq="B")
    weights = _weights(index)
    policy = RecursiveCushionPolicy(
        CushionCandidate(
            "trend_recovery",
            bull_multiplier=100.0,
            bear_multiplier=6.0,
        ),
        pd.DataFrame({"both_above_sma": True}, index=index),
        weights,
    )
    _, _, diagnostics = policy(index[0], weights.iloc[0], 0.85, 1.00)
    assert diagnostics["accepted_non_cash_cap"] == pytest.approx(1.0)
    assert diagnostics["state"] == "normal"


def test_full_requested_budget_is_not_reduced_by_non_divisor_tier() -> None:
    index = pd.date_range("2024-01-02", periods=1, freq="B")
    weights = _weights(index)
    policy = RecursiveCushionPolicy(
        CushionCandidate(
            "non_divisor",
            bull_multiplier=100.0,
            bear_multiplier=8.0,
            tier_size=0.15,
        ),
        pd.DataFrame({"both_above_sma": True}, index=index),
        weights,
    )
    _, _, diagnostics = policy(index[0], weights.iloc[0], 1.0, 1.0)
    assert diagnostics["requested_non_cash_cap"] == pytest.approx(1.0)
    assert diagnostics["accepted_non_cash_cap"] == pytest.approx(1.0)
