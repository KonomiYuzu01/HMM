from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r10_gde_capital_efficiency import ASSETS
from tools.evaluate_r40_trend_conditioned_cap import (
    TrendCapCandidate,
    TrendConditionedCapPolicy,
    cap_non_cash,
)


def weights(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=index, columns=ASSETS)
    frame["QQQ"] = 0.60
    frame["SEMIS"] = 0.60
    frame["CASH"] = -0.20
    return frame


def candidate() -> TrendCapCandidate:
    return TrendCapCandidate("test", 1.20, -0.19, 100.0, 20.0, 0.05)


def test_extended_cap_preserves_levered_target_when_dual_trend_is_positive() -> None:
    dates = pd.date_range("2026-01-02", periods=1, freq="B")
    target = weights(dates)
    signals = pd.DataFrame(
        {
            "both_above_sma": True,
            "stress": False,
            "prior_growth_drawdown": 0.0,
        },
        index=dates,
    )
    policy = TrendConditionedCapPolicy(candidate(), signals, target)
    implemented, force, metadata = policy(
        dates[0], target.iloc[0], 1.0, 1.0
    )
    assert implemented.drop("CASH").sum() == pytest.approx(1.20)
    assert implemented["CASH"] == pytest.approx(-0.20)
    assert force
    assert metadata["normal_cap_extended"] is True


def test_negative_dual_trend_caps_normal_state_at_one_hundred_percent() -> None:
    dates = pd.date_range("2026-01-02", periods=1, freq="B")
    target = weights(dates)
    signals = pd.DataFrame(
        {
            "both_above_sma": False,
            "stress": False,
            "prior_growth_drawdown": -0.02,
        },
        index=dates,
    )
    policy = TrendConditionedCapPolicy(candidate(), signals, target)
    implemented, force, metadata = policy(
        dates[0], target.iloc[0], 0.95, 1.0
    )
    assert implemented.drop("CASH").sum() == pytest.approx(1.0)
    assert implemented["CASH"] == pytest.approx(0.0)
    assert not force
    assert metadata["normal_cap_extended"] is False


def test_cap_non_cash_scales_assets_and_balances_cash() -> None:
    dates = pd.date_range("2026-01-02", periods=1, freq="B")
    implemented = cap_non_cash(weights(dates).iloc[0], 0.75)
    assert implemented.drop("CASH").sum() == pytest.approx(0.75)
    assert implemented["CASH"] == pytest.approx(0.25)
    assert implemented.sum() == pytest.approx(1.0)


def test_negative_trend_shock_uses_defensive_bear_multiplier() -> None:
    dates = pd.date_range("2026-01-02", periods=1, freq="B")
    target = weights(dates)
    signals = pd.DataFrame(
        {
            "both_above_sma": False,
            "stress": True,
            "prior_growth_drawdown": -0.12,
        },
        index=dates,
    )
    policy = TrendConditionedCapPolicy(candidate(), signals, target)
    implemented, _, metadata = policy(
        dates[0], target.iloc[0], 0.88, 1.0
    )
    assert metadata["shock_bear_guard"] is True
    assert metadata["effective_bear_multiplier"] == pytest.approx(9.5)
    assert implemented.drop("CASH").sum() == pytest.approx(0.75)


def test_growth_drawdown_closes_only_the_extended_capacity() -> None:
    dates = pd.date_range("2026-01-02", periods=1, freq="B")
    target = weights(dates)
    signals = pd.DataFrame(
        {
            "both_above_sma": True,
            "stress": False,
            "prior_growth_drawdown": -0.06,
        },
        index=dates,
    )
    policy = TrendConditionedCapPolicy(candidate(), signals, target)
    implemented, _, metadata = policy(dates[0], target.iloc[0], 1.0, 1.0)
    assert metadata["normal_cap_extended"] is False
    assert implemented.drop("CASH").sum() == pytest.approx(1.0)
