from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    VolatilityCandidate,
    accepted_risk_multipliers,
    causal_realized_volatility,
    major_drawdown_events,
)


def test_realized_volatility_does_not_use_same_day_return() -> None:
    index = pd.bdate_range("2024-01-01", periods=25)
    returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
    candidate = VolatilityCandidate("test", 21)
    original = causal_realized_volatility(returns, candidate)
    changed = returns.copy()
    changed.iloc[-1] = 0.50

    revised = causal_realized_volatility(changed, candidate)

    assert revised.iloc[-1] == pytest.approx(original.iloc[-1])


def test_maximum_estimator_waits_for_both_windows() -> None:
    index = pd.bdate_range("2024-01-01", periods=64)
    returns = pd.Series(
        np.sin(np.arange(len(index))) / 100.0,
        index=index,
    )
    candidate = VolatilityCandidate("test", 21, 63)

    realized = causal_realized_volatility(returns, candidate)

    assert realized.iloc[-2] != realized.iloc[-2]
    assert np.isfinite(realized.iloc[-1])


def test_multiplier_updates_only_on_existing_trade_dates() -> None:
    index = pd.bdate_range("2024-01-01", periods=30)
    returns = pd.Series(
        np.sin(np.arange(len(index))) / 100.0,
        index=index,
    )
    weights = pd.DataFrame(index=index)
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    daily.loc[index[22], "turnover"] = 0.10
    daily.loc[index[27], "turnover"] = 0.10

    multipliers, diagnostics = accepted_risk_multipliers(
        returns,
        weights,
        daily,
        VolatilityCandidate("test", 21),
    )

    assert multipliers.iloc[:22].eq(BASE_MULTIPLIER).all()
    assert multipliers.iloc[22:27].nunique() == 1
    assert diagnostics.loc[index[23], "base_trade"] == pytest.approx(0.0)


def test_multiplier_is_bounded() -> None:
    index = pd.bdate_range("2024-01-01", periods=50)
    returns = pd.Series(0.0001, index=index)
    weights = pd.DataFrame(index=index)
    daily = pd.DataFrame({"turnover": 1.0}, index=index)

    multipliers, _ = accepted_risk_multipliers(
        returns,
        weights,
        daily,
        VolatilityCandidate("test", 21),
    )

    assert multipliers.min() >= 0.90
    assert multipliers.max() <= 1.20


def test_major_drawdown_events_include_unrecovered_event() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series(
        [0.0, -0.05, -0.06, 0.01, 0.01],
        index=index,
    )

    events = major_drawdown_events(returns)

    assert events == [(index[1], index[-1])]
