from __future__ import annotations

import pandas as pd

from tools.evaluate_r10_conditional_cash_completion import (
    CompletionRule,
    causal_completion_signals,
    completion_target,
)


def test_completion_target_uses_only_positive_cash_and_preserves_sum() -> None:
    base = pd.Series(
        {
            "QQQ": 0.30,
            "SEMIS": 0.20,
            "GOLD": 0.20,
            "CASH": 0.30,
        }
    )
    target, share = completion_target(
        base,
        True,
        CompletionRule(name="test"),
    )
    assert share == 0.10
    assert abs(float(target["QQQ"]) - 0.40) < 1e-12
    assert abs(float(target["CASH"]) - 0.20) < 1e-12
    assert abs(float(target.sum()) - 1.0) < 1e-12


def test_completion_target_does_not_override_defensive_growth_exposure() -> None:
    base = pd.Series(
        {
            "QQQ": 0.10,
            "SEMIS": 0.10,
            "GOLD": 0.20,
            "CASH": 0.60,
        }
    )
    target, share = completion_target(
        base,
        True,
        CompletionRule(name="test"),
    )
    assert share == 0.0
    pd.testing.assert_series_equal(target, base.astype(float))


def test_signal_at_open_does_not_use_same_day_close() -> None:
    index = pd.date_range("2020-01-01", periods=230, freq="B")
    rising = pd.Series(range(100, 330), index=index, dtype=float)
    closes = pd.DataFrame({"QQQ": rising, "SEMIS": rising * 1.1})
    rule = CompletionRule(
        name="test",
        maximum_annual_volatility=10.0,
    )
    before = causal_completion_signals(closes, rule)
    changed = closes.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] *= 0.01
    after = causal_completion_signals(changed, rule)
    pd.testing.assert_series_equal(
        before.loc[index[-1]],
        after.loc[index[-1]],
    )
