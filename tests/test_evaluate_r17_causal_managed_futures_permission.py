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

from evaluate_r17_causal_managed_futures_permission import (
    academic_permission,
    causal_monthly_permission,
    gated_stack,
)


def test_permission_uses_prior_month_signal() -> None:
    index = pd.bdate_range("2024-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "AQMIX": np.linspace(100.0, 150.0, len(index)),
            "BIL": np.linspace(100.0, 102.0, len(index)),
        },
        index=index,
    )
    permission = causal_monthly_permission(prices)
    changed = prices.copy()
    last_date = index[-1]
    changed.loc[last_date, "AQMIX"] = 1.0

    revised = causal_monthly_permission(changed)

    current_month = last_date.to_period("M")
    pd.testing.assert_series_equal(
        permission.loc[permission.index.to_period("M") == current_month],
        revised.loc[revised.index.to_period("M") == current_month],
    )


def test_gate_off_exactly_matches_base() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    base = pd.Series([0.01, -0.02, 0.03], index=index)
    fund = pd.Series([0.02, 0.02, 0.02], index=index)
    cash = pd.Series([0.001, 0.001, 0.001], index=index)
    permission = pd.Series(False, index=index)

    path = gated_stack(
        base,
        fund,
        cash,
        permission,
        0.15,
        financing_spread_bps=100.0,
    )

    pd.testing.assert_series_equal(
        path["candidate_return"],
        base.rename("candidate_return"),
    )


def test_gate_subtracts_financing_and_switch_cost() -> None:
    index = pd.bdate_range("2026-01-01", periods=2)
    base = pd.Series([0.0, 0.0], index=index)
    fund = pd.Series([0.02, 0.02], index=index)
    cash = pd.Series([0.001, 0.001], index=index)
    permission = pd.Series([True, True], index=index)

    path = gated_stack(
        base,
        fund,
        cash,
        permission,
        0.15,
        financing_spread_bps=252.0,
        one_way_cost_bps=10.0,
    )

    expected_excess = 0.02 - 0.001 - 0.0001
    assert path["candidate_return"].iloc[0] == pytest.approx(
        0.15 * expected_excess - 0.00015
    )
    assert path["candidate_return"].iloc[1] == pytest.approx(
        0.15 * expected_excess
    )


def test_switching_cost_is_charged_on_both_transitions() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    zeros = pd.Series(0.0, index=index)
    permission = pd.Series([False, True, False], index=index)

    path = gated_stack(
        zeros,
        zeros,
        zeros,
        permission,
        0.15,
        financing_spread_bps=0.0,
        one_way_cost_bps=10.0,
    )

    assert path["switching_cost"].tolist() == pytest.approx(
        [0.0, 0.00015, 0.00015]
    )


def test_academic_permission_has_twelve_month_lag() -> None:
    index = pd.date_range("2020-01-31", periods=14, freq="ME")
    returns = pd.Series(0.01, index=index)

    permission = academic_permission(returns)

    assert not bool(permission.iloc[11])
    assert bool(permission.iloc[12])
