from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r16_portable_managed_futures import (
    constant_notional_stack,
    monthly_reset_stack,
)


def test_constant_notional_subtracts_cash_and_financing() -> None:
    index = pd.bdate_range("2026-01-01", periods=2)
    base = pd.Series([0.01, 0.01], index=index)
    fund = pd.Series([0.02, 0.02], index=index)
    cash = pd.Series([0.001, 0.001], index=index)

    path = constant_notional_stack(
        base,
        fund,
        cash,
        0.20,
        financing_spread_bps=252.0,
    )

    expected_overlay = 0.02 - 0.001 - 0.0001
    assert path["candidate_return"].iloc[0] == pytest.approx(
        0.01 + 0.20 * expected_overlay
    )


def test_zero_overlay_exactly_matches_base() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    base = pd.Series([0.01, -0.02, 0.03], index=index)
    fund = pd.Series([0.02, 0.02, 0.02], index=index)
    cash = pd.Series([0.001, 0.001, 0.001], index=index)

    path = constant_notional_stack(
        base,
        fund,
        cash,
        0.0,
        financing_spread_bps=100.0,
    )

    pd.testing.assert_series_equal(
        path["candidate_return"],
        base.rename("candidate_return"),
    )


def test_overlay_effect_is_linear_in_declared_notional() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    base = pd.Series([0.0, 0.0, 0.0], index=index)
    fund = pd.Series([0.01, -0.01, 0.02], index=index)
    cash = pd.Series([0.0, 0.0, 0.0], index=index)

    small = constant_notional_stack(
        base,
        fund,
        cash,
        0.10,
        financing_spread_bps=0.0,
    )
    large = constant_notional_stack(
        base,
        fund,
        cash,
        0.20,
        financing_spread_bps=0.0,
    )

    pd.testing.assert_series_equal(
        large["candidate_return"],
        (2.0 * small["candidate_return"]).rename("candidate_return"),
    )


def test_monthly_reset_charges_initial_and_new_month_cost() -> None:
    index = pd.to_datetime(
        ["2026-01-30", "2026-02-02", "2026-02-03"]
    )
    zeros = pd.Series([0.0, 0.0, 0.0], index=index)

    path = monthly_reset_stack(
        zeros,
        zeros,
        zeros,
        0.20,
        financing_spread_bps=0.0,
        one_way_cost_bps=10.0,
    )

    assert path["reset"].tolist() == [True, True, False]
    assert path["trading_cost"].iloc[0] == pytest.approx(0.0002)
    assert path["trading_cost"].iloc[1] > 0.0
    assert path["trading_cost"].iloc[2] == pytest.approx(0.0)


def test_stack_uses_only_common_dates_without_filling() -> None:
    base_index = pd.to_datetime(
        ["2026-01-02", "2026-01-05", "2026-01-06"]
    )
    fund_index = pd.to_datetime(["2026-01-02", "2026-01-06"])
    base = pd.Series([0.0, 0.0, 0.0], index=base_index)
    fund = pd.Series([0.0, 0.0], index=fund_index)
    cash = pd.Series([0.0, 0.0], index=fund_index)

    path = constant_notional_stack(
        base,
        fund,
        cash,
        0.10,
        financing_spread_bps=0.0,
    )

    assert path.index.tolist() == fund_index.tolist()
