from __future__ import annotations

import pytest

from regime_strategy.recursive_trend_cushion import (
    RecursiveTrendCushionParameters,
    calculate_recursive_trend_cushion,
    protected_account_target,
)


PARAMETERS = RecursiveTrendCushionParameters()
R38 = {"QQQ": 0.45, "SMH": 0.22, "GLD": 0.23, "GDE": 0.02, "cash": 0.08}
R11 = {"QQQ": 0.41, "SMH": 0.20, "GLD": 0.21, "GDE": 0.02, "cash": 0.16}


def test_new_high_water_keeps_full_staged_target() -> None:
    state = calculate_recursive_trend_cushion(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
    )
    assert state.accepted_non_cash_cap == 1.0
    assert protected_account_target(staged_target=R38, r11_target=R11, state=state) == R38


def test_bear_trend_drawdown_uses_five_percent_tiers_and_r11_base() -> None:
    state = calculate_recursive_trend_cushion(
        prior_equity=85.0,
        prior_peak=100.0,
        dual_trend_positive=False,
    )
    assert state.requested_non_cash_cap == pytest.approx(0.4470588235294118)
    assert state.accepted_non_cash_cap == pytest.approx(0.40)
    target = protected_account_target(staged_target=R38, r11_target=R11, state=state)
    assert sum(target.values()) == pytest.approx(1.0)
    assert sum(weight for asset, weight in target.items() if asset != "cash") == pytest.approx(0.40)
    assert target["cash"] == pytest.approx(0.60)
    assert target["QQQ"] / target["SMH"] == pytest.approx(R11["QQQ"] / R11["SMH"])


def test_floor_exhaustion_moves_non_cash_to_zero() -> None:
    state = calculate_recursive_trend_cushion(
        prior_equity=81.0,
        prior_peak=100.0,
        dual_trend_positive=False,
    )
    assert state.state == "floor"
    assert state.accepted_non_cash_cap == 0.0
    target = protected_account_target(staged_target=R38, r11_target=R11, state=state)
    assert target["cash"] == pytest.approx(1.0)


def test_rejects_peak_that_omits_current_equity() -> None:
    with pytest.raises(ValueError, match="prior_peak"):
        calculate_recursive_trend_cushion(
            prior_equity=101.0,
            prior_peak=100.0,
            dual_trend_positive=True,
        )
