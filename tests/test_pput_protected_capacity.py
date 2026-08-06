from __future__ import annotations

import pytest

from regime_strategy.pput_protected_capacity import (
    calculate_pput_protected_capacity,
    map_put_contracts,
    select_five_percent_otm_strike,
)


def test_normal_cap_requires_confirmed_hedge() -> None:
    protected = calculate_pput_protected_capacity(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
        hedge_confirmed=True,
    )
    unhedged = calculate_pput_protected_capacity(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
        hedge_confirmed=False,
    )
    assert protected.accepted_non_cash_cap == pytest.approx(1.20)
    assert unhedged.accepted_non_cash_cap == pytest.approx(1.0)


def test_controlled_and_floor_states_do_not_extend() -> None:
    controlled = calculate_pput_protected_capacity(
        prior_equity=85.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        hedge_confirmed=True,
    )
    floor_state = calculate_pput_protected_capacity(
        prior_equity=81.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        hedge_confirmed=True,
    )
    assert controlled.accepted_non_cash_cap == pytest.approx(0.90)
    assert floor_state.accepted_non_cash_cap == pytest.approx(0.0)


def test_500k_spym_mapping_uses_three_contracts() -> None:
    mapping = map_put_contracts(account_equity=500_000.0, underlying_price=90.52)
    assert mapping.contracts == 3
    assert mapping.implemented_coverage == pytest.approx(0.054312)
    assert mapping.coverage_qualified


def test_contract_mapping_fails_closed_when_rounding_misses_band() -> None:
    mapping = map_put_contracts(account_equity=100_000.0, underlying_price=500.0)
    assert not mapping.coverage_qualified


def test_selects_highest_listed_strike_not_above_95_percent_spot() -> None:
    assert select_five_percent_otm_strike(
        underlying_price=90.52,
        listed_put_strikes=[85, 86, 87, 88, 89, 90],
    ) == 85.0
