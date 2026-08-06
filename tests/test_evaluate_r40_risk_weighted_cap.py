from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r40_risk_weighted_cap import apply_risk_weighted_cap


def target() -> pd.Series:
    return pd.Series(
        {
            "SPX": 0.00,
            "QQQ": 0.55,
            "SEMIS": 0.25,
            "BOND": 0.00,
            "GOLD": 0.20,
            "OIL": 0.00,
            "USD": 0.00,
            "CASH": 0.00,
            "VIX_HEDGE": 0.00,
        }
    )


def test_full_gold_risk_matches_total_non_cash_cap() -> None:
    adjusted = apply_risk_weighted_cap(
        target(),
        accepted_cap=0.60,
        normal_non_cash_cap=1.0,
        defensive_risk_coefficient=1.0,
    )
    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[["QQQ", "SEMIS", "GOLD"]].sum() == pytest.approx(0.60)


def test_partial_gold_risk_preserves_gold_and_caps_risk_budget() -> None:
    adjusted = apply_risk_weighted_cap(
        target(),
        accepted_cap=0.60,
        normal_non_cash_cap=1.0,
        defensive_risk_coefficient=0.25,
    )
    risk_budget = adjusted[["QQQ", "SEMIS"]].sum() + 0.25 * adjusted["GOLD"]
    assert adjusted["GOLD"] == pytest.approx(0.20)
    assert risk_budget == pytest.approx(0.60)
    assert adjusted.sum() == pytest.approx(1.0)


def test_normal_extension_allows_only_configured_risk_budget() -> None:
    leveraged = pd.Series(
        {
            **target().to_dict(),
            "QQQ": 0.70,
            "SEMIS": 0.35,
            "CASH": -0.25,
        }
    )
    adjusted = apply_risk_weighted_cap(
        leveraged,
        accepted_cap=1.0,
        normal_non_cash_cap=1.10,
        defensive_risk_coefficient=0.25,
    )
    risk_budget = adjusted[["QQQ", "SEMIS"]].sum() + 0.25 * adjusted["GOLD"]
    assert risk_budget == pytest.approx(1.10)
    assert adjusted.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("accepted", "normal", "gold"),
    [(-0.1, 1.0, 0.5), (1.0, 1.3, 0.5), (1.0, 1.0, 1.1)],
)
def test_invalid_parameters_are_rejected(
    accepted: float,
    normal: float,
    gold: float,
) -> None:
    with pytest.raises(ValueError):
        apply_risk_weighted_cap(
            target(),
            accepted_cap=accepted,
            normal_non_cash_cap=normal,
            defensive_risk_coefficient=gold,
        )
