from __future__ import annotations

import pandas as pd
import pytest

from regime_strategy.r10_overlay import (
    BASE_ASSETS,
    blend_rollout_target,
    r10_band_targets,
    r10_center_target,
    risk_scaled_target,
    target_from_actual_gde,
)


def base_target() -> pd.Series:
    target = pd.Series(0.0, index=BASE_ASSETS)
    target["QQQ"] = 0.40
    target["SEMIS"] = 0.24
    target["GOLD"] = 0.20
    target["CASH"] = 0.16
    return target


def test_center_preserves_gold_sleeve_and_account_weight() -> None:
    base = base_target()
    center = r10_center_target(base, substitution_fraction=0.50)

    assert center["GDE"] == pytest.approx(0.08)
    assert center["GOLD"] == pytest.approx(0.12)
    assert center["GOLD"] + center["GDE"] == pytest.approx(0.20)
    assert center.sum() == pytest.approx(base.sum())


def test_risk_scaled_target_uses_cash_as_financing_leg() -> None:
    scaled = risk_scaled_target(
        base_target(),
        risk_multiplier=1.065,
    )

    assert scaled["QQQ"] == pytest.approx(0.426)
    assert scaled["SEMIS"] == pytest.approx(0.2556)
    assert scaled["GOLD"] == pytest.approx(0.213)
    assert scaled["CASH"] == pytest.approx(0.1054)
    assert scaled.sum() == pytest.approx(1.0)


def test_risk_scaled_target_validates_multiplier() -> None:
    with pytest.raises(ValueError, match="risk_multiplier"):
        risk_scaled_target(base_target(), risk_multiplier=-0.01)


def test_band_preserves_gold_sleeve() -> None:
    center = r10_center_target(
        base_target(),
        substitution_fraction=0.50,
    )
    lower, upper = r10_band_targets(center, no_trade_band=0.02)

    assert lower["GDE"] == pytest.approx(0.06)
    assert upper["GDE"] == pytest.approx(0.10)
    assert lower["GOLD"] + lower["GDE"] == pytest.approx(0.20)
    assert upper["GOLD"] + upper["GDE"] == pytest.approx(0.20)


def test_quarter_rollout_scales_gde_range_to_account() -> None:
    base = base_target()
    center = r10_center_target(base, substitution_fraction=0.50)
    lower, upper = r10_band_targets(center, no_trade_band=0.02)
    account_lower = blend_rollout_target(
        base,
        lower,
        rollout_share=0.25,
    )
    account_upper = blend_rollout_target(
        base,
        upper,
        rollout_share=0.25,
    )

    assert account_lower["GDE"] == pytest.approx(0.015)
    assert account_upper["GDE"] == pytest.approx(0.025)


def test_actual_holding_trades_to_nearest_band_boundary() -> None:
    base = base_target()
    center = r10_center_target(base, substitution_fraction=0.50)
    target = target_from_actual_gde(
        base,
        center,
        rollout_share=0.25,
        no_trade_band=0.02,
        actual_account_gde_weight=0.0,
    )

    assert target["GDE"] == pytest.approx(0.015)
    assert target["GOLD"] + target["GDE"] == pytest.approx(0.20)
