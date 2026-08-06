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

from evaluate_r10_gde_capital_efficiency import ASSETS
from regime_strategy.relative_damage import apply_relative_damage_veto
from tools.evaluate_r39_relative_damage_concentration_veto import (
    apply_relative_damage_concentration_veto,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2025-01-02", periods=90)
    qqq_returns = np.full(len(index), 0.0005)
    semis_returns = qqq_returns.copy()
    semis_returns[-30:] = -0.006
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.exp(np.cumsum(qqq_returns)),
            "SEMIS": 100.0 * np.exp(np.cumsum(semis_returns)),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.15
    weights["SEMIS"] = 0.65
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_damage_budget_caps_semis_and_routes_excess_to_qqq() -> None:
    closes, weights, daily = _inputs()
    adjusted, _, diagnostics = apply_relative_damage_concentration_veto(
        weights,
        daily,
        closes,
    )
    active = diagnostics["relative_damage_guard_active"]
    assert active.any()
    selected = active[active].index[-1]
    growth_before = weights.loc[selected, ["QQQ", "SEMIS"]].sum()
    growth_after = adjusted.loc[selected, ["QQQ", "SEMIS"]].sum()
    assert growth_after == pytest.approx(growth_before)
    assert (
        diagnostics.loc[selected, "implemented_account_relative_loss"]
        == pytest.approx(0.03)
    )
    assert (
        adjusted.loc[selected, "QQQ"] - weights.loc[selected, "QQQ"]
        == pytest.approx(
            weights.loc[selected, "SEMIS"]
            - adjusted.loc[selected, "SEMIS"]
        )
    )
    assert (
        diagnostics.loc[selected, "implemented_account_relative_loss"]
        < diagnostics.loc[selected, "proposed_account_relative_loss"]
    )


def test_veto_clears_when_proposed_concentration_no_longer_violates() -> None:
    closes, weights, daily = _inputs()
    weights.loc[weights.index[-1], ["QQQ", "SEMIS"]] = [0.55, 0.25]
    adjusted, _, diagnostics = apply_relative_damage_concentration_veto(
        weights,
        daily,
        closes,
    )
    assert not bool(
        diagnostics.iloc[-1]["relative_damage_guard_active"]
    )
    pd.testing.assert_series_equal(
        adjusted.iloc[-1],
        weights.iloc[-1],
    )


def test_current_close_cannot_change_current_guard_state() -> None:
    closes, weights, daily = _inputs()
    original, _, original_diagnostics = (
        apply_relative_damage_concentration_veto(
            weights,
            daily,
            closes,
        )
    )
    changed = closes.copy()
    changed.loc[changed.index[-1], ["QQQ", "SEMIS"]] = [1.0, 10_000.0]
    revised, _, revised_diagnostics = (
        apply_relative_damage_concentration_veto(
            weights,
            daily,
            changed,
        )
    )
    pd.testing.assert_series_equal(
        original.iloc[-1],
        revised.iloc[-1],
    )
    assert (
        original_diagnostics.iloc[-1]["relative_damage_guard_active"]
        == revised_diagnostics.iloc[-1]["relative_damage_guard_active"]
    )


def test_invalid_loss_budget_raises() -> None:
    closes, weights, daily = _inputs()
    with pytest.raises(ValueError, match="account loss budget"):
        apply_relative_damage_concentration_veto(
            weights,
            daily,
            closes,
            entry_account_loss_budget=0.0,
        )


def test_active_guard_prevents_damaged_semis_majority() -> None:
    weights = pd.DataFrame(
        {"QQQ": [0.15], "SEMIS": [0.65]},
        index=pd.DatetimeIndex(["2026-07-30"]),
    )
    loss = pd.Series(0.06, index=weights.index)
    adjusted, diagnostics = apply_relative_damage_veto(
        weights,
        loss,
        pd.Series(True, index=weights.index),
        account_loss_budget=0.03,
        minimum_semis_growth_share=0.60,
        maximum_active_semis_growth_share=0.50,
        maximum_share_activation_multiple=1.10,
    )
    assert bool(diagnostics.iloc[0]["relative_damage_guard_active"])
    assert adjusted.iloc[0]["SEMIS"] == pytest.approx(0.40)
    assert adjusted.iloc[0]["QQQ"] == pytest.approx(0.40)
    assert (
        diagnostics.iloc[0]["implemented_semis_growth_share"]
        == pytest.approx(0.50)
    )


def test_invalid_active_semis_share_cap_raises() -> None:
    weights = pd.DataFrame({"QQQ": [0.15], "SEMIS": [0.65]})
    loss = pd.Series([0.05])
    with pytest.raises(ValueError, match="maximum active SMH"):
        apply_relative_damage_veto(
            weights,
            loss,
            pd.Series(True, index=weights.index),
            account_loss_budget=0.03,
            minimum_semis_growth_share=0.60,
            maximum_active_semis_growth_share=0.51,
            maximum_share_activation_multiple=1.10,
        )


def test_loss_budget_acts_before_hard_share_cap() -> None:
    weights = pd.DataFrame({"QQQ": [0.15], "SEMIS": [0.65]})
    loss = pd.Series([0.048])
    adjusted, diagnostics = apply_relative_damage_veto(
        weights,
        loss,
        pd.Series(True, index=weights.index),
        account_loss_budget=0.03,
        minimum_semis_growth_share=0.60,
        maximum_active_semis_growth_share=0.50,
        maximum_share_activation_multiple=1.10,
    )
    assert bool(diagnostics.iloc[0]["relative_damage_guard_active"])
    assert not bool(diagnostics.iloc[0]["maximum_share_guard_active"])
    assert adjusted.iloc[0]["SEMIS"] == pytest.approx(0.625)
    assert (
        diagnostics.iloc[0]["implemented_account_relative_loss"]
        == pytest.approx(0.03)
    )


def test_volatility_acceleration_blocks_only_hard_share_cap() -> None:
    weights = pd.DataFrame({"QQQ": [0.15], "SEMIS": [0.65]})
    loss = pd.Series([0.06])
    adjusted, diagnostics = apply_relative_damage_veto(
        weights,
        loss,
        pd.Series(False, index=weights.index),
        account_loss_budget=0.03,
        minimum_semis_growth_share=0.60,
        maximum_active_semis_growth_share=0.50,
        maximum_share_activation_multiple=1.10,
    )
    assert bool(diagnostics.iloc[0]["relative_damage_guard_active"])
    assert not bool(diagnostics.iloc[0]["maximum_share_guard_active"])
    assert adjusted.iloc[0]["SEMIS"] == pytest.approx(0.50)
    assert (
        diagnostics.iloc[0]["implemented_account_relative_loss"]
        == pytest.approx(0.03)
    )
