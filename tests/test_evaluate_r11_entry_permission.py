from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r11_entry_permission import (
    EntryPermissionPolicy,
    apply_entry_permission,
    build_entry_permission_signals,
    classify_growth_regime,
    permitted_growth_ceiling,
    project_growth_addition,
)


def allocation(growth: float) -> pd.Series:
    values = pd.Series(0.0, index=ASSETS)
    values["QQQ"] = 0.75 * growth
    values["SEMIS"] = 0.25 * growth
    values["GOLD"] = 0.20
    values["CASH"] = 0.80 - growth
    return values


def signals(
    index: pd.Index,
    *,
    semis_126: float,
    qqq_63: float = -0.05,
    semis_63: float = -0.06,
    qqq_20: float = -0.02,
    semis_20: float = -0.03,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qqq_20d_return": qqq_20,
            "semis_20d_return": semis_20,
            "qqq_63d_return": qqq_63,
            "semis_63d_return": semis_63,
            "semis_126d_return": semis_126,
            "vix_term_ratio": 0.95,
        },
        index=index,
    )


def flat_prices(index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(100.0, index=index, columns=ASSETS)
    return frame.copy(), frame.copy()


def test_regime_classifier_separates_liquidity_from_structural_damage() -> None:
    liquidity = pd.Series(
        {
            "qqq_63d_return": 0.05,
            "semis_63d_return": 0.06,
            "semis_126d_return": 0.08,
            "vix_term_ratio": 1.05,
        }
    )
    structural = liquidity.copy()
    structural["semis_126d_return"] = -0.08
    structural["qqq_63d_return"] = -0.05

    assert classify_growth_regime(liquidity) == "liquidity_stress"
    assert classify_growth_regime(structural) == "structural_damage"


def test_growth_projection_never_turns_entry_limit_into_an_exit() -> None:
    current = allocation(0.20)
    desired = allocation(0.80)

    projected = project_growth_addition(current, desired, 0.10)

    assert np.isclose(projected[list(("SPX", "QQQ", "SEMIS"))].sum(), 0.20)
    assert np.isclose(projected.sum(), desired.sum())


def test_permission_tiers_expand_monotonically_with_recovery_evidence() -> None:
    index = pd.bdate_range("2024-01-02", periods=1)
    policy = EntryPermissionPolicy()
    damaged = signals(index, semis_126=-0.05).iloc[0]
    early = signals(
        index,
        semis_126=-0.05,
        qqq_20=0.02,
        semis_20=0.03,
    ).iloc[0]
    confirmed = signals(
        index,
        semis_126=-0.05,
        qqq_20=0.02,
        semis_20=0.03,
        qqq_63=0.05,
        semis_63=0.06,
    ).iloc[0]

    assert permitted_growth_ceiling(damaged, 0.0, policy) == (
        0.10,
        "damage_probe",
    )
    assert permitted_growth_ceiling(early, 0.0, policy) == (
        0.35,
        "early_repair",
    )
    assert permitted_growth_ceiling(confirmed, 0.0, policy) == (
        None,
        "confirmed_recovery",
    )


def test_damage_budget_limits_cumulative_entry_but_passes_exit() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    weights = pd.DataFrame(
        [
            allocation(0.60),
            allocation(0.00),
            allocation(0.50),
            allocation(0.70),
        ],
        index=index,
    )
    daily = pd.DataFrame(
        {"turnover": [1.0, 1.0, 1.0, 1.0], "slippage_cost": 0.0},
        index=index,
    )
    opens, closes = flat_prices(index)

    adjusted, _, diagnostics = apply_entry_permission(
        weights,
        daily,
        opens,
        closes,
        signals(index, semis_126=-0.05),
        EntryPermissionPolicy(damage_growth_budget=0.10),
    )
    growth = adjusted[list(("SPX", "QQQ", "SEMIS"))].sum(axis=1)

    assert np.isclose(growth.iloc[0], 0.10)
    assert np.isclose(growth.iloc[1], 0.00)
    assert np.isclose(growth.iloc[2], 0.10)
    assert np.isclose(growth.iloc[3], 0.10)
    assert bool(diagnostics.iloc[1]["reduction_passed"])
    assert bool(diagnostics.iloc[2]["entry_limited"])


def test_healthy_regime_releases_budget_only_on_a_proposal() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame(
        [allocation(0.60)] * len(index),
        index=index,
    )
    daily = pd.DataFrame(
        {"turnover": [1.0, 0.0, 1.0], "slippage_cost": 0.0},
        index=index,
    )
    regime_signals = signals(index, semis_126=-0.05)
    regime_signals.loc[index[1]:, "semis_126d_return"] = 0.05
    opens, closes = flat_prices(index)

    adjusted, adjusted_daily, diagnostics = apply_entry_permission(
        weights,
        daily,
        opens,
        closes,
        regime_signals,
        EntryPermissionPolicy(damage_growth_budget=0.10),
    )
    growth = adjusted[list(("SPX", "QQQ", "SEMIS"))].sum(axis=1)

    assert np.isclose(growth.iloc[0], 0.10)
    assert np.isclose(growth.iloc[1], 0.10)
    assert np.isclose(adjusted_daily.iloc[1]["turnover"], 0.0)
    assert bool(diagnostics.iloc[1]["state_released"])
    assert not bool(diagnostics.iloc[1]["state_change_trade"])
    assert np.isclose(growth.iloc[2], 0.60)


def test_short_trend_signals_ignore_same_day_price_changes() -> None:
    index = pd.bdate_range("2024-01-02", periods=130)
    weights = pd.DataFrame(
        [allocation(0.60)] * len(index),
        index=index,
    )
    closes = pd.DataFrame(100.0, index=index, columns=ASSETS)
    market = pd.DataFrame(
        {
            "QQQ": np.linspace(80.0, 120.0, len(index)),
            "SEMIS": np.linspace(70.0, 130.0, len(index)),
            "VIX": 18.0,
            "VIX3M": 20.0,
        },
        index=index,
    )
    original = build_entry_permission_signals(weights, closes, market)
    changed = market.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] = [1.0, 1.0]

    revised = build_entry_permission_signals(weights, closes, changed)

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        revised.loc[index[-1]],
    )


def test_unknown_signal_does_not_silently_release_damage_state() -> None:
    index = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame(
        [allocation(0.60), allocation(0.70)],
        index=index,
    )
    daily = pd.DataFrame(
        {"turnover": [1.0, 1.0], "slippage_cost": 0.0},
        index=index,
    )
    regime_signals = signals(index, semis_126=-0.05)
    regime_signals.loc[index[1], "semis_126d_return"] = np.nan
    opens, closes = flat_prices(index)

    adjusted, _, diagnostics = apply_entry_permission(
        weights,
        daily,
        opens,
        closes,
        regime_signals,
        EntryPermissionPolicy(damage_growth_budget=0.10),
    )
    growth = adjusted[list(("SPX", "QQQ", "SEMIS"))].sum(axis=1)

    assert np.isclose(growth.iloc[0], 0.10)
    assert np.isclose(growth.iloc[1], 0.10)
    assert bool(diagnostics.iloc[1]["damage_active"])
