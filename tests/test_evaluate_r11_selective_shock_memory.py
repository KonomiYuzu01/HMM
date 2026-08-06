from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r11_selective_shock_memory import (
    ShockMemoryCandidate,
    apply_selective_shock_memory,
    build_causal_signals,
    candidate_family,
    structural_condition,
)


def base_weights(index: pd.Index) -> pd.DataFrame:
    target = {
        "SPX": 0.00,
        "QQQ": 0.60,
        "SEMIS": 0.20,
        "BOND": 0.00,
        "GOLD": 0.20,
        "OIL": 0.00,
        "USD": 0.00,
        "CASH": 0.00,
        "VIX_HEDGE": 0.00,
    }
    return pd.DataFrame([target] * len(index), index=index)


def base_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "previous_growth_weight": [0.80] * len(index),
            "previous_weighted_growth_return": [0.0] * len(index),
            "qqq_63d_return": [-0.01] * len(index),
            "semis_63d_return": [-0.02] * len(index),
            "semis_126d_return": [-0.03] * len(index),
            "vix_term_ratio": [1.01] * len(index),
        },
        index=index,
    )


def test_candidate_family_is_fixed_and_unique() -> None:
    candidates = candidate_family()

    assert len(candidates) == 20
    assert len({candidate.name for candidate in candidates}) == 20


def test_market_features_use_only_previous_completed_close() -> None:
    index = pd.bdate_range("2024-01-01", periods=66)
    weights = base_weights(index)
    closes = pd.DataFrame(100.0, index=index, columns=ASSETS)
    market = pd.DataFrame(
        {
            "QQQ": np.linspace(90.0, 110.0, len(index)),
            "SEMIS": np.linspace(80.0, 120.0, len(index)),
            "VIX": 20.0,
            "VIX3M": 19.0,
        },
        index=index,
    )
    original = build_causal_signals(weights, closes, market)
    changed = market.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS", "VIX"]] = [
        10.0,
        5.0,
        100.0,
    ]

    revised = build_causal_signals(weights, closes, changed)

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        revised.loc[index[-1]],
    )


@pytest.mark.parametrize(
    ("gate_mode", "qqq", "semis", "expected"),
    [
        ("qqq_weak", -0.01, 0.02, True),
        ("qqq_weak", 0.01, -0.02, False),
        ("either_weak", 0.01, -0.02, True),
        ("both_weak", -0.01, 0.02, False),
        ("both_weak", -0.01, -0.02, True),
    ],
)
def test_structural_condition_gate_modes(
    gate_mode: str,
    qqq: float,
    semis: float,
    expected: bool,
) -> None:
    signal = pd.Series(
        {
            "qqq_63d_return": qqq,
            "semis_63d_return": semis,
            "semis_126d_return": -0.03,
            "vix_term_ratio": 1.01,
        }
    )

    assert structural_condition(signal, gate_mode) is expected


def test_contango_prevents_structural_trigger() -> None:
    signal = pd.Series(
        {
            "qqq_63d_return": -0.10,
            "semis_63d_return": -0.10,
            "semis_126d_return": -0.10,
            "vix_term_ratio": 0.99,
        }
    )

    assert not structural_condition(signal, "both_weak")


def test_semis_medium_break_separates_correction_from_trend_damage() -> None:
    correction = pd.Series(
        {
            "qqq_63d_return": -0.04,
            "semis_63d_return": -0.08,
            "semis_126d_return": 0.03,
            "vix_term_ratio": 1.10,
        }
    )
    trend_damage = correction.copy()
    trend_damage["semis_126d_return"] = -0.03

    assert not structural_condition(
        correction,
        "semis_medium_break",
    )
    assert structural_condition(
        trend_damage,
        "semis_medium_break",
    )


def test_release_requires_two_successful_thursdays() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    weights = base_weights(index)
    daily = pd.DataFrame(
        {
            "turnover": np.zeros(len(index)),
            "slippage_cost": np.zeros(len(index)),
        },
        index=index,
    )
    signals = base_signals(index)
    signals.loc[index[1], "previous_weighted_growth_return"] = -0.04
    signals.loc[index[2]:, "vix_term_ratio"] = 0.99

    implemented, adjusted_daily, diagnostics = (
        apply_selective_shock_memory(
            weights,
            daily,
            signals,
            ShockMemoryCandidate("both_weak", 0.50, "base_growth"),
        )
    )

    first_thursday = pd.Timestamp("2024-01-04")
    second_thursday = pd.Timestamp("2024-01-11")
    assert diagnostics.loc[index[1], "triggered"]
    assert diagnostics.loc[first_thursday, "recovery_confirmations"] == 1
    assert diagnostics.loc[first_thursday, "active"]
    assert diagnostics.loc[second_thursday, "released"]
    assert not diagnostics.loc[second_thursday, "active"]
    assert (
        implemented.loc[
            index[1]:first_thursday,
            ["QQQ", "SEMIS"],
        ].sum(axis=1)
        <= 0.50 + 1e-12
    ).all()
    assert adjusted_daily.loc[index[1], "turnover"] > 0.0
    assert adjusted_daily.loc[second_thursday, "turnover"] > 0.0


def test_failed_weekly_recovery_resets_confirmation() -> None:
    index = pd.bdate_range("2024-01-01", periods=17)
    weights = base_weights(index)
    daily = pd.DataFrame(
        {
            "turnover": np.zeros(len(index)),
            "slippage_cost": np.zeros(len(index)),
        },
        index=index,
    )
    signals = base_signals(index)
    signals.loc[index[1], "previous_weighted_growth_return"] = -0.04
    signals.loc[index[2]:, "vix_term_ratio"] = 0.99
    signals.loc[pd.Timestamp("2024-01-11"), "vix_term_ratio"] = 1.01

    _, _, diagnostics = apply_selective_shock_memory(
        weights,
        daily,
        signals,
        ShockMemoryCandidate("both_weak", 0.50, "base_growth"),
    )

    assert (
        diagnostics.loc[
            pd.Timestamp("2024-01-04"),
            "recovery_confirmations",
        ]
        == 1
    )
    assert (
        diagnostics.loc[
            pd.Timestamp("2024-01-11"),
            "recovery_confirmations",
        ]
        == 0
    )
    assert diagnostics.loc[
        pd.Timestamp("2024-01-18"),
        "recovery_confirmations",
    ] == 1
    assert diagnostics.loc[pd.Timestamp("2024-01-18"), "active"]


def test_trend_recovery_does_not_require_high_base_growth() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    weights = base_weights(index)
    weights.loc[:, ["QQQ", "SEMIS"]] *= 0.60
    weights.loc[:, "CASH"] = 0.32
    daily = pd.DataFrame(
        {
            "turnover": np.zeros(len(index)),
            "slippage_cost": np.zeros(len(index)),
        },
        index=index,
    )
    signals = base_signals(index)
    signals.loc[index[1], "previous_weighted_growth_return"] = -0.04
    signals.loc[index[2]:, "vix_term_ratio"] = 0.99
    signals.loc[index[2]:, "qqq_63d_return"] = 0.01
    signals.loc[index[2]:, "semis_63d_return"] = 0.02

    _, _, diagnostics = apply_selective_shock_memory(
        weights,
        daily,
        signals,
        ShockMemoryCandidate("both_weak", 0.35, "trend_recovery"),
    )

    assert diagnostics.loc[pd.Timestamp("2024-01-04"), "active"]
    assert diagnostics.loc[pd.Timestamp("2024-01-11"), "released"]


def test_reentry_brake_arms_without_selling_and_caps_weak_reentry() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    weights = base_weights(index)
    weights.loc[index[2], ["QQQ", "SEMIS"]] = [0.15, 0.05]
    weights.loc[index[2], "CASH"] = 0.60
    daily = pd.DataFrame(
        {
            "turnover": np.zeros(len(index)),
            "slippage_cost": np.zeros(len(index)),
        },
        index=index,
    )
    daily.loc[index[2], "turnover"] = 1.0
    first_thursday = pd.Timestamp("2024-01-04")
    daily.loc[first_thursday, "turnover"] = 1.0
    signals = base_signals(index)
    trigger_date = index[1]
    signals.loc[trigger_date, "previous_weighted_growth_return"] = -0.04

    implemented, _, diagnostics = apply_selective_shock_memory(
        weights,
        daily,
        signals,
        ShockMemoryCandidate("both_weak", 0.50, "reentry_brake"),
    )

    assert diagnostics.loc[trigger_date, "triggered"]
    assert not diagnostics.loc[trigger_date, "engaged"]
    assert (
        implemented.loc[
            trigger_date,
            ["QQQ", "SEMIS"],
        ].sum()
        == pytest.approx(0.80)
    )
    assert diagnostics.loc[first_thursday, "engaged"]
    assert (
        implemented.loc[
            first_thursday,
            ["QQQ", "SEMIS"],
        ].sum()
        == pytest.approx(0.50)
    )


def test_reentry_brake_releases_on_confirmed_trend_recovery() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    weights = base_weights(index)
    weights.loc[index[2], ["QQQ", "SEMIS"]] = [0.15, 0.05]
    weights.loc[index[2], "CASH"] = 0.60
    daily = pd.DataFrame(
        {
            "turnover": np.ones(len(index)),
            "slippage_cost": np.zeros(len(index)),
        },
        index=index,
    )
    signals = base_signals(index)
    signals.loc[index[1], "previous_weighted_growth_return"] = -0.04
    recovery_date = pd.Timestamp("2024-01-11")
    signals.loc[
        recovery_date,
        ["qqq_63d_return", "semis_63d_return", "vix_term_ratio"],
    ] = [0.01, 0.02, 0.99]

    implemented, _, diagnostics = apply_selective_shock_memory(
        weights,
        daily,
        signals,
        ShockMemoryCandidate("both_weak", 0.50, "reentry_brake"),
    )

    assert diagnostics.loc[pd.Timestamp("2024-01-04"), "engaged"]
    assert diagnostics.loc[recovery_date, "released"]
    assert not diagnostics.loc[recovery_date, "active"]
    assert (
        implemented.loc[
            recovery_date,
            ["QQQ", "SEMIS"],
        ].sum()
        == pytest.approx(0.80)
    )
