from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r12_gold_crash_guard import (
    GoldCrashCandidate,
    candidate_family,
    causal_gold_crash_signals,
    select_candidate,
    selected_cap_fraction,
    simulate_gold_crash_guard,
)


def test_candidate_family_is_small_and_unique() -> None:
    candidates = candidate_family()

    assert len(candidates) == 9
    assert len({candidate.name for candidate in candidates}) == 9


def test_gold_crash_signal_ignores_same_day_data() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    closes = pd.DataFrame(
        {
            "GOLD": np.linspace(100.0, 90.0, len(index)),
            "CASH": np.linspace(100.0, 101.0, len(index)),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "DFII10": np.linspace(1.0, 2.0, len(index)),
            "DTWEXBGS": np.linspace(100.0, 110.0, len(index)),
            "T10YIE": np.linspace(2.0, 2.2, len(index)),
        },
        index=index,
    )
    original = causal_gold_crash_signals(closes, macro)
    changed_closes = closes.copy()
    changed_macro = macro.copy()
    changed_closes.loc[index[-1], "GOLD"] = 1_000.0
    changed_macro.loc[index[-1], "DFII10"] = -99.0
    changed_macro.loc[index[-1], "DTWEXBGS"] = 1.0

    changed = causal_gold_crash_signals(
        changed_closes,
        changed_macro,
    )

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        changed.loc[index[-1]],
    )


def test_macro_headwinds_scale_protection_depth() -> None:
    assert selected_cap_fraction(0.25, 0.0) == pytest.approx(1.0)
    assert selected_cap_fraction(0.25, 1.0) == pytest.approx(0.625)
    assert selected_cap_fraction(0.25, 2.0) == pytest.approx(0.25)


def simple_execution_inputs(
    periods: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2024-01-01", periods=periods)
    target = {
        "SPX": 0.10,
        "QQQ": 0.30,
        "SEMIS": 0.30,
        "BOND": 0.00,
        "GOLD": 0.20,
        "OIL": 0.00,
        "USD": 0.00,
        "CASH": 0.10,
        "VIX_HEDGE": 0.00,
    }
    weights = pd.DataFrame([target] * periods, index=index)
    daily = pd.DataFrame(
        {
            "turnover": [1.0] + [0.0] * (periods - 1),
            "slippage_cost": np.zeros(periods),
        },
        index=index,
    )
    opens = pd.DataFrame(100.0, index=index, columns=ASSETS)
    closes = opens.copy()
    return weights, daily, opens, closes


def test_no_op_reconstructs_weights_and_zero_returns() -> None:
    weights, daily, opens, closes = simple_execution_inputs()
    signals = pd.DataFrame(
        {
            "drawdown_from_high": -0.20,
            "fast_excess_trend": -0.10,
            "macro_headwinds": 2.0,
        },
        index=weights.index,
    )

    result, implemented = simulate_gold_crash_guard(
        weights,
        daily,
        opens,
        closes,
        signals,
        None,
        cost_bps=0.0,
        emergency_slippage_bps=0.0,
        start_date="2024-01-01",
    )

    pd.testing.assert_frame_equal(implemented, weights)
    assert result["net_return"].tolist() == pytest.approx([0.0] * 4)


def test_guard_triggers_and_reenters_after_two_confirmations() -> None:
    weights, daily, opens, closes = simple_execution_inputs()
    signals = pd.DataFrame(
        {
            "drawdown_from_high": [-0.20, -0.20, -0.05, -0.04],
            "fast_excess_trend": [-0.10, -0.05, 0.01, 0.02],
            "macro_headwinds": [2.0, 2.0, 2.0, 2.0],
        },
        index=weights.index,
    )

    result, implemented = simulate_gold_crash_guard(
        weights,
        daily,
        opens,
        closes,
        signals,
        GoldCrashCandidate(-0.12, 0.25),
        cost_bps=0.0,
        emergency_slippage_bps=0.0,
        start_date="2024-01-01",
    )

    assert result["triggered"].tolist() == [True, False, False, False]
    assert result["recovered"].tolist() == [False, False, False, True]
    assert implemented["GOLD"].tolist() == pytest.approx(
        [0.05, 0.05, 0.05, 0.20]
    )
    assert implemented["CASH"].tolist() == pytest.approx(
        [0.25, 0.25, 0.25, 0.10]
    )


def test_guard_does_not_reenter_on_small_bounce() -> None:
    weights, daily, opens, closes = simple_execution_inputs()
    signals = pd.DataFrame(
        {
            "drawdown_from_high": [-0.20, -0.10, -0.09, -0.08],
            "fast_excess_trend": [-0.10, 0.02, 0.03, 0.04],
            "macro_headwinds": [2.0, 2.0, 2.0, 2.0],
        },
        index=weights.index,
    )

    result, implemented = simulate_gold_crash_guard(
        weights,
        daily,
        opens,
        closes,
        signals,
        GoldCrashCandidate(-0.12, 0.25),
        cost_bps=0.0,
        emergency_slippage_bps=0.0,
        start_date="2024-01-01",
    )

    assert not result["recovered"].any()
    assert implemented["GOLD"].tolist() == pytest.approx([0.05] * 4)


def test_base_defense_trade_releases_gold_overlay() -> None:
    weights, daily, opens, closes = simple_execution_inputs()
    weights.loc[weights.index[2], ["SPX", "QQQ", "SEMIS"]] = [
        0.05,
        0.10,
        0.10,
    ]
    weights.loc[weights.index[2], "CASH"] = 0.55
    weights.loc[weights.index[3]] = weights.loc[weights.index[2]]
    daily.loc[weights.index[2], "turnover"] = 0.50
    signals = pd.DataFrame(
        {
            "drawdown_from_high": [-0.20, -0.20, -0.20, -0.20],
            "fast_excess_trend": [-0.10, -0.05, -0.04, -0.03],
            "macro_headwinds": [2.0, 2.0, 2.0, 2.0],
        },
        index=weights.index,
    )

    result, implemented = simulate_gold_crash_guard(
        weights,
        daily,
        opens,
        closes,
        signals,
        GoldCrashCandidate(-0.12, 0.25),
        cost_bps=0.0,
        emergency_slippage_bps=0.0,
        start_date="2024-01-01",
    )

    assert result["released_for_defense"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert implemented.loc[weights.index[2], "GOLD"] == pytest.approx(0.20)


def test_selection_accepts_a_no_action_period_if_another_period_improves() -> None:
    rows = []
    for candidate, deltas in {
        "dd16_floor25": [0.0, 0.001],
        "dd16_floor50": [-0.001, 0.002],
    }.items():
        for (sample, period), delta in zip(
            [
                ("normal_synthetic", "development_2015_2021"),
                ("proxy_synthetic", "early_2007_2014"),
            ],
            deltas,
            strict=True,
        ):
            rows.append(
                {
                    "sample": sample,
                    "period": period,
                    "scenario": "current_liquidity",
                    "candidate": candidate,
                    "cagr_delta": delta,
                    "sharpe_delta": delta,
                    "max_drawdown_delta": 0.0,
                }
            )

    selected, ranking = select_candidate(pd.DataFrame(rows))

    assert selected == "dd16_floor25"
    assert bool(
        ranking.loc[
            ranking["candidate"].eq("dd16_floor25"),
            "selection_gate",
        ].iloc[0]
    )
