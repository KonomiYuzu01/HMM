from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r11_post_shock_cooldown import (
    CooldownCandidate,
    apply_post_shock_cooldown,
    candidate_family,
    cap_growth_to_cash,
    causal_growth_shock_signals,
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


def test_candidate_family_is_fixed_and_unique() -> None:
    candidates = candidate_family()

    assert len(candidates) == 27
    assert len({candidate.name for candidate in candidates}) == 27


def test_growth_shock_signal_uses_only_completed_session() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    weights = base_weights(index)
    closes = pd.DataFrame(100.0, index=index, columns=ASSETS)
    original = causal_growth_shock_signals(weights, closes)
    changed = closes.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] = [50.0, 40.0]

    revised = causal_growth_shock_signals(weights, changed)

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        revised.loc[index[-1]],
    )


def test_growth_cap_preserves_mix_and_account_weight() -> None:
    weights = base_weights(pd.Index([0])).iloc[0]

    adjusted = cap_growth_to_cash(weights, 0.20)

    assert adjusted["QQQ"] == pytest.approx(0.15)
    assert adjusted["SEMIS"] == pytest.approx(0.05)
    assert adjusted["GOLD"] == pytest.approx(0.20)
    assert adjusted["CASH"] == pytest.approx(0.60)
    assert adjusted.sum() == pytest.approx(1.0)


def test_cooldown_starts_next_day_and_holds_exact_sessions() -> None:
    index = pd.bdate_range("2024-01-01", periods=7)
    weights = base_weights(index)
    daily = pd.DataFrame(
        {
            "turnover": [1.0] + [0.0] * 6,
            "slippage_cost": np.zeros(7),
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "previous_growth_weight": [
                np.nan,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
                0.80,
            ],
            "previous_weighted_growth_return": [
                np.nan,
                -0.04,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
            ],
        },
        index=index,
    )

    implemented, adjusted_daily, diagnostics = (
        apply_post_shock_cooldown(
            weights,
            daily,
            signals,
            CooldownCandidate(-0.03, 5, 0.20),
        )
    )

    assert diagnostics["triggered"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert diagnostics["active"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert implemented.loc[index[1:6], ["QQQ", "SEMIS"]].sum(
        axis=1
    ).tolist() == pytest.approx([0.20] * 5)
    assert adjusted_daily.loc[index[1], "turnover"] > 0.0
    assert (
        adjusted_daily.loc[index[2:6], "turnover"] == 0.0
    ).all()
    assert diagnostics.loc[index[-1], "released"]
    assert adjusted_daily.loc[index[-1], "turnover"] > 0.0


def test_new_shock_resets_remaining_cooldown() -> None:
    index = pd.bdate_range("2024-01-01", periods=8)
    weights = base_weights(index)
    daily = pd.DataFrame(
        {
            "turnover": np.zeros(8),
            "slippage_cost": np.zeros(8),
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "previous_growth_weight": [np.nan] + [0.80] * 7,
            "previous_weighted_growth_return": [
                np.nan,
                -0.04,
                0.01,
                0.01,
                -0.05,
                0.01,
                0.01,
                0.01,
            ],
        },
        index=index,
    )

    _, _, diagnostics = apply_post_shock_cooldown(
        weights,
        daily,
        signals,
        CooldownCandidate(-0.03, 3, 0.20),
    )

    assert diagnostics.loc[index[1], "triggered"]
    assert diagnostics.loc[index[4], "triggered"]
    assert diagnostics.loc[index[6], "active"]
    assert diagnostics.loc[index[7], "released"]
