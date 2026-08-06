from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r26_pulse_relative_tilt_risk_veto import (
    apply_pulse_gated_industry_momentum,
    causal_pulse_gated_industry_momentum,
    pulse_cooldown_state,
)


def test_pulse_expires_after_four_sessions() -> None:
    index = pd.bdate_range("2025-01-02", periods=8)
    high = pd.Series(
        [False, True, True, True, True, True, True, False],
        index=index,
    )
    relative = pd.Series(-0.01, index=index)
    state = pulse_cooldown_state(high, relative)
    assert state["pulse_veto_active"].sum() == 4
    assert state["pulse_veto_expired"].sum() == 1


def test_two_positive_confirmations_release_early() -> None:
    index = pd.bdate_range("2025-01-02", periods=6)
    high = pd.Series(
        [False, True, True, True, True, False],
        index=index,
    )
    relative = pd.Series(
        [0.0, -0.01, 0.01, 0.02, 0.01, 0.0],
        index=index,
    )
    state = pulse_cooldown_state(high, relative)
    assert state["pulse_veto_active"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert state["pulse_veto_recovered"].sum() == 1


def test_continuous_high_episode_cannot_rearm_after_expiry() -> None:
    index = pd.bdate_range("2025-01-02", periods=10)
    high = pd.Series(
        [False] + [True] * 8 + [False],
        index=index,
    )
    relative = pd.Series(-0.01, index=index)
    state = pulse_cooldown_state(high, relative)
    assert state["high_volatility_episode_entry"].sum() == 1
    assert state["pulse_veto_active"].sum() == 4


def test_exit_then_new_high_episode_can_rearm() -> None:
    index = pd.bdate_range("2025-01-02", periods=12)
    high = pd.Series(
        [False, True, True, True, True, True, False, True, True, True, True, False],
        index=index,
    )
    relative = pd.Series(-0.01, index=index)
    state = pulse_cooldown_state(high, relative)
    assert state["high_volatility_episode_entry"].sum() == 2
    assert state["pulse_veto_active"].sum() == 8


def test_pulse_application_preserves_growth_budget() -> None:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(26)
    common = rng.normal(0.0004, 0.008, len(index))
    common[-100:] = rng.normal(-0.0002, 0.045, 100)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0, 0.002, len(index))
            ),
            "SEMIS": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0003, 0.004, len(index))
            ),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {"QQQ": 0.45, "SEMIS": 0.45, "CASH": 0.10},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    adjusted, _, diagnostics = apply_pulse_gated_industry_momentum(
        weights,
        daily,
        closes,
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert np.allclose(adjusted.sum(axis=1), weights.sum(axis=1))


def test_current_close_cannot_change_current_pulse_target() -> None:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(260)
    common = rng.normal(0.0004, 0.012, len(index))
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0, 0.002, len(index))
            ),
            "SEMIS": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0003, 0.004, len(index))
            ),
        },
        index=index,
    )
    before = causal_pulse_gated_industry_momentum(closes, index)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_pulse_gated_industry_momentum(changed, index)
    assert np.isclose(
        before.iloc[-1]["semis_growth_share"],
        after.iloc[-1]["semis_growth_share"],
    )
