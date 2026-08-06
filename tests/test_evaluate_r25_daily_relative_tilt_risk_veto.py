from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r25_daily_relative_tilt_risk_veto import (
    apply_daily_gated_industry_momentum,
    causal_daily_gated_industry_momentum,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(25)
    common = rng.normal(0.0004, 0.008, len(index))
    common[-100:] = rng.normal(-0.0002, 0.045, 100)
    qqq = 100.0 * np.cumprod(
        1.0 + common + rng.normal(0.0, 0.002, len(index))
    )
    semis = 100.0 * np.cumprod(
        1.0 + common + rng.normal(0.0003, 0.004, len(index))
    )
    closes = pd.DataFrame({"QQQ": qqq, "SEMIS": semis}, index=index)
    weights = pd.DataFrame(
        {"QQQ": 0.45, "SEMIS": 0.45, "CASH": 0.10},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily, closes


def test_current_close_cannot_change_current_daily_gate() -> None:
    _, _, closes = _inputs()
    before = causal_daily_gated_industry_momentum(
        closes,
        closes.index,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_daily_gated_industry_momentum(
        changed,
        changed.index,
    )
    assert np.isclose(
        before.iloc[-1]["semis_growth_share"],
        after.iloc[-1]["semis_growth_share"],
    )
    assert (
        before.iloc[-1]["volatility_state"]
        == after.iloc[-1]["volatility_state"]
    )


def test_high_volatility_days_are_neutral_between_monthly_updates() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_daily_gated_industry_momentum(
        closes,
        closes.index,
    )
    active_non_update = (
        diagnostics["volatility_gate_active"]
        & ~diagnostics["momentum_update"]
    )
    assert active_non_update.any()
    assert np.allclose(
        diagnostics.loc[active_non_update, "semis_growth_share"],
        0.50,
    )


def test_inactive_daily_gate_preserves_monthly_signal() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_daily_gated_industry_momentum(
        closes,
        closes.index,
    )
    inactive = ~diagnostics["volatility_gate_active"]
    assert inactive.any()
    assert np.allclose(
        diagnostics.loc[inactive, "semis_growth_share"],
        diagnostics.loc[inactive, "ungated_semis_growth_share"],
    )


def test_gate_transition_creates_execution_update() -> None:
    weights, daily, closes = _inputs()
    _, updated, diagnostics = apply_daily_gated_industry_momentum(
        weights,
        daily,
        closes,
    )
    transitions = diagnostics["volatility_gate_change"]
    assert transitions.any()
    assert updated.loc[transitions, "turnover"].gt(0.0).all()


def test_daily_gate_preserves_growth_budget() -> None:
    weights, daily, closes = _inputs()
    adjusted, _, diagnostics = apply_daily_gated_industry_momentum(
        weights,
        daily,
        closes,
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert np.allclose(adjusted.sum(axis=1), weights.sum(axis=1))
