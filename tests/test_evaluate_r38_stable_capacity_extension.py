from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from tools.evaluate_r10_gde_capital_efficiency import ASSETS
import tools.evaluate_r38_stable_capacity_extension as stable


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2016-01-04", periods=1_200)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.exp(np.cumsum(np.full(len(index), 0.0004))),
            "SEMIS": 100.0
            * np.exp(np.cumsum(np.full(len(index), 0.0006))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.50
    weights["SEMIS"] = 0.45
    weights["GOLD"] = 0.25
    weights["CASH"] = -0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_only_stable_state_can_use_extra_capacity(monkeypatch) -> None:
    closes, weights, daily = _inputs()
    index = closes.index
    eligible = pd.Series(
        np.resize(np.array([True, False]), len(index)),
        index=index,
    )

    def fake_schedule(*args, **kwargs):
        del args, kwargs
        implemented = weights.copy()
        implemented["QQQ"] *= 1.025
        implemented["SEMIS"] *= 1.025
        implemented["GOLD"] *= 1.025
        implemented["CASH"] = 1.0 - implemented[
            ["QQQ", "SEMIS", "GOLD"]
        ].sum(axis=1)
        diagnostics = pd.DataFrame(
            {
                "effective_incremental_permission": eligible,
                "implemented_cash_weight": implemented["CASH"],
                "implemented_non_cash_weight": implemented[
                    ["QQQ", "SEMIS", "GOLD"]
                ].sum(axis=1),
                "state_multiplier_semantic_error": 0.0,
            },
            index=index,
        )
        return implemented, daily.copy(), diagnostics

    monkeypatch.setattr(
        stable.accel,
        "accelerating_volatility_schedule",
        fake_schedule,
    )
    adjusted, _, diagnostics = stable.stable_capacity_schedule(
        weights,
        daily,
        closes,
        stable_cash_floor=-0.23,
    )
    non_cash = adjusted.drop(columns="CASH").sum(axis=1)
    assert non_cash.loc[eligible].le(1.23 + 1e-12).all()
    assert non_cash.loc[~eligible].le(1.20 + 1e-12).all()
    assert diagnostics.loc[
        eligible, "state_cash_floor"
    ].eq(-0.23).all()
    assert diagnostics.loc[
        ~eligible, "state_cash_floor"
    ].eq(-0.20).all()


def test_current_close_cannot_change_current_open_target() -> None:
    closes, weights, daily = _inputs()
    before, _, _ = stable.stable_capacity_schedule(
        weights,
        daily,
        closes,
        stable_cash_floor=-0.23,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = stable.stable_capacity_schedule(
        weights,
        daily,
        changed,
        stable_cash_floor=-0.23,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])


def test_rollout_returns_are_daily_account_blend() -> None:
    index = pd.bdate_range("2026-01-05", periods=2)
    r11 = pd.Series([0.01, -0.02], index=index)
    candidate = pd.Series([0.03, 0.02], index=index)
    result = stable.rollout_returns(r11, candidate, 0.50)
    assert np.allclose(result, [0.02, 0.0])


def test_selector_uses_smallest_passing_capacity() -> None:
    rows = []
    for floor in stable.STABLE_CASH_FLOORS:
        rows.append(
            {
                "stable_cash_floor": floor,
                "target_cagr_pass": floor <= -0.23,
                "drawdown_pass": True,
            }
        )
    selected = stable.select_smallest_passing_floor(
        pd.DataFrame(rows)
    )
    assert selected == -0.23
