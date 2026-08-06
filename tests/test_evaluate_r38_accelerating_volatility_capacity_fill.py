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
from tools.evaluate_r38_accelerating_volatility_capacity_fill import (
    accelerating_volatility_schedule,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2016-01-04", periods=1_200)
    qqq_returns = np.full(len(index), 0.0004)
    semis_returns = np.full(len(index), 0.0006)
    semis_returns[-80:] = np.resize(
        np.array([0.025, -0.022]),
        80,
    )
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.exp(np.cumsum(qqq_returns)),
            "SEMIS": 100.0 * np.exp(np.cumsum(semis_returns)),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_current_close_cannot_change_current_open_target() -> None:
    closes, weights, daily = _inputs()
    before, _, _ = accelerating_volatility_schedule(
        weights,
        daily,
        closes,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = accelerating_volatility_schedule(
        weights,
        daily,
        changed,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])


def test_accelerating_volatility_blocks_only_extra_capacity() -> None:
    closes, weights, daily = _inputs()
    _, _, diagnostics = accelerating_volatility_schedule(
        weights,
        daily,
        closes,
    )
    blocked = diagnostics["volatility_acceleration_block"]
    assert blocked.any()
    assert diagnostics.loc[
        blocked, "state_active_multiplier"
    ].eq(1.30).all()
    assert diagnostics.loc[
        ~blocked, "state_active_multiplier"
    ].eq(1.35).all()


def test_cash_and_non_cash_limits_hold() -> None:
    closes, weights, daily = _inputs()
    adjusted, _, diagnostics = accelerating_volatility_schedule(
        weights,
        daily,
        closes,
    )
    assert adjusted["CASH"].ge(-0.20 - 1e-12).all()
    assert diagnostics["implemented_non_cash_weight"].le(
        1.20 + 1e-12
    ).all()

