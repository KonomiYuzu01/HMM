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
from tools.evaluate_r37_bounded_semiconductor_tilt import (
    CASH_FLOOR,
    MAX_TILT,
    _configure,
)
import tools.evaluate_r23_high_volatility_tilt_gate as r23
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=1_500)
    qqq = np.full(len(index), 0.0003)
    semis = np.full(len(index), 0.0008)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.cumprod(1.0 + qqq),
            "SEMIS": 100.0 * np.cumprod(1.0 + semis),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_bounded_tilt_preserves_growth_budget() -> None:
    _configure()
    closes, weights, daily = _inputs()
    scheduled, execution, _ = r33.one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    tilted, _, diagnostics = r23.apply_gated_industry_momentum(
        scheduled,
        execution,
        closes,
        max_tilt=MAX_TILT,
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert diagnostics["semis_growth_share"].between(
        0.45 - 1e-12,
        0.55 + 1e-12,
    ).all()
    assert np.allclose(
        tilted["QQQ"] + tilted["SEMIS"],
        scheduled["QQQ"] + scheduled["SEMIS"],
    )


def test_current_close_cannot_change_current_open_tilt() -> None:
    _configure()
    closes, weights, daily = _inputs()
    scheduled, execution, _ = r33.one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    before, _, _ = r23.apply_gated_industry_momentum(
        scheduled,
        execution,
        closes,
        max_tilt=MAX_TILT,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = r23.apply_gated_industry_momentum(
        scheduled,
        execution,
        changed,
        max_tilt=MAX_TILT,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])
