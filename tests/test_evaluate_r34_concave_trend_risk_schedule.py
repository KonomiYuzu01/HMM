from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from tools.evaluate_r34_concave_trend_risk_schedule import (
    ACTIVE_MULTIPLIER,
    BASE_MULTIPLIER,
    CASH_FLOOR,
    _configure,
)
from tools.evaluate_r33_one_session_unlevered_shock_brake import (
    one_session_shock_schedule,
)


def _trending_inputs(
    gross: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.001, np.arange(len(index))),
            "SEMIS": 100.0
            * np.power(1.0012, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = gross * 0.40
    weights["SEMIS"] = gross * 0.40
    weights["GOLD"] = gross * 0.20
    weights["CASH"] = 1.0 - gross
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_low_base_risk_is_scaled_not_filled() -> None:
    _configure()
    closes, weights, daily = _trending_inputs(0.20)
    _, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    permitted = diagnostics["effective_incremental_permission"]
    expected = 0.20 * BASE_MULTIPLIER * ACTIVE_MULTIPLIER
    assert np.allclose(
        diagnostics.loc[
            permitted, "implemented_non_cash_weight"
        ],
        expected,
    )
    assert expected < 1.0 - CASH_FLOOR


def test_high_base_risk_saturates_at_120_percent() -> None:
    _configure()
    closes, weights, daily = _trending_inputs(1.00)
    implemented, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    permitted = diagnostics["effective_incremental_permission"]
    assert np.allclose(
        diagnostics.loc[
            permitted, "implemented_non_cash_weight"
        ],
        1.20,
    )
    assert implemented["CASH"].ge(CASH_FLOOR - 1e-12).all()


def test_current_close_cannot_change_current_open_risk() -> None:
    _configure()
    closes, weights, daily = _trending_inputs(0.80)
    before, _, _ = one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = one_session_shock_schedule(
        weights,
        daily,
        changed,
        cash_floor=CASH_FLOOR,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])
