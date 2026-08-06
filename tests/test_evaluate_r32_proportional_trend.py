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
from evaluate_r32_proportional_trend import (
    ACTIVE_MULTIPLIER,
    BASE_MULTIPLIER,
    CASH_FLOOR,
    _configure,
    proportional_trend_schedule,
)


def _inputs(
    *,
    growth: float,
    periods: int = 1_300,
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2018-01-02", periods=periods)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.0 + growth, np.arange(periods)),
            "SEMIS": 100.0
            * np.power(1.0 + growth * 1.1, np.arange(periods)),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return index, closes, weights, daily


def test_r32_never_uses_pulse_veto() -> None:
    _configure()
    _, closes, weights, daily = _inputs(growth=0.001)
    _, _, diagnostics = proportional_trend_schedule(
        weights,
        daily,
        closes,
        pulse_enabled=True,
    )
    assert not diagnostics["pulse_veto_active"].astype(bool).any()


def test_r32_trend_off_returns_to_base1070() -> None:
    _configure()
    _, closes, weights, daily = _inputs(growth=-0.001)
    implemented, _, diagnostics = proportional_trend_schedule(
        weights,
        daily,
        closes,
    )
    assert not diagnostics["trend_permission"].any()
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    assert np.allclose(
        implemented[non_cash].sum(axis=1),
        BASE_MULTIPLIER,
    )


def test_r32_trend_on_is_proportional_and_cash_capped() -> None:
    _configure()
    _, closes, weights, daily = _inputs(growth=0.001)
    implemented, _, diagnostics = proportional_trend_schedule(
        weights,
        daily,
        closes,
    )
    permitted = diagnostics["effective_incremental_permission"]
    expected = min(
        BASE_MULTIPLIER * ACTIVE_MULTIPLIER,
        1.0 - CASH_FLOOR,
    )
    assert np.allclose(
        diagnostics.loc[permitted, "implemented_non_cash_weight"],
        expected,
    )
    assert implemented["CASH"].ge(CASH_FLOOR - 1e-12).all()


def test_current_close_cannot_change_current_open_risk() -> None:
    _configure()
    _, closes, weights, daily = _inputs(growth=0.001)
    before, _, _ = proportional_trend_schedule(
        weights,
        daily,
        closes,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = proportional_trend_schedule(
        weights,
        daily,
        changed,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])
