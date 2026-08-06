from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r12_volatility_managed_risk import BASE_MULTIPLIER
from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r30_trend_risk_budget_pulse import (
    effective_incremental_permission,
    maximum_true_run,
    requested_fill_multiplier,
    risk_budget_schedule,
)


def test_requested_multiplier_fills_declared_cash_budget() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame(
        {
            "QQQ": [0.50, 0.40, 0.00],
            "SEMIS": [0.50, 0.40, 0.00],
            "CASH": [0.00, 0.20, 1.00],
        },
        index=index,
    )
    multiplier = requested_fill_multiplier(
        weights,
        cash_floor=-0.25,
    )
    assert np.isclose(multiplier.iloc[0], 1.25)
    assert np.isclose(multiplier.iloc[1], 1.5625)
    assert np.isclose(multiplier.iloc[2], BASE_MULTIPLIER)


def test_pulse_only_revokes_incremental_permission() -> None:
    index = pd.bdate_range("2024-01-02", periods=5)
    trend = pd.Series([True, True, False, True, True], index=index)
    pulse = pd.Series([False, True, False, False, True], index=index)
    effective = effective_incremental_permission(trend, pulse)
    assert effective.tolist() == [True, False, False, True, False]


def test_maximum_true_run_counts_consecutive_sessions() -> None:
    values = pd.Series(
        [False, True, True, False, True, True, True, True, False]
    )
    assert maximum_true_run(values) == 4


def test_falling_market_cannot_add_risk_above_r11() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    qqq = 200.0 * np.power(0.999, np.arange(len(index)))
    semis = 250.0 * np.power(0.998, np.arange(len(index)))
    closes = pd.DataFrame(
        {"QQQ": qqq, "SEMIS": semis},
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    weights["CASH"] = 0.00
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    implemented, _, diagnostics = risk_budget_schedule(
        weights,
        daily,
        closes,
    )
    assert not diagnostics["trend_permission"].any()
    assert np.allclose(
        diagnostics["implemented_non_cash_weight"],
        diagnostics["base_r11_non_cash_weight"],
    )
    assert implemented["CASH"].ge(-0.25 - 1e-12).all()


def test_current_close_cannot_change_current_open_budget() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    rng = np.random.default_rng(30)
    qqq_returns = rng.normal(0.0005, 0.012, len(index))
    semis_returns = rng.normal(0.0006, 0.018, len(index))
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.cumprod(1.0 + qqq_returns),
            "SEMIS": 100.0 * np.cumprod(1.0 + semis_returns),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    weights["CASH"] = 0.00
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    before, _, before_diag = risk_budget_schedule(
        weights,
        daily,
        closes,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, after_diag = risk_budget_schedule(
        weights,
        daily,
        changed,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])
    assert (
        before_diag.iloc[-1]["effective_incremental_permission"]
        == after_diag.iloc[-1]["effective_incremental_permission"]
    )
