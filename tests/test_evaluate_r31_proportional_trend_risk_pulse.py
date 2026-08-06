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
from evaluate_r12_volatility_managed_risk import BASE_MULTIPLIER
from evaluate_r31_proportional_trend_risk_pulse import (
    ACTIVE_MULTIPLIER,
    causal_absolute_recovery_pulse,
    proportional_risk_schedule,
)


def test_pair_recovery_uses_absolute_not_relative_return() -> None:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(31)
    common = rng.normal(0.0004, 0.012, len(index))
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.cumprod(1.0 + common),
            "SEMIS": 100.0
            * np.cumprod(1.0 + common + rng.normal(0.0, 0.003, len(index))),
        },
        index=index,
    )
    diagnostics = causal_absolute_recovery_pulse(closes, index)
    expected = (
        closes.pct_change(fill_method=None).mean(axis=1).shift(1)
    )
    assert np.allclose(
        diagnostics["prior_pair_return"].dropna(),
        expected.reindex(index).dropna(),
    )


def test_proportional_increment_cannot_fill_low_base_risk() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.001, np.arange(len(index))),
            "SEMIS": 100.0 * np.power(1.0012, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.008
    weights["SEMIS"] = 0.008
    weights["CASH"] = 0.984
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    implemented, _, diagnostics = proportional_risk_schedule(
        weights,
        daily,
        closes,
        pulse_enabled=False,
    )
    permitted = diagnostics["effective_incremental_permission"]
    expected_gross = (
        0.016 * BASE_MULTIPLIER * ACTIVE_MULTIPLIER
    )
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    assert np.allclose(
        implemented.loc[permitted, non_cash].sum(axis=1),
        expected_gross,
    )
    assert implemented.loc[permitted, "CASH"].min() > 0.95


def test_trend_off_returns_exactly_to_r11() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(0.999, np.arange(len(index))),
            "SEMIS": 100.0 * np.power(0.998, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    _, _, diagnostics = proportional_risk_schedule(
        weights,
        daily,
        closes,
    )
    assert not diagnostics["trend_permission"].any()
    assert np.allclose(
        diagnostics["implemented_non_cash_weight"],
        diagnostics["base_r11_non_cash_weight"],
    )


def test_current_close_cannot_change_current_open_risk() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_300)
    rng = np.random.default_rng(310)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.cumprod(1.0 + rng.normal(0.0005, 0.012, len(index))),
            "SEMIS": 100.0
            * np.cumprod(1.0 + rng.normal(0.0006, 0.018, len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    before, _, _ = proportional_risk_schedule(weights, daily, closes)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = proportional_risk_schedule(weights, daily, changed)
    assert np.allclose(before.iloc[-1], after.iloc[-1])
