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
from evaluate_r33_one_session_unlevered_shock_brake import (
    BASE_MULTIPLIER,
    CASH_FLOOR,
    SHOCK_MULTIPLIER,
    _configure,
    causal_one_session_shock,
    one_session_shock_schedule,
)
import evaluate_r33_one_session_unlevered_shock_brake as r33


def _inputs(
    periods: int = 1_400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=periods)
    rng = np.random.default_rng(33)
    returns = rng.normal(0.0005, 0.005, periods)
    returns[1_200:1_220] = rng.normal(-0.003, 0.05, 20)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.cumprod(1.0 + returns),
            "SEMIS": 100.0
            * np.cumprod(
                1.0 + returns + rng.normal(0.0, 0.002, periods)
            ),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_shock_state_last_at_most_one_session() -> None:
    closes, _, _ = _inputs()
    diagnostics = causal_one_session_shock(
        closes,
        closes.index,
    )
    state = diagnostics["pulse_veto_active"].astype(bool)
    groups = state.ne(state.shift()).cumsum()
    assert int(state.groupby(groups).sum().max()) <= 1
    assert state.any()


def test_shock_day_uses_unlevered_absolute_multiplier() -> None:
    _configure()
    closes, weights, daily = _inputs()
    _, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
    )
    shock = diagnostics["pulse_veto_active"].astype(bool)
    assert shock.any()
    assert np.allclose(
        diagnostics.loc[
            shock, "requested_absolute_multiplier"
        ],
        SHOCK_MULTIPLIER,
    )


def test_nontrend_nonshock_returns_to_base_multiplier() -> None:
    _configure()
    index = pd.bdate_range("2018-01-02", periods=1_300)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(0.999, np.arange(len(index))),
            "SEMIS": 100.0
            * np.power(0.998, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    _, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
    )
    selected = (
        ~diagnostics["trend_permission"].astype(bool)
        & ~diagnostics["pulse_veto_active"].astype(bool)
    )
    assert np.allclose(
        diagnostics.loc[
            selected, "requested_absolute_multiplier"
        ],
        BASE_MULTIPLIER,
    )


def test_current_close_cannot_change_current_open_risk() -> None:
    _configure()
    closes, weights, daily = _inputs()
    before, _, _ = one_session_shock_schedule(
        weights,
        daily,
        closes,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = one_session_shock_schedule(
        weights,
        daily,
        changed,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])


def test_cash_floor_and_state_semantics_hold() -> None:
    _configure()
    closes, weights, daily = _inputs()
    implemented, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
    )
    assert implemented["CASH"].ge(CASH_FLOOR - 1e-12).all()
    assert diagnostics["state_multiplier_semantic_error"].le(1e-12).all()


def test_configure_changes_all_schedule_multipliers() -> None:
    try:
        _configure(
            base_multiplier=1.075,
            active_multiplier=1.25,
            shock_multiplier=1.05,
        )
        assert r33.BASE_MULTIPLIER == 1.075
        assert r33.ACTIVE_MULTIPLIER == 1.25
        assert r33.SHOCK_MULTIPLIER == 1.05
    finally:
        _configure(
            base_multiplier=BASE_MULTIPLIER,
            active_multiplier=1.20,
            shock_multiplier=SHOCK_MULTIPLIER,
        )
