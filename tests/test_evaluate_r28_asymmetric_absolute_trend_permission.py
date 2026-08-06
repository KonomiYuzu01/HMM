from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r28_asymmetric_absolute_trend_permission import (
    apply_asymmetric_absolute_industry_momentum,
    asymmetric_absolute_permission,
    causal_asymmetric_absolute_industry_momentum,
)


def test_negative_trend_revokes_between_scheduled_updates() -> None:
    index = pd.bdate_range("2025-01-02", periods=6)
    excess = pd.Series(
        [0.10, 0.08, -0.01, 0.02, 0.03, 0.04],
        index=index,
    )
    scheduled = pd.Series(
        [True, False, False, False, False, True],
        index=index,
    )
    state = asymmetric_absolute_permission(excess, scheduled)
    assert state["absolute_trend_permission"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    assert state.iloc[2]["absolute_trend_permission_revoked"]


def test_positive_trend_cannot_reenter_before_scheduled_update() -> None:
    index = pd.bdate_range("2025-01-02", periods=5)
    excess = pd.Series(
        [-0.01, 0.02, 0.03, 0.04, 0.05],
        index=index,
    )
    scheduled = pd.Series(
        [True, False, False, False, True],
        index=index,
    )
    state = asymmetric_absolute_permission(excess, scheduled)
    assert state["absolute_trend_permission"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(28)
    common = rng.normal(0.0005, 0.010, len(index))
    common[-120:] = rng.normal(-0.001, 0.012, 120)
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
            "CASH": 100.0
            * np.cumprod(np.full(len(index), 1.0 + 0.00015)),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {"QQQ": 0.45, "SEMIS": 0.45, "CASH": 0.10},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily, closes


def test_current_close_cannot_change_current_asymmetric_target() -> None:
    _, _, closes = _inputs()
    before = causal_asymmetric_absolute_industry_momentum(
        closes,
        closes.index,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_asymmetric_absolute_industry_momentum(
        changed,
        changed.index,
    )
    assert np.isclose(
        before.iloc[-1]["semis_growth_share"],
        after.iloc[-1]["semis_growth_share"],
    )
    assert (
        before.iloc[-1]["absolute_trend_permission"]
        == after.iloc[-1]["absolute_trend_permission"]
    )


def test_asymmetric_permission_preserves_growth_budget() -> None:
    weights, daily, closes = _inputs()
    adjusted, _, diagnostics = (
        apply_asymmetric_absolute_industry_momentum(
            weights,
            daily,
            closes,
        )
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert np.allclose(adjusted.sum(axis=1), weights.sum(axis=1))
