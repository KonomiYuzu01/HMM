from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r27_absolute_trend_permission import (
    apply_absolute_permitted_industry_momentum,
    causal_absolute_permitted_industry_momentum,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(27)
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


def test_negative_absolute_trend_forces_neutral_pair() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_absolute_permitted_industry_momentum(
        closes,
        closes.index,
    )
    blocked = ~diagnostics["absolute_trend_permission"]
    assert blocked.any()
    assert np.allclose(
        diagnostics.loc[blocked, "semis_growth_share"],
        0.50,
    )


def test_positive_permission_can_preserve_active_momentum() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_absolute_permitted_industry_momentum(
        closes,
        closes.index,
    )
    selected = (
        diagnostics["absolute_trend_permission"]
        & ~diagnostics["pulse_veto_active"]
    )
    assert selected.any()
    assert np.allclose(
        diagnostics.loc[selected, "semis_growth_share"],
        diagnostics.loc[
            selected, "raw_momentum_semis_growth_share"
        ],
    )


def test_current_close_cannot_change_current_permission_target() -> None:
    _, _, closes = _inputs()
    before = causal_absolute_permitted_industry_momentum(
        closes,
        closes.index,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_absolute_permitted_industry_momentum(
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


def test_absolute_permission_preserves_growth_budget() -> None:
    weights, daily, closes = _inputs()
    adjusted, _, diagnostics = (
        apply_absolute_permitted_industry_momentum(
            weights,
            daily,
            closes,
        )
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert np.allclose(adjusted.sum(axis=1), weights.sum(axis=1))
