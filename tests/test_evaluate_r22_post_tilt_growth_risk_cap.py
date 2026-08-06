from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r22_post_tilt_growth_risk_cap import (
    apply_post_tilt_growth_risk_cap,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=100)
    rng = np.random.default_rng(7)
    qqq = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.025, len(index)))
    semis = 100.0 * np.cumprod(
        1.0 + rng.normal(0.0007, 0.040, len(index))
    )
    closes = pd.DataFrame({"QQQ": qqq, "SEMIS": semis}, index=index)
    weights = pd.DataFrame(
        {"QQQ": 0.45, "SEMIS": 0.45, "CASH": 0.10},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily, closes


def test_post_tilt_cap_never_increases_growth() -> None:
    weights, daily, closes = _inputs()
    _, _, diagnostics = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        closes,
        target_volatility=0.20,
    )
    assert (
        diagnostics["growth_after_post_tilt_cap"]
        <= diagnostics["growth_before_post_tilt_cap"] + 1e-12
    ).all()


def test_released_growth_moves_exactly_to_cash() -> None:
    weights, daily, closes = _inputs()
    _, _, diagnostics = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        closes,
        target_volatility=0.20,
    )
    assert np.allclose(
        diagnostics["growth_released_to_cash"],
        diagnostics["post_tilt_cash_increase"],
    )


def test_current_close_cannot_change_current_risk_target() -> None:
    weights, daily, closes = _inputs()
    before, _, _ = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        closes,
        target_volatility=0.20,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        changed,
        target_volatility=0.20,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])


def test_cap_preserves_total_weight() -> None:
    weights, daily, closes = _inputs()
    capped, _, _ = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        closes,
        target_volatility=0.20,
    )
    assert np.allclose(capped.sum(axis=1), 1.0)


def test_high_risk_path_activates_cap() -> None:
    weights, daily, closes = _inputs()
    _, _, diagnostics = apply_post_tilt_growth_risk_cap(
        weights,
        daily,
        closes,
        target_volatility=0.20,
    )
    assert diagnostics["post_tilt_risk_cap_active"].any()
