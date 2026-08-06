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
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
from tools.evaluate_r38_accelerating_volatility_capacity_fill import (
    accelerating_volatility_schedule,
)
import tools.evaluate_r38_convex_semiconductor_overlay as r38
from tools.build_r38_production_overlay import (
    validate_generation_request,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2018-01-02", periods=1_400)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.0005, np.arange(len(index))),
            "SEMIS": 100.0
            * np.power(1.0008, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.50
    weights["SEMIS"] = 0.35
    weights["GOLD"] = 0.15
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily, closes


def test_production_sequence_respects_cash_and_overlay_budgets() -> None:
    weights, daily, closes = _inputs()
    scheduled, execution, risk_diagnostics = (
        accelerating_volatility_schedule(
            weights,
            daily,
            closes,
            cash_floor=-0.20,
            low_vol_active_multiplier=1.375,
            high_vol_active_multiplier=1.30,
        )
    )
    tilted, tilted_daily, diagnostics = (
        r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution,
            closes,
            max_tilt=0.10,
        )
    )
    limited, _, _ = enforce_daily_cash_target_limit(
        tilted,
        tilted_daily,
        cash_floor=-0.20,
    )
    assert limited["CASH"].ge(-0.20 - 1e-12).all()
    assert np.allclose(limited.sum(axis=1), 1.0)
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert diagnostics["overlay_absolute_deviation"].le(
        0.10 + 1e-12
    ).all()
    assert risk_diagnostics["state_active_multiplier"].isin(
        (1.30, 1.375)
    ).all()


def test_latest_close_cannot_change_latest_open_target() -> None:
    weights, daily, closes = _inputs()

    def target(frame: pd.DataFrame) -> pd.Series:
        scheduled, execution, _ = accelerating_volatility_schedule(
            weights,
            daily,
            frame,
            cash_floor=-0.20,
            low_vol_active_multiplier=1.375,
            high_vol_active_multiplier=1.30,
        )
        tilted, _, _ = r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution,
            frame,
            max_tilt=0.10,
        )
        return tilted.iloc[-1]

    before = target(closes)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = target(changed)
    assert np.allclose(before, after)


def test_intraday_reuse_requires_synchronized_latest_complete_inputs() -> None:
    latest = pd.Timestamp("2026-07-28")
    synchronized = {
        "expected": latest,
        "R9": latest,
        "R11": latest,
        "core": latest,
        "GDE": latest,
    }
    validate_generation_request(
        synchronized,
        "INTRADAY_BLOCKED",
        reuse_latest_complete_inputs=True,
    )

    stale = dict(synchronized)
    stale["GDE"] = pd.Timestamp("2026-07-27")
    with np.testing.assert_raises(RuntimeError):
        validate_generation_request(
            stale,
            "INTRADAY_BLOCKED",
            reuse_latest_complete_inputs=True,
        )


def test_intraday_generation_remains_blocked_without_explicit_reuse() -> None:
    latest = pd.Timestamp("2026-07-28")
    synchronized = {
        "expected": latest,
        "R9": latest,
        "R11": latest,
        "core": latest,
        "GDE": latest,
    }
    with np.testing.assert_raises(RuntimeError):
        validate_generation_request(
            synchronized,
            "INTRADAY_BLOCKED",
            reuse_latest_complete_inputs=False,
        )
