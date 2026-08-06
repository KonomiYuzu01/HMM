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
from tools.evaluate_r38_convex_semiconductor_overlay import (
    OVERLAY_FRACTION,
    apply_convex_semiconductor_overlay,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2017-01-02", periods=1_500)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.0003, np.arange(len(index))),
            "SEMIS": 100.0
            * np.power(1.0008, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.10
    weights["SEMIS"] = 0.70
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_convex_overlay_stays_between_base_and_signal() -> None:
    closes, weights, daily = _inputs()
    adjusted, _, diagnostics = apply_convex_semiconductor_overlay(
        weights,
        daily,
        closes,
        max_tilt=OVERLAY_FRACTION,
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert diagnostics["overlay_absolute_deviation"].le(
        OVERLAY_FRACTION + 1e-12
    ).all()
    assert diagnostics["semis_growth_share"].ge(
        diagnostics["overlay_endpoint_low"] - 1e-12
    ).all()
    assert diagnostics["semis_growth_share"].le(
        diagnostics["overlay_endpoint_high"] + 1e-12
    ).all()
    assert np.allclose(
        adjusted["QQQ"] + adjusted["SEMIS"],
        weights["QQQ"] + weights["SEMIS"],
    )


def test_zero_overlay_is_exact_identity() -> None:
    closes, weights, daily = _inputs()
    adjusted, _, _ = apply_convex_semiconductor_overlay(
        weights,
        daily,
        closes,
        max_tilt=0.0,
    )
    assert np.allclose(adjusted[ASSETS], weights[ASSETS])


def test_current_close_cannot_change_current_open_overlay() -> None:
    closes, weights, daily = _inputs()
    before, _, _ = apply_convex_semiconductor_overlay(
        weights,
        daily,
        closes,
        max_tilt=OVERLAY_FRACTION,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after, _, _ = apply_convex_semiconductor_overlay(
        weights,
        daily,
        changed,
        max_tilt=OVERLAY_FRACTION,
    )
    assert np.allclose(before.iloc[-1], after.iloc[-1])
