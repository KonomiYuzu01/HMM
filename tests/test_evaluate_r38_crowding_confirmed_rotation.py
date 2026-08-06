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
from tools.evaluate_r38_crowding_confirmed_rotation import (
    apply_confirmed_rotation,
    causal_confirmed_unwind_state,
)


def _inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    index = pd.bdate_range("2014-01-02", periods=1_700)
    returns = np.full(len(index), 0.0005)
    returns[-220:-50] = 0.003
    returns[-50:] = -0.004
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.exp(np.cumsum(np.full(len(index), 0.0004))),
            "SEMIS": 100.0 * np.exp(np.cumsum(returns)),
        },
        index=index,
    )
    full = pd.DataFrame(0.0, index=index, columns=ASSETS)
    full["QQQ"] = 0.45
    full["SEMIS"] = 0.45
    full["GOLD"] = 0.25
    full["USD"] = 0.05
    full["CASH"] = -0.20
    reference = full.copy()
    reference["QQQ"] = 0.38
    reference["SEMIS"] = 0.38
    reference["CASH"] = -0.06
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, full, reference, daily


def test_current_close_cannot_change_current_open_unwind_state() -> None:
    closes, _, _, _ = _inputs()
    before = causal_confirmed_unwind_state(closes, closes.index)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_confirmed_unwind_state(changed, changed.index)
    columns = [
        "prior_semis_drawdown_21",
        "confirmed_unwind_active",
    ]
    assert before.iloc[-1][columns].equals(after.iloc[-1][columns])


def test_half_rotation_routes_weight_exactly() -> None:
    closes, full, reference, daily = _inputs()
    adjusted, _, diagnostics = apply_confirmed_rotation(
        full,
        reference,
        daily,
        closes,
        memory_days=84,
        qqq_rotation_fraction=0.50,
    )
    removed = diagnostics["removed_semis_weight"]
    assert np.allclose(
        adjusted["QQQ"] - full["QQQ"],
        removed * 0.50,
    )
    assert np.allclose(
        adjusted["CASH"] - full["CASH"],
        removed * 0.50,
    )
    assert diagnostics["rotation_conservation_error"].le(1e-12).all()


def test_disabled_rotation_is_exact_identity() -> None:
    closes, full, reference, daily = _inputs()
    adjusted, _, _ = apply_confirmed_rotation(
        full,
        reference,
        daily,
        closes,
        rotation_enabled=False,
    )
    assert np.allclose(adjusted[ASSETS], full[ASSETS])
