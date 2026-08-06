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
from tools.evaluate_r38_crowding_fragility_cap import (
    BASE_MULTIPLIER,
    crowding_fragility_risk_schedule,
    causal_crowding_fragility_state,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2014-01-02", periods=1_600)
    semis_returns = np.full(len(index), 0.0005)
    semis_returns[-180:] = 0.003
    semis = 100.0 * np.exp(np.cumsum(semis_returns))
    qqq = 100.0 * np.exp(np.cumsum(np.full(len(index), 0.0004)))
    closes = pd.DataFrame(
        {
            "QQQ": qqq,
            "SEMIS": semis,
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, weights, daily


def test_current_close_cannot_change_current_open_state() -> None:
    closes, _, _ = _inputs()
    before = causal_crowding_fragility_state(
        closes,
        closes.index,
    )
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_crowding_fragility_state(
        changed,
        changed.index,
    )
    columns = [
        "semis_return_126",
        "semis_extension_200",
        "semis_volatility_ratio",
        "crowding_active",
        "fragility_active",
    ]
    assert before.iloc[-1][columns].equals(after.iloc[-1][columns])


def test_disabled_crowding_is_exact_r38_risk_schedule() -> None:
    closes, weights, daily = _inputs()
    candidate, _, diagnostics = crowding_fragility_risk_schedule(
        weights,
        daily,
        closes,
        crowding_enabled=False,
    )
    assert diagnostics["crowding_active"].eq(False).all()
    assert diagnostics["fragility_active"].eq(False).all()
    assert diagnostics["state_multiplier_semantic_error"].le(
        1e-12
    ).all()
    assert candidate["CASH"].ge(-0.20 - 1e-12).all()


def test_active_crowding_never_exceeds_declared_cap() -> None:
    closes, weights, daily = _inputs()
    _, _, diagnostics = crowding_fragility_risk_schedule(
        weights,
        daily,
        closes,
        percentile=0.80,
    )
    selected = diagnostics["crowding_active"]
    assert selected.any()
    assert diagnostics.loc[
        selected,
        "requested_absolute_multiplier",
    ].le(BASE_MULTIPLIER + 1e-12).all()
