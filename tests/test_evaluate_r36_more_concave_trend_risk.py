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
from tools.evaluate_r33_one_session_unlevered_shock_brake import (
    one_session_shock_schedule,
)
from tools.evaluate_r36_more_concave_trend_risk import (
    ACTIVE_MULTIPLIER,
    CASH_FLOOR,
    _configure_helpers,
)


def test_r36_scales_moderate_risk_and_caps_at_118_percent() -> None:
    _configure_helpers()
    index = pd.bdate_range("2018-01-02", periods=1_300)
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.power(1.001, np.arange(len(index))),
            "SEMIS": 100.0
            * np.power(1.0012, np.arange(len(index))),
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.32
    weights["SEMIS"] = 0.32
    weights["GOLD"] = 0.16
    weights["CASH"] = 0.20
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    implemented, _, diagnostics = one_session_shock_schedule(
        weights,
        daily,
        closes,
        cash_floor=CASH_FLOOR,
    )
    permitted = diagnostics["effective_incremental_permission"]
    expected = min(
        0.80 * 1.070 * ACTIVE_MULTIPLIER,
        1.0 - CASH_FLOOR,
    )
    assert np.allclose(
        diagnostics.loc[
            permitted, "implemented_non_cash_weight"
        ],
        expected,
    )
    assert implemented["CASH"].ge(CASH_FLOOR - 1e-12).all()
