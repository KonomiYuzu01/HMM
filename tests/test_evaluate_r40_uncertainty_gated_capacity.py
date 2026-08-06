from __future__ import annotations

import pandas as pd

from tools.evaluate_r38_volatility_conditioned_capacity_fill import (
    volatility_conditioned_schedule,
)


def test_time_varying_multiplier_is_applied_without_lookahead() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    weights = pd.DataFrame(
        {
            "SPX": [0.0] * 3,
            "QQQ": [0.4] * 3,
            "SEMIS": [0.4] * 3,
            "BOND": [0.0] * 3,
            "GOLD": [0.0] * 3,
            "OIL": [0.0] * 3,
            "USD": [0.0] * 3,
            "CASH": [0.2] * 3,
            "VIX_HEDGE": [0.0] * 3,
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": [1.0, 0.0, 0.0]}, index=index)
    closes = pd.DataFrame(
        {
            "QQQ": range(100, 103),
            "SEMIS": range(100, 103),
        },
        index=index,
    )
    multiplier = pd.Series([1.30, 1.35, 1.375], index=index)
    _, _, diagnostics = volatility_conditioned_schedule(
        weights,
        daily,
        closes,
        low_vol_active_multiplier=multiplier,
        high_vol_active_multiplier=1.30,
        base_multiplier=1.0,
        shock_multiplier=1.0,
    )
    assert diagnostics["low_volatility_active_multiplier"].tolist() == [
        1.30,
        1.35,
        1.375,
    ]
