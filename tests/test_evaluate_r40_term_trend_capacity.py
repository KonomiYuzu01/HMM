from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_r40_term_trend_capacity import causal_term_trend_permission


def test_permission_requires_term_and_lagged_long_trend() -> None:
    index = pd.bdate_range("2020-01-01", periods=260)
    up = 100.0 * (1.001 ** np.arange(260))
    closes = pd.DataFrame(
        {
            "QQQ": up,
            "SEMIS": up * 1.01,
            "VIX": 20.0,
            "VIX3M": 22.0,
        },
        index=index,
    )
    permission = causal_term_trend_permission(
        closes,
        term_premium_threshold=1.05,
    )
    assert not permission.iloc[:200].any()
    assert permission.iloc[-1]
    neutral = closes.copy()
    neutral["VIX3M"] = neutral["VIX"]
    assert not causal_term_trend_permission(
        neutral,
        term_premium_threshold=1.0,
    ).any()
