from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_sparse_shock_multiplier import causal_sparse_stress_signals


def prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=120)
    path = 100.0 * (1.0005 ** np.arange(120))
    path[80:85] *= np.array([0.99, 0.96, 0.92, 0.88, 0.84])
    return pd.DataFrame({"QQQ": path, "SEMIS": path * 1.01}, index=index)


def test_sparse_stress_is_causal_and_triggers() -> None:
    closes = prices()
    signals = causal_sparse_stress_signals(
        closes,
        one_day_stress_threshold=-0.04,
        five_day_stress_threshold=-0.09,
    )
    assert signals["stress"].any()
    date = signals.index[signals["stress"]][0]
    altered = closes.copy()
    altered.loc[date, ["QQQ", "SEMIS"]] *= 10.0
    repeated = causal_sparse_stress_signals(
        altered,
        one_day_stress_threshold=-0.04,
        five_day_stress_threshold=-0.09,
    )
    assert repeated.loc[date, "stress"] == signals.loc[date, "stress"]


@pytest.mark.parametrize("one,five", [(-1.0, -0.09), (0.0, -0.09), (-0.04, -1.0), (-0.04, 0.0)])
def test_invalid_thresholds_are_rejected(one: float, five: float) -> None:
    with pytest.raises(ValueError):
        causal_sparse_stress_signals(
            prices(),
            one_day_stress_threshold=one,
            five_day_stress_threshold=five,
        )
