from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_price_shock_multiplier import causal_price_stress_signals


def prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=120)
    path = 100.0 * (1.0005 ** np.arange(120))
    path[80:85] *= np.array([0.99, 0.97, 0.94, 0.91, 0.89])
    return pd.DataFrame({"QQQ": path, "SEMIS": path * 1.01}, index=index)


def test_price_stress_is_causal_and_five_day_loss_triggers() -> None:
    closes = prices()
    signals = causal_price_stress_signals(
        closes,
        five_day_stress_threshold=-0.05,
    )
    triggered = signals.loc[
        signals["prior_five_day_growth_return"].le(-0.05)
    ]
    assert not triggered.empty
    assert triggered["stress"].all()
    date = triggered.index[0]
    altered = closes.copy()
    altered.loc[date, ["QQQ", "SEMIS"]] *= 10.0
    repeated = causal_price_stress_signals(
        altered,
        five_day_stress_threshold=-0.05,
    )
    assert repeated.loc[date, "stress"] == signals.loc[date, "stress"]


@pytest.mark.parametrize("threshold", [-1.0, 0.0])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        causal_price_stress_signals(
            prices(),
            five_day_stress_threshold=threshold,
        )
