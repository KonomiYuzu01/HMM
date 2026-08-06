from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_shock_aware_multiplier import (
    cap_non_cash_exact,
    causal_stress_signals,
)


def prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=420)
    path = 100.0 * (1.0005 ** np.arange(420))
    path[350:355] *= np.array([0.99, 0.97, 0.94, 0.91, 0.89])
    return pd.DataFrame({"QQQ": path, "SEMIS": path * 1.01}, index=index)


def test_stress_signal_uses_only_prior_close() -> None:
    closes = prices()
    signals = causal_stress_signals(
        closes,
        five_day_stress_threshold=-0.05,
    )
    date = signals.index[signals["stress"]][0]
    altered = closes.copy()
    altered.loc[date, ["QQQ", "SEMIS"]] *= 10.0
    repeated = causal_stress_signals(
        altered,
        five_day_stress_threshold=-0.05,
    )
    assert repeated.loc[date, "stress"] == signals.loc[date, "stress"]


def test_five_day_loss_triggers_stress() -> None:
    signals = causal_stress_signals(
        prices(),
        five_day_stress_threshold=-0.05,
    )
    triggered = signals.loc[
        signals["prior_five_day_growth_return"].le(-0.05)
    ]
    assert not triggered.empty
    assert triggered["stress"].all()


def test_exact_cap_reduces_normal_leverage() -> None:
    target = pd.Series(
        {
            "SPX": 0.0,
            "QQQ": 0.7,
            "SEMIS": 0.4,
            "BOND": 0.0,
            "GOLD": 0.1,
            "OIL": 0.0,
            "USD": 0.0,
            "CASH": -0.2,
            "VIX_HEDGE": 0.0,
        }
    )
    capped = cap_non_cash_exact(target, 1.0)
    assert capped.drop("CASH").sum() == pytest.approx(1.0)
    assert capped.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 0.1])
def test_invalid_five_day_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        causal_stress_signals(
            prices(),
            five_day_stress_threshold=threshold,
        )
