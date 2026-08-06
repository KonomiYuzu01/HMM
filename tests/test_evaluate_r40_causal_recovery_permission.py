from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_causal_recovery_permission import causal_recovery_signals


def crash_and_rebound_prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=340)
    path = np.concatenate(
        [
            np.linspace(100.0, 150.0, 260),
            np.linspace(150.0, 120.0, 30),
            np.linspace(120.0, 140.0, 50),
        ]
    )
    return pd.DataFrame({"QQQ": path, "SEMIS": path * 1.01}, index=index)


def test_recovery_permission_uses_only_prior_close_data() -> None:
    closes = crash_and_rebound_prices()
    signals = causal_recovery_signals(
        closes,
        fast_sma_window=10,
        rebound_threshold=0.03,
    )
    first = signals.index[signals["fast_recovery_permission"]][0]
    altered = closes.copy()
    altered.loc[first, ["QQQ", "SEMIS"]] *= 10.0
    repeated = causal_recovery_signals(
        altered,
        fast_sma_window=10,
        rebound_threshold=0.03,
    )
    assert repeated.loc[first, "fast_recovery_permission"] == signals.loc[
        first, "fast_recovery_permission"
    ]


def test_permission_requires_deep_drawdown_and_rebound() -> None:
    closes = crash_and_rebound_prices()
    signals = causal_recovery_signals(
        closes,
        fast_sma_window=10,
        rebound_threshold=0.03,
    )
    active = signals.loc[signals["fast_recovery_permission"]]
    assert not active.empty
    assert active["prior_growth_drawdown"].le(-0.08).all()
    assert active["prior_growth_rebound"].ge(0.03).all()


@pytest.mark.parametrize(
    ("window", "rebound"),
    [(1, 0.03), (10, 0.0), (10, 1.0)],
)
def test_invalid_parameters_are_rejected(window: int, rebound: float) -> None:
    with pytest.raises(ValueError):
        causal_recovery_signals(
            crash_and_rebound_prices(),
            fast_sma_window=window,
            rebound_threshold=rebound,
        )
