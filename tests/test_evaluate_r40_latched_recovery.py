from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_latched_recovery import causal_latched_recovery_signals


def recovery_then_stop_prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=380)
    path = np.concatenate(
        [
            np.linspace(100.0, 150.0, 260),
            np.linspace(150.0, 120.0, 30),
            np.linspace(120.0, 138.0, 45),
            np.linspace(138.0, 130.0, 15),
            np.linspace(130.0, 135.0, 30),
        ]
    )
    return pd.DataFrame({"QQQ": path, "SEMIS": path * 1.01}, index=index)


def test_latch_persists_then_trailing_stop_resets_it() -> None:
    signals = causal_latched_recovery_signals(
        recovery_then_stop_prices(),
        entry_rebound=0.05,
        trailing_stop=0.03,
    )
    assert signals["recovery_entry"].any()
    assert signals["recovery_stop"].any()
    first_stop = signals.index[signals["recovery_stop"]][0]
    prior_date = signals.index[signals.index.get_loc(first_stop) - 1]
    assert bool(signals.loc[prior_date, "latched_recovery_permission"])
    assert not bool(signals.loc[first_stop, "latched_recovery_permission"])


def test_signal_on_date_is_unchanged_by_same_date_close() -> None:
    closes = recovery_then_stop_prices()
    signals = causal_latched_recovery_signals(
        closes,
        entry_rebound=0.05,
        trailing_stop=0.03,
    )
    date = signals.index[signals["recovery_entry"]][0]
    altered = closes.copy()
    altered.loc[date, ["QQQ", "SEMIS"]] *= 0.1
    repeated = causal_latched_recovery_signals(
        altered,
        entry_rebound=0.05,
        trailing_stop=0.03,
    )
    assert repeated.loc[date, "recovery_entry"] == signals.loc[date, "recovery_entry"]


@pytest.mark.parametrize("entry,stop", [(0.0, 0.03), (1.0, 0.03), (0.05, 0.0), (0.05, 1.0)])
def test_invalid_parameters_are_rejected(entry: float, stop: float) -> None:
    with pytest.raises(ValueError):
        causal_latched_recovery_signals(
            recovery_then_stop_prices(),
            entry_rebound=entry,
            trailing_stop=stop,
        )
