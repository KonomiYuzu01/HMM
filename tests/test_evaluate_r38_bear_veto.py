from __future__ import annotations

import pandas as pd

from tools.evaluate_r38_bear_veto import (
    R38BearVetoCandidate,
    r38_bear_veto_state,
)


def test_veto_disables_only_r38_until_dual_trend_confirmation() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    signals = pd.DataFrame(
        {
            "prior_growth_drawdown": [0.0, -0.11, -0.09, -0.08, -0.07, -0.06],
            "both_below_sma": [False, True, False, False, False, False],
            "both_above_sma": [False, False, True, False, True, True],
        },
        index=index,
    )
    candidate = R38BearVetoCandidate("test", recovery_confirmation_days=2)
    diagnostics = r38_bear_veto_state(signals, candidate)
    assert diagnostics.loc[index[1], "state"] == "defense"
    assert diagnostics.loc[index[1], "growth_cap"] == 1.0
    assert not diagnostics.loc[index[1], "r38_incremental_enabled"]
    assert diagnostics.loc[index[2], "recovery_run"] == 1
    assert diagnostics.loc[index[3], "recovery_run"] == 0
    assert diagnostics.loc[index[5], "state"] == "normal"
    assert diagnostics.loc[index[5], "r38_incremental_enabled"]
