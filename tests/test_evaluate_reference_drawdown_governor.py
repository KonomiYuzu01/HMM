from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_reference_drawdown_governor import (
    ReferenceGovernorCandidate,
    causal_reference_signals,
    reference_governor_state,
)


def test_reference_drawdown_is_lagged() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame(
        {"QQQ": [100, 101, 102, 103, 104], "SEMIS": [90, 91, 92, 93, 94]},
        index=index,
    )
    reference = pd.DataFrame(
        {"drawdown": [0.0, -0.01, -0.02, -0.11, -0.12]},
        index=index,
    )
    signals = causal_reference_signals(closes, reference, sma_sessions=2)
    assert signals.loc[index[3], "prior_reference_drawdown"] == -0.02


def test_reference_governor_has_hysteresis_and_confirmation() -> None:
    index = pd.bdate_range("2020-01-01", periods=7)
    signals = pd.DataFrame(
        {
            "prior_reference_drawdown": [0.0, -0.11, -0.08, -0.04, -0.04, -0.04, -0.04],
            "growth_above_sma": [False, False, False, True, False, True, True],
        },
        index=index,
    )
    state = reference_governor_state(
        signals,
        ReferenceGovernorCandidate(
            "test",
            recovery_confirmation_days=2,
            ramp_stage_sessions=2,
        ),
    )
    assert state.loc[index[1], "state"] == "defense"
    assert state.loc[index[4], "recovery_run"] == 0
    assert state.loc[index[6], "state"] == "ramp"
    assert state.loc[index[1], "growth_cap"] == 0.0
    assert not bool(state.loc[index[1], "r38_incremental_enabled"])
