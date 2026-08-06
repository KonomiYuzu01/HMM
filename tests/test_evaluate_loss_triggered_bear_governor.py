from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_loss_triggered_bear_governor import (
    LossTriggeredCandidate,
    loss_triggered_state,
)


def test_one_entry_per_unrecovered_drawdown_episode() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    signals = pd.DataFrame(
        {
            "prior_reference_drawdown": [
                0.0,
                -0.11,
                -0.12,
                -0.12,
                -0.12,
                -0.12,
                -0.12,
                -0.04,
                -0.11,
                -0.12,
            ],
            "growth_above_sma": [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        },
        index=index,
    )
    candidate = LossTriggeredCandidate(
        "test",
        recovery_confirmation_days=2,
        ramp_stage_sessions=1,
    )
    state = loss_triggered_state(signals, candidate)
    assert state.loc[index[:7], "entry"].sum() == 1
    assert state.loc[index[7], "armed"]
    assert state.loc[index[8]:, "entry"].sum() == 1


def test_active_state_is_non_bypassable() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    signals = pd.DataFrame(
        {
            "prior_reference_drawdown": [0.0, -0.11],
            "growth_above_sma": [False, False],
        },
        index=index,
    )
    state = loss_triggered_state(signals, LossTriggeredCandidate("test"))
    assert state.loc[index[1], "growth_cap"] == 0.0
    assert not bool(state.loc[index[1], "r38_incremental_enabled"])
