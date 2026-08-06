from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tools.evaluate_r38_state_dependent_rollout as state


def test_state_share_uses_only_frozen_values() -> None:
    index = pd.bdate_range("2026-01-05", periods=4)
    diagnostics = pd.DataFrame(
        {
            "effective_incremental_permission": [
                True,
                False,
                True,
                False,
            ]
        },
        index=index,
    )
    share = state.state_dependent_share(
        diagnostics,
        0.70,
    )
    assert np.allclose(share, [0.70, 0.25, 0.70, 0.25])


def test_selector_uses_smallest_passing_share() -> None:
    gates = pd.DataFrame(
        {
            "stable_share": state.STABLE_SHARES,
            "target_cagr_pass": [False, True, True],
            "drawdown_pass": [True, True, True],
        }
    )
    assert state.select_smallest_passing_share(gates) == 0.70


def test_invalid_share_order_is_rejected() -> None:
    index = pd.bdate_range("2026-01-05", periods=1)
    diagnostics = pd.DataFrame(
        {"effective_incremental_permission": [True]},
        index=index,
    )
    try:
        state.state_dependent_share(
            diagnostics,
            0.20,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid share order must be rejected")
