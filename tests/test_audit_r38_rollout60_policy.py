from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tools.audit_r38_rollout60_policy as audit


def test_policy_is_fixed_at_sixty_percent() -> None:
    assert audit.CURRENT_SHARE == 0.25
    assert audit.TARGET_SHARE == 0.60


def test_event_metrics_compare_account_paths() -> None:
    index = pd.bdate_range("2020-02-18", "2020-03-23")
    current = pd.Series(
        np.resize(np.array([0.001, -0.0005]), len(index)),
        index=index,
    )
    target = current.copy()
    target.iloc[1] = -0.01
    rows = audit._event_rows(current, target)
    covid = rows.loc[
        rows["event"].eq("covid_acceleration_2020")
    ].iloc[0]
    assert covid["candidate_max_drawdown"] < -0.009
    assert covid["max_drawdown_delta"] < -0.008
