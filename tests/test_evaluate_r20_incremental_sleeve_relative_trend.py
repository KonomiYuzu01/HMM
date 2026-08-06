from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r20_incremental_sleeve_relative_trend import (
    causal_shadow_relative_permission,
)


def test_permission_uses_only_returns_before_current_day() -> None:
    index = pd.bdate_range("2025-01-02", periods=6)
    shadow = pd.Series([0.01, 0.01, 0.01, -0.50, 0.0, 0.0], index=index)
    baseline = pd.Series(0.0, index=index)
    permission = causal_shadow_relative_permission(
        shadow,
        baseline,
        index,
        lookback_days=3,
    )
    assert bool(
        permission.loc[index[3], "shadow_relative_permission"]
    )
    assert not bool(
        permission.loc[index[4], "shadow_relative_permission"]
    )


def test_missing_history_keeps_permission_off() -> None:
    index = pd.bdate_range("2025-01-02", periods=3)
    permission = causal_shadow_relative_permission(
        pd.Series([0.01, 0.01, 0.01], index=index),
        pd.Series(0.0, index=index),
        index,
        lookback_days=3,
    )
    assert not permission["shadow_relative_permission"].any()


def test_shadow_can_reenter_while_actual_layer_is_off() -> None:
    index = pd.bdate_range("2025-01-02", periods=9)
    shadow = pd.Series(
        [-0.02, -0.02, -0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        index=index,
    )
    permission = causal_shadow_relative_permission(
        shadow,
        pd.Series(0.0, index=index),
        index,
        lookback_days=3,
    )
    assert not bool(
        permission.loc[index[4], "shadow_relative_permission"]
    )
    assert bool(
        permission.loc[index[7], "shadow_relative_permission"]
    )


def test_hysteresis_holds_state_inside_band() -> None:
    index = pd.bdate_range("2025-01-02", periods=6)
    relative_logs = np.array([0.0020, 0.0000, 0.0000, -0.0005, 0.0, 0.0])
    shadow = pd.Series(np.expm1(relative_logs), index=index)
    permission = causal_shadow_relative_permission(
        shadow,
        pd.Series(0.0, index=index),
        index,
        lookback_days=2,
        entry_threshold=0.001,
        exit_threshold=-0.001,
    )
    assert bool(
        permission.loc[index[2], "shadow_relative_permission"]
    )
    assert bool(
        permission.loc[index[5], "shadow_relative_permission"]
    )


def test_invalid_hysteresis_thresholds_raise() -> None:
    index = pd.bdate_range("2025-01-02", periods=3)
    with pytest.raises(ValueError):
        causal_shadow_relative_permission(
            pd.Series(0.0, index=index),
            pd.Series(0.0, index=index),
            index,
            lookback_days=2,
            entry_threshold=-0.001,
            exit_threshold=0.001,
        )
