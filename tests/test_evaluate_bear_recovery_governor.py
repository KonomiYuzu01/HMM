from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_bear_recovery_governor import (
    ASSETS,
    BearRecoveryCandidate,
    apply_governor,
    bear_recovery_state,
    causal_bear_signals,
)


def test_causal_signals_do_not_use_same_day_close() -> None:
    index = pd.bdate_range("2020-01-01", periods=8)
    closes = pd.DataFrame(
        {"QQQ": np.arange(100.0, 108.0), "SEMIS": np.arange(90.0, 98.0)},
        index=index,
    )
    original = causal_bear_signals(
        closes,
        sma_sessions=3,
        drawdown_lookback=3,
    )
    changed = closes.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] = 1.0
    revised = causal_bear_signals(
        changed,
        sma_sessions=3,
        drawdown_lookback=3,
    )
    pd.testing.assert_series_equal(original.loc[index[-1]], revised.loc[index[-1]])


def test_recovery_requires_consecutive_confirmations_and_ramps() -> None:
    index = pd.bdate_range("2020-01-01", periods=9)
    signals = pd.DataFrame(
        {
            "prior_growth_drawdown": [0.0, -0.11, -0.12, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04],
            "both_below_sma": [False, True, True, False, False, False, False, False, False],
            "both_above_sma": [False, False, False, True, True, False, True, True, True],
        },
        index=index,
    )
    candidate = BearRecoveryCandidate(
        "test",
        recovery_confirmation_days=3,
        ramp_stage_sessions=2,
    )
    state = bear_recovery_state(signals, candidate)
    assert state.loc[index[1], "state"] == "defense"
    assert not bool(state.loc[index[1], "r38_incremental_enabled"])
    assert state.loc[index[5], "recovery_run"] == 0
    assert state.loc[index[8], "state"] == "ramp"
    assert np.isclose(state.loc[index[8], "growth_cap"], 0.20)


def test_governor_caps_growth_and_conserves_weights() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    base = pd.DataFrame(0.0, index=index, columns=ASSETS)
    base[["QQQ", "SEMIS", "GOLD", "CASH"]] = [0.30, 0.30, 0.10, 0.30]
    blended = base.copy()
    daily = pd.DataFrame(
        {"turnover": [0.0, 0.0], "slippage_cost": [0.0, 0.0]},
        index=index,
    )
    diagnostics = pd.DataFrame(
        {
            "state": ["defense", "normal"],
            "growth_cap": [0.10, 1.0],
            "state_change": [True, True],
            "stage_change": [False, False],
        },
        index=index,
    )
    adjusted, execution, detail = apply_governor(
        blended,
        base,
        daily,
        diagnostics,
        BearRecoveryCandidate("test"),
    )
    assert np.isclose(adjusted.loc[index[0], ["QQQ", "SEMIS"]].sum(), 0.10)
    assert np.isclose(adjusted.loc[index[0]].sum(), 1.0)
    assert np.isclose(adjusted.loc[index[0], "CASH"], 0.80)
    assert execution.loc[index[0], "turnover"] > 0.0
    assert detail["weight_conservation_error"].abs().max() < 1e-12
