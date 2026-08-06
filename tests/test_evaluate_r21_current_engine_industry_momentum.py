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

from evaluate_r21_current_engine_industry_momentum import (
    apply_industry_momentum,
    causal_industry_momentum,
)


def _closes(periods: int = 12) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=periods)
    return pd.DataFrame(
        {
            "QQQ": np.full(periods, 100.0),
            "SEMIS": 100.0 * np.power(1.01, np.arange(periods)),
        },
        index=index,
    )


def test_signal_is_lagged_one_day() -> None:
    closes = _closes(8)
    index = closes.index
    before = causal_industry_momentum(
        closes,
        index,
        lookback_days=3,
        rebalance_days=1,
    )
    changed = closes.copy()
    changed.loc[index[-1], "SEMIS"] *= 0.50
    after = causal_industry_momentum(
        changed,
        index,
        lookback_days=3,
        rebalance_days=1,
    )
    assert before.loc[index[-1], "semis_growth_share"] == pytest.approx(
        after.loc[index[-1], "semis_growth_share"]
    )


def test_growth_budget_is_exactly_conserved() -> None:
    closes = _closes(8)
    index = closes.index
    weights = pd.DataFrame(
        {
            "QQQ": 0.30,
            "SEMIS": 0.50,
            "CASH": 0.20,
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    adjusted, _, diagnostics = apply_industry_momentum(
        weights,
        daily,
        closes,
        rebalance_days=1,
    )
    assert np.allclose(adjusted["QQQ"] + adjusted["SEMIS"], 0.80)
    assert diagnostics["growth_budget_error"].max() <= 1e-12


def test_weights_stay_inside_pair_budget() -> None:
    closes = _closes(10)
    signal = causal_industry_momentum(
        closes,
        closes.index,
        lookback_days=3,
        rebalance_days=1,
    )
    assert signal["semis_growth_share"].between(0.0, 1.0).all()
    assert signal["qqq_growth_share"].between(0.0, 1.0).all()


def test_rebalance_schedule_updates_only_on_anchor_dates() -> None:
    closes = _closes(10)
    signal = causal_industry_momentum(
        closes,
        closes.index,
        lookback_days=3,
        rebalance_days=3,
    )
    expected = [True, False, False, True, False, False, True, False, False, True]
    assert signal["momentum_update"].tolist() == expected


def test_invalid_max_tilt_raises() -> None:
    closes = _closes(5)
    with pytest.raises(ValueError):
        causal_industry_momentum(
            closes,
            closes.index,
            lookback_days=3,
            max_tilt=0.51,
        )
