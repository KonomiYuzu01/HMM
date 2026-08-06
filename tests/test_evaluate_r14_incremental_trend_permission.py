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

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r14_incremental_trend_permission import (
    TrendCandidate,
    causal_trend_permission,
    trend_permission_schedule,
)


def test_sma_permission_uses_only_prior_close() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(100.0, 190.0, len(index)),
        },
        index=index,
    )
    definition = TrendCandidate("qqq_sma200", "qqq_sma200")
    original = causal_trend_permission(closes, definition)
    revised_closes = closes.copy()
    revised_closes.loc[index[-1], "QQQ"] = 1.0

    revised = causal_trend_permission(revised_closes, definition)

    assert revised.iloc[-1] == original.iloc[-1]


def test_both_sma_requires_both_assets() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(200.0, 100.0, len(index)),
        },
        index=index,
    )

    permission = causal_trend_permission(
        closes,
        TrendCandidate("both_sma200", "both_sma200"),
    )

    assert not bool(permission.iloc[-1])


def test_majority_rule_requires_three_positive_horizons() -> None:
    index = pd.bdate_range("2023-01-01", periods=260)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(100.0, 190.0, len(index)),
        },
        index=index,
    )

    permission = causal_trend_permission(
        closes,
        TrendCandidate("majority", "majority_63_252"),
    )

    assert bool(permission.iloc[-1])


def test_inactive_permission_returns_exact_r11_multiplier() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(200.0, 100.0, len(index)),
            "SEMIS": np.linspace(190.0, 100.0, len(index)),
        },
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 0.50
    target["SEMIS"] = 0.30
    target["GOLD"] = 0.20
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": 1.0}, index=index)

    _, _, diagnostics = trend_permission_schedule(
        weights,
        daily,
        closes,
        TrendCandidate("qqq_sma200", "qqq_sma200"),
    )

    assert diagnostics["relative_multiplier"].iloc[-1] == pytest.approx(
        1.0
    )


def test_state_change_creates_execution_update() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.concatenate(
                [np.linspace(100.0, 200.0, 204), [50.0]]
            ),
            "SEMIS": np.linspace(100.0, 190.0, len(index)),
        },
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 0.50
    target["SEMIS"] = 0.30
    target["GOLD"] = 0.20
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": 0.0}, index=index)

    _, execution, diagnostics = trend_permission_schedule(
        weights,
        daily,
        closes,
        TrendCandidate("qqq_sma200", "qqq_sma200"),
    )

    changed = diagnostics["state_change"]
    assert changed.any()
    assert execution.loc[changed, "turnover"].gt(0.0).all()
