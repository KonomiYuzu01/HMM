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
from evaluate_r15_concentration_limited_trend import (
    CombinedCandidate,
    cap_semis_to_qqq,
    combined_schedule,
)


def test_semis_cap_preserves_growth_and_total_weight() -> None:
    weights = pd.DataFrame(
        {
            "QQQ": [0.20],
            "SEMIS": [0.70],
            "GOLD": [0.20],
            "CASH": [-0.10],
        }
    )

    adjusted, excess = cap_semis_to_qqq(weights, 0.60)

    assert adjusted.loc[0, "SEMIS"] == pytest.approx(0.60)
    assert adjusted.loc[0, "QQQ"] == pytest.approx(0.30)
    assert adjusted.loc[0].sum() == pytest.approx(weights.loc[0].sum())
    assert excess.iloc[0] == pytest.approx(0.10)


def test_combined_schedule_enforces_both_caps() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(100.0, 220.0, len(index)),
        },
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 0.20
    target["SEMIS"] = 0.70
    target["GOLD"] = 0.20
    target["CASH"] = -0.10
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": 1.0}, index=index)

    implemented, _, diagnostics = combined_schedule(
        weights,
        daily,
        closes,
        CombinedCandidate(
            "both_sma200_cap60",
            "both_sma200",
            0.60,
        ),
    )

    assert implemented["SEMIS"].max() <= 0.60 + 1e-12
    assert implemented["CASH"].min() >= -0.30 - 1e-12
    assert diagnostics["semis_cap_binding"].any()


def test_current_close_does_not_change_current_schedule() -> None:
    index = pd.bdate_range("2023-01-01", periods=205)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(100.0, 220.0, len(index)),
        },
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 0.40
    target["SEMIS"] = 0.40
    target["GOLD"] = 0.20
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": 1.0}, index=index)
    definition = CombinedCandidate(
        "both_sma200_cap60",
        "both_sma200",
        0.60,
    )
    original, _, original_diagnostics = combined_schedule(
        weights,
        daily,
        closes,
        definition,
    )
    changed = closes.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] = [1.0, 1.0]

    revised, _, revised_diagnostics = combined_schedule(
        weights,
        daily,
        changed,
        definition,
    )

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        revised.loc[index[-1]],
    )
    assert (
        original_diagnostics.loc[index[-1], "trend_permission"]
        == revised_diagnostics.loc[index[-1], "trend_permission"]
    )
