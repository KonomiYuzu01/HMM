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
from evaluate_r13_drawdown_budget import (
    DrawdownCandidate,
    cap_weights_at_cash_floor,
    drawdown_budget_schedule,
)


def test_drawdown_multiplier_uses_frozen_curves() -> None:
    drawdown = pd.Series([0.0, -0.04, -0.06, -0.11])

    linear = DrawdownCandidate("linear12", "linear12")
    tier = DrawdownCandidate("tier5_10", "tier5_10")

    assert linear.relative_multiplier(drawdown).tolist() == pytest.approx(
        [1.10, 0.98, 0.92, 0.77]
    )
    assert tier.relative_multiplier(drawdown).tolist() == pytest.approx(
        [1.10, 1.10, 1.00, 0.75]
    )


def test_cash_floor_preserves_total_weight() -> None:
    weights = pd.DataFrame(
        [
            {
                "SPX": 0.0,
                "QQQ": 0.80,
                "SEMIS": 0.60,
                "BOND": 0.0,
                "GOLD": 0.20,
                "OIL": 0.0,
                "USD": 0.0,
                "CASH": -0.60,
                "VIX_HEDGE": 0.0,
            }
        ]
    )

    adjusted, scale = cap_weights_at_cash_floor(weights, -0.25)

    assert adjusted.loc[0, "CASH"] == pytest.approx(-0.25)
    assert adjusted.loc[0].sum() == pytest.approx(1.0)
    assert scale.iloc[0] < 1.0


def test_schedule_uses_previous_reference_drawdown() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    baseline = pd.DataFrame(
        {"drawdown": [0.0, -0.06, -0.11, -0.02]},
        index=index,
    )
    target = {
        asset: 0.0
        for asset in ASSETS
    }
    target["QQQ"] = 0.50
    target["SEMIS"] = 0.30
    target["GOLD"] = 0.20
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": [1.0, 0.0, 0.0, 0.0]}, index=index)

    _, _, diagnostics = drawdown_budget_schedule(
        baseline,
        weights,
        daily,
        DrawdownCandidate("tier5_10", "tier5_10"),
    )

    assert diagnostics["previous_r11_drawdown"].tolist() == pytest.approx(
        [0.0, 0.0, -0.06, -0.11]
    )
    assert diagnostics["requested_relative_multiplier"].tolist() == (
        pytest.approx([1.10, 1.10, 1.00, 0.75])
    )


def test_linear_schedule_only_updates_on_base_trade() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    baseline = pd.DataFrame(
        {"drawdown": [0.0, -0.02, -0.04, -0.06]},
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 1.0
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": [1.0, 0.0, 1.0, 0.0]}, index=index)

    _, _, diagnostics = drawdown_budget_schedule(
        baseline,
        weights,
        daily,
        DrawdownCandidate("linear12", "linear12"),
    )

    assert diagnostics["execution_update"].tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert diagnostics["accepted_relative_multiplier"].tolist() == (
        pytest.approx([1.10, 1.10, 1.04, 1.04])
    )


def test_tier_schedule_adds_trade_only_when_tier_changes() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    baseline = pd.DataFrame(
        {"drawdown": [0.0, -0.06, -0.07, -0.11]},
        index=index,
    )
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 1.0
    weights = pd.DataFrame([target] * len(index), index=index)
    daily = pd.DataFrame({"turnover": np.zeros(len(index))}, index=index)

    _, _, diagnostics = drawdown_budget_schedule(
        baseline,
        weights,
        daily,
        DrawdownCandidate("tier5_10", "tier5_10"),
    )

    assert diagnostics["execution_update"].tolist() == [
        True,
        False,
        True,
        False,
    ]
