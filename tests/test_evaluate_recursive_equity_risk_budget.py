from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_recursive_equity_risk_budget import (
    RecursiveEquityPolicy,
    RecursiveRiskCandidate,
)
from tools.evaluate_r10_gde_capital_efficiency import ASSETS


def _targets(index: pd.DatetimeIndex) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.40
    weights["GOLD"] = 0.20
    return weights


def test_policy_uses_implemented_equity_high_water_mark_without_reset() -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    trend = pd.DataFrame({"both_above_sma": False}, index=index)
    weights = _targets(index)
    policy = RecursiveEquityPolicy(RecursiveRiskCandidate("test"), trend, weights)

    _, _, first = policy(index[0], weights.loc[index[0]], 1.00, 1.00)
    _, _, soft = policy(index[1], weights.loc[index[1]], 0.93, 1.00)
    _, _, hard = policy(index[2], weights.loc[index[2]], 0.89, 1.00)
    _, _, still_hard = policy(index[3], weights.loc[index[3]], 0.95, 1.00)

    assert first["state"] == "normal"
    assert soft["state"] == "soft"
    assert soft["growth_cap"] == pytest.approx(0.35)
    assert hard["state"] == "defense"
    assert hard["growth_cap"] == pytest.approx(0.0)
    assert still_hard["state"] == "defense"


def test_policy_recovery_requires_dual_trend_and_aborts_on_failure() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    trend = pd.DataFrame(
        {"both_above_sma": [False, False, True, True, True, True, False, True]},
        index=index,
    )
    weights = _targets(index)
    candidate = RecursiveRiskCandidate(
        "test",
        recovery_confirmation_days=2,
        ramp_stage_sessions=1,
    )
    policy = RecursiveEquityPolicy(candidate, trend, weights)
    states: list[str] = []
    caps: list[float] = []
    for date in index:
        _, _, diagnostics = policy(date, weights.loc[date], 0.89, 1.00)
        states.append(str(diagnostics["state"]))
        caps.append(float(diagnostics["growth_cap"]))

    assert states[0] == "defense"
    assert states[3] == "ramp"
    assert caps[3] == pytest.approx(0.20)
    assert states[6] == "defense"


def test_active_policy_uses_r11_target_and_conserves_weight() -> None:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    trend = pd.DataFrame({"both_above_sma": False}, index=index)
    r11 = _targets(index)
    blended = r11.loc[index[1]].copy()
    blended["QQQ"] += 0.10
    blended["CASH"] -= 0.10
    policy = RecursiveEquityPolicy(RecursiveRiskCandidate("test"), trend, r11)

    target, force, diagnostics = policy(index[1], blended, 0.89, 1.00)

    assert force
    assert diagnostics["state"] == "defense"
    assert not diagnostics["r38_incremental_enabled"]
    assert target["QQQ"] + target["SEMIS"] == pytest.approx(0.0)
    assert target.sum() == pytest.approx(1.0)
