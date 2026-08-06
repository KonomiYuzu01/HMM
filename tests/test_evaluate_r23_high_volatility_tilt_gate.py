from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r23_high_volatility_tilt_gate import (
    apply_gated_industry_momentum,
    causal_gated_industry_momentum,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2018-01-02", periods=1_100)
    rng = np.random.default_rng(23)
    common = rng.normal(0.0004, 0.009, len(index))
    common[-80:] = rng.normal(-0.0002, 0.040, 80)
    qqq = 100.0 * np.cumprod(
        1.0 + common + rng.normal(0.0, 0.002, len(index))
    )
    semis = 100.0 * np.cumprod(
        1.0 + common + rng.normal(0.0002, 0.004, len(index))
    )
    closes = pd.DataFrame({"QQQ": qqq, "SEMIS": semis}, index=index)
    weights = pd.DataFrame(
        {"QQQ": 0.45, "SEMIS": 0.45, "CASH": 0.10},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily, closes


def test_current_close_cannot_change_current_gate_target() -> None:
    _, _, closes = _inputs()
    index = closes.index
    before = causal_gated_industry_momentum(closes, index)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 0.50
    after = causal_gated_industry_momentum(changed, index)
    assert np.isclose(
        before.iloc[-1]["semis_growth_share"],
        after.iloc[-1]["semis_growth_share"],
    )
    assert (
        before.iloc[-1]["volatility_state"]
        == after.iloc[-1]["volatility_state"]
    )


def test_high_volatility_gate_returns_pair_to_neutral() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_gated_industry_momentum(
        closes,
        closes.index,
        rebalance_days=1,
    )
    active = diagnostics["volatility_gate_active"]
    assert active.any()
    assert np.allclose(
        diagnostics.loc[active, "semis_growth_share"],
        0.50,
    )


def test_inactive_gate_preserves_ungated_signal() -> None:
    _, _, closes = _inputs()
    diagnostics = causal_gated_industry_momentum(
        closes,
        closes.index,
        rebalance_days=1,
    )
    inactive = ~diagnostics["volatility_gate_active"]
    assert inactive.any()
    assert np.allclose(
        diagnostics.loc[inactive, "semis_growth_share"],
        diagnostics.loc[inactive, "ungated_semis_growth_share"],
    )


def test_gate_preserves_growth_budget_and_total_weight() -> None:
    weights, daily, closes = _inputs()
    adjusted, _, diagnostics = apply_gated_industry_momentum(
        weights,
        daily,
        closes,
    )
    assert diagnostics["growth_budget_error"].le(1e-12).all()
    assert np.allclose(adjusted.sum(axis=1), weights.sum(axis=1))


def test_gate_does_not_create_short_growth_weights() -> None:
    weights, daily, closes = _inputs()
    adjusted, _, _ = apply_gated_industry_momentum(
        weights,
        daily,
        closes,
    )
    assert adjusted[["QQQ", "SEMIS"]].ge(0.0).all().all()
