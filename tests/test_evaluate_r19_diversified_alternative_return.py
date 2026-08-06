from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r19_diversified_alternative_return import (
    constant_notional_blend,
    monthly_reset_blend,
    shapley_contributions,
)


def _series(values: list[float], name: str) -> pd.Series:
    return pd.Series(
        values,
        index=pd.bdate_range("2024-01-02", periods=len(values)),
        name=name,
    )


def _funds(
    aqmix: list[float],
    qspix: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {"AQMIX": aqmix, "QSPIX": qspix},
        index=pd.bdate_range("2024-01-02", periods=len(aqmix)),
    )


def test_constant_blend_uses_fixed_half_weights_and_one_cash_leg() -> None:
    path = constant_notional_blend(
        _series([0.01], "base"),
        _funds([0.02], [0.04]),
        _series([0.001], "cash"),
        0.20,
        financing_spread_bps=0.0,
    )
    expected = 0.01 + 0.20 * (0.5 * 0.02 + 0.5 * 0.04 - 0.001)
    assert np.isclose(path["candidate_return"].iloc[0], expected)


def test_zero_overlay_exactly_matches_base() -> None:
    base = _series([0.01, -0.02, 0.03], "base")
    path = constant_notional_blend(
        base,
        _funds([0.02, 0.01, -0.01], [0.01, -0.01, 0.02]),
        _series([0.001, 0.001, 0.001], "cash"),
        0.0,
        financing_spread_bps=100.0,
    )
    assert np.allclose(path["candidate_return"], base)


def test_shapley_contributions_reconcile_exact_log_increment() -> None:
    path = constant_notional_blend(
        _series([0.01, -0.02], "base"),
        _funds([0.02, 0.01], [0.04, -0.01]),
        _series([0.001, 0.001], "cash"),
        0.30,
        financing_spread_bps=100.0,
    )
    contributions = shapley_contributions(path)
    assert np.allclose(
        contributions["aqmix_log_contribution"]
        + contributions["qspix_log_contribution"],
        contributions["total_relative_log_return"],
    )


def test_monthly_reset_charges_both_fund_trades_once() -> None:
    base = _series([0.0, 0.0], "base")
    path = monthly_reset_blend(
        base,
        _funds([0.0, 0.0], [0.0, 0.0]),
        _series([0.0, 0.0], "cash"),
        0.20,
        financing_spread_bps=0.0,
        one_way_cost_bps=10.0,
    )
    assert path["reset"].tolist() == [True, False]
    assert np.isclose(path["trading_cost"].iloc[0], 0.0002)
    assert np.isclose(path["trading_cost"].iloc[1], 0.0)


def test_common_dates_are_intersected_without_filling() -> None:
    base = _series([0.0, 0.0, 0.0], "base")
    funds = _funds([0.01, np.nan, 0.01], [0.01, 0.01, 0.01])
    path = constant_notional_blend(
        base,
        funds,
        _series([0.0, 0.0, 0.0], "cash"),
        0.20,
        financing_spread_bps=0.0,
    )
    assert path.index.tolist() == [base.index[0], base.index[2]]
