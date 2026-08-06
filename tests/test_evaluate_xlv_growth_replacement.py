from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_xlv_growth_replacement import xlv_growth_replacement


def _inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2024-01-02", periods=3)
    baseline = pd.Series([0.01, -0.02, 0.03], index=index)
    targets = pd.DataFrame(
        {"QQQ": [0.20, 0.10, 0.10], "SEMIS": [0.30, 0.30, 0.30]},
        index=index,
    )
    daily = pd.DataFrame({"turnover": [1.0, 0.0, 1.0]}, index=index)
    closes = pd.DataFrame(
        {"QQQ": [100.0, 110.0, 121.0], "SEMIS": [100.0, 90.0, 99.0]},
        index=index,
    )
    xlv = pd.Series([100.0, 105.0, 115.5], index=index, name="XLV")
    return baseline, targets, daily, closes, xlv


def test_zero_replacement_is_exact_baseline() -> None:
    result = xlv_growth_replacement(*_inputs(), fraction=0.0, xlv_one_way_cost_bps=10.0)
    assert np.allclose(result["baseline_return"], result["candidate_return"])
    assert result["xlv_replacement_weight"].eq(0.0).all()


def test_replacement_is_self_financed_inside_growth_sleeve() -> None:
    result = xlv_growth_replacement(*_inputs(), fraction=0.10, xlv_one_way_cost_bps=0.0)
    assert np.allclose(
        result["xlv_replacement_weight"], result["growth_weight"] * 0.10
    )
    assert result["xlv_replacement_weight"].le(result["growth_weight"]).all()


def test_cost_is_charged_only_on_existing_trade_dates() -> None:
    result = xlv_growth_replacement(*_inputs(), fraction=0.10, xlv_one_way_cost_bps=10.0)
    assert result["xlv_extra_trading_cost"].iloc[0] == 0.00005
    assert result["xlv_extra_trading_cost"].iloc[1] == 0.0
    assert result["xlv_extra_trading_cost"].iloc[2] == 0.0
