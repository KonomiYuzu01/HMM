from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from build_pre2008_crash_proxy import return_index
from evaluate_pre2008_crash_replay import (
    price_metrics,
    protection_label,
    return_metrics,
)
from regime_strategy.ensemble import ModelEnsembleResult, write_ensemble_report


def test_return_index_compounds_without_lookahead() -> None:
    values = return_index(pd.Series([0.10, -0.10]), "TEST")
    assert np.allclose(values.to_numpy(), [110.0, 99.0])


def test_metrics_include_initial_capital_in_drawdown() -> None:
    returns = return_metrics(pd.Series([-0.20, 0.10]))
    prices = price_metrics(pd.Series([100.0, 80.0, 88.0]))
    assert np.isclose(returns["max_drawdown"], -0.20)
    assert np.isclose(prices["max_drawdown"], -0.20)


def test_protection_label_requires_both_absolute_and_relative_control() -> None:
    assert protection_label(-0.15, -0.40) == "avoided_major_crash"
    assert protection_label(-0.25, -0.40) == "partially_protected"
    assert protection_label(-0.32, -0.40) == "not_avoided"


def test_ensemble_report_accepts_history_before_fixed_periods(tmp_path: Path) -> None:
    index = pd.bdate_range("1931-01-02", periods=3)
    result = ModelEnsembleResult(
        daily=pd.DataFrame(
            {
                "net_return": [0.01, -0.02, 0.015],
                "turnover": [0.1, 0.0, 0.0],
                "cost": [0.0001, 0.0, 0.0],
            },
            index=index,
        ),
        weights=pd.DataFrame({"CASH": [1.0, 1.0, 1.0]}, index=index),
        sleeve_weights=pd.DataFrame({"seed": [1.0, 1.0, 1.0]}, index=index),
        member_returns=pd.DataFrame(
            {"seed": [0.01, -0.02, 0.015]}, index=index
        ),
        benchmarks=pd.DataFrame(
            {"SPX": [0.005, -0.01, 0.007]}, index=index
        ),
        asset_returns=pd.DataFrame(
            {"CASH": [0.0001, 0.0001, 0.0001]}, index=index
        ),
    )
    write_ensemble_report(result, tmp_path)
    fixed = pd.read_csv(tmp_path / "fixed_period_metrics.csv")
    assert fixed.empty
