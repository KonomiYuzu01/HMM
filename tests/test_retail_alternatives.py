import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_retail_alternatives.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retail_alternatives", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
funding_targets = MODULE.funding_targets
gde_synthetic_returns = MODULE.gde_synthetic_returns
simulate_gold_substitution = MODULE.simulate_gold_substitution


@pytest.mark.parametrize(
    ("mode", "cash_weight", "sleeve_weight", "expected"),
    [
        ("cash_capped", 0.40, 0.10, (0.60, 0.30, 0.10)),
        ("cash_first", 0.40, 0.10, (0.60, 0.30, 0.10)),
        ("pro_rata", 0.40, 0.10, (0.54, 0.36, 0.10)),
        ("cash_capped", 0.05, 0.10, (0.95, 0.00, 0.05)),
        ("cash_first", 0.05, 0.10, (0.90, 0.00, 0.10)),
        ("pro_rata", 0.05, 0.10, (0.855, 0.045, 0.10)),
        ("cash_capped", -0.05, 0.10, (1.05, -0.05, 0.00)),
        ("cash_first", -0.05, 0.10, (0.90, 0.00, 0.10)),
        ("pro_rata", -0.05, 0.10, (0.945, -0.045, 0.10)),
    ],
)
def test_funding_targets(mode, cash_weight, sleeve_weight, expected):
    actual = funding_targets(cash_weight, sleeve_weight, mode)
    assert np.allclose(actual, expected)
    assert np.isclose(sum(actual), 1.0)


def test_funding_targets_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown funding mode"):
        funding_targets(0.20, 0.10, "unknown")


def test_gde_synthetic_returns_uses_fixed_90_90_formula():
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.02],
            "GLD": [0.02, 0.01],
            "BIL": [0.001, 0.001],
        },
        index=pd.date_range("2024-01-02", periods=2),
    )
    actual = gde_synthetic_returns(returns)
    expected = 0.90 * returns["SPY"] + 0.90 * returns["GLD"] - 0.80 * returns["BIL"]
    assert np.allclose(actual, expected)


def test_zero_gold_substitution_leaves_baseline_unchanged():
    index = pd.date_range("2024-01-02", periods=3)
    baseline = pd.DataFrame(
        {
            "gross_return": [0.01, -0.01, 0.005],
            "net_return": [0.009, -0.011, 0.004],
            "cash_weight": [0.20, 0.20, 0.20],
            "gold_weight": [0.10, 0.10, 0.10],
            "growth_weight": [0.40, 0.40, 0.40],
        },
        index=index,
    )
    path = simulate_gold_substitution(
        baseline,
        pd.Series([0.02, -0.01, 0.01], index=index),
        pd.Series([0.01, 0.00, 0.00], index=index),
        0.0,
    )
    assert np.allclose(path["candidate_net"], baseline["net_return"])
    assert np.allclose(path["execution_cost"], 0.0)


def test_growth_linked_gold_substitution_uses_production_core_scale():
    index = pd.date_range("2024-01-02", periods=2)
    baseline = pd.DataFrame(
        {
            "gross_return": [0.0, 0.0],
            "net_return": [0.0, 0.0],
            "cash_weight": [0.2, 0.2],
            "gold_weight": [0.2, 0.2],
            "growth_weight": [0.4, 0.8],
        },
        index=index,
    )
    path = simulate_gold_substitution(
        baseline,
        pd.Series([0.0, 0.0], index=index),
        pd.Series([0.0, 0.0], index=index),
        1.0,
        gate_mode="growth_linked",
        extra_one_way_cost=0.0,
    )
    assert np.allclose(path["gate_intensity"], [0.5, 1.0])
    assert np.allclose(path["gde_weight"], [0.1, 0.2])
