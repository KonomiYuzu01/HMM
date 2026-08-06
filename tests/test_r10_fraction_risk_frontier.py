from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r10_fraction_risk_frontier import (
    select_largest_feasible,
)


def test_selects_largest_fraction_that_meets_risk_budget() -> None:
    tracking = pd.DataFrame(
        {
            "substitution_fraction": [0.25, 0.33, 0.40, 0.50],
            "probability_all_objectives": [1.0, 0.99, 0.96, 0.80],
            "cagr_delta_p05": [0.004, 0.005, 0.006, 0.007],
            "max_drawdown_p05": [-0.17, -0.175, -0.179, -0.185],
        }
    )

    assert select_largest_feasible(tracking) == pytest.approx(0.40)


def test_raises_when_no_fraction_is_feasible() -> None:
    tracking = pd.DataFrame(
        {
            "substitution_fraction": [0.25],
            "probability_all_objectives": [0.94],
            "cagr_delta_p05": [0.004],
            "max_drawdown_p05": [-0.17],
        }
    )

    with pytest.raises(RuntimeError, match="No fraction"):
        select_largest_feasible(tracking)
