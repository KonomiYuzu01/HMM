from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_staged_production_migration import (
    activation_path_results,
    cumulative_candidate_shares,
    staged_target_plan,
)


def test_cumulative_candidate_shares_honors_spacing() -> None:
    assert np.allclose(
        cumulative_candidate_shares(4, 1),
        [0.25, 0.50, 0.75, 1.00],
    )
    assert np.allclose(
        cumulative_candidate_shares(4, 2),
        [0.25, 0.25, 0.50, 0.50, 0.75, 0.75, 1.00],
    )


def test_staged_plan_reaches_candidate_in_equal_tranches() -> None:
    production = pd.Series({"QQQ": 0.10, "SEMIS": 0.40, "CASH": 0.50})
    candidate = pd.Series({"QQQ": 0.40, "SEMIS": 0.20, "CASH": 0.40})

    targets, trades = staged_target_plan(production, candidate, tranches=4)

    assert np.allclose(targets.iloc[-1], candidate)
    assert np.allclose(trades.sum(), candidate - production)
    assert np.allclose(0.5 * trades.abs().sum(axis=1), 0.075)


def test_activation_path_matches_manual_blend() -> None:
    dates = pd.bdate_range("2025-01-01", periods=5)
    production = pd.Series(0.0, index=dates)
    candidate = pd.Series(0.01, index=dates)

    paths = activation_path_results(
        production,
        candidate,
        tranches=4,
        spacing_sessions=1,
        start="2025-01-01",
        end="2025-12-31",
    )

    expected_staged = np.prod(1.0 + np.array([0.0025, 0.005, 0.0075, 0.01]))
    expected_instant = 1.01**4
    assert np.isclose(
        paths.iloc[0]["relative_transition_return"],
        expected_staged / expected_instant - 1.0,
    )
