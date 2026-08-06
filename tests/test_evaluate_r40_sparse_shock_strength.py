from __future__ import annotations

from tools.evaluate_r40_sparse_shock_strength import (
    SparseShockStrengthCandidate,
)


def test_candidate_changes_only_stress_multiplier() -> None:
    low = SparseShockStrengthCandidate("low", 9.5)
    high = SparseShockStrengthCandidate("high", 12.0)
    assert low.bear_multiplier == 9.5
    assert high.bear_multiplier == 12.0
    assert low.calm_multiplier == high.calm_multiplier == 30.0
    assert low.one_day_stress_threshold == high.one_day_stress_threshold == -0.05
    assert low.five_day_stress_threshold == high.five_day_stress_threshold == -0.09
