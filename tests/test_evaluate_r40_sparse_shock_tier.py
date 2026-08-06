from __future__ import annotations

from tools.evaluate_r40_sparse_shock_tier import SparseShockTierCandidate


def test_tier_candidates_change_only_rounding_granularity() -> None:
    fine = SparseShockTierCandidate("fine", 0.01)
    coarse = SparseShockTierCandidate("coarse", 0.15)
    assert fine.tier_size == 0.01
    assert coarse.tier_size == 0.15
    assert fine.bear_multiplier == coarse.bear_multiplier == 11.0
    assert fine.calm_multiplier == coarse.calm_multiplier == 30.0
    assert fine.one_day_stress_threshold == coarse.one_day_stress_threshold == -0.05
    assert fine.five_day_stress_threshold == coarse.five_day_stress_threshold == -0.09
