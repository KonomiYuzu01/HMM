from __future__ import annotations

from tools.evaluate_r40_stress_normal_cap import StressNormalCapCandidate


def test_candidates_change_only_stress_normal_cap() -> None:
    defensive = StressNormalCapCandidate("defensive", 0.9)
    permissive = StressNormalCapCandidate("permissive", 1.1)
    assert defensive.stress_normal_non_cash_cap == 0.9
    assert permissive.stress_normal_non_cash_cap == 1.1
    assert defensive.bear_multiplier == permissive.bear_multiplier == 11.0
    assert defensive.one_day_stress_threshold == permissive.one_day_stress_threshold == -0.05
