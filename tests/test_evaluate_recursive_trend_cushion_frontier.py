from __future__ import annotations

import pandas as pd

from tools.evaluate_recursive_trend_cushion_frontier import (
    CANDIDATES,
    PRIOR_POINTS,
    TARGET_CEILINGS,
    complete_period,
    select_frontier,
)


def test_frontier_grid_covers_requested_interval_and_boundary_variants() -> None:
    assert TARGET_CEILINGS[0] == 0.16
    assert TARGET_CEILINGS[-1] == 0.20
    assert len(TARGET_CEILINGS) == 9
    assert {candidate.floor_drawdown for candidate in CANDIDATES}.issuperset(
        {-0.155, -0.18, -0.19, -0.20}
    )
    assert all(candidate.bull_multiplier == 100.0 for candidate in CANDIDATES)
    assert {point[1] for point in PRIOR_POINTS}.issuperset(
        {"central", "floor_145", "floor_150"}
    )
    assert any(candidate.name == "floor_190_b950_t050" for candidate in CANDIDATES)


def test_complete_period_accepts_current_live_sample_name() -> None:
    assert complete_period(
        {"periods": {"live_2022_2026": ("2022-01-01", None)}}
    ) == ("live_2022_2026", ("2022-01-01", None))


def test_frontier_maximizes_modern_cagr_only_within_drawdown_ceiling() -> None:
    rows = []
    values = {
        "safer": (-0.15, 0.20, -0.01, 0.14, -0.01),
        "faster": (-0.19, 0.23, -0.005, 0.15, -0.005),
    }
    for candidate, (mdd, modern, modern_delta, proxy, proxy_delta) in values.items():
        for sample in ("pre2008", "normal_synthetic", "proxy_synthetic"):
            rows.append(
                {
                    "candidate": candidate,
                    "sample": sample,
                    "candidate_max_drawdown": mdd if sample == "pre2008" else -0.10,
                    "candidate_cagr": (
                        modern if sample == "normal_synthetic" else proxy
                    ),
                    "cagr_delta": (
                        modern_delta if sample == "normal_synthetic" else proxy_delta
                    ),
                }
            )
    frontier = select_frontier(pd.DataFrame(rows))
    assert frontier.loc[frontier["drawdown_ceiling"].eq(0.16), "candidate"].item() == "safer"
    assert frontier.loc[frontier["drawdown_ceiling"].eq(0.20), "candidate"].item() == "faster"
