from __future__ import annotations

import pandas as pd

from tools.evaluate_r10_high_fraction_band_frontier import (
    select_highest_feasible,
)


def test_selects_highest_fraction_then_smallest_band() -> None:
    rows = []
    full_periods = {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }
    for sample, period in full_periods.items():
        for scenario in ("current_liquidity", "cost_stress"):
            for fraction in (0.40, 0.50):
                for band in (0.0025, 0.01):
                    rows.append(
                        {
                            "sample": sample,
                            "period": period,
                            "scenario": scenario,
                            "substitution_fraction": fraction,
                            "gde_no_trade_band": band,
                            "cagr_delta": 0.01,
                            "sharpe_delta": 0.01,
                            "candidate_max_drawdown": -0.17,
                        }
                    )
    tracking = pd.DataFrame(
        [
            {
                "substitution_fraction": fraction,
                "gde_no_trade_band": band,
                "probability_all_objectives": 0.99,
                "max_drawdown_p05": -0.175,
            }
            for fraction in (0.40, 0.50)
            for band in (0.0025, 0.01)
        ]
    )

    assert select_highest_feasible(
        pd.DataFrame(rows),
        tracking,
    ) == (0.50, 0.0025)
