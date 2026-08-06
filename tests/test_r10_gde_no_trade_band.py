from __future__ import annotations

import pandas as pd

from tools.evaluate_r10_gde_no_trade_band import (
    select_smallest_effective_band,
)


def test_selects_smallest_effective_nonzero_band() -> None:
    rows = []
    for sample, period in {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }.items():
        for scenario in ("current_liquidity", "cost_stress"):
            for band in (0.0, 0.0025, 0.005):
                rows.append(
                    {
                        "sample": sample,
                        "period": period,
                        "scenario": scenario,
                        "gde_no_trade_band": band,
                        "candidate_cagr": (
                            0.10
                            if band == 0.0
                            else 0.1001
                        ),
                        "candidate_sharpe": (
                            1.0
                            if band == 0.0
                            else 1.001
                        ),
                    }
                )
    tracking = pd.DataFrame(
        {
            "gde_no_trade_band": [0.0, 0.0025, 0.005],
            "probability_all_objectives": [0.99, 0.99, 0.99],
            "max_drawdown_p05": [-0.17, -0.17, -0.17],
        }
    )

    assert select_smallest_effective_band(
        pd.DataFrame(rows),
        tracking,
    ) == 0.0025
