from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r11_cross_asset_shock_panel import (
    assign_market_clusters,
    cluster_means,
)


def test_market_clustering_does_not_count_same_selloff_repeatedly() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=20)
    events = pd.DataFrame(
        {
            "asset": ["XLK", "IGV", "XBI", "TAN"],
            "event_date": [
                calendar[2],
                calendar[4],
                calendar[7],
                calendar[13],
            ],
            "relative_log_return": [0.01, 0.02, -0.01, 0.03],
        }
    )

    clustered = assign_market_clusters(events, calendar)
    means = cluster_means(clustered)

    assert clustered["cluster"].tolist() == [0, 0, 0, 1]
    assert len(means) == 2
    assert means.iloc[0]["events"] == 3
