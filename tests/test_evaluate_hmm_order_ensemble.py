from __future__ import annotations

import pandas as pd

from tools.evaluate_hmm_order_ensemble import doubled_trading_cost


def test_doubled_trading_cost_leaves_financing_unchanged() -> None:
    frame = pd.DataFrame(
        {
            "net_return": [0.01, -0.02],
            "trading_cost": [0.001, 0.002],
            "financing_cost": [0.003, 0.004],
        }
    )
    result = doubled_trading_cost(frame)
    pd.testing.assert_series_equal(
        result, pd.Series([0.009, -0.022], name=None), check_names=False
    )
