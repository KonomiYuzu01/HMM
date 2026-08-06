from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r40_fast_crisis_trend import (
    TREND_ASSETS,
    causal_fast_trend_overlay,
)


def prices(rows: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2020-01-01", periods=rows)
    data = {
        asset: 100.0 * (1.001 ** np.arange(rows))
        for asset in TREND_ASSETS
    }
    data["CASH"] = 100.0 * (1.0001 ** np.arange(rows))
    closes = pd.DataFrame(data, index=index)
    opens = closes.copy()
    return opens, closes


def test_overlay_is_causal_and_observes_notional_limits() -> None:
    opens, closes = prices()
    result = causal_fast_trend_overlay(
        opens=opens,
        closes=closes,
        dates=opens.index,
        horizons=(20, 60, 120),
        annual_risk_target=0.08,
    )
    assert result.loc[: opens.index[120], "gross_notional"].eq(0.0).all()
    assert result.loc[opens.index[121], "gross_notional"] > 0.0
    assert result["gross_notional"].max() <= 2.50 + 1e-12
    position_columns = [f"position_{asset}" for asset in TREND_ASSETS]
    assert result[position_columns].abs().max().max() <= 0.75 + 1e-12


def test_overlay_charges_turnover_costs() -> None:
    opens, closes = prices()
    result = causal_fast_trend_overlay(
        opens=opens,
        closes=closes,
        dates=opens.index,
        horizons=(20, 60),
        annual_risk_target=0.06,
        one_way_cost_bps=5.0,
    )
    assert (result["cost"] >= 0.0).all()
    assert result["cost"].sum() > 0.0
    assert np.allclose(result["net_return"], result["gross_return"] - result["cost"])


@pytest.mark.parametrize("risk", [0.0, -0.01])
def test_nonpositive_risk_is_rejected(risk: float) -> None:
    opens, closes = prices()
    with pytest.raises(ValueError):
        causal_fast_trend_overlay(
            opens=opens,
            closes=closes,
            dates=opens.index,
            horizons=(20, 60),
            annual_risk_target=risk,
        )
