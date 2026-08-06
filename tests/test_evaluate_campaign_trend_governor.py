from __future__ import annotations

import pandas as pd

from tools.evaluate_campaign_loss_governor import campaign_loss_state
from tools.evaluate_campaign_trend_governor import CampaignTrendCandidate


def test_dual_trend_recovery_is_required_and_ramp_can_abort() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    signals = pd.DataFrame(
        {
            "prior_reference_equity": [1.00, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
            "prior_reference_drawdown": [0.00] * 8,
            "growth_above_sma": [True] * 8,
            "prior_growth_drawdown": [0.00] * 8,
            "market_recovery_quality": [False, False, True, True, False, True, True, True],
        },
        index=index,
    )
    candidate = CampaignTrendCandidate(
        "test",
        recovery_confirmation_days=2,
        ramp_stage_sessions=1,
    )
    diagnostics = campaign_loss_state(signals, candidate)
    assert diagnostics.loc[index[1], "state"] == "defense"
    assert diagnostics.loc[index[3], "state"] == "ramp"
    assert diagnostics.loc[index[4], "state"] == "defense"
    assert diagnostics.loc[index[6], "state"] == "ramp"
    assert not diagnostics.loc[index[6], "r38_incremental_enabled"]
