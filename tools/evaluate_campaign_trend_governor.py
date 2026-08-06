from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
import tools.evaluate_campaign_loss_governor as campaign
import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/campaign_trend_governor")


@dataclass(frozen=True)
class CampaignTrendCandidate:
    name: str
    entry_drawdown: float = -0.10
    recovery_confirmation_days: int = 10
    ramp_stage_sessions: int = 7

    def __post_init__(self) -> None:
        if not -1.0 < self.entry_drawdown < 0.0:
            raise ValueError("entry_drawdown must be between -1 and 0")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = CampaignTrendCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="entry_08", entry_drawdown=-0.08),
    replace(CENTRAL, name="entry_12", entry_drawdown=-0.12),
    replace(CENTRAL, name="confirm_05", recovery_confirmation_days=5),
    replace(CENTRAL, name="confirm_15", recovery_confirmation_days=15),
    replace(CENTRAL, name="ramp_03", ramp_stage_sessions=3),
    replace(CENTRAL, name="ramp_14", ramp_stage_sessions=14),
)


def causal_campaign_trend_signals(
    closes: pd.DataFrame,
    reference_path: pd.DataFrame,
    *,
    sma_sessions: int = reference.SMA_SESSIONS,
) -> pd.DataFrame:
    signals = campaign.causal_campaign_signals(
        closes,
        reference_path,
        sma_sessions=sma_sessions,
    )
    trend = bear.causal_bear_signals(
        closes,
        sma_sessions=sma_sessions,
        drawdown_lookback=max(bear.MARKET_DRAWDOWN_LOOKBACK, sma_sessions),
    )
    signals["market_recovery_quality"] = trend["both_above_sma"]
    return signals


def main() -> None:
    original_output = reference.OUTPUT
    original_central = reference.CENTRAL
    original_neighbors = reference.NEIGHBORS
    original_signals = reference.causal_reference_signals
    original_state = reference.reference_governor_state
    try:
        reference.OUTPUT = OUTPUT
        reference.CENTRAL = CENTRAL  # type: ignore[assignment]
        reference.NEIGHBORS = NEIGHBORS  # type: ignore[assignment]
        reference.causal_reference_signals = causal_campaign_trend_signals
        reference.reference_governor_state = campaign.campaign_loss_state  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.causal_reference_signals = original_signals
        reference.reference_governor_state = original_state


if __name__ == "__main__":
    main()
