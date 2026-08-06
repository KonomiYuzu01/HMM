from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/campaign_loss_governor")
MARKET_PEAK_SESSIONS = 252
MIN_MARKET_PEAK_SESSIONS = 60
BASE_CAUSAL_REFERENCE_SIGNALS = reference.causal_reference_signals


@dataclass(frozen=True)
class CampaignLossCandidate:
    name: str
    entry_drawdown: float = -0.10
    market_release_drawdown: float = -0.10
    recovery_confirmation_days: int = 10
    ramp_stage_sessions: int = 14

    def __post_init__(self) -> None:
        if not -1.0 < self.entry_drawdown < 0.0:
            raise ValueError("entry_drawdown must be between -1 and 0")
        if not -1.0 < self.market_release_drawdown < 0.0:
            raise ValueError("market_release_drawdown must be between -1 and 0")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = CampaignLossCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="entry_08", entry_drawdown=-0.08),
    replace(CENTRAL, name="entry_12", entry_drawdown=-0.12),
    replace(CENTRAL, name="release_08", market_release_drawdown=-0.08),
    replace(CENTRAL, name="release_12", market_release_drawdown=-0.12),
    replace(CENTRAL, name="confirm_05", recovery_confirmation_days=5),
    replace(CENTRAL, name="confirm_15", recovery_confirmation_days=15),
    replace(CENTRAL, name="ramp_07", ramp_stage_sessions=7),
    replace(CENTRAL, name="ramp_21", ramp_stage_sessions=21),
)


def causal_campaign_signals(
    closes: pd.DataFrame,
    reference_path: pd.DataFrame,
    *,
    sma_sessions: int = reference.SMA_SESSIONS,
) -> pd.DataFrame:
    signals = BASE_CAUSAL_REFERENCE_SIGNALS(
        closes,
        reference_path,
        sma_sessions=sma_sessions,
    )
    if "equity" not in reference_path:
        raise ValueError("Reference path is missing equity")
    prices = closes[["QQQ", "SEMIS"]].astype(float)
    growth_return = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    growth_index = (1.0 + growth_return).cumprod()
    prior_growth = growth_index.shift(1)
    prior_peak = growth_index.rolling(
        MARKET_PEAK_SESSIONS,
        min_periods=MIN_MARKET_PEAK_SESSIONS,
    ).max().shift(1)
    signals["prior_growth_drawdown"] = (prior_growth / prior_peak - 1.0).fillna(-1.0)
    signals["prior_reference_equity"] = reference_path["equity"].shift(1)
    signals["prior_reference_equity"] = (
        signals["prior_reference_equity"].ffill().fillna(1.0)
    )
    return signals


def campaign_loss_state(
    signals: pd.DataFrame,
    candidate: CampaignLossCandidate,
) -> pd.DataFrame:
    state = "normal"
    recovery_run = 0
    ramp_day = 0
    campaign_peak: float | None = None
    rows: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        reference_equity = float(signal["prior_reference_equity"])
        if campaign_peak is None:
            campaign_peak = reference_equity
        if state == "normal":
            campaign_peak = max(campaign_peak, reference_equity)
        campaign_drawdown = reference_equity / campaign_peak - 1.0
        entry = bool(
            state == "normal" and campaign_drawdown <= candidate.entry_drawdown
        )
        if "market_recovery_quality" in signal.index:
            release_quality = bool(signal["market_recovery_quality"])
        else:
            release_quality = bool(
                signal["growth_above_sma"]
                and float(signal["prior_growth_drawdown"])
                >= candidate.market_release_drawdown
            )
        previous_state = state
        previous_stage = (
            min(
                ramp_day // candidate.ramp_stage_sessions,
                len(reference.RAMP_CAPS) - 1,
            )
            if state == "ramp"
            else -1
        )
        if entry:
            state = "defense"
            recovery_run = 0
            ramp_day = 0
        elif state == "defense":
            recovery_run = recovery_run + 1 if release_quality else 0
            if recovery_run >= candidate.recovery_confirmation_days:
                state = "ramp"
                ramp_day = 0
        elif state == "ramp":
            if not release_quality:
                state = "defense"
                recovery_run = 0
                ramp_day = 0
            else:
                ramp_day += 1
                if ramp_day >= candidate.ramp_stage_sessions * len(
                    reference.RAMP_CAPS
                ):
                    state = "normal"
                    recovery_run = 0
                    ramp_day = 0
                    campaign_peak = reference_equity

        stage = -1
        growth_cap = 1.0
        if state == "defense":
            growth_cap = 0.0
        elif state == "ramp":
            stage = min(
                ramp_day // candidate.ramp_stage_sessions,
                len(reference.RAMP_CAPS) - 1,
            )
            growth_cap = reference.RAMP_CAPS[stage]
        rows.append(
            {
                "state": state,
                "entry": entry,
                "state_change": state != previous_state,
                "stage_change": state == "ramp" and stage != previous_stage,
                "release_quality": release_quality,
                "recovery_run": recovery_run,
                "ramp_day": ramp_day,
                "campaign_peak": campaign_peak,
                "campaign_drawdown": campaign_drawdown,
                "growth_cap": growth_cap,
                "r38_incremental_enabled": state == "normal",
            }
        )
    return signals.join(pd.DataFrame(rows, index=signals.index))


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
        reference.causal_reference_signals = causal_campaign_signals
        reference.reference_governor_state = campaign_loss_state  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.causal_reference_signals = original_signals
        reference.reference_governor_state = original_state


if __name__ == "__main__":
    main()
