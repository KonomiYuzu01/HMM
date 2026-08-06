from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/r38_bear_veto")


@dataclass(frozen=True)
class R38BearVetoCandidate:
    name: str
    drawdown_trigger: float = -0.10
    recovery_confirmation_days: int = 20

    def __post_init__(self) -> None:
        if not -1.0 < self.drawdown_trigger < 0.0:
            raise ValueError("drawdown_trigger must be between -1 and 0")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")


CENTRAL = R38BearVetoCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="trigger_08", drawdown_trigger=-0.08),
    replace(CENTRAL, name="trigger_12", drawdown_trigger=-0.12),
    replace(CENTRAL, name="confirm_10", recovery_confirmation_days=10),
    replace(CENTRAL, name="confirm_30", recovery_confirmation_days=30),
)


def causal_r38_veto_signals(
    closes: pd.DataFrame,
    reference_path: pd.DataFrame,
    *,
    sma_sessions: int = reference.SMA_SESSIONS,
) -> pd.DataFrame:
    del reference_path
    return bear.causal_bear_signals(
        closes,
        sma_sessions=sma_sessions,
        drawdown_lookback=max(bear.MARKET_DRAWDOWN_LOOKBACK, sma_sessions),
    )


def r38_bear_veto_state(
    signals: pd.DataFrame,
    candidate: R38BearVetoCandidate,
) -> pd.DataFrame:
    state = "normal"
    recovery_run = 0
    rows: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        entry_condition = bool(
            float(signal["prior_growth_drawdown"])
            <= candidate.drawdown_trigger
            and bool(signal["both_below_sma"])
        )
        previous_state = state
        if entry_condition:
            state = "defense"
            recovery_run = 0
        elif state == "defense":
            recovery_run = recovery_run + 1 if signal["both_above_sma"] else 0
            if recovery_run >= candidate.recovery_confirmation_days:
                state = "normal"
                recovery_run = 0
        rows.append(
            {
                "state": state,
                "entry": entry_condition and previous_state == "normal",
                "state_change": state != previous_state,
                "stage_change": False,
                "recovery_run": recovery_run,
                "ramp_day": 0,
                "growth_cap": 1.0,
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
        reference.causal_reference_signals = causal_r38_veto_signals
        reference.reference_governor_state = r38_bear_veto_state  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.causal_reference_signals = original_signals
        reference.reference_governor_state = original_state


if __name__ == "__main__":
    main()
