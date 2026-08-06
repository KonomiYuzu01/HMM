from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/loss_triggered_bear_governor")


@dataclass(frozen=True)
class LossTriggeredCandidate:
    name: str
    entry_drawdown: float = -0.10
    rearm_drawdown: float = -0.05
    recovery_confirmation_days: int = 10
    ramp_stage_sessions: int = 14

    def __post_init__(self) -> None:
        if not -1.0 < self.entry_drawdown < self.rearm_drawdown < 0.0:
            raise ValueError("drawdown thresholds must provide rearm hysteresis")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = LossTriggeredCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="entry_08", entry_drawdown=-0.08),
    replace(CENTRAL, name="entry_12", entry_drawdown=-0.12),
    replace(CENTRAL, name="rearm_03", rearm_drawdown=-0.03),
    replace(CENTRAL, name="rearm_07", rearm_drawdown=-0.07),
    replace(CENTRAL, name="confirm_05", recovery_confirmation_days=5),
    replace(CENTRAL, name="confirm_15", recovery_confirmation_days=15),
    replace(CENTRAL, name="ramp_07", ramp_stage_sessions=7),
    replace(CENTRAL, name="ramp_21", ramp_stage_sessions=21),
)


def loss_triggered_state(
    signals: pd.DataFrame,
    candidate: LossTriggeredCandidate,
) -> pd.DataFrame:
    state = "normal"
    armed = True
    recovery_run = 0
    ramp_day = 0
    rows: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        reference_drawdown = float(signal["prior_reference_drawdown"])
        if not armed and reference_drawdown >= candidate.rearm_drawdown:
            armed = True
        entry = bool(armed and reference_drawdown <= candidate.entry_drawdown)
        release_quality = bool(signal["growth_above_sma"])
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
            armed = False
            recovery_run = 0
            ramp_day = 0
        elif state == "defense":
            recovery_run = recovery_run + 1 if release_quality else 0
            if recovery_run >= candidate.recovery_confirmation_days:
                state = "ramp"
                ramp_day = 0
        elif state == "ramp":
            ramp_day += 1
            if ramp_day >= candidate.ramp_stage_sessions * len(reference.RAMP_CAPS):
                state = "normal"
                recovery_run = 0
                ramp_day = 0

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
                "armed": armed,
                "entry": entry,
                "state_change": state != previous_state,
                "stage_change": state == "ramp" and stage != previous_stage,
                "release_quality": release_quality,
                "recovery_run": recovery_run,
                "ramp_day": ramp_day,
                "growth_cap": growth_cap,
                "r38_incremental_enabled": state == "normal",
            }
        )
    return signals.join(pd.DataFrame(rows, index=signals.index))


def main() -> None:
    original_output = reference.OUTPUT
    original_central = reference.CENTRAL
    original_neighbors = reference.NEIGHBORS
    original_state = reference.reference_governor_state
    try:
        reference.OUTPUT = OUTPUT
        reference.CENTRAL = CENTRAL  # type: ignore[assignment]
        reference.NEIGHBORS = NEIGHBORS  # type: ignore[assignment]
        reference.reference_governor_state = loss_triggered_state  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.reference_governor_state = original_state


if __name__ == "__main__":
    main()
