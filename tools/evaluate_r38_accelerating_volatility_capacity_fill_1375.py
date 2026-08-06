from __future__ import annotations

from pathlib import Path

import pandas as pd

import tools.evaluate_r38_accelerating_volatility_capacity_fill as accel
import tools.evaluate_r38_volatility_conditioned_capacity_fill as vc


OUTPUT = Path(
    "output/r38_accelerating_volatility_capacity_fill_1375"
)
CANDIDATE = "r38_accelerating_volatility_capacity_fill_1375"
LOW_VOL_ACTIVE_MULTIPLIER = 1.375
HIGH_VOL_ACTIVE_MULTIPLIER = 1.30
CUMULATIVE_TRIALS = 184


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    low_vol_active_multiplier: float = (
        LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    cash_floor: float = vc.CASH_FLOOR,
    overlay_fraction: float = vc.OVERLAY_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return accel.simulate_candidate(
        settings,
        scenario,
        low_vol_active_multiplier=low_vol_active_multiplier,
        high_vol_active_multiplier=high_vol_active_multiplier,
        cash_floor=cash_floor,
        overlay_fraction=overlay_fraction,
    )


def _definitions() -> list[tuple[str, float, float]]:
    return [
        ("central", 1.375, 1.30),
        ("low1350", 1.35, 1.30),
        ("low1400", 1.40, 1.30),
        ("high1275", 1.375, 1.275),
        ("high1325", 1.375, 1.325),
        ("identity_r38", 1.30, 1.30),
        ("identity_fixed1375", 1.375, 1.375),
    ]


def main() -> None:
    original_output = vc.OUTPUT
    original_candidate = vc.CANDIDATE
    original_trials = vc.CUMULATIVE_TRIALS
    original_simulate = vc.simulate_candidate
    original_definitions = vc._definitions
    vc.OUTPUT = OUTPUT
    vc.CANDIDATE = CANDIDATE
    vc.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    vc.simulate_candidate = simulate_candidate
    vc._definitions = _definitions
    try:
        vc.main()
    finally:
        vc.OUTPUT = original_output
        vc.CANDIDATE = original_candidate
        vc.CUMULATIVE_TRIALS = original_trials
        vc.simulate_candidate = original_simulate
        vc._definitions = original_definitions


if __name__ == "__main__":
    main()

