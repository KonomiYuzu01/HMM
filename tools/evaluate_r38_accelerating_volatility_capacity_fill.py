from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r18_incremental_volatility_permission import (
    causal_volatility_permission,
)
import tools.evaluate_r38_volatility_conditioned_capacity_fill as vc
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path(
    "output/r38_accelerating_volatility_capacity_fill"
)
CANDIDATE = "r38_accelerating_volatility_capacity_fill"
CUMULATIVE_TRIALS = 172


def accelerating_volatility_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = vc.CASH_FLOOR,
    low_vol_active_multiplier: float | pd.Series = (
        vc.LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        vc.HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    base_multiplier: float = vc.BASE_MULTIPLIER,
    shock_multiplier: float = vc.SHOCK_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = vc.r33.causal_one_session_shock

    def shock_with_acceleration_state(
        state_closes: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        diagnostics = original(state_closes, index)
        stable = causal_volatility_permission(
            state_closes
        ).reindex(index).fillna(False)
        diagnostics["volatility_gate_active"] = ~stable
        diagnostics["volatility_acceleration_block"] = ~stable
        return diagnostics

    vc.r33.causal_one_session_shock = (
        shock_with_acceleration_state
    )
    try:
        implemented, execution_daily, diagnostics = (
            vc.volatility_conditioned_schedule(
                weights,
                base_daily,
                closes,
                cash_floor=cash_floor,
                low_vol_active_multiplier=(
                    low_vol_active_multiplier
                ),
                high_vol_active_multiplier=(
                    high_vol_active_multiplier
                ),
                base_multiplier=base_multiplier,
                shock_multiplier=shock_multiplier,
            )
        )
    finally:
        vc.r33.causal_one_session_shock = original
    diagnostics["volatility_acceleration_block"] = diagnostics[
        "high_volatility_active"
    ]
    return implemented, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    low_vol_active_multiplier: float = (
        vc.LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        vc.HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    cash_floor: float = vc.CASH_FLOOR,
    overlay_fraction: float = vc.OVERLAY_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = r38.r37.r33.one_session_shock_schedule

    def schedule(
        weights: pd.DataFrame,
        base_daily: pd.DataFrame,
        closes: pd.DataFrame,
        *,
        cash_floor: float = cash_floor,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return accelerating_volatility_schedule(
            weights,
            base_daily,
            closes,
            cash_floor=cash_floor,
            low_vol_active_multiplier=low_vol_active_multiplier,
            high_vol_active_multiplier=high_vol_active_multiplier,
        )

    r38.r37.r33.one_session_shock_schedule = schedule
    try:
        return r38.simulate_candidate(
            settings,
            scenario,
            base_multiplier=vc.BASE_MULTIPLIER,
            active_multiplier=low_vol_active_multiplier,
            shock_multiplier=vc.SHOCK_MULTIPLIER,
            cash_floor=cash_floor,
            overlay_fraction=overlay_fraction,
        )
    finally:
        r38.r37.r33.one_session_shock_schedule = original


def main() -> None:
    original_output = vc.OUTPUT
    original_candidate = vc.CANDIDATE
    original_trials = vc.CUMULATIVE_TRIALS
    original_simulate = vc.simulate_candidate
    vc.OUTPUT = OUTPUT
    vc.CANDIDATE = CANDIDATE
    vc.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    vc.simulate_candidate = simulate_candidate
    try:
        vc.main()
    finally:
        vc.OUTPUT = original_output
        vc.CANDIDATE = original_candidate
        vc.CUMULATIVE_TRIALS = original_trials
        vc.simulate_candidate = original_simulate


if __name__ == "__main__":
    main()
