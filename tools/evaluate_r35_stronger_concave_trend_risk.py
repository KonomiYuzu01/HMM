from __future__ import annotations

from pathlib import Path

import pandas as pd

import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r34_concave_trend_risk_schedule as r34


OUTPUT = Path("output/r35_stronger_concave_trend_risk")
CANDIDATE = "base1070_relative130_cash20_one_day_unlevered_shock"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.25, 1.35)
SHOCK_NEIGHBORS = (0.95, 1.05)
CASH_NEIGHBORS = (-0.18, -0.22)
CUMULATIVE_TRIALS = 108


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    return r34.simulate_candidate(
        settings,
        scenario,
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
        cash_floor=cash_floor,
    )


def _configure_helpers() -> None:
    r34.OUTPUT = OUTPUT
    r34.CANDIDATE = CANDIDATE
    r34.BASE_MULTIPLIER = BASE_MULTIPLIER
    r34.ACTIVE_MULTIPLIER = ACTIVE_MULTIPLIER
    r34.SHOCK_MULTIPLIER = SHOCK_MULTIPLIER
    r34.CASH_FLOOR = CASH_FLOOR
    r34.BASE_NEIGHBORS = BASE_NEIGHBORS
    r34.ACTIVE_NEIGHBORS = ACTIVE_NEIGHBORS
    r34.SHOCK_NEIGHBORS = SHOCK_NEIGHBORS
    r34.CASH_NEIGHBORS = CASH_NEIGHBORS
    r34.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r34._configure(
        base_multiplier=BASE_MULTIPLIER,
        active_multiplier=ACTIVE_MULTIPLIER,
        shock_multiplier=SHOCK_MULTIPLIER,
    )


def main() -> None:
    _configure_helpers()
    r30.OUTPUT = OUTPUT
    r30.CASH_FLOOR = CASH_FLOOR
    r30.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r30.simulate_candidate = simulate_candidate
    r30.main()
    neighborhoods = r34._neighborhood_rows()
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    r34._rewrite_candidate_labels()
    r34._rewrite_acceptance(neighborhoods)
    print("\nR35 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR35 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR35 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
