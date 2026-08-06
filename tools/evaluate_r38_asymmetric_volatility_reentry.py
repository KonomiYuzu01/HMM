from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r38_accelerating_volatility_capacity_fill as accel
import tools.evaluate_r38_convex_semiconductor_overlay as r38
import tools.evaluate_r38_volatility_conditioned_capacity_fill as vc


OUTPUT = Path("output/r38_asymmetric_volatility_reentry")
CANDIDATE = "r38_asymmetric_volatility_reentry5"
CONFIRMATION_DAYS = 5
CUMULATIVE_TRIALS = 180


def asymmetric_stability_permission(
    raw_stable: pd.Series,
    *,
    confirmation_days: int = CONFIRMATION_DAYS,
) -> pd.Series:
    if confirmation_days < 1:
        raise ValueError("confirmation_days must be positive")
    raw = raw_stable.fillna(False).astype(bool)
    accepted = pd.Series(False, index=raw.index, dtype=bool)
    state = False
    stable_run = 0
    for position, value in enumerate(raw.to_numpy(dtype=bool)):
        if not value:
            state = False
            stable_run = 0
        elif state:
            stable_run += 1
        else:
            stable_run += 1
            if stable_run >= confirmation_days:
                state = True
        accepted.iloc[position] = state
    return accepted.rename("asymmetric_stability_permission")


def accelerating_volatility_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = vc.CASH_FLOOR,
    low_vol_active_multiplier: float = (
        vc.LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        vc.HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    confirmation_days: int = CONFIRMATION_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = accel.causal_volatility_permission

    def confirmed_permission(
        state_closes: pd.DataFrame,
    ) -> pd.Series:
        raw = original(state_closes)
        return asymmetric_stability_permission(
            raw,
            confirmation_days=confirmation_days,
        )

    accel.causal_volatility_permission = confirmed_permission
    try:
        implemented, execution_daily, diagnostics = (
            accel.accelerating_volatility_schedule(
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
            )
        )
    finally:
        accel.causal_volatility_permission = original
    diagnostics["reentry_confirmation_days"] = confirmation_days
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
    confirmation_days: int = CONFIRMATION_DAYS,
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
            confirmation_days=confirmation_days,
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


def _reentry_neighborhood() -> pd.DataFrame:
    settings = r21._build_samples()["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    r11_path, _, _ = simulate_fixed_r11(settings, scenario)
    rows: list[dict[str, object]] = []
    for days in (3, 5, 7):
        trial, diagnostics = simulate_candidate(
            settings,
            scenario,
            confirmation_days=days,
        )
        delta = metric_delta(
            r11_path["net_return"],
            trial["net_return"],
        )
        rows.append(
            {
                "confirmation_days": days,
                "state_changes": int(
                    diagnostics[
                        "high_volatility_state_change"
                    ].sum()
                ),
                **delta,
                "point_target_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= (
                        delta["baseline_max_drawdown"]
                        - vc.R11_DRAWDOWN_TOLERANCE
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


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

    reentry = _reentry_neighborhood()
    reentry.to_csv(
        OUTPUT / "reentry_confirmation_neighborhood.csv",
        index=False,
    )
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    raw_diagnostics = pd.read_csv(
        accel.OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    reentry_pass = bool(
        reentry["point_target_pass"].astype(bool).all()
    )
    switching_pass = bool(
        diagnostics["high_volatility_state_change"].sum()
        < raw_diagnostics[
            "high_volatility_state_change"
        ].sum()
    )
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    acceptance.loc[len(acceptance)] = {
        "gate": "reentry_confirmation_neighborhood_pass",
        "passed": reentry_pass,
    }
    acceptance.loc[len(acceptance)] = {
        "gate": "state_switching_reduced",
        "passed": switching_pass,
    }
    research = bool(
        acceptance.loc[
            acceptance["gate"].ne("research_pass"),
            "passed",
        ]
        .astype(bool)
        .all()
    )
    acceptance.loc[
        acceptance["gate"].eq("research_pass"),
        "passed",
    ] = research
    acceptance.to_csv(acceptance_path, index=False)
    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("research_pass"),
        "value",
    ] = str(research)
    summary.to_csv(summary_path, index=False)
    print("\nReentry confirmation neighborhood:")
    print(reentry.round(6).to_string(index=False))
    print("\nFinal acceptance:")
    print(acceptance.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

