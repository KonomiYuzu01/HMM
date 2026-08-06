from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r31_proportional_trend_risk_pulse as r31


OUTPUT = Path("output/r32_proportional_trend")
CANDIDATE = "base1070_both_sma200_relative120_cash25_no_pulse"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.20
CASH_FLOOR = -0.25
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.15, 1.25)
CASH_NEIGHBORS = (-0.20, -0.30)
CUMULATIVE_TRIALS = 105
MAX_DRAWDOWN_TOLERANCE = 0.005


def proportional_trend_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = CASH_FLOOR,
    pulse_enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    implemented, execution_daily, diagnostics = (
        r31.proportional_risk_schedule(
            weights,
            base_daily,
            closes,
            cash_floor=cash_floor,
            pulse_enabled=False,
        )
    )
    if diagnostics["pulse_veto_active"].astype(bool).any():
        raise AssertionError("R32 must not use a pulse veto")
    return implemented, execution_daily, diagnostics


def _configure(
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
) -> None:
    r30.BASE_MULTIPLIER = base_multiplier
    r31.BASE_MULTIPLIER = base_multiplier
    r31.ACTIVE_MULTIPLIER = active_multiplier
    r30.risk_budget_schedule = proportional_trend_schedule


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
    )
    return r30.simulate_candidate(
        settings,
        scenario,
        cash_floor=cash_floor,
        pulse_enabled=False,
    )


def _rewrite_candidate_labels() -> None:
    for path in OUTPUT.glob("*.csv"):
        frame = pd.read_csv(path)
        changed = False
        if "candidate" in frame:
            frame["candidate"] = CANDIDATE
            changed = True
        if path.name == "summary.csv" and set(frame.columns) >= {
            "Unnamed: 0",
            "value",
        }:
            selected = frame["Unnamed: 0"].eq("candidate")
            frame.loc[selected, "value"] = CANDIDATE
            changed = True
        if changed:
            frame.to_csv(path, index=False)


def _neighborhood_rows() -> pd.DataFrame:
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    definitions = [
        (
            "base1065",
            BASE_NEIGHBORS[0],
            ACTIVE_MULTIPLIER,
            CASH_FLOOR,
            "base",
        ),
        (
            "base1075",
            BASE_NEIGHBORS[1],
            ACTIVE_MULTIPLIER,
            CASH_FLOOR,
            "base",
        ),
        (
            "active115",
            BASE_MULTIPLIER,
            ACTIVE_NEIGHBORS[0],
            CASH_FLOOR,
            "active",
        ),
        (
            "active125",
            BASE_MULTIPLIER,
            ACTIVE_NEIGHBORS[1],
            CASH_FLOOR,
            "active",
        ),
        (
            "cash20",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            CASH_NEIGHBORS[0],
            "cash",
        ),
        (
            "cash30",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            CASH_NEIGHBORS[1],
            "cash",
        ),
    ]
    rows: list[dict[str, object]] = []
    try:
        for label, base, active, cash, family in definitions:
            trial, _ = simulate_candidate(
                settings,
                scenario,
                base_multiplier=base,
                active_multiplier=active,
                cash_floor=cash,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            rows.append(
                {
                    "neighborhood": label,
                    "family": family,
                    "base_multiplier": base,
                    "active_multiplier": active,
                    "cash_floor": cash,
                    "pulse_enabled": False,
                    **delta,
                    "relative_positive": bool(
                        delta["candidate_cagr"]
                        > delta["baseline_cagr"]
                    ),
                    "point_target_pass": bool(
                        delta["candidate_cagr"] >= 0.25
                        and delta["candidate_max_drawdown"]
                        >= (
                            delta["baseline_max_drawdown"]
                            - MAX_DRAWDOWN_TOLERANCE
                        )
                    ),
                }
            )
    finally:
        _configure()
    return pd.DataFrame(rows)


def _rewrite_acceptance(neighborhoods: pd.DataFrame) -> None:
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    neighborhood_pass = True
    for family in ("base", "active", "cash"):
        selected = neighborhoods.loc[
            neighborhoods["family"].eq(family)
        ]
        neighborhood_pass = bool(
            neighborhood_pass
            and selected["relative_positive"].astype(bool).all()
            and selected["point_target_pass"].astype(bool).any()
        )
    acceptance.loc[
        acceptance["gate"].eq("parameter_neighborhood_pass"),
        "passed",
    ] = neighborhood_pass
    acceptance.loc[
        acceptance["gate"].eq("pulse_duration_pass"),
        "gate",
    ] = "no_pulse_rule_pass"
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    no_pulse_pass = bool(
        ~diagnostics["pulse_veto_active"].astype(bool).any()
    )
    acceptance.loc[
        acceptance["gate"].eq("no_pulse_rule_pass"),
        "passed",
    ] = no_pulse_pass
    non_production = acceptance["gate"].ne("production_pass")
    production = bool(
        acceptance.loc[non_production, "passed"].astype(bool).all()
    )
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = production
    acceptance.to_csv(acceptance_path, index=False)

    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("production_pass"),
        "value",
    ] = str(production)
    summary.to_csv(summary_path, index=False)


def main() -> None:
    _configure()
    r30.OUTPUT = OUTPUT
    r30.CASH_FLOOR = CASH_FLOOR
    r30.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r30.main()
    neighborhoods = _neighborhood_rows()
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    _rewrite_candidate_labels()
    _rewrite_acceptance(neighborhoods)
    print("\nR32 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR32 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR32 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
