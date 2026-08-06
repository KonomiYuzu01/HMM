from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33


OUTPUT = Path("output/r34_concave_trend_risk_schedule")
CANDIDATE = "base1070_relative125_cash20_one_day_unlevered_shock"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.25
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.20, 1.30)
SHOCK_NEIGHBORS = (0.95, 1.05)
CASH_NEIGHBORS = (-0.18, -0.22)
CUMULATIVE_TRIALS = 107
MAX_DRAWDOWN_TOLERANCE = 0.005

_ORIGINAL_R30_SIMULATE_CANDIDATE = r30.simulate_candidate


def _configure(
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
) -> None:
    r33._configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )
    r30.risk_budget_schedule = r33.one_session_shock_schedule


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
    _configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )
    return _ORIGINAL_R30_SIMULATE_CANDIDATE(
        settings,
        scenario,
        cash_floor=cash_floor,
        pulse_enabled=True,
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
    settings = r21._build_samples()["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    definitions = [
        (
            f"base{int(round(BASE_NEIGHBORS[0] * 1000)):04d}",
            BASE_NEIGHBORS[0],
            ACTIVE_MULTIPLIER,
            SHOCK_MULTIPLIER,
            CASH_FLOOR,
            "base",
        ),
        (
            f"base{int(round(BASE_NEIGHBORS[1] * 1000)):04d}",
            BASE_NEIGHBORS[1],
            ACTIVE_MULTIPLIER,
            SHOCK_MULTIPLIER,
            CASH_FLOOR,
            "base",
        ),
        (
            f"active{int(round(ACTIVE_NEIGHBORS[0] * 100)):03d}",
            BASE_MULTIPLIER,
            ACTIVE_NEIGHBORS[0],
            SHOCK_MULTIPLIER,
            CASH_FLOOR,
            "active",
        ),
        (
            f"active{int(round(ACTIVE_NEIGHBORS[1] * 100)):03d}",
            BASE_MULTIPLIER,
            ACTIVE_NEIGHBORS[1],
            SHOCK_MULTIPLIER,
            CASH_FLOOR,
            "active",
        ),
        (
            f"shock{int(round(SHOCK_NEIGHBORS[0] * 100)):03d}",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            SHOCK_NEIGHBORS[0],
            CASH_FLOOR,
            "shock",
        ),
        (
            f"shock{int(round(SHOCK_NEIGHBORS[1] * 100)):03d}",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            SHOCK_NEIGHBORS[1],
            CASH_FLOOR,
            "shock",
        ),
        (
            f"cash{int(round(abs(CASH_NEIGHBORS[0]) * 100)):02d}",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            SHOCK_MULTIPLIER,
            CASH_NEIGHBORS[0],
            "cash",
        ),
        (
            f"cash{int(round(abs(CASH_NEIGHBORS[1]) * 100)):02d}",
            BASE_MULTIPLIER,
            ACTIVE_MULTIPLIER,
            SHOCK_MULTIPLIER,
            CASH_NEIGHBORS[1],
            "cash",
        ),
    ]
    rows: list[dict[str, object]] = []
    try:
        for label, base, active, shock, cash, family in definitions:
            trial, _ = simulate_candidate(
                settings,
                scenario,
                base_multiplier=base,
                active_multiplier=active,
                shock_multiplier=shock,
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
                    "shock_multiplier": shock,
                    "cash_floor": cash,
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
        _configure(
            base_multiplier=BASE_MULTIPLIER,
            active_multiplier=ACTIVE_MULTIPLIER,
            shock_multiplier=SHOCK_MULTIPLIER,
        )
    return pd.DataFrame(rows)


def _rewrite_acceptance(neighborhoods: pd.DataFrame) -> None:
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    neighborhood_pass = True
    for family in ("base", "active", "shock", "cash"):
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
    ] = "one_session_shock_duration_pass"
    acceptance.loc[
        acceptance["gate"].eq(
            "disabled_increment_equals_r11_pass"
        ),
        "gate",
    ] = "state_multiplier_semantics_pass"
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "one_session_shock_duration_pass"
        ),
        "passed",
    ] = (
        r30.maximum_true_run(
            diagnostics["pulse_veto_active"]
        )
        <= 1
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "state_multiplier_semantics_pass"
        ),
        "passed",
    ] = bool(
        diagnostics["state_multiplier_semantic_error"]
        .le(1e-12)
        .all()
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "absolute_risk_cap_pass",
        "passed": bool(
            diagnostics["implemented_non_cash_weight"]
            .le(1.0 - CASH_FLOOR + 1e-12)
            .all()
            and diagnostics["pre_gde_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all()
        ),
    }
    non_production = acceptance["gate"].ne("production_pass")
    production = bool(
        acceptance.loc[non_production, "passed"].astype(bool).all()
    )
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = production
    acceptance.to_csv(path, index=False)
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
    r30.simulate_candidate = simulate_candidate
    r30.main()
    neighborhoods = _neighborhood_rows()
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    _rewrite_candidate_labels()
    _rewrite_acceptance(neighborhoods)
    print("\nR34 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR34 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR34 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
