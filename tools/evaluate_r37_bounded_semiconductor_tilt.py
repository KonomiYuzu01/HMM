from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import (
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    GDE_FRACTION,
    simulate_fixed_r11,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r23_high_volatility_tilt_gate as r23
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r34_concave_trend_risk_schedule as r34
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33


OUTPUT = Path("output/r37_bounded_semiconductor_tilt")
CANDIDATE = "r35_with_bounded_semiconductor_tilt5"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
MAX_TILT = 0.05
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.25, 1.35)
SHOCK_NEIGHBORS = (0.95, 1.05)
CASH_NEIGHBORS = (-0.18, -0.22)
TILT_NEIGHBORS = (0.025, 0.075)
CUMULATIVE_TRIALS = 110
MAX_DRAWDOWN_TOLERANCE = 0.005


def _configure(
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
) -> None:
    r34._configure(
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
    max_tilt: float = MAX_TILT,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    _configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, risk_diagnostics = (
        r33.one_session_shock_schedule(
            weights,
            daily,
            closes,
            cash_floor=cash_floor,
        )
    )
    tilted, tilted_daily, tilt_diagnostics = (
        r23.apply_gated_industry_momentum(
            scheduled,
            execution_daily,
            closes,
            max_tilt=max_tilt,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
        opens,
        closes,
        guard_for_multiplier(
            base_multiplier,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    limited, limited_daily, limit_diagnostics = (
        enforce_daily_cash_target_limit(
            guard_weights,
            guard_daily,
            cash_floor=cash_floor,
        )
    )
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=(
            scenario.base_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        gde_one_way_cost_bps=(
            scenario.gde_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        financing_spread_bps=(
            scenario.financing_spread_bps  # type: ignore[attr-defined]
        ),
        weights_override=limited,
        daily_override=limited_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = risk_diagnostics.reindex(trial.index).join(
        tilt_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_tilt",
    ).join(
        limit_diagnostics.reindex(trial.index),
        how="left",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["pre_gde_cash_weight"] = limited.reindex(
        trial.index
    )["CASH"]
    diagnostics["post_guard_semis_weight"] = guard_weights.reindex(
        trial.index
    )["SEMIS"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


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
        ("base1065", 1.065, 1.30, 1.00, -0.20, 0.05, "base"),
        ("base1075", 1.075, 1.30, 1.00, -0.20, 0.05, "base"),
        ("active125", 1.070, 1.25, 1.00, -0.20, 0.05, "active"),
        ("active135", 1.070, 1.35, 1.00, -0.20, 0.05, "active"),
        ("shock095", 1.070, 1.30, 0.95, -0.20, 0.05, "shock"),
        ("shock105", 1.070, 1.30, 1.05, -0.20, 0.05, "shock"),
        ("cash18", 1.070, 1.30, 1.00, -0.18, 0.05, "cash"),
        ("cash22", 1.070, 1.30, 1.00, -0.22, 0.05, "cash"),
        ("tilt025", 1.070, 1.30, 1.00, -0.20, 0.025, "tilt"),
        ("tilt075", 1.070, 1.30, 1.00, -0.20, 0.075, "tilt"),
    ]
    rows: list[dict[str, object]] = []
    try:
        for (
            label,
            base,
            active,
            shock,
            cash,
            tilt,
            family,
        ) in definitions:
            trial, _ = simulate_candidate(
                settings,
                scenario,
                base_multiplier=base,
                active_multiplier=active,
                shock_multiplier=shock,
                cash_floor=cash,
                max_tilt=tilt,
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
                    "max_tilt": tilt,
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
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    neighborhood_pass = True
    for family in ("base", "active", "shock", "cash", "tilt"):
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
    extra = {
        "absolute_risk_cap_pass": bool(
            diagnostics["implemented_non_cash_weight"]
            .le(1.20 + 1e-12)
            .all()
            and diagnostics["pre_gde_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all()
        ),
        "growth_budget_conservation_pass": bool(
            diagnostics["growth_budget_error"]
            .fillna(0.0)
            .le(1e-12)
            .all()
        ),
        "bounded_tilt_pass": bool(
            diagnostics["semis_growth_share"]
            .between(0.45 - 1e-12, 0.55 + 1e-12)
            .all()
        ),
    }
    for gate, passed in extra.items():
        acceptance.loc[len(acceptance)] = {
            "gate": gate,
            "passed": passed,
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
    print("\nR37 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR37 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR37 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
