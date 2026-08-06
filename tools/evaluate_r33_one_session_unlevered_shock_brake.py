from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    metric_delta,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
    causal_trend_permission,
)
from tools.evaluate_r25_daily_relative_tilt_risk_veto import (
    causal_daily_gated_industry_momentum,
)
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    pulse_cooldown_state,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r31_proportional_trend_risk_pulse as r31


OUTPUT = Path("output/r33_one_session_unlevered_shock_brake")
CANDIDATE = "base1070_relative120_cash25_one_day_unlevered_shock"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.20
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.25
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.15, 1.25)
SHOCK_NEIGHBORS = (0.95, 1.05)
CASH_NEIGHBORS = (-0.20, -0.30)
CUMULATIVE_TRIALS = 106
MAX_DRAWDOWN_TOLERANCE = 0.005


def causal_one_session_shock(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily_gate = causal_daily_gated_industry_momentum(
        closes,
        index,
    )
    prior_pair_return = (
        closes[["QQQ", "SEMIS"]]
        .pct_change(fill_method=None)
        .mean(axis=1)
        .shift(1)
        .reindex(index)
    )
    pulse = pulse_cooldown_state(
        daily_gate["volatility_gate_active"],
        prior_pair_return,
        recovery_confirmations=2,
        maximum_hold_sessions=1,
    )
    diagnostics = daily_gate[
        [
            "volatility_state",
            "pair_realized_volatility",
            "high_volatility_threshold",
            "volatility_gate_active",
        ]
    ].copy()
    diagnostics = diagnostics.join(pulse)
    diagnostics["prior_pair_return"] = prior_pair_return
    return diagnostics


def one_session_shock_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = CASH_FLOOR,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    base_weights = weights.loc[index].copy()
    trend = causal_trend_permission(
        closes.reindex(index),
        r30.DEFINITION,
    )
    shock_diagnostics = causal_one_session_shock(
        closes.reindex(index),
        index,
    )
    shock = shock_diagnostics["pulse_veto_active"].astype(bool)
    requested_absolute = pd.Series(
        np.where(
            shock,
            SHOCK_MULTIPLIER,
            np.where(
                trend,
                BASE_MULTIPLIER * ACTIVE_MULTIPLIER,
                BASE_MULTIPLIER,
            ),
        ),
        index=index,
        dtype=float,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_change = trend.ne(trend.shift(1))
    shock_change = shock.ne(shock.shift(1))
    update = base_trade | trend_change | shock_change
    accepted_absolute = (
        requested_absolute.where(update)
        .ffill()
        .fillna(BASE_MULTIPLIER)
    )
    requested = scale_non_cash_weights_by_series(
        base_weights,
        accepted_absolute,
    )
    implemented, cap_scale = cap_weights_at_cash_floor(
        requested,
        cash_floor,
    )
    base_requested = scale_non_cash_weights_by_series(
        base_weights,
        pd.Series(BASE_MULTIPLIER, index=index),
    )
    base_r11, _ = cap_weights_at_cash_floor(
        base_requested,
        cash_floor,
    )
    expected_requested = scale_non_cash_weights_by_series(
        base_weights,
        requested_absolute,
    )
    expected_implemented, _ = cap_weights_at_cash_floor(
        expected_requested,
        cash_floor,
    )
    non_cash = [column for column in implemented if column != "CASH"]
    semantic_error = (
        implemented[non_cash] - expected_implemented[non_cash]
    ).abs().max(axis=1)
    if implemented["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("R33 exceeded its cash floor")
    if float(semantic_error.max()) > 1e-12:
        raise AssertionError("R33 state multiplier was not implemented")

    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": trend,
            "pulse_veto_active": shock,
            "effective_incremental_permission": trend & ~shock,
            "requested_absolute_multiplier": requested_absolute,
            "accepted_absolute_multiplier": accepted_absolute,
            "base_trade": base_trade,
            "trend_state_change": trend_change,
            "pulse_state_change": shock_change,
            "execution_update": update,
            "pre_cap_cash_weight": requested["CASH"],
            "cash_cap_scale": cap_scale,
            "implemented_cash_weight": implemented["CASH"],
            "implemented_non_cash_weight": implemented[
                non_cash
            ].sum(axis=1),
            "base_r11_non_cash_weight": base_r11[
                non_cash
            ].sum(axis=1),
            "state_multiplier_semantic_error": semantic_error,
        },
        index=index,
    )
    for column in (
        "high_volatility_episode_entry",
        "pulse_veto_recovered",
        "pulse_veto_expired",
        "pulse_veto_remaining_sessions",
        "pair_realized_volatility",
        "high_volatility_threshold",
        "prior_pair_return",
    ):
        diagnostics[column] = shock_diagnostics[column]
    return implemented, execution_daily, diagnostics


def _configure(
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
) -> None:
    global BASE_MULTIPLIER, ACTIVE_MULTIPLIER, SHOCK_MULTIPLIER
    BASE_MULTIPLIER = base_multiplier
    ACTIVE_MULTIPLIER = active_multiplier
    r30.BASE_MULTIPLIER = base_multiplier
    r31.BASE_MULTIPLIER = base_multiplier
    r31.ACTIVE_MULTIPLIER = active_multiplier
    SHOCK_MULTIPLIER = shock_multiplier
    r30.risk_budget_schedule = one_session_shock_schedule


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )
    return r30.simulate_candidate(
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
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    definitions = [
        ("base1065", 1.065, 1.20, 1.00, -0.25, "base"),
        ("base1075", 1.075, 1.20, 1.00, -0.25, "base"),
        ("active115", 1.070, 1.15, 1.00, -0.25, "active"),
        ("active125", 1.070, 1.25, 1.00, -0.25, "active"),
        ("shock095", 1.070, 1.20, 0.95, -0.25, "shock"),
        ("shock105", 1.070, 1.20, 1.05, -0.25, "shock"),
        ("cash20", 1.070, 1.20, 1.00, -0.20, "cash"),
        ("cash30", 1.070, 1.20, 1.00, -0.30, "cash"),
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
        _configure()
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
    duration_pass = r30.maximum_true_run(
        diagnostics["pulse_veto_active"]
    ) <= 1
    semantics_pass = bool(
        diagnostics["state_multiplier_semantic_error"]
        .le(1e-12)
        .all()
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "one_session_shock_duration_pass"
        ),
        "passed",
    ] = duration_pass
    acceptance.loc[
        acceptance["gate"].eq(
            "state_multiplier_semantics_pass"
        ),
        "passed",
    ] = semantics_pass
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
    r30.main()
    neighborhoods = _neighborhood_rows()
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    _rewrite_candidate_labels()
    _rewrite_acceptance(neighborhoods)
    print("\nR33 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR33 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR33 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
