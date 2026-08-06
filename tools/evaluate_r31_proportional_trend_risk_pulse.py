from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    metric_delta,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    simulate_fixed_r11,
)
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
    causal_trend_permission,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r25_daily_relative_tilt_risk_veto import (
    causal_daily_gated_industry_momentum,
)
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    pulse_cooldown_state,
)
import tools.evaluate_r30_trend_risk_budget_pulse as r30


OUTPUT = Path("output/r31_proportional_trend_risk_pulse")
CANDIDATE = "both_sma200_relative120_absolute_recovery_pulse4"
ACTIVE_MULTIPLIER = 1.20
ACTIVE_NEIGHBORS = (1.15, 1.25)
CASH_FLOOR = -0.25
CUMULATIVE_TRIALS = 104
MAX_DRAWDOWN_TOLERANCE = 0.005


def causal_absolute_recovery_pulse(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily_gate = causal_daily_gated_industry_momentum(
        closes,
        index,
    )
    pair_return = (
        closes[["QQQ", "SEMIS"]]
        .pct_change(fill_method=None)
        .mean(axis=1)
        .shift(1)
        .reindex(index)
    )
    pulse = pulse_cooldown_state(
        daily_gate["volatility_gate_active"],
        pair_return,
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
    diagnostics["prior_pair_return"] = pair_return
    return diagnostics


def proportional_risk_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = CASH_FLOOR,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    base_weights = weights.loc[index].copy()
    trend = causal_trend_permission(
        closes.reindex(index),
        r30.DEFINITION,
    )
    pulse_diagnostics = causal_absolute_recovery_pulse(
        closes.reindex(index),
        index,
    )
    pulse_active = pulse_diagnostics["pulse_veto_active"].astype(bool)
    if not pulse_enabled:
        pulse_active = pd.Series(False, index=index)
    effective = r30.effective_incremental_permission(
        trend,
        pulse_active,
    )
    requested_relative = pd.Series(
        np.where(effective, ACTIVE_MULTIPLIER, 1.0),
        index=index,
        dtype=float,
    )
    requested_absolute = BASE_MULTIPLIER * requested_relative
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_change = trend.ne(trend.shift(1))
    pulse_change = pulse_active.ne(pulse_active.shift(1))
    update = base_trade | trend_change | pulse_change
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
    non_cash = [column for column in implemented if column != "CASH"]
    if implemented["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("Proportional risk exceeded cash floor")
    if (
        implemented.loc[~effective, non_cash]
        - base_r11.loc[~effective, non_cash]
    ).abs().max().max() > 1e-12:
        raise AssertionError("Disabled increment did not return to R11")
    if (
        implemented.loc[effective, non_cash].sum(axis=1)
        + 1e-12
        < base_r11.loc[effective, non_cash].sum(axis=1)
    ).any():
        raise AssertionError("Proportional increment reduced R11")

    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": trend,
            "pulse_veto_active": pulse_active,
            "effective_incremental_permission": effective,
            "relative_multiplier": requested_relative,
            "fill_absolute_multiplier": (
                BASE_MULTIPLIER * ACTIVE_MULTIPLIER
            ),
            "requested_absolute_multiplier": requested_absolute,
            "accepted_absolute_multiplier": accepted_absolute,
            "base_trade": base_trade,
            "trend_state_change": trend_change,
            "pulse_state_change": pulse_change,
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
        diagnostics[column] = pulse_diagnostics[column]
    return implemented, execution_daily, diagnostics


def _rewrite_candidate_labels() -> None:
    for path in OUTPUT.glob("*.csv"):
        frame = pd.read_csv(path)
        changed = False
        if "candidate" in frame:
            frame["candidate"] = CANDIDATE
            changed = True
        if path.name == "summary.csv" and set(
            frame.columns
        ) >= {"Unnamed: 0", "value"}:
            selected = frame["Unnamed: 0"].eq("candidate")
            frame.loc[selected, "value"] = CANDIDATE
            changed = True
        if changed:
            frame.to_csv(path, index=False)


def _append_active_neighborhoods() -> None:
    global ACTIVE_MULTIPLIER
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    rows: list[dict[str, object]] = []
    central = ACTIVE_MULTIPLIER
    try:
        for active in ACTIVE_NEIGHBORS:
            ACTIVE_MULTIPLIER = active
            trial, _ = r30.simulate_candidate(
                settings,
                scenario,
                cash_floor=CASH_FLOOR,
                pulse_enabled=True,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            rows.append(
                {
                    "neighborhood": (
                        f"active{int(round(active * 100)):03d}"
                    ),
                    "cash_floor": CASH_FLOOR,
                    "pulse_enabled": True,
                    "active_multiplier": active,
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
        ACTIVE_MULTIPLIER = central
    path = OUTPUT / "parameter_neighborhood.csv"
    neighborhoods = pd.read_csv(path)
    neighborhoods["active_multiplier"] = ACTIVE_MULTIPLIER
    neighborhoods = pd.concat(
        [neighborhoods, pd.DataFrame(rows)],
        ignore_index=True,
    )
    neighborhoods.to_csv(path, index=False)


def _rewrite_neighborhood_gate() -> None:
    neighborhoods = pd.read_csv(
        OUTPUT / "parameter_neighborhood.csv"
    )
    cash = neighborhoods.loc[
        neighborhoods["neighborhood"].isin(("cash20", "cash30"))
    ]
    active = neighborhoods.loc[
        neighborhoods["neighborhood"].isin(
            ("active115", "active125")
        )
    ]
    neighborhood_pass = bool(
        cash["relative_positive"].astype(bool).all()
        and cash["point_target_pass"].astype(bool).any()
        and active["relative_positive"].astype(bool).all()
        and active["point_target_pass"].astype(bool).any()
    )
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    acceptance.loc[
        acceptance["gate"].eq("parameter_neighborhood_pass"),
        "passed",
    ] = neighborhood_pass
    production = acceptance.loc[
        ~acceptance["gate"].eq("production_pass"),
        "passed",
    ].astype(bool).all()
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = bool(production)
    acceptance.to_csv(path, index=False)
    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("production_pass"),
        "value",
    ] = str(bool(production))
    summary.to_csv(summary_path, index=False)


def main() -> None:
    r30.OUTPUT = OUTPUT
    r30.CASH_FLOOR = CASH_FLOOR
    r30.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r30.risk_budget_schedule = proportional_risk_schedule
    r30.main()
    _append_active_neighborhoods()
    _rewrite_candidate_labels()
    _rewrite_neighborhood_gate()
    print("\nR31 parameter neighborhoods:")
    print(
        pd.read_csv(OUTPUT / "parameter_neighborhood.csv")
        .round(6)
        .to_string(index=False)
    )
    print("\nR31 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR31 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
