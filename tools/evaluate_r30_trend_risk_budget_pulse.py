from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    robustness_rows,
    simulate_fixed_r11,
)
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
    TrendCandidate,
    causal_trend_permission,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    causal_pulse_gated_industry_momentum,
)


OUTPUT = Path("output/r30_trend_risk_budget_pulse")
DEFINITION = TrendCandidate("both_sma200", "both_sma200")
CASH_FLOOR = -0.25
NEIGHBOR_CASH_FLOORS = (-0.20, -0.30)
CUMULATIVE_TRIALS = 103
MAX_DRAWDOWN_TOLERANCE = 0.005


def requested_fill_multiplier(
    weights: pd.DataFrame,
    *,
    cash_floor: float,
) -> pd.Series:
    if not -1.0 < cash_floor <= 0.0:
        raise ValueError("cash_floor must be in (-1, 0]")
    non_cash = [column for column in weights if column != "CASH"]
    if not non_cash:
        raise ValueError("At least one non-cash asset is required")
    gross = weights[non_cash].sum(axis=1)
    requested = pd.Series(BASE_MULTIPLIER, index=weights.index)
    positive = gross.gt(1e-12)
    requested.loc[positive] = (
        (1.0 - cash_floor) / gross.loc[positive]
    ).clip(lower=BASE_MULTIPLIER)
    return requested


def effective_incremental_permission(
    trend_permission: pd.Series,
    pulse_active: pd.Series,
) -> pd.Series:
    index = trend_permission.index.intersection(pulse_active.index)
    return (
        trend_permission.reindex(index).fillna(False).astype(bool)
        & ~pulse_active.reindex(index).fillna(False).astype(bool)
    ).rename("effective_incremental_permission")


def risk_budget_schedule(
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
    permission = causal_trend_permission(
        closes.reindex(index),
        DEFINITION,
    )
    pulse_diagnostics = causal_pulse_gated_industry_momentum(
        closes.reindex(index),
        index,
    )
    pulse_active = pulse_diagnostics["pulse_veto_active"].astype(bool)
    if not pulse_enabled:
        pulse_active = pd.Series(False, index=index)
    effective = effective_incremental_permission(
        permission,
        pulse_active,
    )
    fill_multiplier = requested_fill_multiplier(
        base_weights,
        cash_floor=cash_floor,
    )
    requested_absolute = fill_multiplier.where(
        effective,
        BASE_MULTIPLIER,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_change = permission.ne(permission.shift(1))
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
    base_r11_requested = scale_non_cash_weights_by_series(
        base_weights,
        pd.Series(BASE_MULTIPLIER, index=index),
    )
    base_r11, _ = cap_weights_at_cash_floor(
        base_r11_requested,
        cash_floor,
    )
    if implemented["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("Risk budget exceeded its cash floor")
    non_cash = [column for column in implemented if column != "CASH"]
    if (
        implemented.loc[~effective, non_cash]
        - base_r11.loc[~effective, non_cash]
    ).abs().max().max() > 1e-12:
        raise AssertionError(
            "Disabled incremental permission did not return to R11"
        )
    if (
        implemented.loc[effective, non_cash].sum(axis=1)
        + 1e-12
        < base_r11.loc[effective, non_cash].sum(axis=1)
    ).any():
        raise AssertionError("Incremental permission reduced R11 risk")

    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": permission,
            "pulse_veto_active": pulse_active,
            "effective_incremental_permission": effective,
            "fill_absolute_multiplier": fill_multiplier,
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
    ):
        diagnostics[column] = pulse_diagnostics[column]
    return implemented, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    cash_floor: float = CASH_FLOOR,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, diagnostics = risk_budget_schedule(
        weights,
        daily,
        closes,
        cash_floor=cash_floor,
        pulse_enabled=pulse_enabled,
    )
    guard_daily, guard_weights = simulate_guard(
        scheduled,
        execution_daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
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
    diagnostics = diagnostics.reindex(trial.index).join(
        limit_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_post_guard",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["pre_gde_cash_weight"] = limited.reindex(
        trial.index
    )["CASH"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def maximum_true_run(values: pd.Series) -> int:
    state = values.fillna(False).astype(bool)
    if not state.any():
        return 0
    groups = state.ne(state.shift()).cumsum()
    return int(state.groupby(groups).sum().max())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    paths: dict[str, pd.Series] = {}
    baselines: dict[str, pd.Series] = {}
    diagnostics_by_sample: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            trial, diagnostics = simulate_candidate(settings, scenario)
            common = baseline.index.intersection(trial.index)
            for period, (period_start, period_end) in periods.items():
                selected = common[
                    (common >= period_start) & (common <= period_end)
                ]
                baseline_returns = baseline.loc[
                    selected, "net_return"
                ]
                candidate_returns = trial.loc[
                    selected, "net_return"
                ]
                metric_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "candidate": (
                            "both_sma200_fill_cash25_pulse4"
                        ),
                        "period": period,
                        "annualized_relative_log_return": float(
                            (
                                np.log1p(candidate_returns)
                                - np.log1p(baseline_returns)
                            ).mean()
                            * 252.0
                        ),
                        **metric_delta(
                            baseline_returns,
                            candidate_returns,
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                baselines[sample] = baseline["net_return"]
                paths[sample] = trial["net_return"]
                diagnostics_by_sample[sample] = diagnostics
                baseline.to_csv(
                    OUTPUT / f"{sample}_r11_baseline_daily.csv",
                    index_label="date",
                )
                trial.to_csv(
                    OUTPUT / f"{sample}_candidate_daily.csv",
                    index_label="date",
                )
                diagnostics.to_csv(
                    OUTPUT / f"{sample}_diagnostics.csv",
                    index_label="date",
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        relative = relative_log_return(
            baselines[sample],
            paths[sample],
        )[:, None]
        for block_days in (21, 63, 126):
            check = circular_family_reality_check(
                relative,
                0,
                block_days,
            )
            family_p = float(
                check["familywise_reality_check_p_value"]
            )
            family_rows.append(
                {
                    "sample": sample,
                    "candidate": (
                        "both_sma200_fill_cash25_pulse4"
                    ),
                    "declared_family_size": 1,
                    "cumulative_trials": CUMULATIVE_TRIALS,
                    **check,
                    "cumulative_trial_adjusted_p_value": min(
                        1.0,
                        family_p * CUMULATIVE_TRIALS,
                    ),
                }
            )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    year_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        years, events = robustness_rows(
            sample,
            "both_sma200_fill_cash25_pulse4",
            baselines[sample],
            paths[sample],
        )
        year_rows.extend(years)
        event_rows.extend(events)
    years = pd.DataFrame(year_rows)
    events = pd.DataFrame(event_rows)
    years.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    events.to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )

    normal_settings = samples["normal_synthetic"]
    central_scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(
        normal_settings,
        central_scenario,
    )
    neighborhood_rows: list[dict[str, object]] = []
    for label, cash_floor, pulse_enabled in (
        ("cash20", -0.20, True),
        ("cash30", -0.30, True),
        ("no_pulse", CASH_FLOOR, False),
    ):
        trial, _ = simulate_candidate(
            normal_settings,
            central_scenario,
            cash_floor=cash_floor,
            pulse_enabled=pulse_enabled,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        neighborhood_rows.append(
            {
                "neighborhood": label,
                "cash_floor": cash_floor,
                "pulse_enabled": pulse_enabled,
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
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    annual_relative = pd.DataFrame(
        {
            sample: pd.Series(
                relative_log_return(
                    baselines[sample],
                    paths[sample],
                ),
                index=baselines[sample]
                .index.intersection(paths[sample].index),
            )
            .groupby(
                baselines[sample]
                .index.intersection(paths[sample].index)
                .year
            )
            .sum()
            for sample in ("normal_synthetic", "proxy_synthetic")
        }
    )
    annual_relative.to_csv(
        OUTPUT / "annual_relative_log_return.csv",
        index_label="year",
    )

    def metric(
        sample: str,
        scenario: str,
        period: str,
    ) -> pd.Series:
        selected = metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["scenario"].eq(scenario)
            & metrics["period"].eq(period)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one metric for {sample}/{scenario}/{period}"
            )
        return selected.iloc[0]

    complete = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
    )
    development = metric(
        "normal_synthetic",
        "current_liquidity",
        "development_2015_2021",
    )
    holdout = metric(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
    )
    stress = metric(
        "normal_synthetic",
        "cost_stress",
        "complete_2015_2026",
    )
    proxy_complete = metric(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
    )
    proxy_early = metric(
        "proxy_synthetic",
        "current_liquidity",
        "early_2006_2014",
    )
    proxy_late = metric(
        "proxy_synthetic",
        "current_liquidity",
        "late_2015_2026",
    )
    live = metric(
        "normal_live",
        "current_liquidity",
        "live_2022_2026",
    )
    normal_family = family.loc[
        family["sample"].eq("normal_synthetic")
    ]
    family_pass = bool(
        normal_family[
            "cumulative_trial_adjusted_p_value"
        ].lt(0.05).all()
    )
    robustness_pass = bool(
        years.loc[
            years["sample"].eq("normal_synthetic"),
            "annualized_relative_log_return",
        ]
        .gt(0.0)
        .all()
        and events.loc[
            events["sample"].eq("normal_synthetic"),
            "annualized_relative_log_return",
        ]
        .gt(0.0)
        .all()
    )
    normal_annual = annual_relative["normal_synthetic"].dropna()
    year_breadth_pass = bool(
        normal_annual.loc[2015:2021].gt(0.0).sum() >= 5
        and normal_annual.loc[2022:2025].gt(0.0).sum() >= 3
    )
    ridge = neighborhoods.loc[
        neighborhoods["neighborhood"].isin(("cash20", "cash30"))
    ]
    neighborhood_pass = bool(
        ridge["relative_positive"].all()
        and ridge["point_target_pass"].any()
    )
    diagnostics = diagnostics_by_sample["normal_synthetic"]
    cash_limit_pass = bool(
        diagnostics["implemented_cash_weight"].ge(
            CASH_FLOOR - 1e-12
        ).all()
        and diagnostics["pre_gde_cash_weight"].ge(
            CASH_FLOOR - 1e-12
        ).all()
    )
    pulse_duration_pass = (
        maximum_true_run(diagnostics["pulse_veto_active"]) <= 4
    )
    disabled_equals_r11_pass = bool(
        (
            diagnostics.loc[
                ~diagnostics[
                    "effective_incremental_permission"
                ].astype(bool),
                "implemented_non_cash_weight",
            ]
            - diagnostics.loc[
                ~diagnostics[
                    "effective_incremental_permission"
                ].astype(bool),
                "base_r11_non_cash_weight",
            ]
        )
        .abs()
        .le(1e-12)
        .all()
    )
    tolerance = MAX_DRAWDOWN_TOLERANCE
    gates = {
        "complete_cagr_at_least_25pct": float(
            complete["candidate_cagr"]
        )
        >= 0.25,
        "complete_drawdown_within_tolerance": float(
            complete["candidate_max_drawdown"]
        )
        >= float(complete["baseline_max_drawdown"]) - tolerance,
        "development_relative_positive": float(
            development["annualized_relative_log_return"]
        )
        > 0.0,
        "development_drawdown_within_tolerance": float(
            development["candidate_max_drawdown"]
        )
        >= float(development["baseline_max_drawdown"]) - tolerance,
        "holdout_relative_positive": float(
            holdout["annualized_relative_log_return"]
        )
        > 0.0,
        "holdout_drawdown_within_tolerance": float(
            holdout["candidate_max_drawdown"]
        )
        >= float(holdout["baseline_max_drawdown"]) - tolerance,
        "stress_relative_positive": float(
            stress["annualized_relative_log_return"]
        )
        > 0.0,
        "stress_drawdown_within_tolerance": float(
            stress["candidate_max_drawdown"]
        )
        >= float(stress["baseline_max_drawdown"]) - tolerance,
        "proxy_complete_relative_positive": float(
            proxy_complete["annualized_relative_log_return"]
        )
        > 0.0,
        "proxy_early_relative_positive": float(
            proxy_early["annualized_relative_log_return"]
        )
        > 0.0,
        "proxy_late_relative_positive": float(
            proxy_late["annualized_relative_log_return"]
        )
        > 0.0,
        "proxy_drawdown_within_tolerance": float(
            proxy_complete["candidate_max_drawdown"]
        )
        >= float(proxy_complete["baseline_max_drawdown"]) - tolerance,
        "live_relative_positive": float(
            live["annualized_relative_log_return"]
        )
        > 0.0,
        "live_drawdown_within_tolerance": float(
            live["candidate_max_drawdown"]
        )
        >= float(live["baseline_max_drawdown"]) - tolerance,
        "leave_one_out_pass": robustness_pass,
        "year_breadth_pass": year_breadth_pass,
        "parameter_neighborhood_pass": neighborhood_pass,
        "multiple_testing_pass": family_pass,
        "cash_hard_limit_pass": cash_limit_pass,
        "pulse_duration_pass": pulse_duration_pass,
        "disabled_increment_equals_r11_pass": (
            disabled_equals_r11_pass
        ),
    }
    acceptance = pd.DataFrame(
        [{"gate": gate, "passed": bool(value)} for gate, value in gates.items()]
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "production_pass",
        "passed": bool(all(gates.values())),
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    pd.Series(
        {
            "candidate": "both_sma200_fill_cash25_pulse4",
            "complete_cagr": complete["candidate_cagr"],
            "complete_max_drawdown": (
                complete["candidate_max_drawdown"]
            ),
            "r11_max_drawdown": complete["baseline_max_drawdown"],
            "development_relative_log_return": (
                development["annualized_relative_log_return"]
            ),
            "holdout_relative_log_return": (
                holdout["annualized_relative_log_return"]
            ),
            "stress_cagr": stress["candidate_cagr"],
            "stress_max_drawdown": stress["candidate_max_drawdown"],
            "proxy_cagr": proxy_complete["candidate_cagr"],
            "proxy_max_drawdown": (
                proxy_complete["candidate_max_drawdown"]
            ),
            "live_cagr": live["candidate_cagr"],
            "live_max_drawdown": live["candidate_max_drawdown"],
            "minimum_target_cash": diagnostics[
                "implemented_cash_weight"
            ].min(),
            "minimum_pre_gde_cash": diagnostics[
                "pre_gde_cash_weight"
            ].min(),
            "production_pass": bool(all(gates.values())),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")

    print("R30 summary:")
    print(pd.read_csv(OUTPUT / "summary.csv").to_string(index=False))
    print("\nParameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print(f"\nR30 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
