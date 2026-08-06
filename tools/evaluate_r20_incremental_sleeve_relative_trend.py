from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    metric_delta,
    relative_log_return,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r11_post_shock_cooldown import sample_definitions
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    robustness_rows,
    simulate_fixed_r11,
)
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TrendCandidate,
    causal_trend_permission,
    simulate_candidate as simulate_r14,
)


OUTPUT = Path("output/r20_incremental_sleeve_relative_trend")
ACTIVE_MULTIPLIER = 1.18
CASH_FLOOR = -0.25
LOOKBACK_DAYS = 63
CUMULATIVE_TRIALS = 69
TRADING_EPSILON = 1e-14
DEFINITION = TrendCandidate("both_sma200", "both_sma200")


def causal_shadow_relative_permission(
    shadow_returns: pd.Series,
    baseline_returns: pd.Series,
    index: pd.DatetimeIndex,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    entry_threshold: float = 0.0,
    exit_threshold: float = 0.0,
) -> pd.DataFrame:
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least two")
    if entry_threshold < exit_threshold:
        raise ValueError("entry_threshold cannot be below exit_threshold")
    aligned = pd.concat(
        [
            shadow_returns.rename("shadow"),
            baseline_returns.rename("baseline"),
        ],
        axis=1,
        join="inner",
    ).reindex(index)
    relative_log_return = (
        np.log1p(aligned["shadow"]) - np.log1p(aligned["baseline"])
    )
    trailing = (
        relative_log_return.rolling(
            lookback_days,
            min_periods=lookback_days,
        )
        .sum()
        .shift(1)
    )
    state = False
    values: list[bool] = []
    for value in trailing:
        if pd.isna(value):
            state = False
        elif state and float(value) < exit_threshold:
            state = False
        elif not state and float(value) > entry_threshold:
            state = True
        values.append(state)
    return pd.DataFrame(
        {
            "shadow_relative_log_return": relative_log_return,
            "prior_63d_shadow_relative_log_return": trailing,
            "shadow_relative_permission": values,
        },
        index=index,
    )


def relative_trend_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    shadow_permission: pd.DataFrame,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if active_multiplier < 1.0:
        raise ValueError("active_multiplier cannot reduce R11")
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
        .intersection(shadow_permission.index)
    )
    trend = causal_trend_permission(
        closes.reindex(index),
        DEFINITION,
    )
    shadow = shadow_permission.reindex(index)[
        "shadow_relative_permission"
    ].fillna(False)
    permission = trend & shadow
    requested_relative = pd.Series(
        np.where(permission, active_multiplier, 1.0),
        index=index,
        dtype=float,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    state_change = permission.ne(permission.shift(1))
    update = base_trade | state_change
    accepted_relative = (
        requested_relative.where(update).ffill().fillna(1.0)
    )
    if accepted_relative.lt(1.0).any():
        raise AssertionError("R20 reduced risk below R11")
    absolute = BASE_MULTIPLIER * accepted_relative
    requested = scale_non_cash_weights_by_series(
        weights.loc[index],
        absolute,
    )
    capped, cap_scale = cap_weights_at_cash_floor(
        requested,
        cash_floor,
    )
    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = shadow_permission.reindex(index).copy()
    diagnostics["absolute_trend_permission"] = trend
    diagnostics["combined_permission"] = permission
    diagnostics["requested_relative_multiplier"] = requested_relative
    diagnostics["accepted_relative_multiplier"] = accepted_relative
    diagnostics["absolute_multiplier"] = absolute
    diagnostics["cash_cap_scale"] = cap_scale
    diagnostics["implemented_cash_weight"] = capped["CASH"]
    diagnostics["base_trade"] = base_trade
    diagnostics["state_change"] = state_change
    diagnostics["execution_update"] = update
    return capped, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    shadow_permission: pd.DataFrame,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, diagnostics = relative_trend_schedule(
        weights,
        daily,
        closes,
        shadow_permission,
        active_multiplier=active_multiplier,
        cash_floor=cash_floor,
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
        weights_override=guard_weights,
        daily_override=guard_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = diagnostics.reindex(trial.index)
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def _build_samples() -> dict[str, dict[str, object]]:
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(
        PROXY_OPEN_CLOSE
    )
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(
        NORMAL_DIRECTORY
    )
    proxy_weights, proxy_daily = load_strategy_inputs(
        PROXY_DIRECTORY
    )
    return sample_definitions(
        normal_weights,
        normal_daily,
        proxy_weights,
        proxy_daily,
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = _build_samples()
    central_scenario = COST_SCENARIOS[0]
    baselines: dict[str, pd.Series] = {}
    frozen_permissions: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        baseline, _, _ = simulate_fixed_r11(
            settings,
            central_scenario,
        )
        shadow, _ = simulate_r14(
            settings,
            central_scenario,
            DEFINITION,
            active_multiplier=ACTIVE_MULTIPLIER,
            cash_floor=CASH_FLOOR,
        )
        index = baseline.index.intersection(shadow.index)
        baselines[sample] = baseline.loc[index, "net_return"]
        frozen_permissions[sample] = (
            causal_shadow_relative_permission(
                shadow.loc[index, "net_return"],
                baseline.loc[index, "net_return"],
                index,
            )
        )
        frozen_permissions[sample].to_csv(
            OUTPUT / f"{sample}_frozen_shadow_permission.csv",
            index_label="date",
        )

    metric_rows: list[dict[str, object]] = []
    paths: dict[str, pd.Series] = {}
    diagnostics_by_sample: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            trial, diagnostics = simulate_candidate(
                settings,
                scenario,
                frozen_permissions[sample],
            )
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
                        "candidate": "shadow_relative_63",
                        "period": period,
                        "annualized_relative_log_return": float(
                            (
                                np.log1p(candidate_returns)
                                - np.log1p(baseline_returns)
                            ).mean()
                            * 252
                        ),
                        **metric_delta(
                            baseline_returns,
                            candidate_returns,
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                paths[sample] = trial["net_return"]
                diagnostics_by_sample[sample] = diagnostics
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
                    "candidate": "shadow_relative_63",
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
            "shadow_relative_63",
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

    settings = samples["normal_synthetic"]
    baseline, _, _ = simulate_fixed_r11(settings, central_scenario)
    neighborhood_rows: list[dict[str, object]] = []
    for label, active in (("active115", 1.15), ("active121", 1.21)):
        trial, _ = simulate_candidate(
            settings,
            central_scenario,
            frozen_permissions["normal_synthetic"],
            active_multiplier=active,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        neighborhood_rows.append(
            {
                "dimension": "active_multiplier",
                "neighborhood": label,
                "value": active,
                **delta,
                "economic_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= delta["baseline_max_drawdown"]
                ),
            }
        )
    for label, entry, exit_ in (
        ("hysteresis_plus_10bps", 0.001, -0.001),
        ("threshold_minus_10bps", -0.001, -0.001),
    ):
        permission = causal_shadow_relative_permission(
            frozen_permissions["normal_synthetic"][
                "shadow_relative_log_return"
            ].pipe(lambda x: np.expm1(x)).rename("shadow_proxy"),
            pd.Series(
                0.0,
                index=frozen_permissions["normal_synthetic"].index,
                name="zero_baseline",
            ),
            frozen_permissions["normal_synthetic"].index,
            entry_threshold=entry,
            exit_threshold=exit_,
        )
        trial, _ = simulate_candidate(
            settings,
            central_scenario,
            permission,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        neighborhood_rows.append(
            {
                "dimension": "relative_threshold",
                "neighborhood": label,
                "value": entry,
                **delta,
                "economic_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= delta["baseline_max_drawdown"]
                ),
            }
        )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    normal_diagnostics = diagnostics_by_sample["normal_synthetic"]
    risk_floor_pass = bool(
        normal_diagnostics["accepted_relative_multiplier"].ge(1.0).all()
    )
    declared_cash_hard_limit_pass = bool(
        normal_diagnostics["implemented_cash_weight"].ge(-0.18).all()
        and normal_diagnostics["post_guard_cash_weight"].ge(-0.18).all()
    )
    central = metrics.loc[
        (metrics["sample"] == "normal_synthetic")
        & (metrics["scenario"] == "current_liquidity")
    ]
    stress = metrics.loc[
        (metrics["sample"] == "normal_synthetic")
        & (metrics["scenario"] == "cost_stress")
    ]
    proxy = metrics.loc[
        (metrics["sample"] == "proxy_synthetic")
        & (metrics["scenario"] == "current_liquidity")
    ]

    def row(frame: pd.DataFrame, period: str) -> pd.Series:
        selected = frame.loc[frame["period"] == period]
        if len(selected) != 1:
            raise ValueError(f"Expected one metric row for {period}")
        return selected.iloc[0]

    complete = row(central, "complete_2015_2026")
    development = row(central, "development_2015_2021")
    holdout = row(central, "holdout_2022_2025")
    stress_complete = row(stress, "complete_2015_2026")
    proxy_complete = row(proxy, "complete_2006_2026")
    proxy_early = row(proxy, "early_2006_2014")
    proxy_late = row(proxy, "late_2015_2026")
    family_pass = bool(
        family.loc[
            family["sample"] == "normal_synthetic",
            "cumulative_trial_adjusted_p_value",
        ].le(0.05).all()
    )
    robustness_pass = bool(
        years["annualized_relative_log_return"].gt(0.0).all()
        and events["annualized_relative_log_return"].gt(0.0).all()
    )
    neighborhood_pass = bool(
        neighborhoods.groupby("dimension")["economic_pass"].any().all()
    )
    statistical_economic_pass = bool(
        complete["candidate_cagr"] >= 0.25
        and complete["candidate_max_drawdown"]
        >= complete["baseline_max_drawdown"]
        and development["annualized_relative_log_return"] > 0.0
        and holdout["annualized_relative_log_return"] > 0.0
        and stress_complete["candidate_cagr"] >= 0.25
        and stress_complete["candidate_max_drawdown"]
        >= stress_complete["baseline_max_drawdown"]
        and proxy_complete["annualized_relative_log_return"] > 0.0
        and proxy_early["annualized_relative_log_return"] > 0.0
        and proxy_late["annualized_relative_log_return"] > 0.0
        and family_pass
        and robustness_pass
        and neighborhood_pass
        and risk_floor_pass
    )
    acceptance = pd.DataFrame(
        [
            {
                "candidate": "shadow_relative_63",
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
                "stress_cagr": stress_complete["candidate_cagr"],
                "stress_max_drawdown": (
                    stress_complete["candidate_max_drawdown"]
                ),
                "proxy_cagr": proxy_complete["candidate_cagr"],
                "proxy_max_drawdown": (
                    proxy_complete["candidate_max_drawdown"]
                ),
                "family_multiple_testing_pass": family_pass,
                "leave_one_out_pass": robustness_pass,
                "parameter_neighborhood_pass": neighborhood_pass,
                "never_below_r11_risk_pass": risk_floor_pass,
                "declared_cash_hard_limit_pass": (
                    declared_cash_hard_limit_pass
                ),
                "statistical_economic_pass": (
                    statistical_economic_pass
                ),
                "production_pass": bool(
                    statistical_economic_pass
                    and declared_cash_hard_limit_pass
                ),
            }
        ]
    )
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    print(acceptance.round(6).to_string(index=False))
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
