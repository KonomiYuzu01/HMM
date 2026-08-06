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
from tools.evaluate_r14_incremental_trend_permission import (
    TrendCandidate,
    trend_permission_schedule,
)


OUTPUT = Path("output/r21_current_engine_industry_momentum")
ACTIVE_MULTIPLIER = 1.15
CASH_FLOOR = -0.25
MOMENTUM_DAYS = 126
REBALANCE_DAYS = 21
MAX_TILT = 0.50
CUMULATIVE_TRIALS = 90
DEFINITION = TrendCandidate("both_sma200", "both_sma200")


def causal_industry_momentum(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    lookback_days: int = MOMENTUM_DAYS,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
) -> pd.DataFrame:
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least two")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be positive")
    if not 0.0 <= max_tilt <= 0.50:
        raise ValueError("max_tilt must be between zero and 0.50")
    missing = [
        asset for asset in ("QQQ", "SEMIS") if asset not in closes
    ]
    if missing:
        raise ValueError(f"Missing industry-momentum assets: {missing}")
    log_returns = np.log(
        closes[["QQQ", "SEMIS"]]
        .astype(float)
        .div(closes[["QQQ", "SEMIS"]].shift(1))
    )
    relative = log_returns["SEMIS"] - log_returns["QQQ"]
    trailing_sum = (
        relative.rolling(
            lookback_days,
            min_periods=lookback_days,
        )
        .sum()
        .shift(1)
    )
    trailing_std = (
        relative.rolling(
            lookback_days,
            min_periods=lookback_days,
        )
        .std(ddof=1)
        .shift(1)
    )
    standardized = trailing_sum.div(
        trailing_std.clip(lower=1e-6) * np.sqrt(lookback_days)
    )
    score = np.tanh(standardized)
    raw_tilt = (0.50 + max_tilt * score).clip(
        lower=0.50 - max_tilt,
        upper=0.50 + max_tilt,
    )
    selected = pd.DataFrame(
        {
            "prior_relative_log_return": trailing_sum.reindex(index),
            "prior_relative_volatility": trailing_std.reindex(index),
            "relative_momentum_score": score.reindex(index),
            "raw_semis_growth_share": raw_tilt.reindex(index),
        },
        index=index,
    )
    scheduled_update = pd.Series(
        np.arange(len(index)) % rebalance_days == 0,
        index=index,
        dtype=bool,
    )
    scheduled_tilt = (
        selected["raw_semis_growth_share"]
        .where(scheduled_update)
        .ffill()
        .fillna(0.50)
    )
    selected["momentum_update"] = scheduled_update
    selected["semis_growth_share"] = scheduled_tilt
    selected["qqq_growth_share"] = 1.0 - scheduled_tilt
    return selected


def apply_industry_momentum(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    diagnostics = causal_industry_momentum(
        closes,
        index,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
    )
    adjusted = weights.loc[index].copy()
    growth_total = adjusted["QQQ"] + adjusted["SEMIS"]
    adjusted["SEMIS"] = (
        growth_total * diagnostics["semis_growth_share"]
    )
    adjusted["QQQ"] = growth_total - adjusted["SEMIS"]
    budget_error = (
        adjusted["QQQ"] + adjusted["SEMIS"] - growth_total
    ).abs()
    if float(budget_error.max()) > 1e-12:
        raise AssertionError("Industry momentum changed growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Industry momentum created short growth weight")
    updated_daily = execution_daily.loc[index].copy()
    update = diagnostics["momentum_update"]
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics["growth_budget_before"] = growth_total
    diagnostics["growth_budget_after"] = (
        adjusted["QQQ"] + adjusted["SEMIS"]
    )
    diagnostics["growth_budget_error"] = budget_error
    return adjusted, updated_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, trend_diagnostics = (
        trend_permission_schedule(
            weights,
            daily,
            closes,
            DEFINITION,
            active_multiplier=active_multiplier,
            cash_floor=cash_floor,
        )
    )
    tilted, tilted_daily, momentum_diagnostics = (
        apply_industry_momentum(
            scheduled,
            execution_daily,
            closes,
            rebalance_days=rebalance_days,
            max_tilt=max_tilt,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
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
    diagnostics = trend_diagnostics.reindex(trial.index).join(
        momentum_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_momentum",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["post_guard_semis_weight"] = guard_weights.reindex(
        trial.index
    )["SEMIS"]
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
    metric_rows: list[dict[str, object]] = []
    paths: dict[str, pd.Series] = {}
    baselines: dict[str, pd.Series] = {}
    diagnostics_by_sample: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            trial, diagnostics = simulate_candidate(
                settings,
                scenario,
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
                        "candidate": "industry_momentum_126",
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
                    "candidate": "industry_momentum_126",
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
            "industry_momentum_126",
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
    neighborhoods = (
        ("active_multiplier", "active112", 1.12, REBALANCE_DAYS, MAX_TILT),
        ("active_multiplier", "active118", 1.18, REBALANCE_DAYS, MAX_TILT),
        ("max_tilt", "max_tilt40", ACTIVE_MULTIPLIER, REBALANCE_DAYS, 0.40),
        ("rebalance_days", "rebalance20", ACTIVE_MULTIPLIER, 20, MAX_TILT),
        ("rebalance_days", "rebalance22", ACTIVE_MULTIPLIER, 22, MAX_TILT),
    )
    neighborhood_rows: list[dict[str, object]] = []
    for dimension, label, active, rebalance, max_tilt in neighborhoods:
        trial, _ = simulate_candidate(
            normal_settings,
            central_scenario,
            active_multiplier=active,
            rebalance_days=rebalance,
            max_tilt=max_tilt,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        neighborhood_rows.append(
            {
                "dimension": dimension,
                "neighborhood": label,
                "active_multiplier": active,
                "rebalance_days": rebalance,
                "max_tilt": max_tilt,
                **delta,
                "economic_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= delta["baseline_max_drawdown"]
                ),
            }
        )
    neighborhood = pd.DataFrame(neighborhood_rows)
    neighborhood.to_csv(
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
        neighborhood.groupby("dimension")["economic_pass"].any().all()
    )
    normal_years = annual_relative["normal_synthetic"].dropna()
    development_years = normal_years.loc[2015:2021]
    holdout_years = normal_years.loc[2022:2025]
    year_breadth_pass = bool(
        development_years.gt(0.0).sum() >= 4
        and holdout_years.gt(0.0).sum() >= 4
    )
    diagnostics = diagnostics_by_sample["normal_synthetic"]
    budget_conservation_pass = bool(
        diagnostics["growth_budget_error"].fillna(0.0).le(1e-12).all()
    )
    declared_cash_hard_limit_pass = bool(
        diagnostics["implemented_cash_weight"].ge(-0.18).all()
        and diagnostics["post_guard_cash_weight"].ge(-0.18).all()
    )
    statistical_economic_pass = bool(
        complete["candidate_cagr"] >= 0.25
        and complete["candidate_max_drawdown"]
        >= complete["baseline_max_drawdown"]
        and development["annualized_relative_log_return"] > 0.0
        and development["candidate_max_drawdown"]
        >= development["baseline_max_drawdown"]
        and holdout["annualized_relative_log_return"] > 0.0
        and holdout["candidate_max_drawdown"]
        >= holdout["baseline_max_drawdown"]
        and stress_complete["candidate_cagr"] >= 0.25
        and stress_complete["candidate_max_drawdown"]
        >= stress_complete["baseline_max_drawdown"]
        and proxy_complete["annualized_relative_log_return"] > 0.0
        and proxy_complete["candidate_max_drawdown"]
        >= proxy_complete["baseline_max_drawdown"]
        and proxy_early["annualized_relative_log_return"] > 0.0
        and proxy_late["annualized_relative_log_return"] > 0.0
        and family_pass
        and robustness_pass
        and neighborhood_pass
        and year_breadth_pass
        and budget_conservation_pass
    )
    acceptance = pd.DataFrame(
        [
            {
                "candidate": "industry_momentum_126",
                "complete_cagr": complete["candidate_cagr"],
                "complete_max_drawdown": (
                    complete["candidate_max_drawdown"]
                ),
                "r11_max_drawdown": complete["baseline_max_drawdown"],
                "development_relative_log_return": (
                    development["annualized_relative_log_return"]
                ),
                "development_max_drawdown": (
                    development["candidate_max_drawdown"]
                ),
                "holdout_relative_log_return": (
                    holdout["annualized_relative_log_return"]
                ),
                "holdout_max_drawdown": (
                    holdout["candidate_max_drawdown"]
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
                "year_breadth_pass": year_breadth_pass,
                "growth_budget_conservation_pass": (
                    budget_conservation_pass
                ),
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
