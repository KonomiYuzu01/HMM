from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_hybrid import constrained_reality_check
from evaluate_relative_momentum import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "zero_entry_tail_relative_risk_validation"
BASELINE = "paper_core_robust_vol_guarded_floor_ensemble"
CANDIDATE = "paper_core_zero_entry_tail_relative_risk_ensemble"
BASELINE_COST15 = "paper_core_robust_vol_guarded_floor_ensemble_cost15"
CANDIDATE_COST15 = "paper_core_zero_entry_tail_relative_risk_ensemble_cost15"
BASELINE_2012 = "paper_core_robust_vol_guarded_floor_ensemble_2012"
CANDIDATE_2012 = "paper_core_zero_entry_tail_relative_risk_ensemble_2012"
GROWTH_CANDIDATE = "paper_core_zero_entry_growth_reallocation_ensemble"
GROWTH_CANDIDATE_COST15 = (
    "paper_core_zero_entry_growth_reallocation_ensemble_cost15"
)
GROWTH_CANDIDATE_2012 = (
    "paper_core_zero_entry_growth_reallocation_ensemble_2012"
)
STATE_CANDIDATE = "paper_core_zero_entry_state_switch_ensemble"
STATE_CANDIDATE_COST15 = "paper_core_zero_entry_state_switch_ensemble_cost15"
STATE_CANDIDATE_2012 = "paper_core_zero_entry_state_switch_ensemble_2012"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "holdout_2022_present": ("2022-01-01", None),
    "full_2015_present": ("2015-01-01", None),
}
PARAMETER_CANDIDATES = [
    "paper_core_zero_entry_state_switch_fast_ensemble",
    STATE_CANDIDATE,
    "paper_core_zero_entry_state_switch_slow_ensemble",
    "paper_core_zero_entry_state_switch_vol18_ensemble",
    "paper_core_zero_entry_state_switch_vol22_ensemble",
]


def load_daily(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def relative_total_return(candidate: pd.Series, baseline: pd.Series) -> float:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    return float(
        (1.0 + aligned.iloc[:, 0]).prod()
        / (1.0 + aligned.iloc[:, 1]).prod()
        - 1.0
    )


def member_trigger_audit() -> pd.DataFrame:
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for seed in (7, 42, 123):
        member_path = Path("members") / f"seed_{seed}"
        candidate_regimes = pd.read_csv(
            OUTPUT_ROOT / CANDIDATE / member_path / "regimes.csv",
            index_col=0,
            parse_dates=True,
        )
        candidate_daily = pd.read_csv(
            OUTPUT_ROOT / CANDIDATE / member_path / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        baseline_daily = pd.read_csv(
            OUTPUT_ROOT / BASELINE / member_path / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        active = candidate_regimes[
            candidate_regimes["asset_risk_overlay_active"].eq(1)
            & candidate_regimes["asset_relative_risk_multiplier"].lt(1.0 - 1e-9)
        ]
        regime_dates = candidate_regimes.index
        for date, diagnostics in active.iterrows():
            position = int(regime_dates.get_loc(date))
            next_date = (
                regime_dates[position + 1]
                if position + 1 < len(regime_dates)
                else candidate_daily.index[-1] + pd.Timedelta(days=1)
            )
            candidate_window = candidate_daily.loc[
                (candidate_daily.index >= date) & (candidate_daily.index < next_date),
                "net_return",
            ]
            baseline_window = baseline_daily.loc[
                (baseline_daily.index >= date) & (baseline_daily.index < next_date),
                "net_return",
            ]
            rows.append(
                {
                    "seed": seed,
                    "entry_date": date,
                    "next_review_date": next_date,
                    "smh_multiplier": diagnostics[
                        "asset_relative_risk_multiplier"
                    ],
                    "qqq_effective_volatility": diagnostics[
                        "asset_aware_QQQ_effective_volatility"
                    ],
                    "smh_effective_volatility": diagnostics[
                        "asset_aware_SEMIS_effective_volatility"
                    ],
                    "candidate_window_return": float(
                        (1.0 + candidate_window).prod() - 1.0
                    ),
                    "baseline_window_return": float(
                        (1.0 + baseline_window).prod() - 1.0
                    ),
                    "relative_window_return": relative_total_return(
                        candidate_window, baseline_window
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["entry_date", "seed"])


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    names = [
        BASELINE,
        CANDIDATE,
        BASELINE_COST15,
        CANDIDATE_COST15,
        BASELINE_2012,
        CANDIDATE_2012,
        GROWTH_CANDIDATE,
        GROWTH_CANDIDATE_COST15,
        GROWTH_CANDIDATE_2012,
        STATE_CANDIDATE,
        STATE_CANDIDATE_COST15,
        STATE_CANDIDATE_2012,
    ]
    daily = {name: load_daily(name) for name in names}

    metric_rows: list[dict[str, float | str]] = []
    for strategy in names[:4]:
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(
                        daily[strategy].loc[start:end, "net_return"]
                    ),
                }
            )
    for strategy in (BASELINE_2012, CANDIDATE_2012):
        metric_rows.append(
            {
                "strategy": strategy,
                "period": "extended_2012_2025",
                **performance_metrics(daily[strategy].loc[:"2025", "net_return"]),
            }
        )
        for period, (start, end) in list(PERIODS.items())[:3]:
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(
                        daily[strategy].loc[start:end, "net_return"]
                    ),
                }
            )
    for strategy in (GROWTH_CANDIDATE, GROWTH_CANDIDATE_COST15):
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(
                        daily[strategy].loc[start:end, "net_return"]
                    ),
                }
            )
    metric_rows.append(
        {
            "strategy": GROWTH_CANDIDATE_2012,
            "period": "extended_2012_2025",
            **performance_metrics(
                daily[GROWTH_CANDIDATE_2012].loc[:"2025", "net_return"]
            ),
        }
    )
    for period, (start, end) in list(PERIODS.items())[:3]:
        metric_rows.append(
            {
                "strategy": GROWTH_CANDIDATE_2012,
                "period": period,
                **performance_metrics(
                    daily[GROWTH_CANDIDATE_2012].loc[start:end, "net_return"]
                ),
            }
        )
    for strategy in (STATE_CANDIDATE, STATE_CANDIDATE_COST15):
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(
                        daily[strategy].loc[start:end, "net_return"]
                    ),
                }
            )
    metric_rows.append(
        {
            "strategy": STATE_CANDIDATE_2012,
            "period": "extended_2012_2025",
            **performance_metrics(
                daily[STATE_CANDIDATE_2012].loc[:"2025", "net_return"]
            ),
        }
    )
    for period, (start, end) in list(PERIODS.items())[:3]:
        metric_rows.append(
            {
                "strategy": STATE_CANDIDATE_2012,
                "period": period,
                **performance_metrics(
                    daily[STATE_CANDIDATE_2012].loc[start:end, "net_return"]
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    def metric(strategy: str, period: str, field: str) -> float:
        return float(metrics.loc[(strategy, period), field])

    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr_delta": metric(
                    CANDIDATE, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE, "complete_2015_2025", "cagr"),
                "complete_sharpe_delta": metric(
                    CANDIDATE, "complete_2015_2025", "sharpe"
                )
                - metric(BASELINE, "complete_2015_2025", "sharpe"),
                "complete_volatility_delta": metric(
                    CANDIDATE, "complete_2015_2025", "annual_volatility"
                )
                - metric(BASELINE, "complete_2015_2025", "annual_volatility"),
                "complete_max_drawdown": metric(
                    CANDIDATE, "complete_2015_2025", "max_drawdown"
                ),
                "development_cagr_delta": metric(
                    CANDIDATE, "development_2015_2021", "cagr"
                )
                - metric(BASELINE, "development_2015_2021", "cagr"),
                "holdout_cagr_delta": metric(
                    CANDIDATE, "holdout_2022_2025", "cagr"
                )
                - metric(BASELINE, "holdout_2022_2025", "cagr"),
                "cost15_complete_cagr_delta": metric(
                    CANDIDATE_COST15, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE_COST15, "complete_2015_2025", "cagr"),
                "start2012_extended_cagr_delta": metric(
                    CANDIDATE_2012, "extended_2012_2025", "cagr"
                )
                - metric(BASELINE_2012, "extended_2012_2025", "cagr"),
                "cagr_non_degradation_pass": int(
                    metric(CANDIDATE, "complete_2015_2025", "cagr")
                    >= metric(BASELINE, "complete_2015_2025", "cagr")
                ),
                "sharpe_non_degradation_pass": int(
                    metric(CANDIDATE, "complete_2015_2025", "sharpe")
                    >= metric(BASELINE, "complete_2015_2025", "sharpe")
                ),
                "drawdown_18pct_pass": int(
                    metric(CANDIDATE, "complete_2015_2025", "max_drawdown")
                    >= -0.18
                ),
                "development_strict_non_degradation_pass": int(
                    metric(CANDIDATE, "development_2015_2021", "cagr")
                    >= metric(BASELINE, "development_2015_2021", "cagr")
                ),
                "holdout_non_degradation_pass": int(
                    metric(CANDIDATE, "holdout_2022_2025", "cagr")
                    >= metric(BASELINE, "holdout_2022_2025", "cagr")
                ),
                "cost15_non_degradation_pass": int(
                    metric(CANDIDATE_COST15, "complete_2015_2025", "cagr")
                    >= metric(BASELINE_COST15, "complete_2015_2025", "cagr")
                ),
            }
        ],
        index=[CANDIDATE],
    )
    pass_columns = [column for column in acceptance if column.endswith("_pass")]
    acceptance["strict_overall_pass"] = (
        acceptance[pass_columns].all(axis=1).astype(int)
    )
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    growth_acceptance = pd.DataFrame(
        [
            {
                "complete_cagr_delta": metric(
                    GROWTH_CANDIDATE, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE, "complete_2015_2025", "cagr"),
                "complete_sharpe_delta": metric(
                    GROWTH_CANDIDATE, "complete_2015_2025", "sharpe"
                )
                - metric(BASELINE, "complete_2015_2025", "sharpe"),
                "complete_volatility_delta": metric(
                    GROWTH_CANDIDATE,
                    "complete_2015_2025",
                    "annual_volatility",
                )
                - metric(BASELINE, "complete_2015_2025", "annual_volatility"),
                "complete_max_drawdown": metric(
                    GROWTH_CANDIDATE, "complete_2015_2025", "max_drawdown"
                ),
                "development_cagr_delta": metric(
                    GROWTH_CANDIDATE, "development_2015_2021", "cagr"
                )
                - metric(BASELINE, "development_2015_2021", "cagr"),
                "holdout_cagr_delta": metric(
                    GROWTH_CANDIDATE, "holdout_2022_2025", "cagr"
                )
                - metric(BASELINE, "holdout_2022_2025", "cagr"),
                "cost15_complete_cagr_delta": metric(
                    GROWTH_CANDIDATE_COST15, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE_COST15, "complete_2015_2025", "cagr"),
                "start2012_extended_cagr_delta": metric(
                    GROWTH_CANDIDATE_2012, "extended_2012_2025", "cagr"
                )
                - metric(BASELINE_2012, "extended_2012_2025", "cagr"),
            }
        ],
        index=[GROWTH_CANDIDATE],
    )
    growth_acceptance["complete_cagr_pass"] = (
        growth_acceptance["complete_cagr_delta"] >= 0.0
    ).astype(int)
    growth_acceptance["complete_sharpe_pass"] = (
        growth_acceptance["complete_sharpe_delta"] >= 0.0
    ).astype(int)
    growth_acceptance["drawdown_18pct_pass"] = (
        growth_acceptance["complete_max_drawdown"] >= -0.18
    ).astype(int)
    growth_acceptance["development_pass"] = (
        growth_acceptance["development_cagr_delta"] >= 0.0
    ).astype(int)
    growth_acceptance["holdout_pass"] = (
        growth_acceptance["holdout_cagr_delta"] >= 0.0
    ).astype(int)
    growth_acceptance["cost15_pass"] = (
        growth_acceptance["cost15_complete_cagr_delta"] >= 0.0
    ).astype(int)
    growth_acceptance["strict_overall_pass"] = growth_acceptance[
        [column for column in growth_acceptance if column.endswith("_pass")]
    ].all(axis=1).astype(int)
    growth_acceptance.to_csv(DESTINATION / "growth_acceptance.csv")

    state_acceptance = pd.DataFrame(
        [
            {
                "complete_cagr_delta": metric(
                    STATE_CANDIDATE, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE, "complete_2015_2025", "cagr"),
                "complete_sharpe_delta": metric(
                    STATE_CANDIDATE, "complete_2015_2025", "sharpe"
                )
                - metric(BASELINE, "complete_2015_2025", "sharpe"),
                "complete_volatility_delta": metric(
                    STATE_CANDIDATE, "complete_2015_2025", "annual_volatility"
                )
                - metric(BASELINE, "complete_2015_2025", "annual_volatility"),
                "complete_max_drawdown": metric(
                    STATE_CANDIDATE, "complete_2015_2025", "max_drawdown"
                ),
                "development_cagr_delta": metric(
                    STATE_CANDIDATE, "development_2015_2021", "cagr"
                )
                - metric(BASELINE, "development_2015_2021", "cagr"),
                "holdout_cagr_delta": metric(
                    STATE_CANDIDATE, "holdout_2022_2025", "cagr"
                )
                - metric(BASELINE, "holdout_2022_2025", "cagr"),
                "cost15_complete_cagr_delta": metric(
                    STATE_CANDIDATE_COST15, "complete_2015_2025", "cagr"
                )
                - metric(BASELINE_COST15, "complete_2015_2025", "cagr"),
                "start2012_extended_cagr_delta": metric(
                    STATE_CANDIDATE_2012, "extended_2012_2025", "cagr"
                )
                - metric(BASELINE_2012, "extended_2012_2025", "cagr"),
            }
        ],
        index=[STATE_CANDIDATE],
    )
    for field in (
        "complete_cagr_delta",
        "complete_sharpe_delta",
        "development_cagr_delta",
        "holdout_cagr_delta",
        "cost15_complete_cagr_delta",
        "start2012_extended_cagr_delta",
    ):
        state_acceptance[f"{field}_pass"] = (
            state_acceptance[field] >= 0.0
        ).astype(int)
    state_acceptance["drawdown_18pct_pass"] = (
        state_acceptance["complete_max_drawdown"] >= -0.18
    ).astype(int)
    state_acceptance["strict_overall_pass"] = state_acceptance[
        [column for column in state_acceptance if column.endswith("_pass")]
    ].all(axis=1).astype(int)
    state_acceptance.to_csv(DESTINATION / "state_acceptance.csv")

    bootstrap = pd.DataFrame(
        {
            "2015_start_complete": circular_block_bootstrap(
                daily[CANDIDATE].loc["2015":"2025", "net_return"],
                daily[BASELINE].loc["2015":"2025", "net_return"],
            ),
            "2012_start_extended": circular_block_bootstrap(
                daily[CANDIDATE_2012].loc[:"2025", "net_return"],
                daily[BASELINE_2012].loc[:"2025", "net_return"],
            ),
        }
    ).T
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")
    growth_bootstrap = pd.DataFrame(
        {
            "2015_start_complete": circular_block_bootstrap(
                daily[GROWTH_CANDIDATE].loc["2015":"2025", "net_return"],
                daily[BASELINE].loc["2015":"2025", "net_return"],
            ),
            "2012_start_extended": circular_block_bootstrap(
                daily[GROWTH_CANDIDATE_2012].loc[:"2025", "net_return"],
                daily[BASELINE_2012].loc[:"2025", "net_return"],
            ),
        }
    ).T
    growth_bootstrap.to_csv(DESTINATION / "growth_bootstrap.csv")
    state_bootstrap = pd.DataFrame(
        {
            "2015_start_complete": circular_block_bootstrap(
                daily[STATE_CANDIDATE].loc["2015":"2025", "net_return"],
                daily[BASELINE].loc["2015":"2025", "net_return"],
            ),
            "2012_start_extended": circular_block_bootstrap(
                daily[STATE_CANDIDATE_2012].loc[:"2025", "net_return"],
                daily[BASELINE_2012].loc[:"2025", "net_return"],
            ),
        }
    ).T
    state_bootstrap.to_csv(DESTINATION / "state_bootstrap.csv")
    block_sensitivity = pd.DataFrame(
        {
            f"block_{block_days}": circular_block_bootstrap(
                daily[STATE_CANDIDATE].loc["2015":"2025", "net_return"],
                daily[BASELINE].loc["2015":"2025", "net_return"],
                block_days=block_days,
            )
            for block_days in (5, 21, 63, 126)
        }
    ).T
    block_sensitivity.to_csv(DESTINATION / "state_block_sensitivity.csv")

    reality, ranking = constrained_reality_check(
        daily[BASELINE].loc["2015":"2025", "net_return"],
        CANDIDATE,
        destination=DESTINATION,
    )
    reality.to_csv(DESTINATION / "multiple_testing_reality_check.csv")
    ranking.to_csv(DESTINATION / "risk_feasible_candidate_ranking.csv")
    growth_reality, _ = constrained_reality_check(
        daily[BASELINE].loc["2015":"2025", "net_return"],
        GROWTH_CANDIDATE,
        destination=DESTINATION,
    )
    growth_reality.to_csv(
        DESTINATION / "growth_multiple_testing_reality_check.csv"
    )
    state_reality, _ = constrained_reality_check(
        daily[BASELINE].loc["2015":"2025", "net_return"],
        STATE_CANDIDATE,
        destination=DESTINATION,
    )
    state_reality.to_csv(
        DESTINATION / "state_multiple_testing_reality_check.csv"
    )

    parameter_rows: list[dict[str, float | str]] = []
    for strategy in PARAMETER_CANDIDATES:
        strategy_metrics = pd.read_csv(
            OUTPUT_ROOT / strategy / "fixed_period_metrics.csv"
        )
        ensemble_metrics = strategy_metrics[
            strategy_metrics["strategy"].eq("ENSEMBLE")
        ].set_index("period")
        full_metrics = pd.read_csv(
            OUTPUT_ROOT / strategy / "metrics.csv", index_col=0
        ).loc["ENSEMBLE"]
        parameter_rows.append(
            {
                "strategy": strategy,
                "full_cagr": full_metrics["cagr"],
                "full_sharpe": full_metrics["sharpe"],
                "max_drawdown": full_metrics["max_drawdown"],
                "development_cagr": ensemble_metrics.loc[
                    "development_2015_2021", "cagr"
                ],
                "holdout_cagr": ensemble_metrics.loc[
                    "holdout_2022_2025", "cagr"
                ],
                "complete_cagr": ensemble_metrics.loc[
                    "complete_2015_2025", "cagr"
                ],
                "complete_sharpe": ensemble_metrics.loc[
                    "complete_2015_2025", "sharpe"
                ],
            }
        )
    pd.DataFrame(parameter_rows).set_index("strategy").to_csv(
        DESTINATION / "state_parameter_robustness.csv"
    )

    fixed_pair = pd.concat(
        [
            daily[STATE_CANDIDATE].loc["2015":"2025", "net_return"].rename(
                "candidate"
            ),
            daily[BASELINE].loc["2015":"2025", "net_return"].rename(
                "baseline"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()
    jackknife_rows: list[dict[str, float | int]] = []
    for excluded_year in sorted(fixed_pair.index.year.unique()):
        sample = fixed_pair[fixed_pair.index.year != excluded_year]
        relative_log_return = (
            np.log1p(sample["candidate"]) - np.log1p(sample["baseline"])
        )
        jackknife_rows.append(
            {
                "excluded_year": int(excluded_year),
                "annualized_relative_return": float(
                    np.expm1(relative_log_return.mean() * 252)
                ),
            }
        )
    pd.DataFrame(jackknife_rows).set_index("excluded_year").to_csv(
        DESTINATION / "state_leave_one_year_out.csv"
    )

    subperiods = {
        "2015_2018": ("2015", "2018"),
        "2019_2021": ("2019", "2021"),
        "2022_2023": ("2022", "2023"),
        "2024_2025": ("2024", "2025"),
    }
    subperiod_rows: list[dict[str, float | str]] = []
    for period, (start, end) in subperiods.items():
        candidate_metrics = performance_metrics(
            fixed_pair.loc[start:end, "candidate"]
        )
        baseline_metrics = performance_metrics(fixed_pair.loc[start:end, "baseline"])
        subperiod_rows.append(
            {
                "period": period,
                "candidate_cagr": candidate_metrics["cagr"],
                "baseline_cagr": baseline_metrics["cagr"],
                "cagr_delta": candidate_metrics["cagr"]
                - baseline_metrics["cagr"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "baseline_sharpe": baseline_metrics["sharpe"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "baseline_max_drawdown": baseline_metrics["max_drawdown"],
            }
        )
    pd.DataFrame(subperiod_rows).set_index("period").to_csv(
        DESTINATION / "state_subperiods.csv"
    )

    triggers = member_trigger_audit()
    triggers.to_csv(DESTINATION / "member_trigger_windows.csv", index=False)
    trigger_summary = pd.DataFrame(
        [
            {
                "member_trigger_count": len(triggers),
                "unique_trigger_dates": triggers["entry_date"].nunique(),
                "mean_smh_multiplier": triggers["smh_multiplier"].mean(),
                "median_relative_window_return": triggers[
                    "relative_window_return"
                ].median(),
                "positive_relative_window_share": triggers[
                    "relative_window_return"
                ].gt(0.0).mean(),
                "sum_member_relative_log_return": float(
                    np.log1p(triggers["relative_window_return"]).sum()
                ),
            }
        ],
        index=[CANDIDATE],
    )
    trigger_summary.to_csv(DESTINATION / "trigger_summary.csv")

    annual = pd.concat(
        {
            "baseline": daily[BASELINE].loc["2015":"2025", "net_return"],
            "candidate": daily[CANDIDATE].loc["2015":"2025", "net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_baseline"] = annual["candidate"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    common = pd.concat(
        [
            daily[CANDIDATE]["net_return"].rename("start_2015"),
            daily[CANDIDATE_2012].loc["2015":, "net_return"].rename("start_2012"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    start_stability = pd.DataFrame(
        [
            {
                "daily_return_correlation": common.corr().iloc[0, 1],
                "start_2015_common_cagr": performance_metrics(
                    common["start_2015"]
                )["cagr"],
                "start_2012_common_cagr": performance_metrics(
                    common["start_2012"]
                )["cagr"],
                "start_2015_common_max_drawdown": performance_metrics(
                    common["start_2015"]
                )["max_drawdown"],
                "start_2012_common_max_drawdown": performance_metrics(
                    common["start_2012"]
                )["max_drawdown"],
            }
        ],
        index=[CANDIDATE],
    )
    start_stability.to_csv(DESTINATION / "start_date_stability.csv")

    current_event = pd.DataFrame(
        [
            {
                "baseline_2026_07_15_17_return": float(
                    (1.0 + daily[BASELINE].loc["2026-07-15":"2026-07-17", "net_return"]).prod()
                    - 1.0
                ),
                "candidate_2026_07_15_17_return": float(
                    (1.0 + daily[CANDIDATE].loc["2026-07-15":"2026-07-17", "net_return"]).prod()
                    - 1.0
                ),
                "current_qqq_target": pd.read_csv(
                    OUTPUT_ROOT / CANDIDATE / "next_target_weights.csv",
                    index_col=0,
                ).loc["QQQ", "ensemble_current_sleeve_weight"],
                "current_smh_target": pd.read_csv(
                    OUTPUT_ROOT / CANDIDATE / "next_target_weights.csv",
                    index_col=0,
                ).loc["SEMIS", "ensemble_current_sleeve_weight"],
                "current_cash_target": pd.read_csv(
                    OUTPUT_ROOT / CANDIDATE / "next_target_weights.csv",
                    index_col=0,
                ).loc["CASH", "ensemble_current_sleeve_weight"],
            }
        ],
        index=[CANDIDATE],
    )
    current_event.to_csv(DESTINATION / "current_event.csv")

    print("Acceptance:")
    print(acceptance.round(6).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string())
    print("\nGrowth acceptance:")
    print(growth_acceptance.round(6).to_string())
    print("\nGrowth bootstrap:")
    print(growth_bootstrap.round(6).to_string())
    print("\nState acceptance:")
    print(state_acceptance.round(6).to_string())
    print("\nState bootstrap:")
    print(state_bootstrap.round(6).to_string())
    print("\nState block sensitivity:")
    print(block_sensitivity.round(6).to_string())
    print("\nFamily-wise Reality Check:")
    print(reality.round(6).to_string())
    print("\nGrowth Family-wise Reality Check:")
    print(growth_reality.round(6).to_string())
    print("\nState Family-wise Reality Check:")
    print(state_reality.round(6).to_string())
    print("\nTrigger summary:")
    print(trigger_summary.round(6).to_string())
    print("\nStart-date stability:")
    print(start_stability.round(6).to_string())
    print("\nCurrent event:")
    print(current_event.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
