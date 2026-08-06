from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_reentry_redesign import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "correction_veto_validation"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
SCENARIOS = {
    "normal": {
        "step2": "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "floor40": (
            "paper_core_growth_gold20_floor40_daily_risk_netted_ensemble"
        ),
        "candidate": (
            "paper_core_growth_gold20_correction_veto63_exact_"
            "floor_guard_netted_ensemble"
        ),
    },
    "cost15": {
        "step2": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15"
        ),
        "floor40": (
            "paper_core_growth_gold20_floor40_daily_risk_"
            "netted_ensemble_cost15"
        ),
        "candidate": (
            "paper_core_growth_gold20_correction_veto63_exact_"
            "floor_guard_netted_ensemble_cost15"
        ),
    },
    "extended": {
        "step2": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
        ),
        "floor40": (
            "paper_core_growth_gold20_floor40_daily_risk_"
            "netted_ensemble_2012"
        ),
        "candidate": (
            "paper_core_growth_gold20_correction_veto63_exact_"
            "floor_guard_netted_ensemble_2012"
        ),
    },
}


def load(directory: str, filename: str = "daily_returns.csv") -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def capture_ratio(
    strategy: pd.Series,
    benchmark: pd.Series,
    positive: bool,
) -> float:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    mask = aligned["benchmark"] > 0.0 if positive else aligned["benchmark"] < 0.0
    return float(
        aligned.loc[mask, "strategy"].mean()
        / aligned.loc[mask, "benchmark"].mean()
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    frames = {
        scenario: {
            strategy: load(directory)
            for strategy, directory in directories.items()
        }
        for scenario, directories in SCENARIOS.items()
    }
    prices = pd.read_csv(
        "data/prices_recovery_quality.csv",
        index_col=0,
        parse_dates=True,
    )
    growth = prices[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1)

    metric_rows: list[dict[str, float | str]] = []
    for scenario, scenario_frames in frames.items():
        periods = (
            {"extended_2012_2025": ("2012-01-01", "2025-12-31")}
            if scenario == "extended"
            else PERIODS
        )
        for strategy, frame in scenario_frames.items():
            for period, (start, end) in periods.items():
                sample = frame.loc[start:end]
                metrics = performance_metrics(sample["net_return"])
                aligned_growth = growth.reindex(sample.index)
                up_capture = capture_ratio(
                    sample["net_return"], aligned_growth, True
                )
                down_capture = capture_ratio(
                    sample["net_return"], aligned_growth, False
                )
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "period": period,
                        **metrics,
                        "annualized_cost_drag": float(
                            sample["cost"].mean() * 252.0
                        ),
                        "annualized_turnover": float(
                            sample["turnover"].mean() * 252.0
                        ),
                        "up_capture": up_capture,
                        "down_capture": down_capture,
                        "capture_spread": up_capture - down_capture,
                    }
                )
    metrics = pd.DataFrame(metric_rows).set_index(
        ["scenario", "strategy", "period"]
    )
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    bootstrap_rows: list[dict[str, float | str]] = []
    samples = {
        "normal": slice("2015-01-01", "2025-12-31"),
        "cost15": slice("2015-01-01", "2025-12-31"),
        "extended": slice("2012-01-01", "2025-12-31"),
    }
    for scenario, sample in samples.items():
        candidate = frames[scenario]["candidate"].loc[
            sample, "net_return"
        ]
        for comparator in ("step2", "floor40"):
            baseline = frames[scenario][comparator].loc[
                sample, "net_return"
            ]
            for block_days in (21, 63, 126):
                result = circular_block_bootstrap(
                    candidate,
                    baseline,
                    block_days=block_days,
                )
                bootstrap_rows.append(
                    {
                        "scenario": scenario,
                        "comparator": comparator,
                        "block_days": block_days,
                        **result.to_dict(),
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index(
        ["scenario", "comparator", "block_days"]
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    primary = slice("2015-01-01", "2025-12-31")
    normal = {
        name: frame.loc[primary]
        for name, frame in frames["normal"].items()
    }
    weights = {
        name: load(SCENARIOS["normal"][name], "weights.csv").loc[primary]
        for name in ("step2", "floor40", "candidate")
    }
    growth_exposure = {
        name: frame[["QQQ", "SEMIS"]].sum(axis=1)
        for name, frame in weights.items()
    }
    candidate_log = np.log1p(normal["candidate"]["net_return"])
    comparison_rows: list[dict[str, float | str]] = []
    for comparator in ("step2", "floor40"):
        comparison_log = np.log1p(normal[comparator]["net_return"])
        relative_log = candidate_log - comparison_log
        exposure_delta = (
            growth_exposure["candidate"] - growth_exposure[comparator]
        )
        changed = exposure_delta.abs() > 0.001
        comparison_rows.append(
            {
                "comparison": f"candidate_vs_{comparator}",
                "annualized_relative_log_return": float(
                    relative_log.mean() * 252.0
                ),
                "growth_up_day_contribution": float(
                    relative_log.where(growth.loc[primary] > 0.0, 0.0).mean()
                    * 252.0
                ),
                "growth_down_day_contribution": float(
                    relative_log.where(growth.loc[primary] < 0.0, 0.0).mean()
                    * 252.0
                ),
                "changed_exposure_day_contribution": float(
                    relative_log.where(changed, 0.0).mean() * 252.0
                ),
                "fraction_changed_exposure_days": float(changed.mean()),
                "average_growth_exposure_delta": float(
                    exposure_delta.mean()
                ),
                "incremental_annualized_cost_drag": float(
                    (
                        normal["candidate"]["cost"]
                        - normal[comparator]["cost"]
                    ).mean()
                    * 252.0
                ),
            }
        )
    pd.DataFrame(comparison_rows).set_index("comparison").to_csv(
        DESTINATION / "mechanism_decomposition.csv"
    )

    floor_relative_log = (
        candidate_log - np.log1p(normal["floor40"]["net_return"])
    )
    veto_active = (
        growth_exposure["candidate"]
        < growth_exposure["floor40"] - 0.001
    )
    episode_ids = veto_active.ne(veto_active.shift(fill_value=False)).cumsum()
    episode_rows: list[dict[str, float | int | str]] = []
    for episode_id, episode_mask in veto_active.groupby(episode_ids):
        if not bool(episode_mask.iloc[0]):
            continue
        dates = episode_mask.index
        episode_relative_log = floor_relative_log.loc[dates]
        episode_growth = growth.reindex(dates).dropna()
        episode_rows.append(
            {
                "episode_id": int(episode_id),
                "start": dates[0].date().isoformat(),
                "end": dates[-1].date().isoformat(),
                "days": len(dates),
                "growth_return": float(
                    (1.0 + episode_growth).prod() - 1.0
                ),
                "candidate_minus_floor40_return": float(
                    np.expm1(episode_relative_log.sum())
                ),
                "annualized_full_sample_log_contribution": float(
                    episode_relative_log.sum()
                    / len(floor_relative_log)
                    * 252.0
                ),
                "average_candidate_growth_exposure": float(
                    growth_exposure["candidate"].loc[dates].mean()
                ),
                "average_floor40_growth_exposure": float(
                    growth_exposure["floor40"].loc[dates].mean()
                ),
            }
        )
    episodes = pd.DataFrame(episode_rows).sort_values(
        "annualized_full_sample_log_contribution",
        ascending=False,
    )
    episodes.to_csv(DESTINATION / "veto_episodes.csv", index=False)
    total_floor_relative_log = float(floor_relative_log.sum())
    leave_one_out = episodes[
        [
            "episode_id",
            "start",
            "end",
            "days",
            "annualized_full_sample_log_contribution",
        ]
    ].copy()
    leave_one_out["annualized_relative_log_without_episode"] = (
        total_floor_relative_log
        - (
            leave_one_out["annualized_full_sample_log_contribution"]
            * len(floor_relative_log)
            / 252.0
        )
    ) / len(floor_relative_log) * 252.0
    leave_one_out.to_csv(
        DESTINATION / "leave_one_episode_out.csv",
        index=False,
    )

    annual = pd.concat(
        {
            name: frame["net_return"]
            for name, frame in normal.items()
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_step2"] = (
        annual["candidate"] - annual["step2"]
    )
    annual["candidate_minus_floor40"] = (
        annual["candidate"] - annual["floor40"]
    )
    annual.to_csv(DESTINATION / "annual_returns.csv")
    year_rows: list[dict[str, float | int]] = []
    total_days = len(floor_relative_log)
    for year in sorted(floor_relative_log.index.year.unique()):
        year_mask = floor_relative_log.index.year == year
        year_sum = float(floor_relative_log.loc[year_mask].sum())
        year_rows.append(
            {
                "omitted_year": int(year),
                "year_contribution_full_sample_annualized": (
                    year_sum / total_days * 252.0
                ),
                "annualized_relative_log_without_year": float(
                    (
                        floor_relative_log.sum() - year_sum
                    )
                    / (total_days - int(year_mask.sum()))
                    * 252.0
                ),
            }
        )
    leave_one_year_out = pd.DataFrame(year_rows).set_index("omitted_year")
    leave_one_year_out.to_csv(
        DESTINATION / "leave_one_year_out.csv"
    )
    positive_episode_contributions = episodes.loc[
        episodes["annualized_full_sample_log_contribution"] > 0.0,
        "annualized_full_sample_log_contribution",
    ].sort_values(ascending=False)
    episode_summary = pd.DataFrame(
        [
            {
                "episode_count": len(episodes),
                "positive_episode_count": int(
                    (
                        episodes[
                            "annualized_full_sample_log_contribution"
                        ]
                        > 0.0
                    ).sum()
                ),
                "negative_episode_count": int(
                    (
                        episodes[
                            "annualized_full_sample_log_contribution"
                        ]
                        < 0.0
                    ).sum()
                ),
                "largest_positive_share_of_episode_net": float(
                    positive_episode_contributions.iloc[0]
                    / episodes[
                        "annualized_full_sample_log_contribution"
                    ].sum()
                ),
                "top_two_positive_share_of_episode_net": float(
                    positive_episode_contributions.iloc[:2].sum()
                    / episodes[
                        "annualized_full_sample_log_contribution"
                    ].sum()
                ),
                "relative_log_without_largest_episode": float(
                    leave_one_out.iloc[0][
                        "annualized_relative_log_without_episode"
                    ]
                ),
                "relative_log_without_top_two_episodes": float(
                    floor_relative_log.mean() * 252.0
                    - positive_episode_contributions.iloc[:2].sum()
                ),
                "minimum_leave_one_year_out_relative_log": float(
                    leave_one_year_out[
                        "annualized_relative_log_without_year"
                    ].min()
                ),
            }
        ],
        index=["correction_veto63_exact_floor_guard"],
    )
    episode_summary.to_csv(
        DESTINATION / "episode_robustness_summary.csv"
    )

    seed_metric_rows: list[dict[str, float | int | str]] = []
    for seed in (7, 42, 123):
        for strategy, directory in SCENARIOS["normal"].items():
            member = load(
                f"{directory}/members/seed_{seed}",
                "daily_returns.csv",
            )
            for period, (start, end) in PERIODS.items():
                sample = member.loc[start:end]
                seed_metric_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "period": period,
                        **performance_metrics(sample["net_return"]),
                        "annualized_cost_drag": float(
                            sample["cost"].mean() * 252.0
                        ),
                    }
                )
    seed_metrics = pd.DataFrame(seed_metric_rows).set_index(
        ["seed", "strategy", "period"]
    )
    seed_metrics.to_csv(DESTINATION / "seed_metrics.csv")
    seed_delta_rows: list[dict[str, float | int | str]] = []
    for seed in (7, 42, 123):
        for period in PERIODS:
            candidate_seed = seed_metrics.loc[
                (seed, "candidate", period)
            ]
            for comparator in ("step2", "floor40"):
                comparator_seed = seed_metrics.loc[
                    (seed, comparator, period)
                ]
                seed_delta_rows.append(
                    {
                        "seed": seed,
                        "period": period,
                        "comparator": comparator,
                        "cagr_delta": float(
                            candidate_seed["cagr"]
                            - comparator_seed["cagr"]
                        ),
                        "sharpe_delta": float(
                            candidate_seed["sharpe"]
                            - comparator_seed["sharpe"]
                        ),
                        "max_drawdown_delta": float(
                            candidate_seed["max_drawdown"]
                            - comparator_seed["max_drawdown"]
                        ),
                    }
                )
    pd.DataFrame(seed_delta_rows).set_index(
        ["seed", "period", "comparator"]
    ).to_csv(DESTINATION / "seed_deltas.csv")

    normal_complete = metrics.loc[
        ("normal", slice(None), "complete_2015_2025"), :
    ].droplevel(["scenario", "period"])
    cost_complete = metrics.loc[
        ("cost15", slice(None), "complete_2015_2025"), :
    ].droplevel(["scenario", "period"])
    extended_complete = metrics.loc[
        ("extended", slice(None), "extended_2012_2025"), :
    ].droplevel(["scenario", "period"])
    development = metrics.loc[
        ("normal", slice(None), "development_2015_2021"), :
    ].droplevel(["scenario", "period"])
    holdout = metrics.loc[
        ("normal", slice(None), "holdout_2022_2025"), :
    ].droplevel(["scenario", "period"])
    candidate_bootstrap = bootstrap.xs(
        "floor40", level="comparator"
    )
    acceptance = pd.DataFrame(
        [
            {
                "development_cagr_delta_vs_step2": float(
                    development.loc["candidate", "cagr"]
                    - development.loc["step2", "cagr"]
                ),
                "holdout_cagr_delta_vs_step2": float(
                    holdout.loc["candidate", "cagr"]
                    - holdout.loc["step2", "cagr"]
                ),
                "complete_cagr_delta_vs_floor40": float(
                    normal_complete.loc["candidate", "cagr"]
                    - normal_complete.loc["floor40", "cagr"]
                ),
                "cost15_cagr_delta_vs_floor40": float(
                    cost_complete.loc["candidate", "cagr"]
                    - cost_complete.loc["floor40", "cagr"]
                ),
                "extended_cagr_delta_vs_floor40": float(
                    extended_complete.loc["candidate", "cagr"]
                    - extended_complete.loc["floor40", "cagr"]
                ),
                "development_capture_spread_delta_vs_step2": float(
                    development.loc["candidate", "capture_spread"]
                    - development.loc["step2", "capture_spread"]
                ),
                "holdout_capture_spread_delta_vs_step2": float(
                    holdout.loc["candidate", "capture_spread"]
                    - holdout.loc["step2", "capture_spread"]
                ),
                "candidate_max_drawdown": float(
                    normal_complete.loc["candidate", "max_drawdown"]
                ),
                "minimum_probability_positive_vs_floor40": float(
                    candidate_bootstrap["probability_positive"].min()
                ),
                "minimum_lower_95_vs_floor40": float(
                    candidate_bootstrap["lower_95"].min()
                ),
                "all_point_estimates_positive": int(
                    min(
                        normal_complete.loc["candidate", "cagr"]
                        - normal_complete.loc["floor40", "cagr"],
                        cost_complete.loc["candidate", "cagr"]
                        - cost_complete.loc["floor40", "cagr"],
                        extended_complete.loc["candidate", "cagr"]
                        - extended_complete.loc["floor40", "cagr"],
                    )
                    > 0.0
                ),
                "development_and_holdout_capture_spread_positive": int(
                    development.loc["candidate", "capture_spread"]
                    > development.loc["step2", "capture_spread"]
                    and holdout.loc["candidate", "capture_spread"]
                    > holdout.loc["step2", "capture_spread"]
                ),
                "max_drawdown_above_minus_15pct": int(
                    normal_complete.loc["candidate", "max_drawdown"] >= -0.15
                ),
                "all_bootstrap_lower_bounds_positive_vs_floor40": int(
                    (candidate_bootstrap["lower_95"] > 0.0).all()
                ),
            }
        ],
        index=["correction_veto63_exact_floor_guard"],
    )
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    print(metrics.round(6).to_string())
    print("\nBootstrap")
    print(bootstrap.round(6).to_string())
    print("\nAcceptance")
    print(acceptance.round(6).to_string())


if __name__ == "__main__":
    main()
