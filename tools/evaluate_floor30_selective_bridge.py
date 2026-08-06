from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_reentry_redesign import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "floor30_selective_bridge_validation"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
DIRECTORIES = {
    "normal": {
        "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "floor30": "paper_core_growth_gold20_floor30_daily_risk_netted_ensemble",
        "floor30_bridge": (
            "paper_core_growth_gold20_floor30_selective_bridge_"
            "daily_hmm_netted_ensemble"
        ),
    },
    "cost15": {
        "step2_netted": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15"
        ),
        "floor30": (
            "paper_core_growth_gold20_floor30_daily_risk_"
            "netted_ensemble_cost15"
        ),
        "floor30_bridge": (
            "paper_core_growth_gold20_floor30_selective_bridge_"
            "daily_hmm_netted_ensemble_cost15"
        ),
    },
    "extended": {
        "step2_netted": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
        ),
        "floor30": (
            "paper_core_growth_gold20_floor30_daily_risk_"
            "netted_ensemble_2012"
        ),
        "floor30_bridge": (
            "paper_core_growth_gold20_floor30_selective_bridge_"
            "daily_hmm_netted_ensemble_2012"
        ),
    },
}
COMPARISONS = {
    "floor30_vs_step2": ("floor30", "step2_netted"),
    "bridge_increment": ("floor30_bridge", "floor30"),
    "combined_vs_step2": ("floor30_bridge", "step2_netted"),
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
        for scenario, directories in DIRECTORIES.items()
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
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "period": period,
                        **performance_metrics(sample["net_return"]),
                        "annualized_cost_drag": float(
                            sample["cost"].mean() * 252.0
                        ),
                        "average_daily_turnover": float(
                            sample["turnover"].mean()
                        ),
                    }
                )
    metrics = pd.DataFrame(metric_rows).set_index(
        ["scenario", "strategy", "period"]
    )
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    bootstrap_rows = []
    samples = {
        "normal": slice("2015-01-01", "2025-12-31"),
        "cost15": slice("2015-01-01", "2025-12-31"),
        "extended": slice("2012-01-01", "2025-12-31"),
    }
    for comparison, (candidate, baseline) in COMPARISONS.items():
        for scenario, sample in samples.items():
            result = circular_block_bootstrap(
                frames[scenario][candidate].loc[sample, "net_return"],
                frames[scenario][baseline].loc[sample, "net_return"],
            )
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "scenario": scenario,
                    **result.to_dict(),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index(
        ["comparison", "scenario"]
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    primary = slice("2015-01-01", "2025-12-31")
    decomposition_rows: list[dict[str, float | str]] = []
    normal = frames["normal"]
    for comparison, (candidate, baseline) in COMPARISONS.items():
        aligned = pd.concat(
            [
                normal[candidate].loc[primary, "net_return"].rename("candidate"),
                normal[baseline].loc[primary, "net_return"].rename("baseline"),
                growth.loc[primary].rename("growth"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        candidate_weights = load(
            DIRECTORIES["normal"][candidate], "weights.csv"
        ).reindex(aligned.index)
        baseline_weights = load(
            DIRECTORIES["normal"][baseline], "weights.csv"
        ).reindex(aligned.index)
        exposure_change = (
            candidate_weights[["QQQ", "SEMIS"]].sum(axis=1)
            - baseline_weights[["QQQ", "SEMIS"]].sum(axis=1)
        )
        changed = exposure_change.abs() > 0.001
        relative_log = (
            np.log1p(aligned["candidate"])
            - np.log1p(aligned["baseline"])
        )
        decomposition_rows.append(
            {
                "comparison": comparison,
                "annualized_relative_log_return": float(
                    relative_log.mean() * 252.0
                ),
                "growth_up_day_contribution": float(
                    relative_log.where(aligned["growth"] > 0.0, 0.0).mean()
                    * 252.0
                ),
                "growth_down_day_contribution": float(
                    relative_log.where(aligned["growth"] < 0.0, 0.0).mean()
                    * 252.0
                ),
                "changed_exposure_day_contribution": float(
                    relative_log.where(changed, 0.0).mean() * 252.0
                ),
                "unchanged_exposure_day_contribution": float(
                    relative_log.where(~changed, 0.0).mean() * 252.0
                ),
                "fraction_changed_exposure_days": float(changed.mean()),
                "average_growth_exposure_change": float(
                    exposure_change.mean()
                ),
                "up_capture_change": (
                    capture_ratio(
                        aligned["candidate"], aligned["growth"], True
                    )
                    - capture_ratio(
                        aligned["baseline"], aligned["growth"], True
                    )
                ),
                "down_capture_change": (
                    capture_ratio(
                        aligned["candidate"], aligned["growth"], False
                    )
                    - capture_ratio(
                        aligned["baseline"], aligned["growth"], False
                    )
                ),
            }
        )
    pd.DataFrame(decomposition_rows).set_index("comparison").to_csv(
        DESTINATION / "mechanism_decomposition.csv"
    )

    annual = pd.concat(
        {
            strategy: frame.loc[primary, "net_return"]
            for strategy, frame in normal.items()
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual.to_csv(DESTINATION / "annual_returns.csv")

    acceptance_rows = []
    for comparison, (candidate, baseline) in COMPARISONS.items():
        primary_candidate = metrics.loc[
            ("normal", candidate, "complete_2015_2025")
        ]
        primary_baseline = metrics.loc[
            ("normal", baseline, "complete_2015_2025")
        ]
        development_delta = float(
            metrics.loc[
                ("normal", candidate, "development_2015_2021"), "cagr"
            ]
            - metrics.loc[
                ("normal", baseline, "development_2015_2021"), "cagr"
            ]
        )
        holdout_delta = float(
            metrics.loc[
                ("normal", candidate, "holdout_2022_2025"), "cagr"
            ]
            - metrics.loc[
                ("normal", baseline, "holdout_2022_2025"), "cagr"
            ]
        )
        cost_delta = float(
            metrics.loc[
                ("cost15", candidate, "complete_2015_2025"), "cagr"
            ]
            - metrics.loc[
                ("cost15", baseline, "complete_2015_2025"), "cagr"
            ]
        )
        extended_delta = float(
            metrics.loc[
                ("extended", candidate, "extended_2012_2025"), "cagr"
            ]
            - metrics.loc[
                ("extended", baseline, "extended_2012_2025"), "cagr"
            ]
        )
        bootstrap_row = bootstrap.loc[(comparison, "normal")]
        acceptance_rows.append(
            {
                "comparison": comparison,
                "complete_cagr_delta": float(
                    primary_candidate["cagr"] - primary_baseline["cagr"]
                ),
                "development_cagr_delta": development_delta,
                "holdout_cagr_delta": holdout_delta,
                "cost15_cagr_delta": cost_delta,
                "extended_cagr_delta": extended_delta,
                "candidate_max_drawdown": float(
                    primary_candidate["max_drawdown"]
                ),
                "bootstrap_probability_positive": float(
                    bootstrap_row["probability_positive"]
                ),
                "bootstrap_lower_95": float(bootstrap_row["lower_95"]),
                "all_point_estimates_positive": int(
                    min(
                        development_delta,
                        holdout_delta,
                        cost_delta,
                        extended_delta,
                    )
                    >= 0.0
                ),
                "max_drawdown_above_minus_15pct": int(
                    primary_candidate["max_drawdown"] >= -0.15
                ),
                "bootstrap_lower_bound_positive": int(
                    bootstrap_row["lower_95"] > 0.0
                ),
            }
        )
    pd.DataFrame(acceptance_rows).set_index("comparison").to_csv(
        DESTINATION / "acceptance.csv"
    )


if __name__ == "__main__":
    main()
