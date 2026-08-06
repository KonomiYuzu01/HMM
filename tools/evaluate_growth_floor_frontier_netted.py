from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_reentry_redesign import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "growth_floor_frontier_netted_validation"
LEVEL_DIRECTORIES = {
    0.20: "paper_core_growth_gold20_daily_risk_netted_ensemble",
    0.25: "paper_core_growth_gold20_floor25_daily_risk_netted_ensemble",
    0.30: "paper_core_growth_gold20_floor30_daily_risk_netted_ensemble",
    0.35: "paper_core_growth_gold20_floor35_daily_risk_netted_ensemble",
    0.40: "paper_core_growth_gold20_floor40_daily_risk_netted_ensemble",
}
SCENARIOS = {
    "normal": {
        "baseline": "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "floor40": "paper_core_growth_gold20_floor40_daily_risk_netted_ensemble",
    },
    "cost15": {
        "baseline": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15"
        ),
        "floor40": (
            "paper_core_growth_gold20_floor40_daily_risk_"
            "netted_ensemble_cost15"
        ),
    },
    "extended": {
        "baseline": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
        ),
        "floor40": (
            "paper_core_growth_gold20_floor40_daily_risk_"
            "netted_ensemble_2012"
        ),
    },
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
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
    level_frames = {
        level: load(directory)
        for level, directory in LEVEL_DIRECTORIES.items()
    }
    scenario_frames = {
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

    frontier_rows = []
    for level, frame in level_frames.items():
        row: dict[str, float] = {"floor_share": level}
        for period, (start, end) in PERIODS.items():
            metrics = performance_metrics(frame.loc[start:end, "net_return"])
            for metric in (
                "cagr",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "calmar",
            ):
                row[f"{period}_{metric}"] = float(metrics[metric])
        frontier_rows.append(row)
    frontier = pd.DataFrame(frontier_rows).set_index("floor_share")
    baseline_development_sharpe = float(
        frontier.loc[0.20, "development_2015_2021_sharpe"]
    )
    frontier["development_drawdown_pass"] = (
        frontier["development_2015_2021_max_drawdown"] >= -0.15
    ).astype(int)
    frontier["development_sharpe_pass"] = (
        frontier["development_2015_2021_sharpe"]
        >= baseline_development_sharpe - 0.02
    ).astype(int)
    frontier["eligible"] = frontier[
        ["development_drawdown_pass", "development_sharpe_pass"]
    ].all(axis=1).astype(int)
    eligible = frontier[frontier["eligible"].eq(1)]
    selected_floor = float(
        eligible["development_2015_2021_cagr"].idxmax()
    )
    frontier["selected"] = (frontier.index == selected_floor).astype(int)
    frontier.to_csv(DESTINATION / "frontier.csv")

    metric_rows: list[dict[str, float | str]] = []
    for scenario, frames in scenario_frames.items():
        periods = (
            {"extended_2012_2025": ("2012-01-01", "2025-12-31")}
            if scenario == "extended"
            else PERIODS
        )
        for strategy, frame in frames.items():
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
    for scenario, sample in samples.items():
        for block_days in (21, 63, 126):
            result = circular_block_bootstrap(
                scenario_frames[scenario]["floor40"].loc[
                    sample, "net_return"
                ],
                scenario_frames[scenario]["baseline"].loc[
                    sample, "net_return"
                ],
                block_days=block_days,
            )
            bootstrap_rows.append(
                {
                    "scenario": scenario,
                    "block_days": block_days,
                    **result.to_dict(),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index(
        ["scenario", "block_days"]
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    primary = slice("2015-01-01", "2025-12-31")
    baseline = scenario_frames["normal"]["baseline"].loc[primary]
    candidate = scenario_frames["normal"]["floor40"].loc[primary]
    aligned = pd.concat(
        [
            candidate["net_return"].rename("candidate"),
            baseline["net_return"].rename("baseline"),
            growth.loc[primary].rename("growth"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    baseline_weights = load(
        SCENARIOS["normal"]["baseline"], "weights.csv"
    ).reindex(aligned.index)
    candidate_weights = load(
        SCENARIOS["normal"]["floor40"], "weights.csv"
    ).reindex(aligned.index)
    baseline_growth_exposure = baseline_weights[["QQQ", "SEMIS"]].sum(axis=1)
    candidate_growth_exposure = candidate_weights[["QQQ", "SEMIS"]].sum(axis=1)
    exposure_change = candidate_growth_exposure - baseline_growth_exposure
    changed = exposure_change.abs() > 0.001
    relative_log = (
        np.log1p(aligned["candidate"]) - np.log1p(aligned["baseline"])
    )
    up_capture_delta = (
        capture_ratio(aligned["candidate"], aligned["growth"], True)
        - capture_ratio(aligned["baseline"], aligned["growth"], True)
    )
    down_capture_delta = (
        capture_ratio(aligned["candidate"], aligned["growth"], False)
        - capture_ratio(aligned["baseline"], aligned["growth"], False)
    )
    decomposition = pd.DataFrame(
        [
            {
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
                "fraction_changed_exposure_days": float(changed.mean()),
                "baseline_average_growth_exposure": float(
                    baseline_growth_exposure.mean()
                ),
                "candidate_average_growth_exposure": float(
                    candidate_growth_exposure.mean()
                ),
                "up_capture_delta": up_capture_delta,
                "down_capture_delta": down_capture_delta,
                "capture_spread_delta": (
                    up_capture_delta - down_capture_delta
                ),
                "incremental_annualized_cost_drag": float(
                    (candidate["cost"] - baseline["cost"]).mean() * 252.0
                ),
            }
        ],
        index=["floor40_vs_step2"],
    )
    decomposition.to_csv(DESTINATION / "mechanism_decomposition.csv")

    annual = pd.concat(
        {
            "baseline": baseline["net_return"],
            "floor40": candidate["net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["floor40_minus_baseline"] = annual["floor40"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_returns.csv")

    bridge = load(
        "paper_core_growth_gold20_floor40_selective_bridge_"
        "daily_hmm_netted_ensemble"
    )
    bridge_metrics = []
    for period, (start, end) in PERIODS.items():
        bridge_row = performance_metrics(bridge.loc[start:end, "net_return"])
        floor_row = performance_metrics(
            level_frames[0.40].loc[start:end, "net_return"]
        )
        bridge_metrics.append(
            {
                "period": period,
                "cagr_delta": float(
                    bridge_row["cagr"] - floor_row["cagr"]
                ),
                "sharpe_delta": float(
                    bridge_row["sharpe"] - floor_row["sharpe"]
                ),
                "max_drawdown_delta": float(
                    bridge_row["max_drawdown"]
                    - floor_row["max_drawdown"]
                ),
            }
        )
    pd.DataFrame(bridge_metrics).set_index("period").to_csv(
        DESTINATION / "bridge_rejection.csv"
    )

    normal_primary = metrics.loc[
        ("normal", slice(None), "complete_2015_2025"), :
    ].droplevel(["scenario", "period"])
    cost_primary = metrics.loc[
        ("cost15", slice(None), "complete_2015_2025"), :
    ].droplevel(["scenario", "period"])
    extended_primary = metrics.loc[
        ("extended", slice(None), "extended_2012_2025"), :
    ].droplevel(["scenario", "period"])
    acceptance = pd.DataFrame(
        [
            {
                "selected_floor": selected_floor,
                "complete_cagr_delta": float(
                    normal_primary.loc["floor40", "cagr"]
                    - normal_primary.loc["baseline", "cagr"]
                ),
                "development_cagr_delta": float(
                    frontier.loc[
                        0.40, "development_2015_2021_cagr"
                    ]
                    - frontier.loc[
                        0.20, "development_2015_2021_cagr"
                    ]
                ),
                "holdout_cagr_delta": float(
                    frontier.loc[0.40, "holdout_2022_2025_cagr"]
                    - frontier.loc[0.20, "holdout_2022_2025_cagr"]
                ),
                "cost15_cagr_delta": float(
                    cost_primary.loc["floor40", "cagr"]
                    - cost_primary.loc["baseline", "cagr"]
                ),
                "extended_cagr_delta": float(
                    extended_primary.loc["floor40", "cagr"]
                    - extended_primary.loc["baseline", "cagr"]
                ),
                "candidate_max_drawdown": float(
                    normal_primary.loc["floor40", "max_drawdown"]
                ),
                "minimum_bootstrap_probability_positive": float(
                    bootstrap["probability_positive"].min()
                ),
                "minimum_bootstrap_lower_95": float(
                    bootstrap["lower_95"].min()
                ),
                "all_point_estimates_positive": int(
                    min(
                        [
                            frontier.loc[
                                0.40, "development_2015_2021_cagr"
                            ]
                            - frontier.loc[
                                0.20, "development_2015_2021_cagr"
                            ],
                            frontier.loc[
                                0.40, "holdout_2022_2025_cagr"
                            ]
                            - frontier.loc[
                                0.20, "holdout_2022_2025_cagr"
                            ],
                            cost_primary.loc["floor40", "cagr"]
                            - cost_primary.loc["baseline", "cagr"],
                            extended_primary.loc["floor40", "cagr"]
                            - extended_primary.loc["baseline", "cagr"],
                        ]
                    )
                    >= 0.0
                ),
                "max_drawdown_above_minus_15pct": int(
                    normal_primary.loc["floor40", "max_drawdown"] >= -0.15
                ),
                "all_bootstrap_lower_bounds_positive": int(
                    (bootstrap["lower_95"] > 0.0).all()
                ),
            }
        ],
        index=["floor40"],
    )
    acceptance.to_csv(DESTINATION / "acceptance.csv")


if __name__ == "__main__":
    main()
