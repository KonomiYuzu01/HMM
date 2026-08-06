from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_drawdown_uncertainty import paired_circular_block_bootstrap
from regime_strategy.report import performance_metrics


ROOT = Path("output/open_execution_validation")
DESTINATION = Path("output/jump_aware_validation")
SCENARIOS = {
    "normal_2015": (
        "gold_20_daily_asset_cap_daily.csv",
        "gold_20_jump_aware_daily_cap_daily.csv",
        "2015-01-01",
    ),
    "cost15_2015": (
        "gold_20_daily_asset_cap_cost15_daily.csv",
        "gold_20_jump_aware_daily_cap_cost15_daily.csv",
        "2015-01-01",
    ),
    "extended_2012": (
        "gold_20_daily_asset_cap_2012_daily.csv",
        "gold_20_jump_aware_daily_cap_2012_daily.csv",
        "2012-01-01",
    ),
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def load(path: str, start: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, index_col=0, parse_dates=True).loc[start:]


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for scenario, (baseline_path, candidate_path, start) in SCENARIOS.items():
        baseline = load(baseline_path, start)
        candidate = load(candidate_path, start)
        frames[scenario] = (baseline, candidate)
        for strategy, frame in (("baseline", baseline), ("jump_aware", candidate)):
            metric_rows.append(
                {
                    "scenario": scenario,
                    "strategy": strategy,
                    **performance_metrics(frame["net_return"]),
                    "annualized_cost": float(frame["cost"].mean() * 252.0),
                    "annualized_open_turnover": float(
                        frame["open_turnover"].mean() * 252.0
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows).set_index(["scenario", "strategy"])
    metrics.to_csv(DESTINATION / "scenario_metrics.csv")

    baseline, candidate = frames["normal_2015"]
    period_rows = []
    for period, (start, end) in PERIODS.items():
        for strategy, frame in (("baseline", baseline), ("jump_aware", candidate)):
            period_rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **performance_metrics(frame.loc[start:end, "net_return"]),
                }
            )
    periods = pd.DataFrame(period_rows).set_index(["period", "strategy"])
    periods.to_csv(DESTINATION / "period_metrics.csv")

    annual_rows = []
    aligned = pd.concat(
        [
            baseline["net_return"].rename("baseline"),
            candidate["net_return"].rename("jump_aware"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    for year, frame in aligned.groupby(aligned.index.year):
        baseline_return = float((1.0 + frame["baseline"]).prod() - 1.0)
        candidate_return = float((1.0 + frame["jump_aware"]).prod() - 1.0)
        annual_rows.append(
            {
                "year": int(year),
                "baseline_return": baseline_return,
                "jump_aware_return": candidate_return,
                "return_delta": candidate_return - baseline_return,
            }
        )
    annual = pd.DataFrame(annual_rows).set_index("year")
    annual.to_csv(DESTINATION / "annual_comparison.csv")

    bootstrap_rows = []
    for block_length in (21, 63, 126):
        simulations = paired_circular_block_bootstrap(
            aligned["jump_aware"].to_numpy(),
            aligned["baseline"].to_numpy(),
            block_length=block_length,
            simulations=10_000,
            seed=20260723 + block_length,
        )
        bootstrap_rows.append(
            {
                "block_length": block_length,
                "median_cagr_delta": float(simulations["cagr_delta"].median()),
                "cagr_delta_2_5pct": float(
                    simulations["cagr_delta"].quantile(0.025)
                ),
                "cagr_delta_97_5pct": float(
                    simulations["cagr_delta"].quantile(0.975)
                ),
                "probability_cagr_improvement": float(
                    (simulations["cagr_delta"] > 0.0).mean()
                ),
                "probability_drawdown_no_worse": float(
                    (simulations["max_drawdown_delta"] >= 0.0).mean()
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index("block_length")
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    cagr = lambda scenario, strategy: float(metrics.loc[(scenario, strategy), "cagr"])
    mdd = lambda scenario, strategy: float(
        metrics.loc[(scenario, strategy), "max_drawdown"]
    )
    period_cagr = lambda period, strategy: float(
        periods.loc[(period, strategy), "cagr"]
    )
    acceptance = pd.Series(
        {
            "complete_cagr_non_degradation_pass": int(
                cagr("normal_2015", "jump_aware")
                >= cagr("normal_2015", "baseline")
            ),
            "development_cagr_non_degradation_pass": int(
                period_cagr("development_2015_2021", "jump_aware")
                >= period_cagr("development_2015_2021", "baseline")
            ),
            "holdout_cagr_non_degradation_pass": int(
                period_cagr("holdout_2022_2025", "jump_aware")
                >= period_cagr("holdout_2022_2025", "baseline")
            ),
            "cost15_cagr_non_degradation_pass": int(
                cagr("cost15_2015", "jump_aware")
                >= cagr("cost15_2015", "baseline")
            ),
            "extended_cagr_non_degradation_pass": int(
                cagr("extended_2012", "jump_aware")
                >= cagr("extended_2012", "baseline")
            ),
            "mdd_non_degradation_pass": int(
                mdd("normal_2015", "jump_aware")
                >= mdd("normal_2015", "baseline") - 1e-8
            ),
            "all_bootstrap_probabilities_above_50pct": int(
                (bootstrap["probability_cagr_improvement"] > 0.50).all()
            ),
        },
        name="value",
    )
    acceptance["all_directional_checks_pass"] = int(acceptance.all())
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    print("Scenario metrics:")
    print(metrics.round(6).to_string())
    print("\nPeriod metrics:")
    print(periods[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nAnnual return deltas:")
    print(annual.round(6).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string())
    print("\nAcceptance:")
    print(acceptance.to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
