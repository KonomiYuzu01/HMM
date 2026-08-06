from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


ROOT = Path("output")
DESTINATION = ROOT / "structural_diversification_validation"
STRATEGIES = {
    "production_primary": "paper_core_zero_entry_growth_reallocation_ensemble",
    "gold_10": "paper_core_zero_entry_growth_gold10_ensemble",
    "gold_15": "paper_core_zero_entry_growth_gold15_ensemble",
    "gold_20": "paper_core_zero_entry_growth_gold20_ensemble",
    "gold_20_lev105": "paper_core_zero_entry_growth_gold20_lev105_ensemble",
    "gold_20_lev110": "paper_core_zero_entry_growth_gold20_lev110_ensemble",
    "gold_20_lev110_accountcap": "paper_core_zero_entry_growth_gold20_lev110_accountcap_ensemble",
    "bond_15": "paper_core_zero_entry_growth_bond15_ensemble",
    "gold_bond_15": "paper_core_zero_entry_growth_goldbond15_ensemble",
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}
STRESS_STRATEGIES = {
    "production_primary_cost15": "paper_core_zero_entry_growth_reallocation_ensemble_cost15",
    "gold_15_cost15": "paper_core_zero_entry_growth_gold15_ensemble_cost15",
    "gold_20_cost15": "paper_core_zero_entry_growth_gold20_ensemble_cost15",
    "gold_20_lev110_cost15": "paper_core_zero_entry_growth_gold20_lev110_ensemble_cost15",
}
EXTENDED_STRATEGIES = {
    "production_primary_2012": "paper_core_zero_entry_growth_reallocation_ensemble_2012",
    "gold_15_2012": "paper_core_zero_entry_growth_gold15_ensemble_2012",
    "gold_20_2012": "paper_core_zero_entry_growth_gold20_ensemble_2012",
    "gold_20_lev110_2012": "paper_core_zero_entry_growth_gold20_lev110_ensemble_2012",
}


def relative_annual_return(candidate: pd.Series, reference: pd.Series) -> float:
    aligned = pd.concat([candidate, reference], axis=1, join="inner").dropna()
    years = len(aligned) / 252.0
    return float(
        (
            (1.0 + aligned.iloc[:, 0]).prod()
            / (1.0 + aligned.iloc[:, 1]).prod()
        )
        ** (1.0 / years)
        - 1.0
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {
        label: pd.read_csv(
            ROOT / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )["net_return"]
        for label, directory in STRATEGIES.items()
    }
    metric_rows = []
    for strategy, returns in daily.items():
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    reference_metrics = metrics.loc["production_primary"]
    relative_rows = []
    for strategy in STRATEGIES:
        if strategy == "production_primary":
            continue
        for period in PERIODS:
            candidate = metrics.loc[(strategy, period)]
            reference = reference_metrics.loc[period]
            relative_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "cagr_delta": candidate["cagr"] - reference["cagr"],
                    "sharpe_delta": candidate["sharpe"] - reference["sharpe"],
                    "max_drawdown_delta": candidate["max_drawdown"]
                    - reference["max_drawdown"],
                }
            )
    relative = pd.DataFrame(relative_rows).set_index(["strategy", "period"])
    relative.to_csv(DESTINATION / "relative_metrics.csv")

    annual_rows = []
    reference = daily["production_primary"].loc["2015":"2025"]
    for strategy in STRATEGIES:
        if strategy == "production_primary":
            continue
        candidate = daily[strategy].reindex(reference.index)
        for year in sorted(reference.index.year.unique()):
            mask = reference.index.year == year
            annual_rows.append(
                {
                    "strategy": strategy,
                    "year": year,
                    "relative_annual_return": relative_annual_return(
                        candidate.loc[mask], reference.loc[mask]
                    ),
                }
            )
    annual = pd.DataFrame(annual_rows).set_index(["strategy", "year"])
    annual.to_csv(DESTINATION / "annual_relative_return.csv")

    leave_rows = []
    for strategy in STRATEGIES:
        if strategy == "production_primary":
            continue
        candidate = daily[strategy].reindex(reference.index)
        for omitted_year in sorted(reference.index.year.unique()):
            keep = reference.index.year != omitted_year
            leave_rows.append(
                {
                    "strategy": strategy,
                    "omitted_year": omitted_year,
                    "relative_annualized_return": relative_annual_return(
                        candidate.loc[keep], reference.loc[keep]
                    ),
                }
            )
    leave = pd.DataFrame(leave_rows).set_index(["strategy", "omitted_year"])
    leave.to_csv(DESTINATION / "leave_one_year_out.csv")

    stress_rows = []
    for strategy, directory in STRESS_STRATEGIES.items():
        returns = pd.read_csv(
            ROOT / directory / "daily_returns.csv", index_col=0, parse_dates=True
        ).loc["2015":"2025", "net_return"]
        stress_rows.append({"strategy": strategy, **performance_metrics(returns)})
    stress = pd.DataFrame(stress_rows).set_index("strategy")
    stress.to_csv(DESTINATION / "cost15_complete_metrics.csv")

    extended_rows = []
    for strategy, directory in EXTENDED_STRATEGIES.items():
        returns = pd.read_csv(
            ROOT / directory / "daily_returns.csv", index_col=0, parse_dates=True
        ).loc[:"2025", "net_return"]
        extended_rows.append(
            {"strategy": strategy, **performance_metrics(returns)}
        )
    extended = pd.DataFrame(extended_rows).set_index("strategy")
    extended.to_csv(DESTINATION / "extended_2012_2025_metrics.csv")

    prices = pd.read_csv(
        "data/prices_high_cagr.csv", index_col=0, parse_dates=True
    )
    benchmark_returns = prices[["SPX", "QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    benchmark_rows = []
    for asset in benchmark_returns:
        for period, (start, end) in PERIODS.items():
            if period == "full_2015_present":
                continue
            benchmark_rows.append(
                {
                    "benchmark": asset,
                    "period": period,
                    **performance_metrics(benchmark_returns.loc[start:end, asset]),
                }
            )
    benchmarks = pd.DataFrame(benchmark_rows).set_index(["benchmark", "period"])
    benchmarks.to_csv(DESTINATION / "benchmark_metrics.csv")

    print(metrics[["cagr", "sharpe", "max_drawdown", "calmar"]].round(6))
    print("\nCandidate minus production primary:")
    print(relative.round(6))
    print("\nLeave-one-year-out relative-return ranges:")
    print(
        leave.groupby(level="strategy")["relative_annualized_return"]
        .agg(["min", "median", "max"])
        .round(6)
    )
    print("\n15 bps cost, complete 2015-2025:")
    print(stress[["cagr", "sharpe", "max_drawdown"]].round(6))
    print("\nExtended 2012-2025:")
    print(extended[["cagr", "sharpe", "max_drawdown"]].round(6))
    print("\nUnmanaged benchmarks:")
    print(benchmarks[["cagr", "sharpe", "max_drawdown"]].round(6))
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
