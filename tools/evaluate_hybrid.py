from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_relative_momentum import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "hybrid_validation"
BASELINE = "paper_core_growth"
HYBRID = "paper_hmm_trend_hybrid"
BASELINE_COST15 = "paper_core_growth_cost15"
HYBRID_COST15 = "paper_hmm_trend_hybrid_cost15"
PERIODS = {
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def load_daily(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def constrained_reality_check(
    baseline: pd.Series,
    selected_name: str,
    maximum_drawdown: float = -0.18,
    block_days: int = 21,
    samples: int = 5_000,
    seed: int = 20_260_722,
    destination: Path = DESTINATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relative: dict[str, pd.Series] = {}
    eligibility_rows: list[dict[str, float | int | str]] = []
    for path in sorted(OUTPUT_ROOT.glob("*/daily_returns.csv")):
        name = path.parent.name
        if (
            name == BASELINE
            or "_cost15" in name
            or name.endswith("_2012")
            or name.endswith("_validation")
            or not (Path("config") / f"{name}.yaml").exists()
        ):
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if "net_return" not in frame:
            continue
        aligned = pd.concat(
            [frame["net_return"], baseline], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 0.95 * len(baseline):
            continue
        candidate_metrics = performance_metrics(aligned.iloc[:, 0])
        eligible = candidate_metrics["max_drawdown"] >= maximum_drawdown
        eligibility_rows.append(
            {
                "strategy": name,
                "eligible": int(eligible),
                "cagr": candidate_metrics["cagr"],
                "sharpe": candidate_metrics["sharpe"],
                "max_drawdown": candidate_metrics["max_drawdown"],
            }
        )
        if eligible:
            relative[name] = (
                np.log1p(frame["net_return"].reindex(baseline.index))
                - np.log1p(baseline)
            )
    eligibility = pd.DataFrame(eligibility_rows).set_index("strategy")
    matrix_frame = pd.DataFrame(relative).dropna(axis=1).dropna()
    if selected_name not in matrix_frame:
        raise RuntimeError("Hybrid is absent from the risk-feasible candidate family")
    matrix = matrix_frame.to_numpy(dtype=float)
    observed = matrix.mean(axis=0) * 252
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    count = len(centered)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    selected_index = matrix_frame.columns.get_loc(selected_name)
    selected_observed = observed[selected_index]
    maximum_statistics = np.empty(samples)
    selected_statistics = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:count] % count
        statistic = centered[indices].mean(axis=0) * 252
        maximum_statistics[sample] = statistic.max()
        selected_statistics[sample] = statistic[selected_index]
    ranking = pd.DataFrame(
        {"annualized_relative_log_return": observed}, index=matrix_frame.columns
    ).sort_values("annualized_relative_log_return", ascending=False)
    summary = pd.DataFrame(
        [
            {
                "annualized_relative_log_return": selected_observed,
                "nominal_one_sided_p_value": float(
                    (selected_statistics >= selected_observed).mean()
                ),
                "risk_feasible_familywise_p_value": float(
                    (maximum_statistics >= selected_observed).mean()
                ),
                "risk_feasible_candidate_count": matrix.shape[1],
                "rank_within_risk_feasible_family": int(
                    ranking.index.get_loc(selected_name) + 1
                ),
                "drawdown_constraint": maximum_drawdown,
                "block_days": block_days,
                "bootstrap_samples": samples,
            }
        ],
        index=[selected_name],
    )
    destination.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(destination / "candidate_family_eligibility.csv")
    return summary, ranking


def window_statistics(returns: pd.Series, start: str, end: str) -> dict[str, float]:
    sample = returns.loc[start:end]
    equity = (1.0 + sample).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_volatility": float(sample.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    names = [BASELINE, HYBRID, BASELINE_COST15, HYBRID_COST15]
    daily = {name: load_daily(name) for name in names}
    prices = pd.read_csv("data/prices_high_cagr.csv", index_col=0, parse_dates=True)
    benchmark_returns = prices[["SPX", "QQQ", "SEMIS"]].pct_change(fill_method=None)

    rows: list[dict[str, float | str]] = []
    for strategy, frame in daily.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(frame.loc[start:end, "net_return"]),
                }
            )
    for benchmark in benchmark_returns:
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": benchmark,
                    "period": period,
                    **performance_metrics(benchmark_returns.loc[start:end, benchmark]),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    baseline_complete = metrics.loc[(BASELINE, "complete_2015_2025")]
    hybrid_complete = metrics.loc[(HYBRID, "complete_2015_2025")]
    baseline_development = metrics.loc[(BASELINE, "development_2015_2021")]
    hybrid_development = metrics.loc[(HYBRID, "development_2015_2021")]
    baseline_holdout = metrics.loc[(BASELINE, "holdout_2022_2025")]
    hybrid_holdout = metrics.loc[(HYBRID, "holdout_2022_2025")]
    baseline_stress = metrics.loc[(BASELINE_COST15, "complete_2015_2025")]
    hybrid_stress = metrics.loc[(HYBRID_COST15, "complete_2015_2025")]
    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr": hybrid_complete["cagr"],
                "cagr_delta_vs_baseline": hybrid_complete["cagr"]
                - baseline_complete["cagr"],
                "complete_sharpe_delta": hybrid_complete["sharpe"]
                - baseline_complete["sharpe"],
                "complete_max_drawdown": hybrid_complete["max_drawdown"],
                "development_cagr_delta": hybrid_development["cagr"]
                - baseline_development["cagr"],
                "holdout_cagr_delta": hybrid_holdout["cagr"]
                - baseline_holdout["cagr"],
                "cost15_cagr_delta": hybrid_stress["cagr"]
                - baseline_stress["cagr"],
                "cost15_max_drawdown": hybrid_stress["max_drawdown"],
                "cagr_plus_1pp_pass": int(
                    hybrid_complete["cagr"] >= baseline_complete["cagr"] + 0.01
                ),
                "drawdown_18pct_pass": int(
                    hybrid_complete["max_drawdown"] >= -0.18
                ),
                "development_no_degradation_pass": int(
                    hybrid_development["cagr"] >= baseline_development["cagr"]
                ),
                "holdout_no_degradation_pass": int(
                    hybrid_holdout["cagr"] >= baseline_holdout["cagr"]
                ),
                "cost15_cagr_plus_1pp_pass": int(
                    hybrid_stress["cagr"] >= baseline_stress["cagr"] + 0.01
                ),
                "cost15_drawdown_18pct_pass": int(
                    hybrid_stress["max_drawdown"] >= -0.18
                ),
            }
        ],
        index=[HYBRID],
    )
    pass_columns = [column for column in acceptance if column.endswith("_pass")]
    acceptance["overall_pass"] = acceptance[pass_columns].all(axis=1).astype(int)
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    sample = slice("2015-01-01", "2025-12-31")
    bootstrap = pd.DataFrame(
        [
            circular_block_bootstrap(
                daily[HYBRID].loc[sample, "net_return"],
                daily[BASELINE].loc[sample, "net_return"],
            )
        ],
        index=[HYBRID],
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    reality, ranking = constrained_reality_check(
        daily[BASELINE].loc[sample, "net_return"], HYBRID
    )
    reality.to_csv(DESTINATION / "multiple_testing_reality_check.csv")
    ranking.to_csv(DESTINATION / "risk_feasible_candidate_ranking.csv")

    annual = pd.concat(
        {
            "baseline": daily[BASELINE].loc[sample, "net_return"],
            "hybrid": daily[HYBRID].loc[sample, "net_return"],
            "baseline_cost15": daily[BASELINE_COST15].loc[sample, "net_return"],
            "hybrid_cost15": daily[HYBRID_COST15].loc[sample, "net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["hybrid_minus_baseline"] = annual["hybrid"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    crisis_periods = {
        "china_brexit_2015_2016": ("2015-05-28", "2016-06-27"),
        "q4_2018": ("2018-10-01", "2018-12-31"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "inflation_bear_2022": ("2022-01-03", "2022-12-30"),
        "tariff_selloff_2025": ("2025-02-19", "2025-04-08"),
    }
    crisis_rows: list[dict[str, float | str]] = []
    for window, (start, end) in crisis_periods.items():
        for strategy in [BASELINE, HYBRID]:
            crisis_rows.append(
                {
                    "window": window,
                    "strategy": strategy,
                    **window_statistics(daily[strategy]["net_return"], start, end),
                }
            )
    crisis = pd.DataFrame(crisis_rows).set_index(["window", "strategy"])
    crisis.to_csv(DESTINATION / "crisis_windows.csv")

    component = daily[HYBRID].loc[sample]
    diagnostics = pd.DataFrame(
        [
            {
                "hmm_trend_daily_return_correlation": float(
                    component["hmm_return"].corr(component["trend_return"])
                ),
                "annual_wins_vs_baseline": int(
                    (annual["hybrid_minus_baseline"] > 0.0).sum()
                ),
                "annual_periods": len(annual),
                "annualized_total_cost": float(component["cost"].mean() * 252),
                "annualized_outer_cost": float(
                    component["outer_cost"].mean() * 252
                ),
                "mean_hmm_sleeve_weight": float(
                    pd.read_csv(
                        OUTPUT_ROOT / HYBRID / "sleeve_weights.csv",
                        index_col=0,
                        parse_dates=True,
                    ).loc[sample, "HMM"].mean()
                ),
            }
        ],
        index=[HYBRID],
    )
    diagnostics.to_csv(DESTINATION / "diagnostics.csv")

    print(metrics.round(4).to_string())
    print("\nAcceptance:")
    print(acceptance.round(4).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(4).to_string())
    print("\nRisk-feasible Reality Check:")
    print(reality.round(4).to_string())
    print("\nDiagnostics:")
    print(diagnostics.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
