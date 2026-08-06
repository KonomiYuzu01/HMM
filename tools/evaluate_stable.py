from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_hybrid import constrained_reality_check, window_statistics
from evaluate_relative_momentum import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "stable_validation"
BASELINE = "paper_core_growth"
CANDIDATE = "paper_core_stable"
BASELINE_COST15 = "paper_core_growth_cost15"
CANDIDATE_COST15 = "paper_core_stable_cost15"
BASELINE_2012 = "paper_core_growth_2012"
CANDIDATE_2012 = "paper_core_stable_2012"
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


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    names = [
        BASELINE,
        CANDIDATE,
        BASELINE_COST15,
        CANDIDATE_COST15,
        BASELINE_2012,
        CANDIDATE_2012,
    ]
    daily = {name: load_daily(name) for name in names}
    prices = pd.read_csv("data/prices_high_cagr.csv", index_col=0, parse_dates=True)
    benchmarks = prices[["SPX", "QQQ", "SEMIS"]].pct_change(fill_method=None)

    rows: list[dict[str, float | str]] = []
    for strategy in names[:4]:
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(daily[strategy].loc[start:end, "net_return"]),
                }
            )
    for benchmark in benchmarks:
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": benchmark,
                    "period": period,
                    **performance_metrics(benchmarks.loc[start:end, benchmark]),
                }
            )
    for strategy in [BASELINE_2012, CANDIDATE_2012]:
        rows.append(
            {
                "strategy": strategy,
                "period": "extended_2012_2025",
                **performance_metrics(daily[strategy].loc[:"2025", "net_return"]),
            }
        )
    metrics = pd.DataFrame(rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    baseline_complete = metrics.loc[(BASELINE, "complete_2015_2025")]
    candidate_complete = metrics.loc[(CANDIDATE, "complete_2015_2025")]
    baseline_development = metrics.loc[(BASELINE, "development_2015_2021")]
    candidate_development = metrics.loc[(CANDIDATE, "development_2015_2021")]
    baseline_holdout = metrics.loc[(BASELINE, "holdout_2022_2025")]
    candidate_holdout = metrics.loc[(CANDIDATE, "holdout_2022_2025")]
    baseline_stress = metrics.loc[(BASELINE_COST15, "complete_2015_2025")]
    candidate_stress = metrics.loc[(CANDIDATE_COST15, "complete_2015_2025")]
    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr": candidate_complete["cagr"],
                "cagr_delta_vs_baseline": candidate_complete["cagr"]
                - baseline_complete["cagr"],
                "complete_sharpe_delta": candidate_complete["sharpe"]
                - baseline_complete["sharpe"],
                "complete_max_drawdown": candidate_complete["max_drawdown"],
                "development_cagr_delta": candidate_development["cagr"]
                - baseline_development["cagr"],
                "holdout_cagr_delta": candidate_holdout["cagr"]
                - baseline_holdout["cagr"],
                "cost15_cagr": candidate_stress["cagr"],
                "cost15_cagr_delta_vs_baseline": candidate_stress["cagr"]
                - baseline_stress["cagr"],
                "cost15_max_drawdown": candidate_stress["max_drawdown"],
                "cagr_plus_1pp_pass": int(
                    candidate_complete["cagr"] >= baseline_complete["cagr"] + 0.01
                ),
                "drawdown_18pct_pass": int(
                    candidate_complete["max_drawdown"] >= -0.18
                ),
                "development_no_degradation_pass": int(
                    candidate_development["cagr"] >= baseline_development["cagr"]
                ),
                "holdout_no_degradation_pass": int(
                    candidate_holdout["cagr"] >= baseline_holdout["cagr"]
                ),
                "cost15_cagr_plus_1pp_pass": int(
                    candidate_stress["cagr"] >= baseline_stress["cagr"] + 0.01
                ),
                "cost15_drawdown_18pct_pass": int(
                    candidate_stress["max_drawdown"] >= -0.18
                ),
            }
        ],
        index=[CANDIDATE],
    )
    normal_columns = [
        "cagr_plus_1pp_pass",
        "drawdown_18pct_pass",
        "development_no_degradation_pass",
        "holdout_no_degradation_pass",
    ]
    acceptance["normal_cost_overall_pass"] = (
        acceptance[normal_columns].all(axis=1).astype(int)
    )
    acceptance["strict_including_cost15_pass"] = (
        acceptance[[*normal_columns, "cost15_cagr_plus_1pp_pass", "cost15_drawdown_18pct_pass"]]
        .all(axis=1)
        .astype(int)
    )
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    sample = slice("2015-01-01", "2025-12-31")
    bootstrap = pd.DataFrame(
        {
            "2015_start_vs_baseline": circular_block_bootstrap(
                daily[CANDIDATE].loc[sample, "net_return"],
                daily[BASELINE].loc[sample, "net_return"],
            ),
            "2012_start_vs_baseline": circular_block_bootstrap(
                daily[CANDIDATE_2012].loc[:"2025", "net_return"],
                daily[BASELINE_2012].loc[:"2025", "net_return"],
            ),
        }
    ).T
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    reality, ranking = constrained_reality_check(
        daily[BASELINE].loc[sample, "net_return"],
        CANDIDATE,
        destination=DESTINATION,
    )
    reality.to_csv(DESTINATION / "multiple_testing_reality_check.csv")
    ranking.to_csv(DESTINATION / "risk_feasible_candidate_ranking.csv")

    annual = pd.concat(
        {
            "baseline": daily[BASELINE].loc[sample, "net_return"],
            "stable": daily[CANDIDATE].loc[sample, "net_return"],
            "baseline_cost15": daily[BASELINE_COST15].loc[sample, "net_return"],
            "stable_cost15": daily[CANDIDATE_COST15].loc[sample, "net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["stable_minus_baseline"] = annual["stable"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    common = pd.concat(
        [
            daily[CANDIDATE]["net_return"].rename("start_2015"),
            daily[CANDIDATE_2012].loc["2015":, "net_return"].rename("start_2012"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    regimes_2015 = pd.read_csv(
        OUTPUT_ROOT / CANDIDATE / "regimes.csv", index_col=0, parse_dates=True
    )
    regimes_2012 = pd.read_csv(
        OUTPUT_ROOT / CANDIDATE_2012 / "regimes.csv", index_col=0, parse_dates=True
    ).loc["2015":]
    aligned_regimes = pd.concat(
        [
            regimes_2015["risk_on_leverage"].rename("start_2015"),
            regimes_2012["risk_on_leverage"].rename("start_2012"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    start_stability = pd.DataFrame(
        [
            {
                "daily_return_correlation": float(common.corr().iloc[0, 1]),
                "identical_daily_return_share": float(
                    np.isclose(common["start_2015"], common["start_2012"], atol=1e-12).mean()
                ),
                "post_april_2015_identical_share": float(
                    np.isclose(
                        common.loc["2015-04":, "start_2015"],
                        common.loc["2015-04":, "start_2012"],
                        atol=1e-12,
                    ).mean()
                ),
                "risk_on_decision_agreement": float(
                    (aligned_regimes["start_2015"] == aligned_regimes["start_2012"]).mean()
                ),
                "start_2015_common_cagr": performance_metrics(common["start_2015"])[
                    "cagr"
                ],
                "start_2012_common_cagr": performance_metrics(common["start_2012"])[
                    "cagr"
                ],
            }
        ],
        index=[CANDIDATE],
    )
    start_stability.to_csv(DESTINATION / "start_date_stability.csv")

    crisis_periods = {
        "china_brexit_2015_2016": ("2015-05-28", "2016-06-27"),
        "q4_2018": ("2018-10-01", "2018-12-31"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "inflation_bear_2022": ("2022-01-03", "2022-12-30"),
        "tariff_selloff_2025": ("2025-02-19", "2025-04-08"),
    }
    crisis_rows: list[dict[str, float | str]] = []
    for window, (start, end) in crisis_periods.items():
        for strategy in [BASELINE, CANDIDATE]:
            crisis_rows.append(
                {
                    "window": window,
                    "strategy": strategy,
                    **window_statistics(daily[strategy]["net_return"], start, end),
                }
            )
    pd.DataFrame(crisis_rows).set_index(["window", "strategy"]).to_csv(
        DESTINATION / "crisis_windows.csv"
    )

    diagnostics = pd.DataFrame(
        [
            {
                "annual_wins_vs_baseline": int(
                    (annual["stable_minus_baseline"] > 0.0).sum()
                ),
                "annual_periods": len(annual),
                "annualized_cost": float(
                    daily[CANDIDATE].loc[sample, "cost"].mean() * 252
                ),
                "cost15_annualized_cost": float(
                    daily[CANDIDATE_COST15].loc[sample, "cost"].mean() * 252
                ),
            }
        ],
        index=[CANDIDATE],
    )
    diagnostics.to_csv(DESTINATION / "diagnostics.csv")

    print(metrics.round(4).to_string())
    print("\nAcceptance:")
    print(acceptance.round(4).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(4).to_string())
    print("\nRisk-feasible Reality Check:")
    print(reality.round(4).to_string())
    print("\nStart-date stability:")
    print(start_stability.round(4).to_string())
    print("\nDiagnostics:")
    print(diagnostics.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
