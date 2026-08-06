from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = (
    OUTPUT
    / "regime_strategy_research_2026-07-25"
    / "r8_long_history_robustness"
)
STRATEGIES = {
    "production": "past_20y_drawdown_backtest_netted",
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble_20y_proxy"
    ),
}
COST15_STRATEGIES = {
    "production": "past_20y_drawdown_backtest_netted_cost15",
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble_20y_proxy_cost15"
    ),
}
PERIODS = {
    "global_financial_crisis": ("2008-01-01", "2009-12-31"),
    "euro_area_stress_2011": ("2011-05-01", "2011-12-31"),
    "china_growth_scare_2015_2016": ("2015-06-01", "2016-03-31"),
    "volmageddon_and_q4_2018": ("2018-02-01", "2018-12-31"),
    "covid_2020": ("2020-01-01", "2020-12-31"),
    "inflation_tightening_2022": ("2022-01-01", "2022-12-31"),
}


def load_returns(directory: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]


def fixed_metrics(
    returns: dict[str, pd.Series],
    scenario: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy, series in returns.items():
        values = performance_metrics(
            series.loc["2006-08-01":"2025-12-31"]
        )
        rows.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                **values,
            }
        )
    return rows


def rolling_relative(
    production: pd.Series,
    candidate: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat(
        [production.rename("production"), candidate.rename("r8")],
        axis=1,
    ).dropna().loc["2006-08-01":"2025-12-31"]
    relative_log = np.log1p(aligned["r8"]) - np.log1p(
        aligned["production"]
    )
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for years, sessions in ((1, 252), (3, 756), (5, 1260)):
        annualized_log = relative_log.rolling(sessions).sum() / years
        annualized = np.expm1(annualized_log).dropna()
        frame = annualized.rename("annualized_relative_return").to_frame()
        frame["window_years"] = years
        rows.append(frame)
        worst_date = annualized.idxmin()
        summaries.append(
            {
                "window_years": years,
                "windows": len(annualized),
                "outperformance_share": float(annualized.gt(0.0).mean()),
                "median_annualized_relative_return": float(
                    annualized.median()
                ),
                "p10_annualized_relative_return": float(
                    annualized.quantile(0.10)
                ),
                "worst_annualized_relative_return": float(
                    annualized.loc[worst_date]
                ),
                "worst_window_end": worst_date.date().isoformat(),
                "best_annualized_relative_return": float(annualized.max()),
            }
        )
    detail = pd.concat(rows).set_index("window_years", append=True)
    summary = pd.DataFrame(summaries).set_index("window_years")
    return detail, summary


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    normal = {
        strategy: load_returns(directory)
        for strategy, directory in STRATEGIES.items()
    }
    cost15 = {
        strategy: load_returns(directory)
        for strategy, directory in COST15_STRATEGIES.items()
    }
    metrics = pd.DataFrame(
        [
            *fixed_metrics(normal, "normal"),
            *fixed_metrics(cost15, "cost15"),
        ]
    ).set_index(["scenario", "strategy"])
    metrics.to_csv(DESTINATION / "fixed_metrics.csv")

    rolling_detail, rolling_summary = rolling_relative(
        normal["production"],
        normal["r8"],
    )
    rolling_detail.to_csv(DESTINATION / "rolling_relative_returns.csv")
    rolling_summary.to_csv(DESTINATION / "rolling_summary.csv")

    episode_rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        for strategy, returns in normal.items():
            episode_rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    episodes = pd.DataFrame(episode_rows).set_index(["period", "strategy"])
    episodes.to_csv(DESTINATION / "stress_period_metrics.csv")

    annual = pd.DataFrame(
        {
            strategy: (1.0 + returns).groupby(returns.index.year).prod() - 1.0
            for strategy, returns in normal.items()
        }
    ).loc[2007:2025]
    annual["r8_minus_production"] = annual["r8"] - annual["production"]
    annual.to_csv(DESTINATION / "annual_returns.csv")

    complete = metrics.xs("normal").copy()
    complete["cagr_delta_vs_production"] = (
        complete["cagr"] - complete.loc["production", "cagr"]
    )
    cost = metrics.xs("cost15").copy()
    cost["cagr_delta_vs_production"] = (
        cost["cagr"] - cost.loc["production", "cagr"]
    )
    print("Fixed 2006-08 through 2025:")
    print(
        complete[
            ["cagr", "sharpe", "max_drawdown", "cagr_delta_vs_production"]
        ].round(6).to_string()
    )
    print("\n15 bps:")
    print(
        cost[
            ["cagr", "sharpe", "max_drawdown", "cagr_delta_vs_production"]
        ].round(6).to_string()
    )
    print("\nRolling relative summary:")
    print(rolling_summary.round(6).to_string())
    print("\nStress-period CAGR deltas:")
    stress = episodes["cagr"].unstack()
    stress["r8_minus_production"] = stress["r8"] - stress["production"]
    print(stress.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
