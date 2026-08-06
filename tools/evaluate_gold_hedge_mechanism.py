from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


ROOT = Path("output")
DESTINATION = ROOT / "gold_hedge_mechanism_validation"
STRATEGIES = {
    "production_primary": "paper_core_zero_entry_growth_reallocation_ensemble",
    "gold_20": "paper_core_zero_entry_growth_gold20_ensemble",
}
WINDOWS = {
    "china_growth_scare": ("2015-08-01", "2016-02-29"),
    "q4_2018": ("2018-10-01", "2018-12-31"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "inflation_rate_shock_2022": ("2022-01-01", "2022-12-31"),
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_high_cagr.csv", index_col=0, parse_dates=True
    ).loc[:"2025"]
    returns = prices[["QQQ", "SEMIS", "GOLD"]].pct_change(fill_method=None).dropna()
    growth_equal = returns[["QQQ", "SEMIS"]].mean(axis=1)
    rolling = pd.DataFrame(
        {
            "correlation_60d": growth_equal.rolling(60).corr(returns["GOLD"]),
            "correlation_252d": growth_equal.rolling(252).corr(returns["GOLD"]),
        }
    ).dropna()
    correlation_summary = rolling.agg(["min", "median", "max"])
    correlation_summary.loc["positive_fraction"] = (rolling > 0.0).mean()
    correlation_summary.to_csv(DESTINATION / "rolling_correlation_summary.csv")

    tail_threshold = growth_equal.quantile(0.05)
    tail = pd.DataFrame(
        {
            "growth_equal_return": growth_equal,
            "gold_return": returns["GOLD"],
        }
    ).loc[growth_equal <= tail_threshold]
    tail_summary = pd.DataFrame(
        {
            "observations": [len(tail)],
            "growth_equal_mean": [tail["growth_equal_return"].mean()],
            "gold_mean": [tail["gold_return"].mean()],
            "gold_positive_fraction": [(tail["gold_return"] > 0.0).mean()],
            "gold_5pct": [tail["gold_return"].quantile(0.05)],
        },
        index=["worst_5pct_growth_days"],
    )
    tail_summary.to_csv(DESTINATION / "growth_tail_gold_behavior.csv")

    daily = {
        label: pd.read_csv(
            ROOT / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )["net_return"]
        for label, directory in STRATEGIES.items()
    }
    window_rows = []
    for window, (start, end) in WINDOWS.items():
        for strategy, strategy_returns in daily.items():
            window_rows.append(
                {
                    "window": window,
                    "strategy": strategy,
                    **performance_metrics(strategy_returns.loc[start:end]),
                    "total_return": float(
                        (1.0 + strategy_returns.loc[start:end]).prod() - 1.0
                    ),
                }
            )
    windows = pd.DataFrame(window_rows).set_index(["window", "strategy"])
    windows.to_csv(DESTINATION / "crisis_windows.csv")
    print("Rolling growth/GLD correlation:")
    print(correlation_summary.round(6).to_string())
    print("\nGLD on the worst 5% growth days:")
    print(tail_summary.round(6).to_string())
    print("\nCrisis windows:")
    print(windows[["total_return", "max_drawdown"]].round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
