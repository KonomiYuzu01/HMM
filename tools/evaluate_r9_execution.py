from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_dual_reentry_execution_delay import delayed_execution
from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/r9_first_principles_validation")
STRATEGIES = {
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
    "broad50": "experiment_r9_broad50",
    "stage35_d10": "experiment_r9_broad50_stage35_d10",
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(refresh=False)
    open_rows: list[dict[str, object]] = []
    for cost_bps in (7.5, 15.0):
        for strategy, directory in STRATEGIES.items():
            frame = simulate_open_execution(
                directory,
                opens,
                closes,
                start_date="2015-01-01",
                end_date="2025-12-31",
                cost_bps=cost_bps,
            )
            for period, (start, end) in PERIODS.items():
                open_rows.append(
                    {
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "period": period,
                        **performance_metrics(
                            frame.loc[start:end, "net_return"]
                        ),
                    }
                )
    open_metrics = pd.DataFrame(open_rows).set_index(
        ["strategy", "cost_bps", "period"]
    )
    open_metrics.to_csv(DESTINATION / "open_execution_metrics.csv")

    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    )
    asset_returns = prices.pct_change(fill_method=None)
    delay_rows: list[dict[str, object]] = []
    for delay in (0, 1, 2):
        for strategy, directory in STRATEGIES.items():
            simulated = delayed_execution(
                directory,
                asset_returns,
                delay,
            )
            for period, (start, end) in PERIODS.items():
                delay_rows.append(
                    {
                        "strategy": strategy,
                        "delay_sessions": delay,
                        "period": period,
                        **performance_metrics(simulated.loc[start:end]),
                    }
                )
    delay_metrics = pd.DataFrame(delay_rows).set_index(
        ["strategy", "delay_sessions", "period"]
    )
    delay_metrics.to_csv(DESTINATION / "execution_delay_metrics.csv")

    print("Next-open execution, complete period:")
    print(
        open_metrics.xs(
            "complete_2015_2025",
            level="period",
        )[["cagr", "sharpe", "max_drawdown"]]
        .round(6)
        .to_string()
    )
    print("\nExecution delay, complete period:")
    print(
        delay_metrics.xs(
            "complete_2015_2025",
            level="period",
        )[["cagr", "sharpe", "max_drawdown"]]
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
