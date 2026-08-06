from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_drawdown_uncertainty import (
    paired_circular_block_bootstrap,
    strategy_summary,
)
from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/risk_capacity_frontier")
STRATEGIES = {
    0.14: "paper_core_growth_gold20_daily_risk_vol14_ensemble",
    0.16: "paper_core_growth_gold20_daily_risk_vol16_ensemble",
    0.18: "paper_core_growth_gold20_daily_risk_vol18_ensemble",
    0.20: "paper_core_growth_gold20_daily_risk_ensemble",
    0.22: "paper_core_growth_gold20_daily_risk_vol22_ensemble",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(False)
    strategy_returns = {}
    metric_rows = []
    for target, directory in STRATEGIES.items():
        frame = simulate_open_execution(directory, opens, closes, cost_bps=7.5)
        strategy_returns[target] = frame["net_return"]
        metric_rows.append(
            {
                "growth_target_volatility": target,
                **performance_metrics(frame["net_return"]),
                "annualized_open_turnover": float(
                    frame["open_turnover"].mean() * 252.0
                ),
                "annualized_cost": float(frame["cost"].mean() * 252.0),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("growth_target_volatility")
    metrics.to_csv(DESTINATION / "frontier_metrics.csv")

    spy = closes["SPX"].pct_change(fill_method=None).loc["2015":"2025"]
    tail_rows = []
    for target, returns in strategy_returns.items():
        aligned = pd.concat(
            [returns.rename("candidate"), spy.rename("baseline")],
            axis=1,
            join="inner",
        ).dropna()
        for block_length in (21, 63, 126):
            simulations = paired_circular_block_bootstrap(
                aligned["candidate"].to_numpy(),
                aligned["baseline"].to_numpy(),
                block_length=block_length,
                simulations=5_000,
                seed=20260723 + block_length,
            )
            summary = strategy_summary(simulations, block_length, "candidate")
            tail_rows.append(
                {
                    "growth_target_volatility": target,
                    "block_length": block_length,
                    "probability_cagr_above_spy": float(
                        (simulations["cagr_delta"] > 0.0).mean()
                    ),
                    **summary,
                }
            )
    tails = pd.DataFrame(tail_rows).set_index(
        ["growth_target_volatility", "block_length"]
    )
    tails.to_csv(DESTINATION / "bootstrap_tail_summary.csv")

    decision_rows = []
    for target in STRATEGIES:
        target_tails = tails.loc[target]
        decision_rows.append(
            {
                "growth_target_volatility": target,
                "historical_cagr": float(metrics.loc[target, "cagr"]),
                "historical_max_drawdown": float(
                    metrics.loc[target, "max_drawdown"]
                ),
                "worst_block_probability_breach_18pct": float(
                    target_tails["probability_breach_18pct"].max()
                ),
                "best_block_probability_breach_18pct": float(
                    target_tails["probability_breach_18pct"].min()
                ),
                "all_blocks_breach_probability_below_25pct": int(
                    (target_tails["probability_breach_18pct"] < 0.25).all()
                ),
                "all_blocks_probability_outperform_spy_above_50pct": int(
                    (target_tails["probability_cagr_above_spy"] > 0.50).all()
                ),
            }
        )
    decision = pd.DataFrame(decision_rows).set_index(
        "growth_target_volatility"
    )
    decision["joint_risk_and_return_pass"] = decision[
        [
            "all_blocks_breach_probability_below_25pct",
            "all_blocks_probability_outperform_spy_above_50pct",
        ]
    ].all(axis=1).astype(int)
    decision.to_csv(DESTINATION / "decision.csv")

    print("Historical frontier:")
    print(
        metrics[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "annualized_open_turnover",
                "annualized_cost",
            ]
        ]
        .round(6)
        .to_string()
    )
    print("\nBootstrap risk/return decision:")
    print(decision.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
