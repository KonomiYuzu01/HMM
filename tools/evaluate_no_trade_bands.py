from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/no_trade_band_validation")
STRATEGIES = {
    "band_1pct": "paper_core_growth_gold20_jump_aware_daily_risk_ensemble",
    "band_2pct": "paper_core_growth_gold20_jump_daily_risk_notrade2_ensemble",
    "band_3pct": "paper_core_growth_gold20_jump_daily_risk_notrade3_ensemble",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(False)
    rows = []
    for strategy, directory in STRATEGIES.items():
        for cost_bps in (7.5, 15.0):
            frame = simulate_open_execution(
                directory,
                opens,
                closes,
                cost_bps=cost_bps,
            )
            rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost_bps,
                    **performance_metrics(frame["net_return"]),
                    "annualized_open_turnover": float(
                        frame["open_turnover"].mean() * 252.0
                    ),
                    "annualized_cost": float(frame["cost"].mean() * 252.0),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["strategy", "cost_bps"])
    metrics.to_csv(DESTINATION / "metrics.csv")

    baseline = metrics.loc[("band_1pct", 7.5)]
    comparison_rows = []
    for strategy in ("band_2pct", "band_3pct"):
        candidate = metrics.loc[(strategy, 7.5)]
        comparison_rows.append(
            {
                "strategy": strategy,
                "cagr_delta": float(candidate["cagr"] - baseline["cagr"]),
                "sharpe_delta": float(candidate["sharpe"] - baseline["sharpe"]),
                "max_drawdown_delta": float(
                    candidate["max_drawdown"] - baseline["max_drawdown"]
                ),
                "turnover_reduction": float(
                    1.0
                    - candidate["annualized_open_turnover"]
                    / baseline["annualized_open_turnover"]
                ),
                "material_cagr_improvement_10bp": int(
                    candidate["cagr"] - baseline["cagr"] >= 0.001
                ),
                "material_turnover_reduction_10pct": int(
                    1.0
                    - candidate["annualized_open_turnover"]
                    / baseline["annualized_open_turnover"]
                    >= 0.10
                ),
                "drawdown_non_degradation_20bp": int(
                    candidate["max_drawdown"]
                    >= baseline["max_drawdown"] - 0.002
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows).set_index("strategy")
    comparison["all_material_checks_pass"] = comparison[
        [
            "material_cagr_improvement_10bp",
            "material_turnover_reduction_10pct",
            "drawdown_non_degradation_20bp",
        ]
    ].all(axis=1).astype(int)
    comparison.to_csv(DESTINATION / "comparison.csv")

    print("Metrics:")
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
    print("\nComparison versus 1% band:")
    print(comparison.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
