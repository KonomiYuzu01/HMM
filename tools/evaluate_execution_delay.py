from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "execution_delay_validation"
STRATEGIES = {
    "baseline": "paper_core_robust_vol_guarded_floor_ensemble",
    "growth_candidate": "paper_core_zero_entry_growth_reallocation_ensemble",
    "gold_leverage_candidate": "paper_core_zero_entry_growth_gold20_lev110_ensemble",
}


def simulate_delayed_targets(
    strategy: str,
    asset_returns: pd.DataFrame,
    delay_sessions: int,
) -> pd.Series:
    weights = pd.read_csv(
        OUTPUT_ROOT / strategy / "weights.csv", index_col=0, parse_dates=True
    )
    daily = pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv", index_col=0, parse_dates=True
    )
    aligned_returns = asset_returns.reindex(weights.index)[weights.columns]
    events: dict[pd.Timestamp, np.ndarray] = {}
    costs: dict[pd.Timestamp, float] = {}
    for position, date in enumerate(weights.index):
        if position > 0 and daily.loc[date, "turnover"] <= 1e-14:
            continue
        delayed_position = position + delay_sessions
        if delayed_position >= len(weights):
            continue
        delayed_date = weights.index[delayed_position]
        events[delayed_date] = weights.loc[date].to_numpy(dtype=float)
        costs[delayed_date] = float(daily.loc[date, "cost"])

    current = np.zeros(len(weights.columns), dtype=float)
    current[weights.columns.get_loc("CASH")] = 1.0
    simulated_returns: list[float] = []
    for date in weights.index:
        trading_cost = 0.0
        if date in events:
            current = events[date].copy()
            trading_cost = costs[date]
        day_returns = aligned_returns.loc[date].to_numpy(dtype=float)
        gross_return = float(current @ day_returns)
        simulated_returns.append(gross_return - trading_cost)
        current = current * (1.0 + day_returns) / (1.0 + gross_return)
    return pd.Series(simulated_returns, index=weights.index, name=strategy)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_high_cagr.csv", index_col=0, parse_dates=True
    )
    asset_returns = prices.pct_change(fill_method=None)
    rows: list[dict[str, float | int | str]] = []
    simulations: dict[tuple[str, int], pd.Series] = {}
    for label, strategy in STRATEGIES.items():
        for delay in (0, 1, 2):
            simulated = simulate_delayed_targets(strategy, asset_returns, delay)
            simulations[(label, delay)] = simulated
            for period, start, end in (
                ("complete_2015_2025", "2015-01-01", "2025-12-31"),
                ("full_2015_present", "2015-01-01", None),
            ):
                rows.append(
                    {
                        "strategy": label,
                        "delay_sessions": delay,
                        "period": period,
                        **performance_metrics(simulated.loc[start:end]),
                    }
                )
    metrics = pd.DataFrame(rows).set_index(
        ["strategy", "delay_sessions", "period"]
    )
    metrics.to_csv(DESTINATION / "metrics.csv")

    relative_rows = []
    for strategy in STRATEGIES:
        if strategy == "baseline":
            continue
        for delay in (0, 1, 2):
            for period, start, end in (
                ("complete_2015_2025", "2015-01-01", "2025-12-31"),
                ("full_2015_present", "2015-01-01", None),
            ):
                candidate = simulations[(strategy, delay)].loc[start:end]
                baseline = simulations[("baseline", delay)].loc[start:end]
                candidate_metrics = performance_metrics(candidate)
                baseline_metrics = performance_metrics(baseline)
                relative_rows.append(
                    {
                        "strategy": strategy,
                        "delay_sessions": delay,
                        "period": period,
                        "candidate_cagr_delta": candidate_metrics["cagr"]
                        - baseline_metrics["cagr"],
                        "candidate_sharpe_delta": candidate_metrics["sharpe"]
                        - baseline_metrics["sharpe"],
                        "candidate_max_drawdown_delta": candidate_metrics[
                            "max_drawdown"
                        ]
                        - baseline_metrics["max_drawdown"],
                    }
                )
    relative = pd.DataFrame(relative_rows).set_index(
        ["strategy", "delay_sessions", "period"]
    )
    relative.to_csv(DESTINATION / "relative_metrics.csv")
    print(metrics.round(6).to_string())
    print("\nCandidate minus baseline:")
    print(relative.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
