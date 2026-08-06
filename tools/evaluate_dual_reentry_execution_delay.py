from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
STRATEGIES = {
    "production": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "inverse_gate": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble"
    ),
    "dual_reentry": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble"
    ),
    "hysteresis19": (
        "paper_core_growth_gold20_dual_reentry_hysteresis19_inverse_"
        "momentum_netted_ensemble"
    ),
    "goodvol": (
        "paper_core_growth_gold20_dual_reentry_goodvol_inverse_momentum_"
        "netted_ensemble"
    ),
    "goodvol_bear20": (
        "paper_core_growth_gold20_dual_reentry_goodvol_bear20_inverse_"
        "momentum_netted_ensemble"
    ),
}


def delayed_execution(
    directory: str,
    asset_returns: pd.DataFrame,
    delay_sessions: int,
    cost_bps: float = 7.5,
) -> pd.Series:
    weights = pd.read_csv(
        OUTPUT / directory / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    returns = asset_returns.reindex(weights.index)[weights.columns]
    events: dict[pd.Timestamp, np.ndarray] = {}
    for position, date in enumerate(weights.index):
        if position > 0 and float(daily.loc[date, "turnover"]) <= 1e-14:
            continue
        delayed_position = position + delay_sessions
        if delayed_position < len(weights):
            events[weights.index[delayed_position]] = weights.loc[
                date
            ].to_numpy(dtype=float)

    current = np.zeros(len(weights.columns), dtype=float)
    current[weights.columns.get_loc("CASH")] = 1.0
    simulated = []
    for date in weights.index:
        trading_cost = 0.0
        if date in events:
            target = events[date]
            turnover = 0.5 * float(np.abs(target - current).sum())
            trading_cost = 2.0 * turnover * cost_bps / 10_000.0
            current = target.copy()
        asset_return = returns.loc[date].to_numpy(dtype=float)
        gross_return = float(current @ asset_return)
        simulated.append(gross_return - trading_cost)
        if 1.0 + gross_return > 1e-12:
            current = current * (1.0 + asset_return) / (1.0 + gross_return)
    return pd.Series(simulated, index=weights.index, name=directory)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    )
    asset_returns = prices.pct_change(fill_method=None)
    simulations = {
        (strategy, delay): delayed_execution(
            directory,
            asset_returns,
            delay,
        )
        for strategy, directory in STRATEGIES.items()
        for delay in (0, 1, 2)
    }
    rows = []
    relative_rows = []
    for delay in (0, 1, 2):
        for period, start, end in (
            ("complete_2015_2025", "2015-01-01", "2025-12-31"),
            ("full_2015_present", "2015-01-01", None),
        ):
            metrics = {}
            for strategy in STRATEGIES:
                sample = simulations[(strategy, delay)].loc[start:end]
                metrics[strategy] = performance_metrics(sample)
                rows.append(
                    {
                        "strategy": strategy,
                        "delay_sessions": delay,
                        "period": period,
                        **metrics[strategy],
                    }
                )
            for candidate, baseline in (
                ("dual_reentry", "production"),
                ("dual_reentry", "inverse_gate"),
                ("hysteresis19", "production"),
                ("hysteresis19", "inverse_gate"),
                ("hysteresis19", "dual_reentry"),
                ("goodvol", "production"),
                ("goodvol", "inverse_gate"),
                ("goodvol", "dual_reentry"),
                ("goodvol_bear20", "production"),
                ("goodvol_bear20", "inverse_gate"),
                ("goodvol_bear20", "dual_reentry"),
            ):
                relative_rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "delay_sessions": delay,
                        "period": period,
                        "cagr_delta": (
                            metrics[candidate]["cagr"]
                            - metrics[baseline]["cagr"]
                        ),
                        "sharpe_delta": (
                            metrics[candidate]["sharpe"]
                            - metrics[baseline]["sharpe"]
                        ),
                        "max_drawdown_delta": (
                            metrics[candidate]["max_drawdown"]
                            - metrics[baseline]["max_drawdown"]
                        ),
                    }
                )
    metrics_frame = pd.DataFrame(rows).set_index(
        ["strategy", "delay_sessions", "period"]
    )
    relative_frame = pd.DataFrame(relative_rows).set_index(
        ["candidate", "baseline", "delay_sessions", "period"]
    )
    metrics_frame.to_csv(DESTINATION / "execution_delay_metrics.csv")
    relative_frame.to_csv(
        DESTINATION / "execution_delay_relative_metrics.csv"
    )
    print(relative_frame.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
