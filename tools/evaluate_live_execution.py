from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/live_execution_validation")
STRATEGIES = {
    "production_baseline": "paper_core_growth_gold20_daily_risk_ensemble",
    "jump_aware_candidate": "paper_core_growth_gold20_jump_aware_daily_risk_ensemble",
}
PERIODS = {
    "frozen_2015_2025": ("2015-01-01", "2025-12-31"),
    "paper_2026": ("2026-01-01", None),
    "recent_july_2026": ("2026-07-01", None),
    "full_2015_present": ("2015-01-01", None),
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(refresh=False)
    simulations = {}
    rows = []
    for strategy, directory in STRATEGIES.items():
        frame = simulate_open_execution(
            directory,
            opens,
            closes,
            start_date="2015-01-01",
            end_date=None,
            cost_bps=7.5,
        )
        simulations[strategy] = frame
        frame.to_csv(DESTINATION / f"{strategy}_daily.csv")
        for period, (start, end) in PERIODS.items():
            sample = frame.loc[start:end, "net_return"]
            if sample.empty:
                continue
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "sessions": int(len(sample)),
                    "period_return": float((1.0 + sample).prod() - 1.0),
                    **performance_metrics(sample),
                }
            )

    benchmark_returns = closes[["SPX", "QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    common_end = min(frame.index.max() for frame in simulations.values())
    for benchmark in benchmark_returns:
        for period, (start, end) in PERIODS.items():
            period_end = min(pd.Timestamp(end), common_end) if end else common_end
            sample = benchmark_returns.loc[start:period_end, benchmark].dropna()
            if sample.empty:
                continue
            rows.append(
                {
                    "period": period,
                    "strategy": f"benchmark_{benchmark}",
                    "sessions": int(len(sample)),
                    "period_return": float((1.0 + sample).prod() - 1.0),
                    **performance_metrics(sample),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["period", "strategy"])
    metrics.to_csv(DESTINATION / "metrics.csv")

    recent = pd.concat(
        {
            strategy: frame.loc["2026-07-01":, "net_return"]
            for strategy, frame in simulations.items()
        },
        axis=1,
    ).dropna()
    recent_equity = (1.0 + recent).cumprod()
    recent_drawdown = recent_equity / recent_equity.cummax() - 1.0
    recent_summary = pd.DataFrame(
        {
            "period_return": (1.0 + recent).prod() - 1.0,
            "max_drawdown": recent_drawdown.min(),
            "worst_day": recent.min(),
        }
    )
    recent_summary.to_csv(DESTINATION / "recent_summary.csv")

    event_series = {
        **{
            strategy: frame.loc["2026":, "net_return"]
            for strategy, frame in simulations.items()
        },
        "benchmark_QQQ": benchmark_returns.loc["2026":, "QQQ"],
        "benchmark_SEMIS": benchmark_returns.loc["2026":, "SEMIS"],
    }
    event_rows = []
    for strategy, returns in event_series.items():
        returns = returns.loc[:common_end].dropna()
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        trough_date = drawdown.idxmin()
        peak_date = equity.loc[:trough_date].idxmax()
        trailing = returns.iloc[-15:]
        trailing_equity = (1.0 + trailing).cumprod()
        trailing_drawdown = trailing_equity / trailing_equity.cummax() - 1.0
        event_rows.append(
            {
                "strategy": strategy,
                "peak_date_before_max_drawdown": peak_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "year_max_drawdown": float(drawdown.min()),
                "current_drawdown": float(drawdown.iloc[-1]),
                "last_15_sessions_return": float((1.0 + trailing).prod() - 1.0),
                "last_15_sessions_max_drawdown": float(trailing_drawdown.min()),
            }
        )
    events = pd.DataFrame(event_rows).set_index("strategy")
    events.to_csv(DESTINATION / "drawdown_events.csv")

    recent_benchmark_returns = benchmark_returns.loc[:common_end].iloc[-60:]
    decoupling = pd.Series(
        {
            "qqq_20d_annualized_volatility": float(
                recent_benchmark_returns["QQQ"].iloc[-20:].std(ddof=1)
                * (252.0**0.5)
            ),
            "smh_20d_annualized_volatility": float(
                recent_benchmark_returns["SEMIS"].iloc[-20:].std(ddof=1)
                * (252.0**0.5)
            ),
            "smh_qqq_20d_volatility_ratio": float(
                recent_benchmark_returns["SEMIS"].iloc[-20:].std(ddof=1)
                / recent_benchmark_returns["QQQ"].iloc[-20:].std(ddof=1)
            ),
            "qqq_smh_20d_correlation": float(
                recent_benchmark_returns["QQQ"]
                .iloc[-20:]
                .corr(recent_benchmark_returns["SEMIS"].iloc[-20:])
            ),
        },
        name="value",
    )
    decoupling.to_csv(DESTINATION / "current_decoupling.csv")

    print(
        metrics[
            ["sessions", "period_return", "cagr", "sharpe", "max_drawdown"]
        ]
        .round(6)
        .to_string()
    )
    print("\nRecent July strategy path:")
    print(recent_summary.round(6).to_string())
    print("\n2026 drawdown events:")
    print(events.round(6).to_string())
    print("\nCurrent decoupling snapshot:")
    print(decoupling.round(6).to_string())
    print(f"\nData through: {common_end.date().isoformat()}")
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
