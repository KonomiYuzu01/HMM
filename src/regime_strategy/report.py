from __future__ import annotations

import json
import os
from pathlib import Path

# Keep Matplotlib's font/config cache inside the writable project tree.
_MPL_CACHE = Path("tmp/matplotlib").resolve()
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import BacktestResult


def performance_metrics(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot calculate metrics from an empty return series")
    equity = (1.0 + clean).cumprod()
    years = len(clean) / annualization
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(clean.std(ddof=1) * np.sqrt(annualization))
    sharpe = float(clean.mean() / clean.std(ddof=1) * np.sqrt(annualization))
    downside = clean.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(annualization))
    sortino = float(clean.mean() * annualization / downside_deviation)
    drawdown = equity / equity.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    return {
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": maximum_drawdown,
        "calmar": cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan,
        "positive_day_rate": float((clean > 0).mean()),
    }


def write_report(
    result: BacktestResult,
    output_dir: str | Path,
    annualization: int = 252,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    strategies = {"REGIME": result.daily["net_return"]}
    strategies.update({name: result.benchmarks[name] for name in result.benchmarks})
    metrics = pd.DataFrame(
        {name: performance_metrics(series, annualization) for name, series in strategies.items()}
    ).T
    cash_returns = result.asset_returns["CASH"]
    for name, series in strategies.items():
        excess = series.reindex(cash_returns.index) - cash_returns
        metrics.loc[name, "cash_excess_sharpe"] = float(
            excess.mean() / excess.std(ddof=1) * np.sqrt(annualization)
        )
    metrics.loc["REGIME", "average_daily_turnover"] = result.daily["turnover"].mean()
    metrics.loc["REGIME", "annualized_cost_drag"] = result.daily["cost"].mean() * annualization

    result.daily.to_csv(destination / "daily_returns.csv")
    result.weights.to_csv(destination / "weights.csv")
    result.regimes.to_csv(destination / "regimes.csv")
    metrics.to_csv(destination / "metrics.csv")
    (destination / "metrics.json").write_text(
        json.dumps(metrics.round(8).to_dict(orient="index"), indent=2, allow_nan=True),
        encoding="utf-8",
    )

    annual_rows: list[dict[str, float | int]] = []
    for year, group in result.daily["net_return"].groupby(result.daily.index.year):
        annual_rows.append({"year": int(year), **performance_metrics(group, annualization)})
    pd.DataFrame(annual_rows).set_index("year").to_csv(destination / "annual_metrics.csv")

    paper_window = result.daily.loc["2023-05-01":"2026-02-28", "net_return"]
    if not paper_window.empty:
        pd.DataFrame(
            [performance_metrics(paper_window, annualization)], index=["paper_window"]
        ).to_csv(destination / "paper_window_metrics.csv")

    fixed_periods = {
        "development_2015_2021": ("2015-01-01", "2021-12-31"),
        "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
        "holdout_2022_present": ("2022-01-01", None),
    }
    period_rows: list[dict[str, float | str]] = []
    for period, (start, end) in fixed_periods.items():
        for name, series in strategies.items():
            sample = series.loc[start:end]
            if sample.empty:
                continue
            period_rows.append(
                {
                    "period": period,
                    "strategy": name,
                    **performance_metrics(sample, annualization),
                }
            )
    pd.DataFrame(period_rows).set_index(["period", "strategy"]).to_csv(
        destination / "fixed_period_metrics.csv"
    )

    cumulative = pd.DataFrame({name: (1.0 + series).cumprod() for name, series in strategies.items()})
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    cumulative.plot(ax=axes[0], title="Out-of-sample growth of $1")
    result.daily["drawdown"].plot(ax=axes[1], color="firebrick", title="Regime strategy drawdown")
    risky_weights = result.weights.drop(columns="CASH")
    risky_weights.plot.area(
        ax=axes[2], title="Risky weights and cash/borrowing sleeve", linewidth=0
    )
    cash_axis = axes[2].twinx()
    result.weights["CASH"].plot(
        ax=cash_axis, color="black", linewidth=1.0, label="CASH / borrowing"
    )
    cash_axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    cash_axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(destination / "performance.png", dpi=160)
    plt.close(fig)
    return metrics
