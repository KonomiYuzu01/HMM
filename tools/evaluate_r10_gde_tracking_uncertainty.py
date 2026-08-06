from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


TRADING_DAYS = 252.0
OUTPUT = Path("output/r10_combined_tail_capital")
ADJUSTED_CLOSE = Path("data/retail_alternatives_adjusted_close.csv")
FRACTIONS = (0.25, 0.50, 0.75)


def gde_implementation_residual(prices: pd.DataFrame) -> pd.Series:
    required = ["GDE", "SPY", "GLD", "BIL"]
    missing = [column for column in required if column not in prices]
    if missing:
        raise ValueError(f"Missing adjusted-close columns: {missing}")
    returns = prices[required].pct_change(fill_method=None)
    synthetic = (
        0.90 * returns["SPY"]
        + 0.90 * returns["GLD"]
        - 0.80 * returns["BIL"]
    )
    return (returns["GDE"] - synthetic).dropna().rename("gde_residual")


def bootstrap_tracking_uncertainty(
    baseline: pd.Series,
    candidate: pd.DataFrame,
    residual: pd.Series,
    *,
    simulations: int = 5_000,
    block_days: int = 21,
    seed: int = 20_260_727,
) -> dict[str, float | int | str]:
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate["net_return"].rename("candidate"),
            candidate["gde_target"].rename("gde_target"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    residual_values = residual.dropna().to_numpy(dtype=float)
    if len(residual_values) < block_days:
        raise ValueError("Residual history is shorter than one block")
    observations = len(aligned)
    blocks_needed = int(np.ceil(observations / block_days))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        len(residual_values),
        size=(simulations, blocks_needed),
    )
    offsets = np.arange(block_days)
    indices = (
        starts[:, :, None] + offsets[None, None, :]
    ) % len(residual_values)
    bootstrapped = (
        residual_values[indices]
        .reshape(simulations, -1)[:, :observations]
        .T
    )
    simulated_returns = (
        aligned["candidate"].to_numpy(dtype=float)[:, None]
        + aligned["gde_target"].to_numpy(dtype=float)[:, None]
        * bootstrapped
    )
    wealth = np.cumprod(1.0 + simulated_returns, axis=0)
    years = observations / TRADING_DAYS
    cagr = wealth[-1] ** (1.0 / years) - 1.0
    peaks = np.maximum.accumulate(wealth, axis=0)
    max_drawdown = np.min(wealth / peaks - 1.0, axis=0)
    volatility = simulated_returns.std(axis=0, ddof=1)
    sharpe = (
        simulated_returns.mean(axis=0)
        / np.where(volatility > 0.0, volatility, np.nan)
        * np.sqrt(TRADING_DAYS)
    )
    baseline_metrics = performance_metrics(aligned["baseline"])
    cagr_delta = cagr - baseline_metrics["cagr"]
    sharpe_delta = sharpe - baseline_metrics["sharpe"]
    within_drawdown = max_drawdown >= -0.18
    all_objectives = (
        (cagr_delta > 0.0)
        & (sharpe_delta >= 0.0)
        & within_drawdown
    )
    return {
        "observations": observations,
        "simulations": simulations,
        "block_days": block_days,
        "seed": seed,
        "residual_start": residual.index.min().date().isoformat(),
        "residual_end": residual.index.max().date().isoformat(),
        "residual_annualized_mean": float(
            residual_values.mean() * TRADING_DAYS
        ),
        "residual_annualized_tracking_error": float(
            residual_values.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ),
        "cagr_delta_p05": float(np.quantile(cagr_delta, 0.05)),
        "cagr_delta_median": float(np.quantile(cagr_delta, 0.50)),
        "cagr_delta_p95": float(np.quantile(cagr_delta, 0.95)),
        "sharpe_p05": float(np.quantile(sharpe, 0.05)),
        "sharpe_median": float(np.quantile(sharpe, 0.50)),
        "sharpe_p95": float(np.quantile(sharpe, 0.95)),
        "max_drawdown_p05": float(
            np.quantile(max_drawdown, 0.05)
        ),
        "max_drawdown_median": float(
            np.quantile(max_drawdown, 0.50)
        ),
        "max_drawdown_p95": float(
            np.quantile(max_drawdown, 0.95)
        ),
        "probability_positive_cagr_delta": float(
            np.mean(cagr_delta > 0.0)
        ),
        "probability_sharpe_not_worse": float(
            np.mean(sharpe_delta >= 0.0)
        ),
        "probability_drawdown_within_18pct": float(
            np.mean(within_drawdown)
        ),
        "probability_all_objectives": float(
            np.mean(all_objectives)
        ),
    }


def main() -> None:
    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    residual = gde_implementation_residual(prices)
    baseline = pd.read_csv(
        OUTPUT / "proxy_synthetic_baseline_daily.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]
    rows: list[dict[str, float | int | str]] = []
    for fraction in FRACTIONS:
        label = int(round(fraction * 100))
        candidate = pd.read_csv(
            OUTPUT
            / f"proxy_synthetic_primary_fraction{label}_daily.csv",
            index_col=0,
            parse_dates=True,
        )
        rows.append(
            {
                "candidate": f"fixed_fraction_{label}",
                "substitution_fraction": fraction,
                **bootstrap_tracking_uncertainty(
                    baseline,
                    candidate,
                    residual,
                ),
            }
        )
    selected = pd.read_csv(
        Path("output/r10_relative_gap_filter")
        / "proxy_synthetic_rel50bp_current_liquidity_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    rows.append(
        {
            "candidate": (
                "selected_rel50bp_cap15_gde25_current_liquidity"
            ),
            "substitution_fraction": 0.25,
            **bootstrap_tracking_uncertainty(
                baseline,
                selected,
                residual,
            ),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT / "gde_tracking_error_bootstrap_by_candidate.csv",
        index=False,
    )
    print(result.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
