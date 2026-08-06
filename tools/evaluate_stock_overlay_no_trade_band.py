from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_individual_stock_overlay import (
    DEFAULT_COST_RATE,
    FIXED_BASKETS,
    POLICIES,
    ActiveRiskCache,
    ActiveSignalCache,
    HistoryCache,
    RiskCache,
    active_dispersion_percentile,
    load_inputs,
    load_stock_prices,
    period_metrics,
    random_baskets,
    simulate_overlay,
)


DESTINATION = Path("output/individual_stock_no_trade_band")
BASE_POLICY = "active_regime_stable_rs63_126_disp67"
TEMPLATE = {**POLICIES[BASE_POLICY]}


def add_policy(name: str, threshold: float) -> None:
    POLICIES[name] = {
        **TEMPLATE,
        "minimum_rebalance_turnover": threshold,
    }


add_policy("no_trade_025", 0.0025)
add_policy("no_trade_050", 0.0050)
add_policy("no_trade_075", 0.0075)
add_policy("no_trade_100", 0.0100)
add_policy("no_trade_150", 0.0150)

POLICY_NAMES = (
    BASE_POLICY,
    "no_trade_025",
    "no_trade_050",
    "no_trade_075",
    "no_trade_100",
    "no_trade_150",
)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics.loc[metrics["basket"].str.startswith("random_")]
    return (
        random.groupby(["period", "policy"])
        .agg(
            median_cagr=("cagr", "median"),
            cagr_10pct=("cagr", lambda values: values.quantile(0.10)),
            median_relative=(
                "annualized_relative_log_return",
                "median",
            ),
            relative_10pct=(
                "annualized_relative_log_return",
                lambda values: values.quantile(0.10),
            ),
            median_max_drawdown=("max_drawdown", "median"),
            drawdown_10pct=(
                "max_drawdown",
                lambda values: values.quantile(0.10),
            ),
            median_stock_weight=("average_stock_weight", "median"),
            median_turnover=("annualized_overlay_turnover", "median"),
        )
        .reset_index()
    )


def main() -> None:
    stock_prices = load_stock_prices(False)
    base, weights, all_returns, states = load_inputs(stock_prices)
    baskets = {**FIXED_BASKETS, **random_baskets(10)}
    dispersion = active_dispersion_percentile(all_returns)
    risk_cache: RiskCache = {}
    active_risk_cache: ActiveRiskCache = {}
    history_cache: HistoryCache = {}
    active_signal_cache: ActiveSignalCache = {}
    rows: list[dict[str, object]] = []
    for basket_name, basket in baskets.items():
        for policy in POLICY_NAMES:
            result, _ = simulate_overlay(
                basket_name,
                basket,
                policy,
                base,
                weights,
                all_returns,
                states,
                stock_prices,
                risk_cache,
                active_risk_cache,
                cost_rate=DEFAULT_COST_RATE,
                precomputed_dispersion_percentile=dispersion,
                history_cache=history_cache,
                active_signal_cache=active_signal_cache,
            )
            rows.extend(
                period_metrics(
                    result,
                    basket_name,
                    policy,
                    0.0,
                    DEFAULT_COST_RATE,
                )
            )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    summary.to_csv(DESTINATION / "summary.csv", index=False)
    print(
        summary.sort_values(
            ["period", "relative_10pct", "median_relative"],
            ascending=[True, False, False],
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
