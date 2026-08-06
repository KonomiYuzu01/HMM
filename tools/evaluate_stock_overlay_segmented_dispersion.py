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
    active_dispersion_percentiles_by_benchmark,
    load_inputs,
    load_stock_prices,
    period_metrics,
    random_baskets,
    simulate_overlay,
)


DESTINATION = Path("output/individual_stock_segmented_dispersion")
BASE_POLICY = "active_regime_stable_rs63_126_disp67"
TEMPLATE = {**POLICIES[BASE_POLICY]}


def add_policy(name: str, **overrides: object) -> None:
    POLICIES[name] = {
        **TEMPLATE,
        "benchmark_specific_dispersion": True,
        **overrides,
    }


add_policy("segmented_disp60", maximum_active_dispersion_percentile=0.60)
add_policy("segmented_disp67")
add_policy("segmented_disp75", maximum_active_dispersion_percentile=0.75)
add_policy(
    "segmented_soft50_disp67",
    active_dispersion_soft_start_percentile=0.50,
)
add_policy(
    "segmented_soft50_disp75",
    maximum_active_dispersion_percentile=0.75,
    active_dispersion_soft_start_percentile=0.50,
)

POLICY_NAMES = (
    BASE_POLICY,
    "segmented_disp60",
    "segmented_disp67",
    "segmented_disp75",
    "segmented_soft50_disp67",
    "segmented_soft50_disp75",
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
    global_dispersion = active_dispersion_percentile(all_returns)
    segmented_dispersion = active_dispersion_percentiles_by_benchmark(
        all_returns
    )
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
                precomputed_dispersion_percentile=global_dispersion,
                precomputed_benchmark_dispersion_percentiles=(
                    segmented_dispersion
                ),
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
