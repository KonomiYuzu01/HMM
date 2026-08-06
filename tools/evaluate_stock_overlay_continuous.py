from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_individual_stock_overlay import (
    DEFAULT_COST_RATE,
    EVALUATION_END,
    EVALUATION_START,
    FIXED_BASKETS,
    POLICIES,
    ActiveSignalCache,
    ActiveRiskCache,
    HistoryCache,
    RiskCache,
    active_dispersion_percentile,
    load_inputs,
    load_stock_prices,
    period_metrics,
    random_baskets,
    simulate_overlay,
)


DESTINATION = Path("output/individual_stock_regime_continuous")
BASE_POLICY = "active_regime_stable_rs63_126_disp67"
POLICY_TEMPLATE = {
    **POLICIES[BASE_POLICY],
}


def add_policy(name: str, **overrides: object) -> None:
    POLICIES[name] = {**POLICY_TEMPLATE, **overrides}


add_policy("continuous_z025", active_signal_full_strength_z=0.25)
add_policy("continuous_z050", active_signal_full_strength_z=0.50)
add_policy("continuous_z075", active_signal_full_strength_z=0.75)
add_policy("continuous_z100", active_signal_full_strength_z=1.00)
add_policy("breadth50", minimum_active_breadth=0.50)
add_policy("breadth67", minimum_active_breadth=0.67)
add_policy(
    "dispersion_soft_33_67",
    active_dispersion_soft_start_percentile=0.33,
)
add_policy(
    "dispersion_soft_50_67",
    active_dispersion_soft_start_percentile=0.50,
)
add_policy("minimum_weight_050", minimum_bucket_stock_weight=0.005)
add_policy("minimum_weight_100", minimum_bucket_stock_weight=0.010)
add_policy(
    "continuous_z050_breadth50",
    active_signal_full_strength_z=0.50,
    minimum_active_breadth=0.50,
)
add_policy(
    "continuous_z050_minimum050",
    active_signal_full_strength_z=0.50,
    minimum_bucket_stock_weight=0.005,
)
add_policy(
    "continuous_z050_soft50_minimum050",
    active_signal_full_strength_z=0.50,
    active_dispersion_soft_start_percentile=0.50,
    minimum_bucket_stock_weight=0.005,
)

POLICY_NAMES = (
    BASE_POLICY,
    "continuous_z025",
    "continuous_z050",
    "continuous_z075",
    "continuous_z100",
    "breadth50",
    "breadth67",
    "dispersion_soft_33_67",
    "dispersion_soft_50_67",
    "minimum_weight_050",
    "minimum_weight_100",
    "continuous_z050_breadth50",
    "continuous_z050_minimum050",
    "continuous_z050_soft50_minimum050",
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
    risk_cache: RiskCache = {}
    active_risk_cache: ActiveRiskCache = {}
    history_cache: HistoryCache = {}
    active_signal_cache: ActiveSignalCache = {}
    dispersion_percentile = active_dispersion_percentile(all_returns)
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
                precomputed_dispersion_percentile=dispersion_percentile,
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
