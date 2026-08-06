from __future__ import annotations

import numpy as np
import pandas as pd

import evaluate_probabilistic_reentry as base


DESTINATION = base.DESTINATION


def bond_disinflation_gate(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> pd.Series:
    returns = prices[["BOND", "CASH"]].pct_change(fill_method=None)
    excess_log_return = (
        np.log1p(returns["BOND"].clip(lower=-0.999999))
        - np.log1p(returns["CASH"].clip(lower=-0.999999))
    )
    gate = excess_log_return.rolling(lookback_days).sum().shift(1) > 0.0
    gate.name = "bond_outperforms_cash_252"
    return gate


def macro_conditioned_inverse_predictions(
    raw: pd.DataFrame,
    gate: pd.Series,
) -> pd.DataFrame:
    inverse = pd.DataFrame(
        {
            "probability": 1.0 - raw["probability"],
            "base_rate": 1.0 - raw["base_rate"],
        },
        index=raw.index,
    )
    active = gate.reindex(inverse.index).fillna(False).astype(bool)
    inverse.loc[~active, "probability"] = inverse.loc[
        ~active, "base_rate"
    ]
    inverse["macro_gate_active"] = active.astype(int)
    return inverse


def main() -> None:
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []

    for sample, settings in base.SAMPLES.items():
        prices, weights, daily = base.load_sample(settings)
        raw = pd.read_csv(
            DESTINATION / f"predictions_{sample}.csv",
            index_col=0,
            parse_dates=True,
        )
        gate = bond_disinflation_gate(prices)
        conditioned = macro_conditioned_inverse_predictions(raw, gate)
        conditioned.to_csv(
            DESTINATION / f"macro_conditioned_predictions_{sample}.csv"
        )
        asset_returns = prices.pct_change(fill_method=None)
        simulations = {
            "macro_inverse_cap10": base.simulate_probe(
                weights,
                daily,
                asset_returns,
                conditioned,
                base.SENSITIVITY_CAP,
            ),
            "macro_inverse_cap20": base.simulate_probe(
                weights,
                daily,
                asset_returns,
                conditioned,
                base.PRIMARY_CAP,
            ),
        }
        robustness_periods = {
            "complete": next(
                bounds
                for name, bounds in settings["periods"].items()
                if name.startswith("complete_")
            ),
            "post_gfc_2010_present": ("2010-01-01", None),
        }
        for delay_days in (0, 1, 2):
            delayed = conditioned[["probability", "base_rate"]].shift(
                delay_days
            )
            for cost_bps in (7.5, 15.0):
                for cap in (0.10, 0.20):
                    robust_simulation = base.simulate_probe(
                        weights,
                        daily,
                        asset_returns,
                        delayed,
                        cap,
                        cost_bps,
                    )
                    common_index = robust_simulation.index
                    for robustness_period, (
                        robustness_start,
                        robustness_end,
                    ) in robustness_periods.items():
                        candidate = robust_simulation.loc[
                            robustness_start:robustness_end,
                            "net_return",
                        ]
                        baseline_robust = daily["net_return"].reindex(
                            common_index
                        ).loc[robustness_start:robustness_end]
                        candidate_metrics = base.performance_metrics(
                            candidate
                        )
                        baseline_metrics = base.performance_metrics(
                            baseline_robust
                        )
                        robustness_rows.append(
                            {
                                "sample": sample,
                                "period": robustness_period,
                                "delay_days": delay_days,
                                "cost_bps": cost_bps,
                                "cap": cap,
                                "cagr_delta": float(
                                    candidate_metrics["cagr"]
                                    - baseline_metrics["cagr"]
                                ),
                                "max_drawdown_delta": float(
                                    candidate_metrics["max_drawdown"]
                                    - baseline_metrics["max_drawdown"]
                                ),
                                "annualized_relative_log_return": float(
                                    (
                                        np.log1p(candidate)
                                        - np.log1p(baseline_robust)
                                    ).mean()
                                    * 252.0
                                ),
                                "active_probe_fraction": float(
                                    robust_simulation.loc[
                                        robustness_start:robustness_end,
                                        "added_qqq_weight",
                                    ].gt(0.0).mean()
                                ),
                            }
                        )
        for strategy, frame in simulations.items():
            frame.to_csv(
                DESTINATION / f"daily_{sample}_{strategy}.csv"
            )

        growth_return = asset_returns[["QQQ", "SEMIS"]].mean(axis=1)
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for period, bounds in periods.items():
            metric_rows.extend(
                base.strategy_rows(
                    sample,
                    period,
                    bounds,
                    daily,
                    simulations,
                    growth_return,
                )
            )

        complete_period = next(
            name for name in periods if name.startswith("complete_")
        )
        start, end = periods[complete_period]
        baseline = daily.loc[start:end, "net_return"]
        for strategy, frame in simulations.items():
            bootstrap_rows.append(
                {
                    "sample": sample,
                    "strategy": strategy,
                    **base.block_bootstrap(
                        frame.loc[start:end, "net_return"],
                        baseline,
                    ),
                }
            )
            relative_log_return = (
                np.log1p(frame["net_return"])
                - np.log1p(frame["base_net_return"])
            )
            for year, values in relative_log_return.groupby(
                relative_log_return.index.year
            ):
                selected = frame.loc[values.index]
                if not selected["added_qqq_weight"].gt(0.0).any():
                    continue
                annual_rows.append(
                    {
                        "sample": sample,
                        "strategy": strategy,
                        "year": int(year),
                        "active_days": int(
                            selected["added_qqq_weight"].gt(0.0).sum()
                        ),
                        "average_added_qqq_weight": float(
                            selected["added_qqq_weight"].mean()
                        ),
                        "relative_log_return": float(values.sum()),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    annual = pd.DataFrame(annual_rows)
    robustness = pd.DataFrame(robustness_rows)
    metrics.to_csv(
        DESTINATION / "macro_conditioned_metrics.csv",
        index=False,
    )
    bootstrap.to_csv(
        DESTINATION / "macro_conditioned_bootstrap.csv",
        index=False,
    )
    annual.to_csv(
        DESTINATION / "macro_conditioned_annual_attribution.csv",
        index=False,
    )
    robustness.to_csv(
        DESTINATION / "macro_conditioned_robustness.csv",
        index=False,
    )
    print("Metrics:")
    print(
        metrics[
            [
                "sample",
                "period",
                "strategy",
                "cagr",
                "max_drawdown",
                "up_capture",
                "down_capture",
                "annualized_incremental_cost",
                "average_added_qqq_weight",
                "active_probe_fraction",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string(index=False))
    print("\nAnnual attribution:")
    print(annual.round(6).to_string(index=False))
    print("\nRobustness:")
    print(robustness.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
