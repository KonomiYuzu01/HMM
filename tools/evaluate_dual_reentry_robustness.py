from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
VARIANTS = {
    "production": (
        "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "past_20y_drawdown_backtest_netted",
    ),
    "inverse_gate": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble",
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_fast10": (
        "paper_core_growth_gold20_dual_reentry_fast10_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_fast10_inverse_momentum_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_fast20": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_fast40": (
        "paper_core_growth_gold20_dual_reentry_fast40_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_fast40_inverse_momentum_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_q85": (
        "paper_core_growth_gold20_dual_reentry_q85_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_q85_inverse_momentum_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_q95": (
        "paper_core_growth_gold20_dual_reentry_q95_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_q95_inverse_momentum_"
        "netted_ensemble_20y_proxy",
    ),
    "dual_equal_bridge": (
        "paper_core_growth_gold20_dual_reentry_equal_bridge_"
        "inverse_momentum_netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_equal_bridge_"
        "inverse_momentum_netted_ensemble_20y_proxy",
    ),
}


def returns(directory: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, (normal_directory, proxy_directory) in VARIANTS.items():
        for sample, series in (
            (
                "normal_2015_2025",
                returns(normal_directory).loc["2015":"2025"],
            ),
            ("proxy_full", returns(proxy_directory)),
        ):
            rows.append(
                {
                    "variant": variant,
                    "sample": sample,
                    **performance_metrics(series),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["variant", "sample"])
    normal_production = metrics.loc[
        ("production", "normal_2015_2025")
    ]
    proxy_production = metrics.loc[("production", "proxy_full")]
    metrics["cagr_delta_vs_production"] = [
        row["cagr"]
        - (
            normal_production["cagr"]
            if sample == "normal_2015_2025"
            else proxy_production["cagr"]
        )
        for (_, sample), row in metrics.iterrows()
    ]
    metrics["mdd_gate_pass"] = metrics["max_drawdown"] >= -0.18
    metrics.to_csv(DESTINATION / "parameter_robustness.csv")
    print(
        metrics[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "cagr_delta_vs_production",
                "mdd_gate_pass",
            ]
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
