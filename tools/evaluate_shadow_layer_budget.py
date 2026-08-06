from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
LAYERS = (
    "production",
    "lowvol_floor",
    "inverse_momentum_gate",
    "zero_entry_bridge",
)
DIRECTORIES = {
    "normal_2015_2025": {
        "production": "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "lowvol_floor": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble"
        ),
        "inverse_momentum_gate": (
            "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
            "netted_ensemble"
        ),
        "zero_entry_bridge": (
            "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
            "netted_ensemble"
        ),
    },
    "cost15_2015_2025": {
        "production": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15"
        ),
        "lowvol_floor": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble_cost15"
        ),
        "inverse_momentum_gate": (
            "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
            "netted_ensemble_cost15"
        ),
        "zero_entry_bridge": (
            "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
            "netted_ensemble_cost15"
        ),
    },
    "extended_2012_2025": {
        "production": (
            "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
        ),
        "lowvol_floor": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble_2012"
        ),
        "inverse_momentum_gate": (
            "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
            "netted_ensemble_2012"
        ),
        "zero_entry_bridge": (
            "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
            "netted_ensemble_2012"
        ),
    },
    "proxy_full": {
        "production": "past_20y_drawdown_backtest_netted",
        "lowvol_floor": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble_20y_proxy"
        ),
        "inverse_momentum_gate": (
            "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
            "netted_ensemble_20y_proxy"
        ),
        "zero_entry_bridge": (
            "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
            "netted_ensemble_20y_proxy"
        ),
    },
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
    for scenario, directories in DIRECTORIES.items():
        previous_metrics: dict[str, float] | None = None
        for layer in LAYERS:
            sample = returns(directories[layer])
            if scenario in {"normal_2015_2025", "cost15_2015_2025"}:
                sample = sample.loc["2015":"2025"]
            elif scenario == "extended_2012_2025":
                sample = sample.loc["2012":"2025"]
            metrics = performance_metrics(sample)
            rows.append(
                {
                    "scenario": scenario,
                    "layer": layer,
                    **metrics,
                    "marginal_cagr": (
                        metrics["cagr"] - previous_metrics["cagr"]
                        if previous_metrics is not None
                        else 0.0
                    ),
                    "marginal_sharpe": (
                        metrics["sharpe"] - previous_metrics["sharpe"]
                        if previous_metrics is not None
                        else 0.0
                    ),
                    "marginal_max_drawdown": (
                        metrics["max_drawdown"]
                        - previous_metrics["max_drawdown"]
                        if previous_metrics is not None
                        else 0.0
                    ),
                }
            )
            previous_metrics = metrics
    budget = pd.DataFrame(rows).set_index(["scenario", "layer"])
    budget.to_csv(DESTINATION / "shadow_layer_budget.csv")
    print(
        budget[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "marginal_cagr",
                "marginal_sharpe",
                "marginal_max_drawdown",
            ]
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
