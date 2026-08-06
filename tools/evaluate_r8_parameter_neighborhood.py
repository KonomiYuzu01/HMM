from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = (
    OUTPUT / "regime_strategy_research_2026-07-25" / "r8_neighborhood"
)
PRODUCTION = "paper_core_growth_gold20_daily_risk_netted_ensemble"
CANDIDATES = {
    "floor40": (
        "paper_core_growth_gold20_dual_reentry_floor40_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
    "floor45_r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
    "floor50": (
        "paper_core_growth_gold20_dual_reentry_floor50_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
    "bridge10": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear10_"
        "inverse_momentum_netted_ensemble"
    ),
    "bridge30": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear30_"
        "inverse_momentum_netted_ensemble"
    ),
    "downside40": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_down40_"
        "inverse_momentum_netted_ensemble"
    ),
    "downside60": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_down60_"
        "inverse_momentum_netted_ensemble"
    ),
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def returns(directory: str, suffix: str = "") -> pd.Series:
    path = OUTPUT / directory / "daily_returns.csv"
    if suffix:
        path = OUTPUT / directory / "members" / suffix / "daily_returns.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["net_return"]


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    production = returns(PRODUCTION)
    production_metrics = {
        period: performance_metrics(production.loc[start:end])
        for period, (start, end) in PERIODS.items()
    }
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for name, directory in CANDIDATES.items():
        candidate = returns(directory)
        for period, (start, end) in PERIODS.items():
            values = performance_metrics(candidate.loc[start:end])
            baseline = production_metrics[period]
            rows.append(
                {
                    "candidate": name,
                    "period": period,
                    **values,
                    "cagr_delta_vs_production": (
                        values["cagr"] - baseline["cagr"]
                    ),
                    "sharpe_delta_vs_production": (
                        values["sharpe"] - baseline["sharpe"]
                    ),
                    "max_drawdown_delta_vs_production": (
                        values["max_drawdown"] - baseline["max_drawdown"]
                    ),
                    "mdd_within_18pct": values["max_drawdown"] >= -0.18,
                }
            )
        for seed in ("seed_7", "seed_42", "seed_123"):
            values = performance_metrics(
                returns(directory, seed).loc["2015-01-01":"2025-12-31"]
            )
            seed_rows.append(
                {
                    "candidate": name,
                    "seed": seed,
                    **values,
                    "cagr_above_production": (
                        values["cagr"]
                        > production_metrics["complete_2015_2025"]["cagr"]
                    ),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["candidate", "period"])
    seeds = pd.DataFrame(seed_rows).set_index(["candidate", "seed"])
    metrics.to_csv(DESTINATION / "metrics.csv")
    seeds.to_csv(DESTINATION / "seed_consistency.csv")

    complete = metrics.xs("complete_2015_2025", level="period")
    decision = pd.Series(
        {
            "tested_neighbors": len(CANDIDATES) - 1,
            "all_neighbors_cagr_above_production": bool(
                complete.drop(index="floor45_r8")[
                    "cagr_delta_vs_production"
                ].gt(0.0).all()
            ),
            "all_neighbors_sharpe_above_production": bool(
                complete.drop(index="floor45_r8")[
                    "sharpe_delta_vs_production"
                ].gt(0.0).all()
            ),
            "all_neighbors_mdd_within_18pct": bool(
                complete.drop(index="floor45_r8")[
                    "mdd_within_18pct"
                ].all()
            ),
            "all_neighbor_seeds_cagr_above_production": bool(
                seeds.drop(index="floor45_r8", level="candidate")[
                    "cagr_above_production"
                ].all()
            ),
            "selection_policy": (
                "diagnostic_only_keep_preregistered_r8_no_neighbor_selection"
            ),
        },
        name="value",
    )
    decision.to_csv(DESTINATION / "decision.csv")

    print("Complete-window neighborhood:")
    print(
        complete[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "cagr_delta_vs_production",
            ]
        ].round(6).to_string()
    )
    print("\nDecision:")
    print(decision.to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
