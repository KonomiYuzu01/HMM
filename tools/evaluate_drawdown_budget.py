from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "drawdown_budget_validation"
STRATEGIES = {
    "production_primary": "paper_core_zero_entry_growth_reallocation_ensemble",
    "tiered_8_12_16": "paper_core_zero_entry_growth_dd_tiered_ensemble",
    "late_10_15": "paper_core_zero_entry_growth_dd_late_ensemble",
    "single_12": "paper_core_zero_entry_growth_dd_single_ensemble",
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | str]] = []
    activation_rows: list[dict[str, float | int | str]] = []
    for label, directory in STRATEGIES.items():
        daily = pd.read_csv(
            OUTPUT_ROOT / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        returns = daily["net_return"]
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": label,
                    "period": period,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
        for member_dir in sorted((OUTPUT_ROOT / directory / "members").glob("seed_*")):
            regimes = pd.read_csv(
                member_dir / "regimes.csv",
                index_col=0,
                parse_dates=True,
            )
            multipliers = regimes.get(
                "drawdown_multiplier",
                pd.Series(1.0, index=regimes.index),
            ).astype(float)
            active = multipliers < 1.0 - 1e-12
            activation_rows.append(
                {
                    "strategy": label,
                    "member": member_dir.name,
                    "scheduled_observations": len(multipliers),
                    "overlay_observations": int(active.sum()),
                    "overlay_fraction": float(active.mean()),
                    "first_activation": (
                        active.index[active][0].date().isoformat()
                        if active.any()
                        else ""
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")
    baseline = metrics.loc["production_primary"]
    relative_rows = []
    for strategy in STRATEGIES:
        if strategy == "production_primary":
            continue
        for period in PERIODS:
            candidate = metrics.loc[(strategy, period)]
            reference = baseline.loc[period]
            relative_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "cagr_delta": candidate["cagr"] - reference["cagr"],
                    "sharpe_delta": candidate["sharpe"] - reference["sharpe"],
                    "max_drawdown_delta": candidate["max_drawdown"]
                    - reference["max_drawdown"],
                    "accepted": int(
                        candidate["cagr"] >= reference["cagr"]
                        and candidate["max_drawdown"] >= reference["max_drawdown"]
                    ),
                }
            )
    relative = pd.DataFrame(relative_rows).set_index(["strategy", "period"])
    relative.to_csv(DESTINATION / "relative_metrics.csv")
    pd.DataFrame(activation_rows).set_index(["strategy", "member"]).to_csv(
        DESTINATION / "activation_summary.csv"
    )
    print(metrics[["cagr", "sharpe", "max_drawdown", "calmar"]].round(6))
    print("\nCandidate minus production primary:")
    print(relative.round(6))
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
