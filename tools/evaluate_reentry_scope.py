from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


ROOT = Path("output")
DESTINATION = ROOT / "reentry_scope_validation"
STRATEGIES = {
    "baseline": "paper_core_robust_vol_guarded_floor_ensemble",
    "zero_growth_entry": "paper_core_zero_entry_growth_reallocation_ensemble",
    "all_risk_on_entries": "paper_core_risk_on_growth_reallocation_ensemble",
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def annualized_relative_return(candidate: pd.Series, reference: pd.Series) -> float:
    aligned = pd.concat([candidate, reference], axis=1, join="inner").dropna()
    years = len(aligned) / 252.0
    relative_wealth = (1.0 + aligned.iloc[:, 0]).prod() / (
        1.0 + aligned.iloc[:, 1]
    ).prod()
    return float(relative_wealth ** (1.0 / years) - 1.0)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {
        label: pd.read_csv(
            ROOT / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )["net_return"]
        for label, directory in STRATEGIES.items()
    }
    rows = []
    for label, returns in daily.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": label,
                    "period": period,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    yearly_rows = []
    broad = daily["all_risk_on_entries"].loc["2015":"2025"]
    narrow = daily["zero_growth_entry"].loc["2015":"2025"]
    for year in sorted(broad.index.year.unique()):
        mask = broad.index.year == year
        yearly_rows.append(
            {
                "year": year,
                "broad_minus_narrow_annual_return": annualized_relative_return(
                    broad.loc[mask], narrow.loc[mask]
                ),
            }
        )
    yearly = pd.DataFrame(yearly_rows).set_index("year")
    yearly.to_csv(DESTINATION / "annual_relative_return.csv")

    leave_rows = []
    for omitted_year in yearly.index:
        keep = broad.index.year != omitted_year
        leave_rows.append(
            {
                "omitted_year": omitted_year,
                "broad_minus_narrow_annualized_return": annualized_relative_return(
                    broad.loc[keep], narrow.loc[keep]
                ),
            }
        )
    leave = pd.DataFrame(leave_rows).set_index("omitted_year")
    leave.to_csv(DESTINATION / "leave_one_year_out.csv")

    trigger_rows = []
    for label, directory in STRATEGIES.items():
        for member_dir in sorted((ROOT / directory / "members").glob("seed_*")):
            regimes = pd.read_csv(
                member_dir / "regimes.csv", index_col=0, parse_dates=True
            )
            active = regimes.get(
                "asset_risk_overlay_active",
                pd.Series(0, index=regimes.index),
            ).eq(1)
            trigger_rows.append(
                {
                    "strategy": label,
                    "member": member_dir.name,
                    "active_reviews": int(active.sum()),
                    "active_fraction": float(active.mean()),
                }
            )
    pd.DataFrame(trigger_rows).set_index(["strategy", "member"]).to_csv(
        DESTINATION / "trigger_summary.csv"
    )

    print(metrics[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nBroad minus narrow annual relative return:")
    print(yearly.round(6).to_string())
    print("\nLeave-one-year-out range:")
    print(leave.agg(["min", "median", "max"]).round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
