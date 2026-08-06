from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


ROOT = Path("output")
DESTINATION = ROOT / "gold_leverage_candidate_validation"
PRIMARY = "paper_core_zero_entry_growth_reallocation_ensemble"
CANDIDATE = "paper_core_zero_entry_growth_gold20_lev110_ensemble"
PRIMARY_COST15 = "paper_core_zero_entry_growth_reallocation_ensemble_cost15"
CANDIDATE_COST15 = "paper_core_zero_entry_growth_gold20_lev110_ensemble_cost15"
PRIMARY_2012 = "paper_core_zero_entry_growth_reallocation_ensemble_2012"
CANDIDATE_2012 = "paper_core_zero_entry_growth_gold20_lev110_ensemble_2012"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def load_returns(directory: str) -> pd.Series:
    return pd.read_csv(
        ROOT / directory / "daily_returns.csv", index_col=0, parse_dates=True
    )["net_return"]


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    primary = load_returns(PRIMARY)
    candidate = load_returns(CANDIDATE)
    metric_rows = []
    for label, returns in (("primary", primary), ("candidate", candidate)):
        for period, (start, end) in PERIODS.items():
            metric_rows.append(
                {
                    "strategy": label,
                    "period": period,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    for label, directory in (
        ("primary_cost15", PRIMARY_COST15),
        ("candidate_cost15", CANDIDATE_COST15),
    ):
        metric_rows.append(
            {
                "strategy": label,
                "period": "complete_2015_2025",
                **performance_metrics(load_returns(directory).loc["2015":"2025"]),
            }
        )
    for label, directory in (
        ("primary_2012", PRIMARY_2012),
        ("candidate_2012", CANDIDATE_2012),
    ):
        metric_rows.append(
            {
                "strategy": label,
                "period": "extended_2012_2025",
                **performance_metrics(load_returns(directory).loc[:"2025"]),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics.csv")

    def value(strategy: str, period: str, field: str) -> float:
        return float(metrics.loc[(strategy, period), field])

    acceptance = {
        "development_cagr_delta": value(
            "candidate", "development_2015_2021", "cagr"
        )
        - value("primary", "development_2015_2021", "cagr"),
        "holdout_cagr_delta": value("candidate", "holdout_2022_2025", "cagr")
        - value("primary", "holdout_2022_2025", "cagr"),
        "complete_cagr_delta": value("candidate", "complete_2015_2025", "cagr")
        - value("primary", "complete_2015_2025", "cagr"),
        "complete_sharpe_delta": value(
            "candidate", "complete_2015_2025", "sharpe"
        )
        - value("primary", "complete_2015_2025", "sharpe"),
        "extended_cagr_delta": value(
            "candidate_2012", "extended_2012_2025", "cagr"
        )
        - value("primary_2012", "extended_2012_2025", "cagr"),
        "extended_sharpe_delta": value(
            "candidate_2012", "extended_2012_2025", "sharpe"
        )
        - value("primary_2012", "extended_2012_2025", "sharpe"),
        "complete_max_drawdown": value(
            "candidate", "complete_2015_2025", "max_drawdown"
        ),
        "cost15_max_drawdown": value(
            "candidate_cost15", "complete_2015_2025", "max_drawdown"
        ),
        "extended_max_drawdown": value(
            "candidate_2012", "extended_2012_2025", "max_drawdown"
        ),
    }
    for field in (
        "development_cagr_delta",
        "holdout_cagr_delta",
        "complete_cagr_delta",
        "complete_sharpe_delta",
        "extended_cagr_delta",
        "extended_sharpe_delta",
    ):
        acceptance[f"{field}_pass"] = int(float(acceptance[field]) >= 0.0)
    for field in (
        "complete_max_drawdown",
        "cost15_max_drawdown",
        "extended_max_drawdown",
    ):
        acceptance[f"{field}_pass"] = int(float(acceptance[field]) >= -0.18)
    acceptance["strict_overall_pass"] = int(
        all(
            bool(value)
            for key, value in acceptance.items()
            if key.endswith("_pass")
        )
    )
    pd.DataFrame([acceptance], index=[CANDIDATE]).to_csv(
        DESTINATION / "acceptance.csv"
    )

    member_rows = []
    for variant, directory in (
        ("normal", CANDIDATE),
        ("cost15", CANDIDATE_COST15),
        ("extended_2012", CANDIDATE_2012),
    ):
        for member_dir in sorted((ROOT / directory / "members").glob("seed_*")):
            member_returns = pd.read_csv(
                member_dir / "daily_returns.csv", index_col=0, parse_dates=True
            )
            evaluation = member_returns["net_return"]
            if variant == "normal" or variant == "cost15":
                evaluation = evaluation.loc["2015":"2025"]
            else:
                evaluation = evaluation.loc[:"2025"]
            regimes = pd.read_csv(
                member_dir / "regimes.csv", index_col=0, parse_dates=True
            )
            gross = regimes["gross_leverage"].astype(float)
            member_rows.append(
                {
                    "variant": variant,
                    "member": member_dir.name,
                    **performance_metrics(evaluation),
                    "scheduled_leverage_fraction": float((gross > 1.0001).mean()),
                    "maximum_scheduled_gross_leverage": float(gross.max()),
                    "annualized_financing_drag": float(
                        member_returns["financing_cost"].mean() * 252.0
                    ),
                }
            )
    members = pd.DataFrame(member_rows).set_index(["variant", "member"])
    members.to_csv(DESTINATION / "member_and_leverage_audit.csv")

    print(metrics[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nAcceptance:")
    print(pd.Series(acceptance).to_string())
    print("\nMember/leverage audit:")
    print(
        members[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "scheduled_leverage_fraction",
                "maximum_scheduled_gross_leverage",
                "annualized_financing_drag",
            ]
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
