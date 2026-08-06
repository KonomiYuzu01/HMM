from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
DIRECTORIES = {
    "normal": (
        "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
    ),
    "cost15": (
        "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_cost15",
    ),
    "extended": (
        "paper_core_growth_gold20_daily_risk_netted_ensemble_2012",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_2012",
    ),
}
BLENDS = (0.0, 0.25, 0.50, 0.75, 1.0)


def returns(directory: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]


def fixed_mix(
    production: pd.Series,
    candidate: pd.Series,
    candidate_share: float,
    rebalance_cost_bps: float,
) -> pd.Series:
    aligned = pd.concat(
        {
            "production": production,
            "candidate": candidate,
        },
        axis=1,
        join="inner",
    ).dropna()
    mixed = (
        (1.0 - candidate_share) * aligned["production"]
        + candidate_share * aligned["candidate"]
    )
    candidate_after_return = (
        candidate_share
        * (1.0 + aligned["candidate"])
        / (1.0 + mixed)
    )
    total_traded_notional = 2.0 * (
        candidate_after_return - candidate_share
    ).abs()
    cost = total_traded_notional * rebalance_cost_bps / 10_000.0
    return (mixed - cost).rename(f"candidate_share_{candidate_share:.2f}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, (production_directory, candidate_directory) in (
        DIRECTORIES.items()
    ):
        production = returns(production_directory)
        candidate = returns(candidate_directory)
        for candidate_share in BLENDS:
            for mix_cost_bps in (0.0, 7.5):
                mixed = fixed_mix(
                    production,
                    candidate,
                    candidate_share,
                    mix_cost_bps,
                )
                periods = (
                    (
                        ("development_2015_2021", "2015", "2021"),
                        ("holdout_2022_2025", "2022", "2025"),
                        ("complete_2015_2025", "2015", "2025"),
                    )
                    if scenario != "extended"
                    else (("extended_2012_2025", "2012", "2025"),)
                )
                for period, start, end in periods:
                    rows.append(
                        {
                            "scenario": scenario,
                            "candidate_share": candidate_share,
                            "mix_rebalance_cost_bps": mix_cost_bps,
                            "period": period,
                            **performance_metrics(mixed.loc[start:end]),
                            "current_transition_one_way_turnover": (
                                candidate_share * 0.35662766137450314
                            ),
                        }
                    )
    metrics = pd.DataFrame(rows).set_index(
        [
            "scenario",
            "candidate_share",
            "mix_rebalance_cost_bps",
            "period",
        ]
    )
    metrics.to_csv(DESTINATION / "production_shadow_blend_frontier.csv")

    comparison = metrics.xs(
        ("normal", 0.0, "complete_2015_2025"),
        level=("scenario", "mix_rebalance_cost_bps", "period"),
    )
    print(
        comparison[
            [
                "cagr",
                "sharpe",
                "max_drawdown",
                "current_transition_one_way_turnover",
            ]
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
