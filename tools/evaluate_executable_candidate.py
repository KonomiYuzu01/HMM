from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_drawdown_uncertainty import (
    paired_circular_block_bootstrap,
    strategy_summary,
)
from regime_strategy.report import performance_metrics


ROOT = Path("output/open_execution_validation")
DESTINATION = Path("output/executable_candidate_validation")
EXTENDED_CANDIDATE_FILE = "gold_20_daily_asset_cap_2012_daily.csv"
CANDIDATE_FILES = {
    "cost_0bps": "gold_20_daily_asset_cap_cost0_daily.csv",
    "cost_7_5bps": "gold_20_daily_asset_cap_daily.csv",
    "cost_15bps": "gold_20_daily_asset_cap_cost15_daily.csv",
    "cost_18bps": "gold_20_daily_asset_cap_cost18_daily.csv",
    "cost_19bps": "gold_20_daily_asset_cap_cost19_daily.csv",
    "cost_25bps": "gold_20_daily_asset_cap_cost25_daily.csv",
    "target_vol_18pct": "gold_20_daily_asset_cap_vol18_daily.csv",
    "target_vol_20pct": "gold_20_daily_asset_cap_daily.csv",
    "target_vol_22pct": "gold_20_daily_asset_cap_vol22_daily.csv",
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def load_returns(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0, parse_dates=True)["net_return"].astype(
        float
    )


def load_spy_returns() -> pd.Series:
    frame = pd.read_csv(
        "data/adjusted_open_close_2011_present.csv",
        index_col=0,
        parse_dates=True,
    )
    return frame["close_SPX"].pct_change(fill_method=None).rename("SPY")


def period_metrics(candidate: pd.Series, spy: pd.Series) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        aligned = pd.concat(
            [candidate.loc[start:end].rename("candidate"), spy.loc[start:end]],
            axis=1,
            join="inner",
        ).dropna()
        for strategy in aligned:
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **performance_metrics(aligned[strategy]),
                }
            )
    return pd.DataFrame(rows).set_index(["period", "strategy"])


def annual_comparison(candidate: pd.Series, spy: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [candidate.rename("candidate"), spy], axis=1, join="inner"
    ).dropna()
    rows = []
    for year, frame in aligned.groupby(aligned.index.year):
        candidate_metrics = performance_metrics(frame["candidate"])
        spy_metrics = performance_metrics(frame["SPY"])
        rows.append(
            {
                "year": int(year),
                "candidate_return": float((1.0 + frame["candidate"]).prod() - 1.0),
                "spy_return": float((1.0 + frame["SPY"]).prod() - 1.0),
                "return_delta": float(
                    (1.0 + frame["candidate"]).prod()
                    - (1.0 + frame["SPY"]).prod()
                ),
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "spy_max_drawdown": spy_metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows).set_index("year")


def bootstrap_validation(
    candidate: pd.Series, spy: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat(
        [candidate.rename("candidate"), spy], axis=1, join="inner"
    ).dropna()
    strategy_rows = []
    relative_rows = []
    for block_length in (21, 63, 126):
        simulations = paired_circular_block_bootstrap(
            aligned["candidate"].to_numpy(),
            aligned["SPY"].to_numpy(),
            block_length=block_length,
            simulations=5_000,
            seed=20260723 + block_length,
        )
        for prefix in ("candidate", "baseline"):
            row = strategy_summary(simulations, block_length, prefix)
            row["strategy"] = "candidate" if prefix == "candidate" else "SPY"
            strategy_rows.append(row)
        relative_rows.append(
            {
                "block_length": block_length,
                "median_cagr_delta": float(simulations["cagr_delta"].median()),
                "cagr_delta_5pct": float(
                    simulations["cagr_delta"].quantile(0.05)
                ),
                "cagr_delta_95pct": float(
                    simulations["cagr_delta"].quantile(0.95)
                ),
                "probability_candidate_cagr_higher": float(
                    (simulations["cagr_delta"] > 0.0).mean()
                ),
                "probability_candidate_drawdown_better": float(
                    (simulations["max_drawdown_delta"] > 0.0).mean()
                ),
            }
        )
    return (
        pd.DataFrame(strategy_rows).set_index(["block_length", "strategy"]),
        pd.DataFrame(relative_rows).set_index("block_length"),
    )


def parameter_tail_validation(spy: pd.Series) -> pd.DataFrame:
    rows = []
    for scenario in ("target_vol_18pct", "target_vol_20pct", "target_vol_22pct"):
        candidate = load_returns(ROOT / CANDIDATE_FILES[scenario])
        aligned = pd.concat(
            [candidate.rename("candidate"), spy], axis=1, join="inner"
        ).dropna()
        for block_length in (21, 63, 126):
            simulations = paired_circular_block_bootstrap(
                aligned["candidate"].to_numpy(),
                aligned["SPY"].to_numpy(),
                block_length=block_length,
                simulations=5_000,
                seed=20260723 + block_length,
            )
            summary = strategy_summary(simulations, block_length, "candidate")
            rows.append({"scenario": scenario, **summary})
    return pd.DataFrame(rows).set_index(["scenario", "block_length"])


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    spy = load_spy_returns().loc["2015":"2025"]
    candidate = load_returns(ROOT / CANDIDATE_FILES["cost_7_5bps"])

    periods = period_metrics(candidate, spy)
    extended_candidate = load_returns(ROOT / EXTENDED_CANDIDATE_FILE)
    extended_spy = load_spy_returns().loc["2012":"2025"]
    extended_aligned = pd.concat(
        [extended_candidate.rename("candidate"), extended_spy],
        axis=1,
        join="inner",
    ).dropna()
    extended_rows = []
    for strategy in extended_aligned:
        extended_rows.append(
            {
                "period": "extended_2012_2025",
                "strategy": strategy,
                **performance_metrics(extended_aligned[strategy]),
            }
        )
    periods = pd.concat(
        [periods.reset_index(), pd.DataFrame(extended_rows)], ignore_index=True
    ).set_index(["period", "strategy"])
    periods.to_csv(DESTINATION / "period_metrics.csv")
    annual = annual_comparison(candidate, spy)
    annual.to_csv(DESTINATION / "annual_comparison.csv")

    sensitivity_rows = []
    for scenario, filename in CANDIDATE_FILES.items():
        metrics = performance_metrics(load_returns(ROOT / filename))
        sensitivity_rows.append({"scenario": scenario, **metrics})
    sensitivity = pd.DataFrame(sensitivity_rows).set_index("scenario")
    sensitivity.to_csv(DESTINATION / "cost_and_parameter_sensitivity.csv")

    open_metrics = pd.read_csv(ROOT / "metrics.csv", index_col=[0, 1])
    member_names = [
        "gold_20_daily_member_seed7",
        "gold_20_daily_member_seed42",
        "gold_20_daily_member_seed123",
    ]
    member_metrics = open_metrics.loc[
        [(name, "open_execution_proxy") for name in member_names]
    ].copy()
    member_metrics.index = ["seed_7", "seed_42", "seed_123"]
    member_metrics.to_csv(DESTINATION / "member_open_metrics.csv")

    strategy_tail, relative = bootstrap_validation(candidate, spy)
    strategy_tail.to_csv(DESTINATION / "bootstrap_tail_summary.csv")
    relative.to_csv(DESTINATION / "bootstrap_relative_summary.csv")
    parameter_tail = parameter_tail_validation(spy)
    parameter_tail.to_csv(DESTINATION / "parameter_bootstrap_tail_summary.csv")

    complete_candidate = periods.loc[("complete_2015_2025", "candidate")]
    complete_spy = periods.loc[("complete_2015_2025", "SPY")]
    development_candidate = periods.loc[("development_2015_2021", "candidate")]
    development_spy = periods.loc[("development_2015_2021", "SPY")]
    holdout_candidate = periods.loc[("holdout_2022_2025", "candidate")]
    holdout_spy = periods.loc[("holdout_2022_2025", "SPY")]
    extended_candidate_metrics = periods.loc[("extended_2012_2025", "candidate")]
    extended_spy_metrics = periods.loc[("extended_2012_2025", "SPY")]
    cost15 = sensitivity.loc["cost_15bps"]
    acceptance = pd.Series(
        {
            "historical_mdd_18pct_pass": int(
                complete_candidate["max_drawdown"] >= -0.18
            ),
            "complete_cagr_above_spy_pass": int(
                complete_candidate["cagr"] > complete_spy["cagr"]
            ),
            "development_cagr_above_spy_pass": int(
                development_candidate["cagr"] > development_spy["cagr"]
            ),
            "holdout_cagr_above_spy_pass": int(
                holdout_candidate["cagr"] > holdout_spy["cagr"]
            ),
            "cost15_cagr_above_spy_pass": int(cost15["cagr"] > complete_spy["cagr"]),
            "cost15_mdd_18pct_pass": int(cost15["max_drawdown"] >= -0.18),
            "extended_cagr_above_spy_pass": int(
                extended_candidate_metrics["cagr"] > extended_spy_metrics["cagr"]
            ),
            "parameter_neighbors_mdd_18pct_pass": int(
                sensitivity.loc[
                    ["target_vol_18pct", "target_vol_20pct", "target_vol_22pct"],
                    "max_drawdown",
                ].min()
                >= -0.18
            ),
            "all_members_mdd_18pct_pass": int(
                (member_metrics["max_drawdown"] >= -0.18).all()
            ),
            "all_members_cagr_above_spy_pass": int(
                (member_metrics["cagr"] > complete_spy["cagr"]).all()
            ),
            "all_bootstrap_blocks_probability_outperform_ge_75pct": int(
                (relative["probability_candidate_cagr_higher"] >= 0.75).all()
            ),
            "robust_spy_outperformance_pass": 0,
            "risk_target_pass": int(
                complete_candidate["max_drawdown"] >= -0.18
                and cost15["max_drawdown"] >= -0.18
            ),
        },
        name="value",
    )
    acceptance["robust_spy_outperformance_pass"] = int(
        acceptance[
            [
                "complete_cagr_above_spy_pass",
                "development_cagr_above_spy_pass",
                "holdout_cagr_above_spy_pass",
                "cost15_cagr_above_spy_pass",
                "extended_cagr_above_spy_pass",
                "all_members_cagr_above_spy_pass",
                "all_bootstrap_blocks_probability_outperform_ge_75pct",
            ]
        ].all()
    )
    acceptance["full_objective_pass"] = int(
        acceptance["risk_target_pass"]
        and acceptance["robust_spy_outperformance_pass"]
    )
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    cost18 = sensitivity.loc["cost_18bps", "cagr"]
    cost19 = sensitivity.loc["cost_19bps", "cagr"]
    spy_cagr = complete_spy["cagr"]
    break_even_bps = 18.0 + float((cost18 - spy_cagr) / (cost18 - cost19))
    operating_limits = pd.Series(
        {
            "historical_mdd_limit": 0.18,
            "observed_mdd": abs(float(complete_candidate["max_drawdown"])),
            "observed_mdd_buffer": 0.18
            - abs(float(complete_candidate["max_drawdown"])),
            "cost_break_even_bps_interpolated": break_even_bps,
            "normal_cost_assumption_bps": 7.5,
            "cost_stress_assumption_bps": 15.0,
        },
        name="value",
    )
    operating_limits.to_csv(DESTINATION / "operating_limits.csv")

    print("Period metrics:")
    print(periods[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nCost and parameter sensitivity:")
    print(sensitivity[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nBootstrap candidate tails:")
    print(strategy_tail.round(6).to_string())
    print("\nBootstrap candidate minus SPY:")
    print(relative.round(6).to_string())
    print("\nMember open-execution metrics:")
    print(member_metrics[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nParameter-neighbor bootstrap tails:")
    print(
        parameter_tail[
            [
                "median_cagr",
                "median_max_drawdown",
                "probability_breach_18pct",
                "probability_breach_20pct",
            ]
        ]
        .round(6)
        .to_string()
    )
    print("\nAcceptance:")
    print(acceptance.to_string())
    print("\nOperating limits:")
    print(operating_limits.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
