from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_drawdown_uncertainty import paired_circular_block_bootstrap
from evaluate_open_execution import load_open_close, simulate_open_execution
from evaluate_open_family_reality_check import STRATEGIES as EXISTING_FAMILY
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "lev110_jump_candidate_validation"
BASELINE = "paper_core_growth_gold20_daily_risk_ensemble"
CANDIDATE = "paper_core_growth_gold20_lev110_target25_jump_daily_risk_ensemble"
BASELINE_2012 = "paper_core_growth_gold20_daily_risk_ensemble_2012"
CANDIDATE_2012 = (
    "paper_core_growth_gold20_lev110_target25_jump_daily_risk_ensemble_2012"
)
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
ADDITIONAL_FAMILY = {
    "residual_momentum": "paper_core_growth_gold20_residual_momentum_ensemble",
    "dynamic_factor_risk": "paper_core_growth_gold20_dynamic_factor_risk_ensemble",
    "downside_semivol": "paper_core_growth_gold20_downside_daily_risk_ensemble",
    "semiskew": "paper_core_growth_gold20_semiskew_daily_risk_ensemble",
    "probability_allocation": (
        "paper_core_growth_gold20_probability_daily_risk_ensemble"
    ),
    "monthly_members": (
        "paper_core_growth_gold20_monthly_members_daily_risk_ensemble"
    ),
    "lev110_target25": (
        "paper_core_growth_gold20_lev110_target25_daily_risk_ensemble"
    ),
    "lev110_target25_jump": CANDIDATE,
}


def metric_rows(
    name: str,
    cost_bps: float,
    returns: pd.Series,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for period, (start, end) in PERIODS.items():
        rows.append(
            {
                "strategy": name,
                "cost_bps": cost_bps,
                "period": period,
                **performance_metrics(returns.loc[start:end]),
            }
        )
    return rows


def paired_summary(
    candidate: pd.Series,
    baseline: pd.Series,
    sample: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative_rows: list[dict[str, float | int | str]] = []
    tail_rows: list[dict[str, float | int | str]] = []
    for block_days in (21, 63, 126):
        simulations = paired_circular_block_bootstrap(
            aligned["candidate"].to_numpy(),
            aligned["baseline"].to_numpy(),
            block_days,
            simulations=5_000,
            seed=20260723 + block_days,
        )
        relative_rows.append(
            {
                "sample": sample,
                "block_days": block_days,
                "probability_candidate_cagr_higher": float(
                    (simulations["cagr_delta"] > 0.0).mean()
                ),
                "median_cagr_delta": float(simulations["cagr_delta"].median()),
                "cagr_delta_5pct": float(simulations["cagr_delta"].quantile(0.05)),
                "cagr_delta_95pct": float(
                    simulations["cagr_delta"].quantile(0.95)
                ),
                "probability_candidate_drawdown_better": float(
                    (simulations["max_drawdown_delta"] > 0.0).mean()
                ),
            }
        )
        for strategy in ("baseline", "candidate"):
            drawdown = simulations[f"{strategy}_max_drawdown"]
            tail_rows.append(
                {
                    "sample": sample,
                    "block_days": block_days,
                    "strategy": strategy,
                    "probability_breach_18pct": float((drawdown < -0.18).mean()),
                    "max_drawdown_5pct": float(drawdown.quantile(0.05)),
                    "median_max_drawdown": float(drawdown.median()),
                }
            )
    return pd.DataFrame(relative_rows), pd.DataFrame(tail_rows)


def family_reality_check(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    baseline: pd.Series,
    additional_family: dict[str, str] | None = None,
    selected_name: str = "lev110_target25_jump",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    family = dict(EXISTING_FAMILY)
    family.update(additional_family or ADDITIONAL_FAMILY)
    family.pop("production_baseline", None)
    relative: dict[str, pd.Series] = {}
    rows: list[dict[str, float | int | str]] = []
    for name, directory in family.items():
        frame = simulate_open_execution(
            directory,
            opens,
            closes,
            start_date="2015-01-01",
            end_date="2025-12-31",
            cost_bps=7.5,
        )
        metrics = performance_metrics(frame["net_return"])
        eligible = metrics["max_drawdown"] >= -0.18
        rows.append({"strategy": name, "eligible": int(eligible), **metrics})
        if eligible:
            aligned = pd.concat(
                [frame["net_return"].rename("candidate"), baseline],
                axis=1,
                join="inner",
            ).dropna()
            relative[name] = np.log1p(aligned["candidate"]) - np.log1p(
                aligned["baseline"]
            )
    eligibility = pd.DataFrame(rows).drop_duplicates("strategy").set_index("strategy")
    matrix = pd.DataFrame(relative).dropna()
    observed = matrix.mean(axis=0) * 252.0
    centered = matrix - matrix.mean(axis=0)
    selected_observed = float(observed[selected_name])
    selected_index = matrix.columns.get_loc(selected_name)
    block_days = 21
    samples = 10_000
    count = len(matrix)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    generator = np.random.default_rng(20260723)
    selected_statistics = np.empty(samples)
    maximum_statistics = np.empty(samples)
    values = centered.to_numpy(dtype=float)
    for sample in range(samples):
        starts = generator.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        statistics = values[indices].mean(axis=0) * 252.0
        selected_statistics[sample] = statistics[selected_index]
        maximum_statistics[sample] = statistics.max()
    ranking = observed.rename("annualized_relative_log_return").sort_values(
        ascending=False
    ).to_frame()
    summary = pd.Series(
        {
            "selected_annualized_relative_log_return": selected_observed,
            "nominal_one_sided_p_value": float(
                (selected_statistics >= selected_observed).mean()
            ),
            "familywise_reality_check_p_value": float(
                (maximum_statistics >= selected_observed).mean()
            ),
            "eligible_family_size": int(matrix.shape[1]),
            "selected_rank": int(ranking.index.get_loc(selected_name) + 1),
            "block_days": block_days,
            "bootstrap_samples": samples,
        },
        name=selected_name,
    )
    return eligibility, ranking, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--candidate-2012", default=CANDIDATE_2012)
    parser.add_argument("--destination", default=str(DESTINATION))
    parser.add_argument("--selected-name", default="lev110_target25_jump")
    args = parser.parse_args()
    candidate = str(args.candidate)
    candidate_2012 = str(args.candidate_2012)
    destination = Path(args.destination)
    selected_name = str(args.selected_name)
    additional_family = dict(ADDITIONAL_FAMILY)
    additional_family[selected_name] = candidate

    destination.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(False)
    frames: dict[tuple[str, float], pd.DataFrame] = {}
    metric_data: list[dict[str, float | str]] = []
    for cost_bps in (0.0, 7.5, 15.0):
        for name, directory in (("baseline", BASELINE), ("candidate", candidate)):
            frame = simulate_open_execution(directory, opens, closes, cost_bps=cost_bps)
            frames[(name, cost_bps)] = frame
            metric_data.extend(metric_rows(name, cost_bps, frame["net_return"]))
    metrics = pd.DataFrame(metric_data).set_index(["period", "strategy", "cost_bps"])
    metrics.to_csv(destination / "metrics_by_period_and_cost.csv")

    baseline_extended = simulate_open_execution(
        BASELINE_2012, opens, closes, start_date="2012-01-01", cost_bps=7.5
    )
    candidate_extended = simulate_open_execution(
        candidate_2012, opens, closes, start_date="2012-01-01", cost_bps=7.5
    )
    extended_metrics = pd.DataFrame(
        {
            "baseline": performance_metrics(baseline_extended["net_return"]),
            "candidate": performance_metrics(candidate_extended["net_return"]),
        }
    ).T
    extended_metrics.to_csv(destination / "extended_2012_2025_metrics.csv")

    annual = pd.DataFrame(
        {
            "baseline": frames[("baseline", 7.5)]["net_return"],
            "candidate": frames[("candidate", 7.5)]["net_return"],
        }
    ).loc["2015":"2025"]
    annual_rows = []
    for year, frame in annual.groupby(annual.index.year):
        annual_rows.append(
            {
                "year": int(year),
                "baseline_return": float((1.0 + frame["baseline"]).prod() - 1.0),
                "candidate_return": float((1.0 + frame["candidate"]).prod() - 1.0),
                "return_delta": float(
                    (1.0 + frame["candidate"]).prod()
                    - (1.0 + frame["baseline"]).prod()
                ),
            }
        )
    pd.DataFrame(annual_rows).set_index("year").to_csv(
        destination / "annual_comparison.csv"
    )

    primary_relative, primary_tail = paired_summary(
        frames[("candidate", 7.5)]["net_return"].loc["2015":"2025"],
        frames[("baseline", 7.5)]["net_return"].loc["2015":"2025"],
        "complete_2015_2025_open",
    )
    extended_relative, extended_tail = paired_summary(
        candidate_extended["net_return"],
        baseline_extended["net_return"],
        "extended_2012_2025_open",
    )
    pd.concat([primary_relative, extended_relative], ignore_index=True).to_csv(
        destination / "paired_bootstrap.csv", index=False
    )
    pd.concat([primary_tail, extended_tail], ignore_index=True).to_csv(
        destination / "tail_bootstrap.csv", index=False
    )

    member_rows = []
    for seed in (7, 42, 123):
        baseline_member = simulate_open_execution(
            f"{BASELINE}/members/seed_{seed}", opens, closes, cost_bps=7.5
        )
        candidate_member = simulate_open_execution(
            f"{candidate}/members/seed_{seed}", opens, closes, cost_bps=7.5
        )
        baseline_metrics = performance_metrics(
            baseline_member.loc["2015":"2025", "net_return"]
        )
        candidate_metrics = performance_metrics(
            candidate_member.loc["2015":"2025", "net_return"]
        )
        member_rows.append(
            {
                "seed": seed,
                "baseline_cagr": baseline_metrics["cagr"],
                "candidate_cagr": candidate_metrics["cagr"],
                "cagr_delta": candidate_metrics["cagr"] - baseline_metrics["cagr"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "candidate_sharpe": candidate_metrics["sharpe"],
            }
        )
    pd.DataFrame(member_rows).set_index("seed").to_csv(
        destination / "member_seed_stability.csv"
    )

    eligibility, ranking, reality = family_reality_check(
        opens,
        closes,
        frames[("baseline", 7.5)]["net_return"].rename("baseline"),
        additional_family,
        selected_name,
    )
    eligibility.to_csv(destination / "family_eligibility.csv")
    ranking.to_csv(destination / "family_incremental_ranking.csv")
    reality.to_csv(destination / "family_reality_check.csv")

    primary = metrics.loc[("complete_2015_2025", slice(None), 7.5)]
    development = metrics.loc[("development_2015_2021", slice(None), 7.5)]
    holdout = metrics.loc[("holdout_2022_2025", slice(None), 7.5)]
    cost15 = metrics.loc[("complete_2015_2025", slice(None), 15.0)]
    acceptance = pd.Series(
        {
            "primary_cagr_delta": float(
                primary.loc["candidate", "cagr"] - primary.loc["baseline", "cagr"]
            ),
            "primary_candidate_max_drawdown": float(
                primary.loc["candidate", "max_drawdown"]
            ),
            "development_cagr_delta": float(
                development.loc["candidate", "cagr"]
                - development.loc["baseline", "cagr"]
            ),
            "holdout_cagr_delta": float(
                holdout.loc["candidate", "cagr"] - holdout.loc["baseline", "cagr"]
            ),
            "cost15_cagr_delta": float(
                cost15.loc["candidate", "cagr"] - cost15.loc["baseline", "cagr"]
            ),
            "extended_cagr_delta": float(
                extended_metrics.loc["candidate", "cagr"]
                - extended_metrics.loc["baseline", "cagr"]
            ),
            "cagr_plus_1pp_pass": int(
                primary.loc["candidate", "cagr"]
                >= primary.loc["baseline", "cagr"] + 0.01
            ),
            "drawdown_18pct_pass": int(
                primary.loc["candidate", "max_drawdown"] >= -0.18
            ),
            "development_pass": int(
                development.loc["candidate", "cagr"]
                >= development.loc["baseline", "cagr"]
            ),
            "holdout_pass": int(
                holdout.loc["candidate", "cagr"] >= holdout.loc["baseline", "cagr"]
            ),
            "cost15_pass": int(
                cost15.loc["candidate", "cagr"] >= cost15.loc["baseline", "cagr"]
            ),
            "extended_pass": int(
                extended_metrics.loc["candidate", "cagr"]
                >= extended_metrics.loc["baseline", "cagr"]
            ),
        },
        name="value",
    )
    acceptance["point_estimate_overall_pass"] = int(
        bool(
            acceptance[
                [
                    "cagr_plus_1pp_pass",
                    "drawdown_18pct_pass",
                    "development_pass",
                    "holdout_pass",
                    "cost15_pass",
                    "extended_pass",
                ]
            ].all()
        )
    )
    acceptance.to_csv(destination / "acceptance.csv")

    print("Point-estimate acceptance:")
    print(acceptance.round(6).to_string())
    print("\nPrimary open metrics:")
    print(primary[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nExtended open metrics:")
    print(extended_metrics[["cagr", "sharpe", "max_drawdown"]].round(6))
    print("\nReality check:")
    print(reality.round(6).to_string())
    print(f"\nArtifacts: {destination.resolve()}")


if __name__ == "__main__":
    main()
