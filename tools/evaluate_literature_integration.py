from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_drawdown_uncertainty import paired_circular_block_bootstrap
from evaluate_open_execution import load_open_close, simulate_open_execution
from evaluate_open_family_reality_check import STRATEGIES as EXISTING_FAMILY
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "literature_integration_validation"
BASELINE = "paper_core_growth_gold20_daily_risk_ensemble"
CANDIDATE = "paper_core_growth_gold20_residual_momentum_ensemble"
BASELINE_2012 = "paper_core_growth_gold20_daily_risk_ensemble_2012"
CANDIDATE_2012 = "paper_core_growth_gold20_residual_momentum_ensemble_2012"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def load_reported(directory: str, start: str, end: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    ).loc[start:end, "net_return"]


def metric_rows(
    name: str,
    execution: str,
    returns: pd.Series,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for period, (start, end) in PERIODS.items():
        rows.append(
            {
                "strategy": name,
                "execution": execution,
                "period": period,
                **performance_metrics(returns.loc[start:end]),
            }
        )
    return rows


def paired_summary(
    candidate: pd.Series,
    baseline: pd.Series,
    sample: str,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
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
                "cagr_delta_95pct": float(simulations["cagr_delta"].quantile(0.95)),
                "probability_candidate_drawdown_better": float(
                    (simulations["max_drawdown_delta"] > 0.0).mean()
                ),
                "median_drawdown_delta": float(
                    simulations["max_drawdown_delta"].median()
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
    return relative_rows, tail_rows


def family_reality_check(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    family = dict(EXISTING_FAMILY)
    family["residual_momentum"] = CANDIDATE
    spy = closes["SPX"].pct_change(fill_method=None).loc["2015":"2025"]
    relative: dict[str, pd.Series] = {}
    rows = []
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
                [frame["net_return"].rename("candidate"), spy.rename("SPY")],
                axis=1,
                join="inner",
            ).dropna()
            relative[name] = np.log1p(aligned["candidate"]) - np.log1p(
                aligned["SPY"]
            )
    eligibility = pd.DataFrame(rows).set_index("strategy")
    matrix = pd.DataFrame(relative).dropna()
    observed = matrix.mean(axis=0).to_numpy(dtype=float) * 252.0
    centered = matrix.to_numpy(dtype=float).copy()
    centered -= centered.mean(axis=0, keepdims=True)
    selected_index = matrix.columns.get_loc("residual_momentum")
    selected_observed = float(observed[selected_index])
    block_days = 21
    samples = 10_000
    count = len(matrix)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    generator = np.random.default_rng(20260723)
    selected_statistics = np.empty(samples)
    maximum_statistics = np.empty(samples)
    for sample in range(samples):
        starts = generator.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        statistic = centered[indices].mean(axis=0) * 252.0
        selected_statistics[sample] = statistic[selected_index]
        maximum_statistics[sample] = statistic.max()
    summary = pd.Series(
        {
            "annualized_relative_log_return_vs_spy": selected_observed,
            "nominal_one_sided_p_value": float(
                (selected_statistics >= selected_observed).mean()
            ),
            "familywise_reality_check_p_value": float(
                (maximum_statistics >= selected_observed).mean()
            ),
            "eligible_family_size": int(matrix.shape[1]),
            "rank_within_eligible_family": int(
                (-pd.Series(observed, index=matrix.columns))
                .rank(method="min")
                .loc["residual_momentum"]
            ),
        },
        name="residual_momentum",
    )
    return eligibility, summary


def mechanism_diagnostics() -> pd.Series:
    frames = []
    for seed in (7, 42, 123):
        frame = pd.read_csv(
            OUTPUT / CANDIDATE / "members" / f"seed_{seed}" / "regimes.csv",
            index_col=0,
            parse_dates=True,
        ).loc["2015":"2025"]
        frames.append(frame)
    diagnostics = pd.concat(frames)
    valid = diagnostics.dropna(subset=["residual_momentum_score"])
    negative = valid["residual_momentum_score"] < 0.0
    return pd.Series(
        {
            "observations": len(valid),
            "negative_signal_share": float(negative.mean()),
            "median_beta": float(valid["residual_momentum_beta"].median()),
            "beta_10pct": float(valid["residual_momentum_beta"].quantile(0.10)),
            "beta_90pct": float(valid["residual_momentum_beta"].quantile(0.90)),
            "mean_controlled_weight_when_negative": float(
                valid.loc[negative, "residual_momentum_controlled_weight"].mean()
            ),
            "mean_score_when_negative": float(
                valid.loc[negative, "residual_momentum_score"].mean()
            ),
        },
        name="value",
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(False)

    baseline_open = simulate_open_execution(BASELINE, opens, closes)
    candidate_open = simulate_open_execution(CANDIDATE, opens, closes)
    baseline_cost15_open = simulate_open_execution(
        BASELINE, opens, closes, cost_bps=15.0
    )
    candidate_cost15_open = simulate_open_execution(
        CANDIDATE, opens, closes, cost_bps=15.0
    )
    baseline_extended_open = simulate_open_execution(
        BASELINE_2012, opens, closes, start_date="2012-01-01"
    )
    candidate_extended_open = simulate_open_execution(
        CANDIDATE_2012, opens, closes, start_date="2012-01-01"
    )
    for name, frame in {
        "baseline_open": baseline_open,
        "candidate_open": candidate_open,
        "baseline_cost15_open": baseline_cost15_open,
        "candidate_cost15_open": candidate_cost15_open,
        "baseline_extended_open": baseline_extended_open,
        "candidate_extended_open": candidate_extended_open,
    }.items():
        frame.to_csv(DESTINATION / f"{name}_daily.csv")

    rows: list[dict[str, float | str]] = []
    for name, directory in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        rows.extend(
            metric_rows(
                name,
                "reported_close_to_close",
                load_reported(directory, "2015-01-01", "2025-12-31"),
            )
        )
    rows.extend(metric_rows("baseline", "open_proxy", baseline_open["net_return"]))
    rows.extend(metric_rows("candidate", "open_proxy", candidate_open["net_return"]))
    rows.extend(
        metric_rows(
            "baseline_cost15", "open_proxy", baseline_cost15_open["net_return"]
        )
    )
    rows.extend(
        metric_rows(
            "candidate_cost15", "open_proxy", candidate_cost15_open["net_return"]
        )
    )
    metrics = pd.DataFrame(rows).set_index(["strategy", "execution", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    extended_metrics = pd.DataFrame(
        {
            "baseline": performance_metrics(baseline_extended_open["net_return"]),
            "candidate": performance_metrics(candidate_extended_open["net_return"]),
        }
    ).T
    extended_metrics.to_csv(DESTINATION / "extended_2012_2025_metrics.csv")

    annual = pd.DataFrame(
        {
            "baseline": baseline_open["net_return"],
            "candidate": candidate_open["net_return"],
        }
    )
    annual_rows = []
    for year, frame in annual.groupby(annual.index.year):
        baseline_metrics = performance_metrics(frame["baseline"])
        candidate_metrics = performance_metrics(frame["candidate"])
        annual_rows.append(
            {
                "year": int(year),
                "baseline_return": float((1.0 + frame["baseline"]).prod() - 1.0),
                "candidate_return": float((1.0 + frame["candidate"]).prod() - 1.0),
                "return_delta": float(
                    (1.0 + frame["candidate"]).prod()
                    - (1.0 + frame["baseline"]).prod()
                ),
                "baseline_max_drawdown": baseline_metrics["max_drawdown"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
            }
        )
    pd.DataFrame(annual_rows).set_index("year").to_csv(
        DESTINATION / "annual_comparison.csv"
    )

    relative_rows, tail_rows = paired_summary(
        candidate_open["net_return"],
        baseline_open["net_return"],
        "complete_2015_2025_open",
    )
    extended_relative, extended_tail = paired_summary(
        candidate_extended_open["net_return"],
        baseline_extended_open["net_return"],
        "extended_2012_2025_open",
    )
    pd.DataFrame(relative_rows + extended_relative).to_csv(
        DESTINATION / "paired_bootstrap.csv", index=False
    )
    pd.DataFrame(tail_rows + extended_tail).to_csv(
        DESTINATION / "tail_bootstrap.csv", index=False
    )

    eligibility, reality = family_reality_check(opens, closes)
    eligibility.to_csv(DESTINATION / "family_eligibility.csv")
    reality.to_csv(DESTINATION / "family_reality_check.csv")
    diagnostics = mechanism_diagnostics()
    diagnostics.to_csv(DESTINATION / "mechanism_diagnostics.csv")

    print(metrics[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nExtended open proxy:")
    print(extended_metrics[["cagr", "sharpe", "max_drawdown"]].round(6))
    print("\nMechanism:")
    print(diagnostics.round(6).to_string())
    print("\nReality check:")
    print(reality.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
