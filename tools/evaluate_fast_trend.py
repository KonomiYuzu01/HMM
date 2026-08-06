from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_relative_momentum import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "fast_trend_validation"
BASELINE = "paper_core_growth"
CANDIDATE = "paper_core_growth_floor_fast_trend"
COST_STRESS = "paper_core_growth_floor_fast_trend_cost15"
BASELINE_COST_STRESS = "paper_core_growth_cost15"
COMPARATORS = [
    BASELINE,
    "paper_core_growth_floor",
    "paper_core_growth_floor_all_asset_trend",
    CANDIDATE,
    BASELINE_COST_STRESS,
    COST_STRESS,
]
PERIODS = {
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def load_daily(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def reality_check(
    baseline: pd.Series,
    selected: pd.Series,
    block_days: int = 21,
    samples: int = 5_000,
    seed: int = 20_260_722,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.dropna()
    relative: dict[str, pd.Series] = {}
    for path in sorted(OUTPUT_ROOT.glob("*/daily_returns.csv")):
        name = path.parent.name
        if (
            name == BASELINE
            or name.endswith("_cost15")
            or name.endswith("_validation")
        ):
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if "net_return" not in frame:
            continue
        aligned = pd.concat(
            [frame["net_return"], baseline], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 0.95 * len(baseline):
            continue
        relative[name] = (
            np.log1p(frame["net_return"].reindex(baseline.index))
            - np.log1p(baseline)
        )
    matrix_frame = pd.DataFrame(relative).dropna(axis=1).dropna()
    if CANDIDATE not in matrix_frame:
        raise RuntimeError("Selected candidate is absent from the Reality Check family")
    matrix = matrix_frame.to_numpy(dtype=float)
    observed = matrix.mean(axis=0) * 252
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    count = len(centered)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    selected_index = matrix_frame.columns.get_loc(CANDIDATE)
    selected_observed = observed[selected_index]
    maximum_statistics = np.empty(samples)
    selected_statistics = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:count] % count
        statistic = centered[indices].mean(axis=0) * 252
        maximum_statistics[sample] = statistic.max()
        selected_statistics[sample] = statistic[selected_index]

    ranking = pd.DataFrame(
        {
            "annualized_relative_log_return": observed,
        },
        index=matrix_frame.columns,
    ).sort_values("annualized_relative_log_return", ascending=False)
    summary = pd.DataFrame(
        [
            {
                "selected_annualized_relative_log_return": selected_observed,
                "nominal_one_sided_p_value": float(
                    (selected_statistics >= selected_observed).mean()
                ),
                "familywise_reality_check_p_value": float(
                    (maximum_statistics >= selected_observed).mean()
                ),
                "candidate_family_size": matrix.shape[1],
                "selected_rank_by_relative_return": int(
                    ranking.index.get_loc(CANDIDATE) + 1
                ),
                "block_days": block_days,
                "bootstrap_samples": samples,
            }
        ],
        index=[CANDIDATE],
    )
    return summary, ranking


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {name: load_daily(name) for name in COMPARATORS}
    rows: list[dict[str, float | str]] = []
    for strategy, frame in daily.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(frame.loc[start:end, "net_return"]),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    baseline_complete = metrics.loc[(BASELINE, "complete_2015_2025")]
    candidate_complete = metrics.loc[(CANDIDATE, "complete_2015_2025")]
    stress_complete = metrics.loc[(COST_STRESS, "complete_2015_2025")]
    baseline_stress_complete = metrics.loc[
        (BASELINE_COST_STRESS, "complete_2015_2025")
    ]
    baseline_development = metrics.loc[(BASELINE, "development_2015_2021")]
    candidate_development = metrics.loc[(CANDIDATE, "development_2015_2021")]
    baseline_holdout = metrics.loc[(BASELINE, "holdout_2022_2025")]
    candidate_holdout = metrics.loc[(CANDIDATE, "holdout_2022_2025")]
    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr": candidate_complete["cagr"],
                "cagr_delta_vs_baseline": candidate_complete["cagr"]
                - baseline_complete["cagr"],
                "complete_max_drawdown": candidate_complete["max_drawdown"],
                "complete_sharpe_delta": candidate_complete["sharpe"]
                - baseline_complete["sharpe"],
                "development_cagr_delta": candidate_development["cagr"]
                - baseline_development["cagr"],
                "holdout_cagr_delta": candidate_holdout["cagr"]
                - baseline_holdout["cagr"],
                "cost15_complete_cagr": stress_complete["cagr"],
                "cost15_cagr_delta_vs_cost15_baseline": stress_complete["cagr"]
                - baseline_stress_complete["cagr"],
                "cagr_plus_1pp_pass": int(
                    candidate_complete["cagr"] >= baseline_complete["cagr"] + 0.01
                ),
                "drawdown_18pct_pass": int(
                    candidate_complete["max_drawdown"] >= -0.18
                ),
                "development_no_degradation_pass": int(
                    candidate_development["cagr"] >= baseline_development["cagr"]
                ),
                "holdout_no_degradation_pass": int(
                    candidate_holdout["cagr"] >= baseline_holdout["cagr"]
                ),
                "cost15_cagr_plus_1pp_pass": int(
                    stress_complete["cagr"]
                    >= baseline_stress_complete["cagr"] + 0.01
                ),
                "cost15_drawdown_18pct_pass": int(
                    stress_complete["max_drawdown"] >= -0.18
                ),
            }
        ],
        index=[CANDIDATE],
    )
    pass_columns = [column for column in acceptance if column.endswith("_pass")]
    acceptance["overall_pass"] = acceptance[pass_columns].all(axis=1).astype(int)
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    sample = slice("2015-01-01", "2025-12-31")
    bootstrap = pd.DataFrame(
        {
            "vs_baseline": circular_block_bootstrap(
                daily[CANDIDATE].loc[sample, "net_return"],
                daily[BASELINE].loc[sample, "net_return"],
            ),
            "vs_growth_floor": circular_block_bootstrap(
                daily[CANDIDATE].loc[sample, "net_return"],
                daily["paper_core_growth_floor"].loc[sample, "net_return"],
            ),
        }
    ).T
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    annual = pd.concat(
        {
            "baseline": daily[BASELINE].loc[sample, "net_return"],
            "candidate": daily[CANDIDATE].loc[sample, "net_return"],
            "cost15": daily[COST_STRESS].loc[sample, "net_return"],
            "baseline_cost15": daily[BASELINE_COST_STRESS].loc[
                sample, "net_return"
            ],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_baseline"] = annual["candidate"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    baseline_sample = daily[BASELINE].loc[sample, "net_return"]
    reality, ranking = reality_check(
        baseline_sample,
        daily[CANDIDATE].loc[sample, "net_return"],
    )
    reality.to_csv(DESTINATION / "multiple_testing_reality_check.csv")
    ranking.to_csv(DESTINATION / "candidate_family_ranking.csv")

    diagnostics = pd.DataFrame(
        [
            {
                "annual_wins_vs_baseline": int(
                    (annual["candidate_minus_baseline"] > 0.0).sum()
                ),
                "annual_periods": len(annual),
                "annualized_average_cost": float(
                    daily[CANDIDATE].loc[sample, "cost"].mean() * 252
                ),
                "cost15_annualized_average_cost": float(
                    daily[COST_STRESS].loc[sample, "cost"].mean() * 252
                ),
            }
        ],
        index=[CANDIDATE],
    )
    diagnostics.to_csv(DESTINATION / "signal_diagnostics.csv")

    print(metrics.round(4).to_string())
    print("\nAcceptance:")
    print(acceptance.round(4).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(4).to_string())
    print("\nMultiple-testing Reality Check:")
    print(reality.round(4).to_string())
    print("\nAnnual returns:")
    print(annual.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
