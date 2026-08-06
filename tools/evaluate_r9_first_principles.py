from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "r9_first_principles_validation"
VARIANTS = {
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
    "broad50": "experiment_r9_broad50",
    "stage30_d5": "experiment_r9_broad50_stage30_d5",
    "stage35_d3": "experiment_r9_broad50_stage35_d3",
    "stage35_d5": "experiment_r9_broad50_stage35_d5",
    "stage35_d10": "experiment_r9_broad50_stage35_d10",
    "stage45_d5": "experiment_r9_broad50_stage45_d5",
}
NORMAL_PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
PROXY_PERIODS = {
    "early_2006_2014": ("2006-08-01", "2014-12-31"),
    "late_2015_2025": ("2015-01-01", "2025-12-31"),
    "complete_proxy": ("2006-08-01", "2025-12-31"),
}


def proxy_directory(directory: str) -> str:
    return f"{directory}_20y_proxy"


def returns(directory: str, member: str | None = None) -> pd.Series:
    root = OUTPUT / directory
    if member is not None:
        root = root / "members" / member
    return pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]


def metrics_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, directory in VARIANTS.items():
        for sample, source, periods in (
            ("normal", directory, NORMAL_PERIODS),
            ("proxy", proxy_directory(directory), PROXY_PERIODS),
        ):
            series = returns(source)
            for period, (start, end) in periods.items():
                rows.append(
                    {
                        "variant": variant,
                        "sample": sample,
                        "period": period,
                        **performance_metrics(series.loc[start:end]),
                    }
                )
    result = pd.DataFrame(rows).set_index(["variant", "sample", "period"])
    deltas: list[float] = []
    for (variant, sample, period), row in result.iterrows():
        baseline = result.loc[("r8", sample, period), "cagr"]
        deltas.append(float(row["cagr"] - baseline))
    result["cagr_delta_vs_r8"] = deltas
    return result


def seed_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, directory in VARIANTS.items():
        for sample, source, start, end in (
            ("normal", directory, "2015-01-01", "2025-12-31"),
            (
                "proxy",
                proxy_directory(directory),
                "2006-08-01",
                "2025-12-31",
            ),
        ):
            for member in ("seed_7", "seed_42", "seed_123"):
                values = performance_metrics(
                    returns(source, member).loc[start:end]
                )
                rows.append(
                    {
                        "variant": variant,
                        "sample": sample,
                        "member": member,
                        **values,
                    }
                )
    return pd.DataFrame(rows).set_index(["variant", "sample", "member"])


def block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_725,
) -> pd.Series:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    count = len(relative)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[sample] = float(relative[indices].mean() * 252.0)
    return pd.Series(
        {
            "annualized_relative_log_return": float(
                relative.mean() * 252.0
            ),
            "lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)),
            "probability_positive": float((estimates > 0.0).mean()),
        }
    )


def bootstrap_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample, directory_transform, start, end in (
        ("normal", lambda value: value, "2015-01-01", "2025-12-31"),
        (
            "proxy",
            proxy_directory,
            "2006-08-01",
            "2025-12-31",
        ),
    ):
        baseline = returns(directory_transform(VARIANTS["r8"])).loc[start:end]
        for variant, directory in VARIANTS.items():
            if variant == "r8":
                continue
            values = block_bootstrap(
                returns(directory_transform(directory)).loc[start:end],
                baseline,
            )
            rows.append(
                {"variant": variant, "sample": sample, **values.to_dict()}
            )
    return pd.DataFrame(rows).set_index(["variant", "sample"])


def family_reality_check(
    sample: str,
    directory_transform,
    start: str,
    end: str,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_726,
) -> pd.Series:
    baseline = returns(directory_transform(VARIANTS["r8"])).loc[start:end]
    relative_columns: list[pd.Series] = []
    for variant, directory in VARIANTS.items():
        if variant == "r8":
            continue
        aligned = pd.concat(
            [
                returns(directory_transform(directory)).loc[start:end],
                baseline,
            ],
            axis=1,
            join="inner",
        ).dropna()
        relative_columns.append(
            (
                np.log1p(aligned.iloc[:, 0])
                - np.log1p(aligned.iloc[:, 1])
            ).rename(variant)
        )
    relative = pd.concat(relative_columns, axis=1, join="inner").dropna()
    observed = relative.mean(axis=0) * 252.0
    centered = relative - relative.mean(axis=0)
    values = centered.to_numpy(dtype=float)
    count = len(values)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    maxima = np.empty(samples)
    for sample_index in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        maxima[sample_index] = float(
            (values[indices].mean(axis=0) * 252.0).max()
        )
    return pd.Series(
        {
            "sample": sample,
            "tested_candidates": relative.shape[1],
            "best_variant": str(observed.idxmax()),
            "best_annualized_relative_log_return": float(observed.max()),
            "reality_check_p_value": float(
                (maxima >= observed.max()).mean()
            ),
        }
    )


def event_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for variant, directory in VARIANTS.items():
        analysis_directory = (
            "r8_defense_event_analysis"
            if variant == "r8"
            else f"{proxy_directory(directory)}_event_analysis"
        )
        path = OUTPUT / analysis_directory / "event_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["variant"] = variant
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).set_index(["variant", "event"])


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in events.index.get_level_values("variant").unique():
        frame = events.xs(variant, level="variant")
        rows.append(
            {
                "variant": variant,
                "median_rebound_capture": float(
                    frame["rebound_capture_ratio"].median()
                ),
                "minimum_rebound_capture": float(
                    frame["rebound_capture_ratio"].min()
                ),
                "median_strategy_event_drawdown": float(
                    frame["strategy_window_max_drawdown"].median()
                ),
                "worst_strategy_event_drawdown": float(
                    frame["strategy_window_max_drawdown"].min()
                ),
                "total_zero_sessions": int(
                    frame["zero_weight_sessions_peak_to_recovery"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def relative_log_return(
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.Series:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    return (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    )


def concentration_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    for sample, directory_transform, start, end in (
        ("normal", lambda value: value, "2015-01-01", "2025-12-31"),
        (
            "proxy",
            proxy_directory,
            "2006-08-01",
            "2025-12-31",
        ),
    ):
        baseline = returns(directory_transform(VARIANTS["r8"])).loc[start:end]
        candidate = returns(
            directory_transform(VARIANTS["stage35_d10"])
        ).loc[start:end]
        relative = relative_log_return(candidate, baseline)
        total = float(relative.sum())
        for year, values in relative.groupby(relative.index.year):
            excluded = relative.loc[relative.index.year != year]
            annual_rows.append(
                {
                    "sample": sample,
                    "excluded_year": int(year),
                    "excluded_relative_log_return": float(values.sum()),
                    "share_of_total_relative_log_return": (
                        float(values.sum() / total)
                        if not np.isclose(total, 0.0)
                        else np.nan
                    ),
                    "remaining_annualized_relative_log_return": float(
                        excluded.mean() * 252.0
                    ),
                    "remaining_positive": bool(excluded.mean() > 0.0),
                }
            )
        first_year = int(relative.index.year.min())
        last_year = int(relative.index.year.max())
        for window_start in range(first_year, last_year - 1):
            window_end = window_start + 2
            values = relative.loc[
                (relative.index.year >= window_start)
                & (relative.index.year <= window_end)
            ]
            if values.empty:
                continue
            rolling_rows.append(
                {
                    "sample": sample,
                    "window_start": window_start,
                    "window_end": window_end,
                    "annualized_relative_log_return": float(
                        values.mean() * 252.0
                    ),
                    "positive": bool(values.mean() > 0.0),
                }
            )
    annual = pd.DataFrame(annual_rows).set_index(
        ["sample", "excluded_year"]
    )
    rolling = pd.DataFrame(rolling_rows).set_index(
        ["sample", "window_start", "window_end"]
    )
    return annual, rolling


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = metrics_table()
    seeds = seed_table()
    bootstrap = bootstrap_table()
    reality = pd.DataFrame(
        [
            family_reality_check(
                "normal",
                lambda value: value,
                "2015-01-01",
                "2025-12-31",
            ),
            family_reality_check(
                "proxy",
                proxy_directory,
                "2006-08-01",
                "2025-12-31",
            ),
        ]
    ).set_index("sample")
    events = event_table()
    event_aggregate = event_summary(events)
    leave_one_year_out, rolling_three_year = concentration_tables()

    metrics.to_csv(DESTINATION / "metrics_by_period.csv")
    seeds.to_csv(DESTINATION / "seed_consistency.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")
    reality.to_csv(DESTINATION / "family_reality_check.csv")
    events.to_csv(DESTINATION / "event_metrics.csv")
    event_aggregate.to_csv(DESTINATION / "event_summary.csv")
    leave_one_year_out.to_csv(
        DESTINATION / "leave_one_year_out.csv"
    )
    rolling_three_year.to_csv(
        DESTINATION / "rolling_three_year.csv"
    )

    print("Complete samples:")
    print(
        pd.concat(
            [
                metrics.xs(
                    ("normal", "complete_2015_2025"),
                    level=("sample", "period"),
                ).assign(sample="normal"),
                metrics.xs(
                    ("proxy", "complete_proxy"),
                    level=("sample", "period"),
                ).assign(sample="proxy"),
            ]
        )[
            [
                "sample",
                "cagr",
                "sharpe",
                "max_drawdown",
                "cagr_delta_vs_r8",
            ]
        ].round(6).to_string()
    )
    print("\nEvent summary:")
    print(event_aggregate.round(6).to_string())
    print("\nFamily reality check:")
    print(reality.round(6).to_string())
    print("\nLeave-one-year-out:")
    print(
        leave_one_year_out.groupby(level="sample")
        .agg(
            minimum_remaining_annualized_relative_log_return=(
                "remaining_annualized_relative_log_return",
                "min",
            ),
            all_remaining_positive=("remaining_positive", "all"),
            maximum_single_year_share=(
                "share_of_total_relative_log_return",
                "max",
            ),
        )
        .round(6)
        .to_string()
    )
    print("\nRolling three-year windows:")
    print(
        rolling_three_year.groupby(level="sample")
        .agg(
            minimum_annualized_relative_log_return=(
                "annualized_relative_log_return",
                "min",
            ),
            positive_windows=("positive", "sum"),
            total_windows=("positive", "count"),
        )
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
