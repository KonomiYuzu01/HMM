from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output/strategy_library_frontier_spa")
BASELINE = Path(
    "output/r32_proportional_trend/"
    "normal_synthetic_r11_baseline_daily.csv"
)
MINIMUM_COVERAGE = 0.95
BOOTSTRAP_SAMPLES = 2_000
BLOCK_DAYS = (21, 63, 126)
EXCLUDED_FRAGMENTS = (
    "proxy",
    "normal_live",
    "cost_stress",
    "stress",
    "cost15",
    "20y",
    "_2012",
    "vx_contract",
)
EXCLUDED_BASELINE_FRAGMENTS = (
    "_r11_baseline_daily.csv",
    "/baseline_daily.csv",
)
TARGET_PATHS = {
    "r11_risk1070_gde10": Path(
        "output/r11_confirmatory_financing_mix/"
        "normal_synthetic_risk1.070_gde10_daily.csv"
    ),
    "r23": Path(
        "output/r23_high_volatility_tilt_gate/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r24": Path(
        "output/r24_declared_cash_hard_limit/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r25": Path(
        "output/r25_daily_relative_tilt_risk_veto/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r26": Path(
        "output/r26_pulse_relative_tilt_risk_veto/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r31": Path(
        "output/r31_proportional_trend_risk_pulse/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r32": Path(
        "output/r32_proportional_trend/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r33": Path(
        "output/r33_one_session_unlevered_shock_brake/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r34": Path(
        "output/r34_concave_trend_risk_schedule/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r35": Path(
        "output/r35_stronger_concave_trend_risk/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r36": Path(
        "output/r36_more_concave_trend_risk/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r38": Path(
        "output/r38_convex_semiconductor_overlay/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r38_active135": Path(
        "output/r38_active135_capacity_fill/"
        "normal_synthetic_candidate_daily.csv"
    ),
    "r38_accel1375": Path(
        "output/r38_accelerating_volatility_capacity_fill_1375/"
        "normal_synthetic_candidate_daily.csv"
    ),
}


def annualized_cagr(returns: pd.Series) -> float:
    values = returns.astype(float)
    return float(
        np.exp(np.log1p(values).sum() * 252.0 / len(values)) - 1.0
    )


def maximum_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _has_net_return(path: Path) -> bool:
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    return "net_return" in header


def path_is_eligible_name(path: Path) -> bool:
    lowered = str(path).lower()
    if any(fragment in lowered for fragment in EXCLUDED_FRAGMENTS):
        return False
    if any(
        fragment in lowered
        for fragment in EXCLUDED_BASELINE_FRAGMENTS
    ):
        return False
    return True


def load_return_series(path: Path) -> pd.Series:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "net_return" not in header:
        raise ValueError(f"{path} has no net_return column")
    if "date" in header:
        date_column = "date"
    else:
        date_column = header[0]
    frame = pd.read_csv(
        path,
        usecols=[date_column, "net_return"],
    )
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame["net_return"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates)
    series = series.loc[series.index.notna()]
    series = series.loc[~series.index.duplicated(keep="last")]
    return series.sort_index().astype(float)


def aligned_candidate(
    candidate: pd.Series,
    baseline: pd.Series,
) -> tuple[pd.Series, float]:
    aligned = candidate.reindex(baseline.index)
    coverage = float(aligned.notna().mean())
    return aligned.fillna(baseline), coverage


def path_digest(values: pd.Series) -> str:
    rounded = np.round(values.to_numpy(dtype=np.float64), 12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def circular_block_components(
    relative: np.ndarray,
    block_days: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if relative.ndim != 2:
        raise ValueError("relative must be observations by candidates")
    observations = relative.shape[0]
    if not 1 <= block_days <= observations:
        raise ValueError("block_days must fit the observations")
    extended = np.vstack(
        [relative, relative[: block_days - 1]]
    )
    cumulative = np.vstack(
        [
            np.zeros((1, relative.shape[1]), dtype=float),
            np.cumsum(extended, axis=0),
        ]
    )
    block_sums = (
        cumulative[block_days : block_days + observations]
        - cumulative[:observations]
    )
    full_blocks, remainder = divmod(observations, block_days)
    if remainder:
        remainder_sums = (
            cumulative[remainder : remainder + observations]
            - cumulative[:observations]
        )
    else:
        remainder_sums = np.empty(
            (0, relative.shape[1]),
            dtype=float,
        )
    return block_sums, remainder_sums, full_blocks, remainder


def circular_block_standard_error(
    relative: np.ndarray,
    block_days: int,
) -> np.ndarray:
    (
        block_sums,
        remainder_sums,
        full_blocks,
        remainder,
    ) = circular_block_components(relative, block_days)
    observations = relative.shape[0]
    variance = full_blocks * block_sums.var(axis=0, ddof=0)
    if remainder:
        variance += remainder_sums.var(axis=0, ddof=0)
    return np.sqrt(variance) / observations


def studentized_spa(
    relative: np.ndarray,
    target_indices: dict[str, int],
    block_days: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 20_260_729,
    batch_size: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations, candidates = relative.shape
    means = relative.mean(axis=0)
    standard_error = circular_block_standard_error(
        relative,
        block_days,
    )
    valid = np.isfinite(standard_error) & (standard_error > 1e-15)
    observed_t = np.full(candidates, np.nan)
    observed_t[valid] = means[valid] / standard_error[valid]
    consistency_threshold = -np.sqrt(
        2.0 * np.log(np.log(observations))
    )
    consistent = valid & (observed_t >= consistency_threshold)
    if not consistent.any():
        raise ValueError("No valid candidates for SPA")
    (
        block_sums,
        remainder_sums,
        full_blocks,
        remainder,
    ) = circular_block_components(relative, block_days)
    blocks_per_sample = full_blocks + int(remainder > 0)
    generator = np.random.default_rng(seed + block_days)
    null_maximum = np.empty(samples, dtype=float)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        starts = generator.integers(
            0,
            observations,
            size=(stop - start, blocks_per_sample),
        )
        totals = block_sums[
            starts[:, :full_blocks]
        ].sum(axis=1)
        if remainder:
            totals += remainder_sums[starts[:, -1]]
        bootstrap_means = totals / observations
        centered = bootstrap_means - means
        statistics = centered[:, consistent] / standard_error[
            consistent
        ]
        null_maximum[start:stop] = np.max(statistics, axis=1)

    rows: list[dict[str, object]] = []
    for name, index in target_indices.items():
        statistic = float(observed_t[index])
        p_value = float(
            (
                np.count_nonzero(null_maximum >= statistic) + 1
            )
            / (samples + 1)
        )
        rows.append(
            {
                "target": name,
                "block_days": block_days,
                "observations": observations,
                "unique_candidate_paths": candidates,
                "consistent_null_candidates": int(
                    consistent.sum()
                ),
                "bootstrap_samples": samples,
                "annualized_relative_log_return": float(
                    means[index] * 252.0
                ),
                "studentized_statistic": statistic,
                "familywise_spa_p_value": p_value,
                "null_maximum_95pct": float(
                    np.quantile(null_maximum, 0.95)
                ),
            }
        )
    ranking = pd.DataFrame(
        {
            "candidate_index": np.arange(candidates),
            "annualized_relative_log_return": means * 252.0,
            "studentized_statistic": observed_t,
            "consistent_null_candidate": consistent,
        }
    )
    return pd.DataFrame(rows), ranking


def pareto_mask(cagr: np.ndarray, drawdown: np.ndarray) -> np.ndarray:
    keep = np.ones(len(cagr), dtype=bool)
    for index in range(len(cagr)):
        dominates = (
            (cagr >= cagr[index])
            & (drawdown >= drawdown[index])
            & (
                (cagr > cagr[index])
                | (drawdown > drawdown[index])
            )
        )
        dominates[index] = False
        if dominates.any():
            keep[index] = False
    return keep


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = load_return_series(BASELINE)
    eligible_rows: list[dict[str, object]] = []
    unique_values: list[np.ndarray] = []
    unique_rows: list[dict[str, object]] = []
    digest_to_index: dict[str, int] = {}
    path_to_index: dict[str, int] = {}
    scanned = 0
    header_matches = 0
    for path in sorted(Path("output").rglob("*.csv")):
        scanned += 1
        if path == BASELINE or not path_is_eligible_name(path):
            continue
        if not _has_net_return(path):
            continue
        header_matches += 1
        try:
            candidate = load_return_series(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        aligned, coverage = aligned_candidate(candidate, baseline)
        if coverage < MINIMUM_COVERAGE:
            continue
        digest = path_digest(aligned)
        if digest not in digest_to_index:
            digest_to_index[digest] = len(unique_values)
            unique_values.append(aligned.to_numpy(dtype=float))
            unique_rows.append(
                {
                    "candidate_index": len(unique_values) - 1,
                    "representative_path": str(path),
                    "digest": digest,
                    "duplicate_paths": 0,
                }
            )
        candidate_index = digest_to_index[digest]
        unique_rows[candidate_index]["duplicate_paths"] = int(
            unique_rows[candidate_index]["duplicate_paths"]
        ) + 1
        path_to_index[str(path)] = candidate_index
        eligible_rows.append(
            {
                "path": str(path),
                "coverage": coverage,
                "candidate_index": candidate_index,
                "digest": digest,
            }
        )
    if not unique_values:
        raise RuntimeError("No eligible strategy paths found")
    matrix = np.column_stack(unique_values)
    relative = np.log1p(matrix) - np.log1p(
        baseline.to_numpy(dtype=float)
    )[:, None]

    unique = pd.DataFrame(unique_rows)
    metrics: list[dict[str, object]] = []
    years = baseline.index.year
    development = baseline.index <= "2021-12-31"
    holdout = (baseline.index >= "2022-01-01") & (
        baseline.index <= "2025-12-31"
    )
    for index in range(matrix.shape[1]):
        returns = pd.Series(matrix[:, index], index=baseline.index)
        relative_series = pd.Series(
            relative[:, index],
            index=baseline.index,
        )
        annual = relative_series.groupby(years).sum()
        metrics.append(
            {
                "candidate_index": index,
                "cagr": annualized_cagr(returns),
                "max_drawdown": maximum_drawdown(returns),
                "annualized_relative_log_return": float(
                    relative_series.mean() * 252.0
                ),
                "development_relative_log_return": float(
                    relative_series.loc[development].mean() * 252.0
                ),
                "holdout_relative_log_return": float(
                    relative_series.loc[holdout].mean() * 252.0
                ),
                "positive_years_2015_2021": int(
                    annual.loc[2015:2021].gt(0.0).sum()
                ),
                "positive_years_2022_2025": int(
                    annual.loc[2022:2025].gt(0.0).sum()
                ),
            }
        )
    metrics_frame = pd.DataFrame(metrics)
    unique = unique.merge(
        metrics_frame,
        on="candidate_index",
        how="left",
    )
    unique["point_target_pass"] = (
        unique["cagr"].ge(0.25)
        & unique["max_drawdown"].ge(
            maximum_drawdown(baseline) - 0.005
        )
    )
    unique["year_breadth_pass"] = (
        unique["positive_years_2015_2021"].ge(5)
        & unique["positive_years_2022_2025"].ge(3)
    )
    unique["pareto_frontier"] = pareto_mask(
        unique["cagr"].to_numpy(),
        unique["max_drawdown"].to_numpy(),
    )

    target_indices: dict[str, int] = {}
    target_resolution: list[dict[str, object]] = []
    for name, path in TARGET_PATHS.items():
        index = path_to_index.get(str(path))
        target_resolution.append(
            {
                "target": name,
                "path": str(path),
                "resolved": index is not None,
                "candidate_index": index,
            }
        )
        if index is not None:
            target_indices[name] = index
    if not target_indices:
        raise RuntimeError("No target strategy resolved in the library")

    spa_rows: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    for block_days in BLOCK_DAYS:
        spa, ranking = studentized_spa(
            relative,
            target_indices,
            block_days,
        )
        spa_rows.append(spa)
        ranking["block_days"] = block_days
        rank_frames.append(ranking)
    spa = pd.concat(spa_rows, ignore_index=True)
    ranking = pd.concat(rank_frames, ignore_index=True)
    ranking = ranking.merge(
        unique[
            [
                "candidate_index",
                "representative_path",
                "cagr",
                "max_drawdown",
                "point_target_pass",
                "year_breadth_pass",
            ]
        ],
        on="candidate_index",
        how="left",
    )
    top = (
        ranking.sort_values(
            ["block_days", "studentized_statistic"],
            ascending=[True, False],
        )
        .groupby("block_days", as_index=False)
        .head(25)
    )

    pd.DataFrame(eligible_rows).to_csv(
        OUTPUT / "eligible_paths.csv",
        index=False,
    )
    unique.sort_values(
        ["point_target_pass", "cagr"],
        ascending=[False, False],
    ).to_csv(OUTPUT / "unique_paths.csv", index=False)
    unique.loc[unique["pareto_frontier"]].sort_values(
        "cagr"
    ).to_csv(OUTPUT / "pareto_frontier.csv", index=False)
    pd.DataFrame(target_resolution).to_csv(
        OUTPUT / "target_resolution.csv",
        index=False,
    )
    spa.to_csv(OUTPUT / "spa_target_audit.csv", index=False)
    top.to_csv(
        OUTPUT / "top_studentized_paths.csv",
        index=False,
    )
    summary = pd.Series(
        {
            "scanned_csv_files": scanned,
            "net_return_header_matches": header_matches,
            "eligible_paths": len(eligible_rows),
            "unique_paths": len(unique),
            "point_target_paths": int(
                unique["point_target_pass"].sum()
            ),
            "point_and_year_breadth_paths": int(
                (
                    unique["point_target_pass"]
                    & unique["year_breadth_pass"]
                ).sum()
            ),
            "pareto_frontier_paths": int(
                unique["pareto_frontier"].sum()
            ),
            "r11_cagr": annualized_cagr(baseline),
            "r11_max_drawdown": maximum_drawdown(baseline),
        },
        name="value",
    )
    summary.to_csv(OUTPUT / "summary.csv")
    print(summary.to_string())
    print("\nTarget SPA:")
    print(spa.round(6).to_string(index=False))
    print("\nPoint target paths:")
    print(
        unique.loc[
            unique["point_target_pass"],
            [
                "representative_path",
                "cagr",
                "max_drawdown",
                "year_breadth_pass",
            ],
        ]
        .sort_values("cagr", ascending=False)
        .head(30)
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
