from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "reentry_redesign_validation"
PRIMARY = slice("2015-01-01", "2025-12-31")
EXTENDED = slice("2012-01-01", "2025-12-31")
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
NORMAL_DIRECTORIES = {
    "step1_baseline": "paper_core_growth_gold20_daily_risk_ensemble_rerun",
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "step3_daily_hmm": "paper_core_growth_gold20_daily_hmm_netted_ensemble",
    "step4_qqq_first": (
        "paper_core_growth_gold20_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step5_quality_gate": (
        "paper_core_growth_gold20_quality_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step6_episode_memory": (
        "paper_core_growth_gold20_quality_memory_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step7_defensive_carry": (
        "paper_core_growth_gold20_quality_memory_defensive_qqq_first_"
        "daily_hmm_netted_ensemble"
    ),
    "redesign_selective_bridge": (
        "paper_core_growth_gold20_selective_bridge_daily_hmm_netted_ensemble"
    ),
}
COST15_DIRECTORIES = {
    name: f"{directory}_cost15"
    for name, directory in NORMAL_DIRECTORIES.items()
    if name in {"step2_netted", "redesign_selective_bridge"}
}
EXTENDED_DIRECTORIES = {
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble_2012",
    "redesign_selective_bridge": (
        "paper_core_growth_gold20_selective_bridge_daily_hmm_"
        "netted_ensemble_2012"
    ),
}


def load_frame(directory: str, filename: str = "daily_returns.csv") -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def load_normal_returns() -> dict[str, pd.DataFrame]:
    frames = {
        name: load_frame(directory)
        for name, directory in NORMAL_DIRECTORIES.items()
    }
    frames["step8_call_convexity"] = pd.read_csv(
        OUTPUT
        / "reentry_sequence_5_8_validation"
        / "step8_option_daily_normal.csv",
        index_col=0,
        parse_dates=True,
    )
    return frames


def circular_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_724,
) -> pd.Series:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    relative = (
        np.log1p(aligned.iloc[:, 0]) - np.log1p(aligned.iloc[:, 1])
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
            "block_days": block_days,
            "samples": samples,
        }
    )


def capture_ratio(
    strategy: pd.Series,
    benchmark: pd.Series,
    direction: str,
) -> float:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    mask = (
        aligned["benchmark"] > 0.0
        if direction == "up"
        else aligned["benchmark"] < 0.0
    )
    return float(
        aligned.loc[mask, "strategy"].mean()
        / aligned.loc[mask, "benchmark"].mean()
    )


def common_metrics(
    frames: dict[str, pd.DataFrame],
    periods: dict[str, tuple[str, str]],
    scenario: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, frame in frames.items():
        for period, (start, end) in periods.items():
            sample = frame.loc[start:end, "net_return"]
            rows.append(
                {
                    "scenario": scenario,
                    "strategy": name,
                    "period": period,
                    **performance_metrics(sample),
                    "annualized_cost_drag": float(
                        frame.loc[start:end, "cost"].mean() * 252.0
                    )
                    if "cost" in frame
                    else float("nan"),
                    "average_daily_turnover": float(
                        frame.loc[start:end, "turnover"].mean()
                    )
                    if "turnover" in frame
                    else float("nan"),
                }
            )
    return rows


def mechanism_decomposition(
    frames: dict[str, pd.DataFrame],
    benchmark: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, frame in frames.items():
        sample = frame.loc[PRIMARY, "net_return"]
        aligned = pd.concat(
            [sample.rename("strategy"), benchmark.rename("growth")],
            axis=1,
            join="inner",
        ).dropna()
        log_returns = np.log1p(aligned["strategy"])
        up = aligned["growth"] > 0.0
        down = aligned["growth"] < 0.0
        weight_directory = (
            NORMAL_DIRECTORIES["step7_defensive_carry"]
            if name == "step8_call_convexity"
            else NORMAL_DIRECTORIES[name]
        )
        weights = load_frame(weight_directory, "weights.csv").loc[PRIMARY]
        growth_exposure = weights["QQQ"] + weights["SEMIS"]
        cost_drag = (
            float(frame.loc[PRIMARY, "cost"].mean() * 252.0)
            if "cost" in frame
            else float("nan")
        )
        rows.append(
            {
                "strategy": name,
                "cagr": performance_metrics(aligned["strategy"])["cagr"],
                "annualized_log_return": float(log_returns.mean() * 252.0),
                "up_day_log_contribution": float(
                    log_returns.where(up, 0.0).mean() * 252.0
                ),
                "down_day_log_contribution": float(
                    log_returns.where(down, 0.0).mean() * 252.0
                ),
                "growth_up_capture": capture_ratio(
                    aligned["strategy"], aligned["growth"], "up"
                ),
                "growth_down_capture": capture_ratio(
                    aligned["strategy"], aligned["growth"], "down"
                ),
                "capture_spread": (
                    capture_ratio(aligned["strategy"], aligned["growth"], "up")
                    - capture_ratio(
                        aligned["strategy"], aligned["growth"], "down"
                    )
                ),
                "average_growth_exposure": float(growth_exposure.mean()),
                "growth_exposure_below_20pct": float(
                    (growth_exposure < 0.20).mean()
                ),
                "annualized_cost_drag": cost_drag,
                "average_daily_turnover": float(
                    frame.loc[PRIMARY, "turnover"].mean()
                )
                if "turnover" in frame
                else float("nan"),
            }
        )
    result = pd.DataFrame(rows).set_index("strategy")
    champion = result.loc["step2_netted"]
    result["cagr_delta_vs_step2"] = result["cagr"] - champion["cagr"]
    result["up_capture_delta_vs_step2"] = (
        result["growth_up_capture"] - champion["growth_up_capture"]
    )
    result["down_capture_delta_vs_step2"] = (
        result["growth_down_capture"] - champion["growth_down_capture"]
    )
    return result


def bridge_events(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = OUTPUT / NORMAL_DIRECTORIES["redesign_selective_bridge"]
    dates: set[pd.Timestamp] = set()
    member_rows: list[dict[str, float | str | pd.Timestamp]] = []
    for path in sorted((directory / "members").glob("seed_*/daily_returns.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        entries = frame[
            frame["daily_hmm_transition_active"].eq(1)
            & frame["daily_hmm_risk_on_candidate"].eq(1)
        ]
        for date, row in entries.iterrows():
            dates.add(date)
            member_rows.append(
                {
                    "member": path.parent.name,
                    "date": date,
                    "blend_fraction": float(
                        row["daily_hmm_reentry_blend_fraction"]
                    ),
                    "quality_confirmations": int(
                        row["daily_hmm_recovery_quality_confirmations"]
                    ),
                }
            )
    aligned = pd.concat(
        [
            candidate["net_return"].rename("candidate"),
            baseline["net_return"].rename("baseline"),
            benchmark.rename("growth"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    event_rows: list[dict[str, float | pd.Timestamp]] = []
    for date in sorted(dates):
        if date not in aligned.index:
            continue
        position = int(aligned.index.get_loc(date))
        row: dict[str, float | pd.Timestamp] = {"date": date}
        for horizon in (1, 5, 21, 63):
            window = aligned.iloc[position : position + horizon]
            if len(window) < horizon:
                continue
            row[f"candidate_minus_baseline_{horizon}d"] = float(
                (1.0 + window["candidate"]).prod()
                / (1.0 + window["baseline"]).prod()
                - 1.0
            )
            row[f"growth_return_{horizon}d"] = float(
                (1.0 + window["growth"]).prod() - 1.0
            )
        event_rows.append(row)
    events = pd.DataFrame(event_rows).set_index("date")
    summary = pd.DataFrame(
        {
            "unique_entry_dates": [len(events)],
            "member_entries": [len(member_rows)],
            **{
                f"mean_candidate_minus_baseline_{horizon}d": [
                    float(
                        events[
                            f"candidate_minus_baseline_{horizon}d"
                        ].mean()
                    )
                ]
                for horizon in (1, 5, 21, 63)
            },
            **{
                f"positive_relative_rate_{horizon}d": [
                    float(
                        (
                            events[
                                f"candidate_minus_baseline_{horizon}d"
                            ]
                            > 0.0
                        ).mean()
                    )
                ]
                for horizon in (1, 5, 21, 63)
            },
        },
        index=["selective_bridge"],
    )
    pd.DataFrame(member_rows).set_index(["member", "date"]).to_csv(
        DESTINATION / "bridge_member_entries.csv"
    )
    return events, summary


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_recovery_quality.csv",
        index_col=0,
        parse_dates=True,
    )
    benchmark = prices[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1)
    normal = load_normal_returns()
    cost15 = {
        name: load_frame(directory)
        for name, directory in COST15_DIRECTORIES.items()
    }
    extended = {
        name: load_frame(directory)
        for name, directory in EXTENDED_DIRECTORIES.items()
    }

    metric_rows = common_metrics(normal, PERIODS, "normal")
    metric_rows.extend(common_metrics(cost15, PERIODS, "cost15"))
    metric_rows.extend(
        common_metrics(
            extended,
            {"extended_2012_2025": ("2012-01-01", "2025-12-31")},
            "extended",
        )
    )
    pd.DataFrame(metric_rows).set_index(
        ["scenario", "strategy", "period"]
    ).to_csv(DESTINATION / "metrics_by_period.csv")

    mechanism_decomposition(normal, benchmark).to_csv(
        DESTINATION / "mechanism_decomposition.csv"
    )

    bootstrap_rows = []
    for scenario, frames, sample in (
        ("normal_2015_2025", normal, PRIMARY),
        ("cost15_2015_2025", cost15, PRIMARY),
        ("extended_2012_2025", extended, EXTENDED),
    ):
        result = circular_block_bootstrap(
            frames["redesign_selective_bridge"].loc[
                sample, "net_return"
            ],
            frames["step2_netted"].loc[sample, "net_return"],
        )
        bootstrap_rows.append({"scenario": scenario, **result.to_dict()})
    pd.DataFrame(bootstrap_rows).set_index("scenario").to_csv(
        DESTINATION / "bootstrap.csv"
    )

    events, event_summary = bridge_events(
        normal["redesign_selective_bridge"],
        normal["step2_netted"],
        benchmark,
    )
    events.to_csv(DESTINATION / "bridge_events.csv")
    event_summary.to_csv(DESTINATION / "bridge_event_summary.csv")


if __name__ == "__main__":
    main()
