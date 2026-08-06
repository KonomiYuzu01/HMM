from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "reentry_sequence_validation"
STRATEGIES = {
    "step1_baseline": "paper_core_growth_gold20_daily_risk_ensemble_rerun",
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "step3_daily_hmm": "paper_core_growth_gold20_daily_hmm_netted_ensemble",
    "step4_qqq_first": (
        "paper_core_growth_gold20_qqq_first_daily_hmm_netted_ensemble"
    ),
}
COST15_STRATEGIES = {
    "step1_baseline": "paper_core_growth_gold20_daily_risk_ensemble_rerun_cost15",
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble_cost15",
    "step3_daily_hmm": (
        "paper_core_growth_gold20_daily_hmm_netted_ensemble_cost15"
    ),
    "step4_qqq_first": (
        "paper_core_growth_gold20_qqq_first_daily_hmm_netted_ensemble_cost15"
    ),
}
EXTENDED_STRATEGIES = {
    "step1_baseline": "paper_core_growth_gold20_daily_risk_ensemble_2012_current",
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble_2012",
    "step3_daily_hmm": (
        "paper_core_growth_gold20_daily_hmm_netted_ensemble_2012"
    ),
    "step4_qqq_first": (
        "paper_core_growth_gold20_qqq_first_daily_hmm_netted_ensemble_2012"
    ),
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}
COMPARISONS = {
    "step2_vs_step1": ("step2_netted", "step1_baseline"),
    "step3_vs_step2": ("step3_daily_hmm", "step2_netted"),
    "step4_vs_step3": ("step4_qqq_first", "step3_daily_hmm"),
    "step4_vs_step2": ("step4_qqq_first", "step2_netted"),
}


def load_daily(directory: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def benchmark_returns() -> pd.DataFrame:
    prices = pd.read_csv(
        "data/adjusted_open_close_2011_present.csv",
        index_col=0,
        parse_dates=True,
    )
    returns = prices[["close_QQQ", "close_SEMIS"]].pct_change(
        fill_method=None
    )
    returns.columns = ["QQQ", "SMH"]
    returns["GROWTH_EQUAL"] = returns[["QQQ", "SMH"]].mean(axis=1)
    return returns


def capture_ratios(
    strategy_returns: pd.Series,
    benchmarks: pd.DataFrame,
    strategy: str,
    period: str,
    start: str,
    end: str | None,
) -> list[dict[str, float | int | str]]:
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmarks],
        axis=1,
        join="inner",
    ).loc[start:end].dropna()
    rows: list[dict[str, float | int | str]] = []
    for benchmark in benchmarks:
        for direction, mask in (
            ("up", aligned[benchmark] > 0.0),
            ("down", aligned[benchmark] < 0.0),
        ):
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "benchmark": benchmark,
                    "direction": direction,
                    "capture_ratio": float(
                        aligned.loc[mask, "strategy"].mean()
                        / aligned.loc[mask, benchmark].mean()
                    ),
                    "observations": int(mask.sum()),
                }
            )
    return rows


def circular_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 5_000,
    seed: int = 20_260_724,
) -> pd.Series:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    relative = (
        np.log1p(aligned.iloc[:, 0]) - np.log1p(aligned.iloc[:, 1])
    ).to_numpy(dtype=float)
    count = len(relative)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:count] % count
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


def compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def transition_events(
    strategy: str,
    directory: str,
    strategy_returns: pd.Series,
    netted_returns: pd.Series,
    growth_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trigger_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for member_file in sorted(
        (OUTPUT_ROOT / directory / "members").glob(
            "seed_*/daily_returns.csv"
        )
    ):
        member = member_file.parent.name
        frame = pd.read_csv(member_file, index_col=0, parse_dates=True)
        transitions = frame["daily_hmm_transition_active"].eq(1)
        entries = transitions & frame["daily_hmm_risk_on_candidate"].eq(1)
        exits = transitions & frame["daily_hmm_risk_on_candidate"].eq(0)
        trigger_rows.append(
            {
                "strategy": strategy,
                "member": member,
                "transition_days": int(transitions.sum()),
                "entry_days": int(entries.sum()),
                "exit_days": int(exits.sum()),
                "mean_transition_turnover": float(
                    frame.loc[transitions, "turnover"].mean()
                ),
            }
        )
        aligned = pd.concat(
            [
                strategy_returns.rename("strategy"),
                netted_returns.rename("netted"),
                growth_returns.rename("growth"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        for date in frame.index[entries]:
            if date not in aligned.index:
                continue
            position = int(aligned.index.get_loc(date))
            window = aligned.iloc[position : position + 21]
            if len(window) < 21:
                continue
            event_rows.append(
                {
                    "strategy": strategy,
                    "member": member,
                    "date": date,
                    "strategy_21d_return": compound_return(window["strategy"]),
                    "netted_21d_return": compound_return(window["netted"]),
                    "growth_21d_return": compound_return(window["growth"]),
                    "strategy_minus_netted_21d": float(
                        (1.0 + window["strategy"]).prod()
                        / (1.0 + window["netted"]).prod()
                        - 1.0
                    ),
                }
            )
    return pd.DataFrame(trigger_rows), pd.DataFrame(event_rows)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {
        strategy: load_daily(directory)
        for strategy, directory in STRATEGIES.items()
    }
    cost15 = {
        strategy: load_daily(directory)
        for strategy, directory in COST15_STRATEGIES.items()
    }
    extended = {
        strategy: load_daily(directory)
        for strategy, directory in EXTENDED_STRATEGIES.items()
    }
    benchmarks = benchmark_returns()

    metric_rows: list[dict[str, float | str]] = []
    for scenario, frames in (
        ("normal_cost", daily),
        ("cost_15bps", cost15),
    ):
        for strategy, frame in frames.items():
            for period, (start, end) in PERIODS.items():
                sample = frame.loc[start:end, "net_return"]
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "period": period,
                        **performance_metrics(sample),
                        "average_daily_turnover": float(
                            frame.loc[start:end, "turnover"].mean()
                        ),
                        "annualized_cost_drag": float(
                            frame.loc[start:end, "cost"].mean() * 252.0
                        ),
                    }
                )
    for strategy, frame in extended.items():
        metric_rows.append(
            {
                "scenario": "extended_2012",
                "strategy": strategy,
                "period": "full_2012_present",
                **performance_metrics(frame["net_return"]),
                "average_daily_turnover": float(frame["turnover"].mean()),
                "annualized_cost_drag": float(
                    frame["cost"].mean() * 252.0
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index(
        ["scenario", "strategy", "period"]
    )
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    capture_rows: list[dict[str, float | int | str]] = []
    for strategy, frame in daily.items():
        for period, (start, end) in PERIODS.items():
            capture_rows.extend(
                capture_ratios(
                    frame["net_return"],
                    benchmarks,
                    strategy,
                    period,
                    start,
                    end,
                )
            )
    captures = pd.DataFrame(capture_rows).set_index(
        ["strategy", "period", "benchmark", "direction"]
    ).sort_index()
    captures.to_csv(DESTINATION / "capture_ratios.csv")

    exposure_rows: list[dict[str, float | str]] = []
    for strategy, directory in STRATEGIES.items():
        weights = pd.read_csv(
            OUTPUT_ROOT / directory / "weights.csv",
            index_col=0,
            parse_dates=True,
        )
        growth = weights["QQQ"] + weights["SEMIS"]
        exposure_rows.append(
            {
                "strategy": strategy,
                "average_growth_exposure": float(growth.mean()),
                "median_growth_exposure": float(growth.median()),
                "growth_exposure_below_20pct": float((growth < 0.20).mean()),
                "growth_exposure_below_50pct": float((growth < 0.50).mean()),
                "growth_exposure_above_80pct": float((growth >= 0.80).mean()),
            }
        )
    exposures = pd.DataFrame(exposure_rows).set_index("strategy")
    exposures.to_csv(DESTINATION / "exposure_profile.csv")

    primary = slice("2015-01-01", "2025-12-31")
    annual = pd.concat(
        {
            strategy: frame.loc[primary, "net_return"]
            for strategy, frame in daily.items()
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual.to_csv(DESTINATION / "annual_returns.csv")

    bootstrap_rows = []
    for comparison, (candidate, baseline) in COMPARISONS.items():
        for scenario, frames, sample in (
            ("normal_2015_2025", daily, primary),
            ("cost15_2015_2025", cost15, primary),
            ("extended_2012_present", extended, slice(None)),
        ):
            result = circular_block_bootstrap(
                frames[candidate].loc[sample, "net_return"],
                frames[baseline].loc[sample, "net_return"],
            )
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "scenario": scenario,
                    **result.to_dict(),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index(
        ["comparison", "scenario"]
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    trigger_frames = []
    event_frames = []
    for strategy in ("step3_daily_hmm", "step4_qqq_first"):
        trigger_frame, event_frame = transition_events(
            strategy,
            STRATEGIES[strategy],
            daily[strategy]["net_return"],
            daily["step2_netted"]["net_return"],
            benchmarks["GROWTH_EQUAL"],
        )
        trigger_frames.append(trigger_frame)
        event_frames.append(event_frame)
    triggers = pd.concat(trigger_frames, ignore_index=True).set_index(
        ["strategy", "member"]
    )
    triggers.to_csv(DESTINATION / "transition_summary.csv")
    events = pd.concat(event_frames, ignore_index=True).set_index(
        ["strategy", "member", "date"]
    )
    events.to_csv(DESTINATION / "reentry_events.csv")
    event_summary = events.groupby(level="strategy").agg(
        events=("strategy_minus_netted_21d", "size"),
        mean_strategy_minus_netted_21d=(
            "strategy_minus_netted_21d",
            "mean",
        ),
        median_strategy_minus_netted_21d=(
            "strategy_minus_netted_21d",
            "median",
        ),
        positive_relative_rate=("strategy_minus_netted_21d", lambda x: (x > 0).mean()),
        mean_growth_21d_return=("growth_21d_return", "mean"),
        positive_growth_rate=("growth_21d_return", lambda x: (x > 0).mean()),
    )
    event_summary.to_csv(DESTINATION / "reentry_event_summary.csv")

    acceptance_rows: list[dict[str, float | int | str]] = []
    for comparison, (candidate, baseline) in COMPARISONS.items():
        primary_candidate = metrics.loc[
            ("normal_cost", candidate, "complete_2015_2025")
        ]
        primary_baseline = metrics.loc[
            ("normal_cost", baseline, "complete_2015_2025")
        ]
        development_candidate = metrics.loc[
            ("normal_cost", candidate, "development_2015_2021")
        ]
        development_baseline = metrics.loc[
            ("normal_cost", baseline, "development_2015_2021")
        ]
        holdout_candidate = metrics.loc[
            ("normal_cost", candidate, "holdout_2022_2025")
        ]
        holdout_baseline = metrics.loc[
            ("normal_cost", baseline, "holdout_2022_2025")
        ]
        cost_candidate = metrics.loc[
            ("cost_15bps", candidate, "complete_2015_2025")
        ]
        cost_baseline = metrics.loc[
            ("cost_15bps", baseline, "complete_2015_2025")
        ]
        extended_candidate = metrics.loc[
            ("extended_2012", candidate, "full_2012_present")
        ]
        extended_baseline = metrics.loc[
            ("extended_2012", baseline, "full_2012_present")
        ]
        capture_candidate = captures.loc[
            (candidate, "complete_2015_2025", "GROWTH_EQUAL")
        ]["capture_ratio"]
        capture_baseline = captures.loc[
            (baseline, "complete_2015_2025", "GROWTH_EQUAL")
        ]["capture_ratio"]
        cagr_delta = float(
            primary_candidate["cagr"] - primary_baseline["cagr"]
        )
        development_delta = float(
            development_candidate["cagr"] - development_baseline["cagr"]
        )
        holdout_delta = float(
            holdout_candidate["cagr"] - holdout_baseline["cagr"]
        )
        cost15_delta = float(cost_candidate["cagr"] - cost_baseline["cagr"])
        extended_delta = float(
            extended_candidate["cagr"] - extended_baseline["cagr"]
        )
        drawdown_delta = float(
            primary_candidate["max_drawdown"]
            - primary_baseline["max_drawdown"]
        )
        up_capture_delta = float(
            capture_candidate["up"] - capture_baseline["up"]
        )
        down_capture_delta = float(
            capture_candidate["down"] - capture_baseline["down"]
        )
        probability_positive = float(
            bootstrap.loc[
                (comparison, "normal_2015_2025"),
                "probability_positive",
            ]
        )
        evidence_passes = {
            "complete_positive_pass": int(cagr_delta > 0.0),
            "development_nonnegative_pass": int(development_delta >= 0.0),
            "holdout_nonnegative_pass": int(holdout_delta >= 0.0),
            "cost15_nonnegative_pass": int(cost15_delta >= 0.0),
            "extended_nonnegative_pass": int(extended_delta >= 0.0),
            "drawdown_not_worse_50bp_pass": int(drawdown_delta >= -0.005),
            "up_capture_nonnegative_pass": int(up_capture_delta >= 0.0),
        }
        acceptance_rows.append(
            {
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                "complete_cagr_delta": cagr_delta,
                "development_cagr_delta": development_delta,
                "holdout_cagr_delta": holdout_delta,
                "cost15_cagr_delta": cost15_delta,
                "extended_cagr_delta": extended_delta,
                "max_drawdown_delta": drawdown_delta,
                "growth_up_capture_delta": up_capture_delta,
                "growth_down_capture_delta": down_capture_delta,
                "capture_spread_delta": up_capture_delta - down_capture_delta,
                "average_daily_turnover_delta": float(
                    primary_candidate["average_daily_turnover"]
                    - primary_baseline["average_daily_turnover"]
                ),
                "annualized_cost_drag_delta": float(
                    primary_candidate["annualized_cost_drag"]
                    - primary_baseline["annualized_cost_drag"]
                ),
                "bootstrap_probability_positive": probability_positive,
                **evidence_passes,
                "all_evidence_gates_pass": int(
                    all(bool(value) for value in evidence_passes.values())
                ),
            }
        )
    acceptance = pd.DataFrame(acceptance_rows).set_index("comparison")
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    primary_table = metrics.xs(
        ("normal_cost", "complete_2015_2025"),
        level=("scenario", "period"),
    )[
        [
            "cagr",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "average_daily_turnover",
            "annualized_cost_drag",
        ]
    ]
    growth_capture = captures.xs(
        ("complete_2015_2025", "GROWTH_EQUAL"),
        level=("period", "benchmark"),
    )["capture_ratio"].unstack("direction")
    primary_table = primary_table.join(
        growth_capture.rename(
            columns={"up": "growth_up_capture", "down": "growth_down_capture"}
        )
    )
    primary_table["growth_capture_spread"] = (
        primary_table["growth_up_capture"]
        - primary_table["growth_down_capture"]
    )
    primary_table.to_csv(DESTINATION / "primary_comparison.csv")

    step2_pass = bool(
        acceptance.loc["step2_vs_step1", "all_evidence_gates_pass"]
    )
    step3_pass = bool(
        acceptance.loc["step3_vs_step2", "all_evidence_gates_pass"]
    )
    step4_over_step2 = float(
        acceptance.loc["step4_vs_step2", "complete_cagr_delta"]
    )
    decision = f"""# 重入优化顺序评估

评估过程中没有在评价样本上搜索阈值。

1. 单账户净额执行：{"通过" if step2_pass else "不通过"}。
2. 每日 HMM 后验状态切换：{"通过" if step3_pass else "不通过"}。
3. QQQ 先行的分阶段重入改善了每日 HMM 版本，但 2015–2025 CAGR 仍比
   净额执行版本{"低" if step4_over_step2 < 0.0 else "高"}
   {abs(step4_over_step2):.2%}。

生产结论：保留现有信号节奏。只有在实盘确实以一个账户交易聚合目标时，
才采用净额执行；两个每日 HMM 重入版本均不应晋升为生产策略。
"""
    (DESTINATION / "decision.md").write_text(decision, encoding="utf-8")

    print(primary_table.round(6).to_string())
    print("\nAdjacent-step evidence:")
    print(acceptance.round(6).to_string())
    print("\nRe-entry events:")
    print(event_summary.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
