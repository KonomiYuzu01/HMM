from __future__ import annotations

from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "reentry_sequence_5_8_validation"
STRATEGIES = {
    "step2_netted": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "step4_qqq_first": (
        "paper_core_growth_gold20_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step5_recovery_quality": (
        "paper_core_growth_gold20_quality_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step6_episode_memory": (
        "paper_core_growth_gold20_quality_memory_qqq_first_daily_hmm_netted_ensemble"
    ),
    "step7_defensive_carry": (
        "paper_core_growth_gold20_quality_memory_defensive_qqq_first_"
        "daily_hmm_netted_ensemble"
    ),
}
EXTENDED_SUFFIX = "_2012"
COST15_SUFFIX = "_cost15"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
COMPARISONS = {
    "step5_vs_step4": ("step5_recovery_quality", "step4_qqq_first"),
    "step5_vs_champion": ("step5_recovery_quality", "step2_netted"),
    "step6_vs_step5": ("step6_episode_memory", "step5_recovery_quality"),
    "step6_vs_champion": ("step6_episode_memory", "step2_netted"),
    "step7_vs_step6": ("step7_defensive_carry", "step6_episode_memory"),
    "step7_vs_champion": ("step7_defensive_carry", "step2_netted"),
    "step8_vs_step7": ("step8_call_convexity", "step7_defensive_carry"),
    "step8_vs_champion": ("step8_call_convexity", "step2_netted"),
}


def load_frame(directory: str, filename: str = "daily_returns.csv") -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def benchmark_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices[["QQQ", "SEMIS"]].pct_change(fill_method=None)
    returns.columns = ["QQQ", "SMH"]
    returns["GROWTH_EQUAL"] = returns.mean(axis=1)
    return returns


def capture_ratios(
    returns: pd.Series,
    benchmarks: pd.DataFrame,
    strategy: str,
    period: str,
    start: str,
    end: str,
) -> list[dict[str, float | int | str]]:
    aligned = pd.concat(
        [returns.rename("strategy"), benchmarks],
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
    if np.allclose(relative, 0.0, atol=1e-12, rtol=0.0):
        return pd.Series(
            {
                "annualized_relative_log_return": 0.0,
                "lower_95": 0.0,
                "upper_95": 0.0,
                "probability_positive": 0.5,
                "block_days": block_days,
                "samples": samples,
            }
        )
    count = len(relative)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=block_count)
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


def black_scholes_call(
    spot: float,
    strike: float,
    years: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    if years <= 0.0:
        return max(spot - strike, 0.0)
    sigma = max(volatility, 0.05)
    root_time = sqrt(years)
    d1 = (
        log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * years
    ) / (sigma * root_time)
    d2 = d1 - sigma * root_time
    return float(
        spot * norm.cdf(d1)
        - strike * exp(-risk_free_rate * years) * norm.cdf(d2)
    )


def simulate_call_overlay(
    base_returns: pd.Series,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    premium_budget_share: float,
    expiry_days: int = 21,
    half_spread_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if not 0.0 < premium_budget_share < 0.05:
        raise ValueError("Option premium budget must be between zero and 5%")
    index = base_returns.index.intersection(weights.index).intersection(prices.index)
    frame = pd.DataFrame(index=index)
    frame["base_return"] = base_returns.reindex(index)
    frame["growth_exposure"] = (
        weights.reindex(index)["QQQ"] + weights.reindex(index)["SEMIS"]
    )
    frame = frame.join(prices[["QQQ", "CASH", "VXN", "BREADTH", "SPX", "CREDIT", "BOND"]])
    frame = frame.dropna()
    frame["cash_return"] = frame["CASH"].pct_change(fill_method=None).fillna(0.0)
    breadth = frame["BREADTH"] / frame["SPX"]
    credit = frame["CREDIT"] / frame["BOND"]
    frame["recovery_quality"] = (
        (breadth.pct_change(21, fill_method=None).shift(1) > 0.0)
        & (credit.pct_change(21, fill_method=None).shift(1) > 0.0)
    )

    base_equity = 1.0
    overlay_cash = 0.0
    option_units = 0.0
    strike = float("nan")
    days_remaining = 0
    previous_combined_equity = 1.0
    entries = 0
    early_exits = 0
    expiries = 0
    total_premium = 0.0
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    previous_row: pd.Series | None = None

    for date, row in frame.iterrows():
        entry = 0
        exit_trade = 0
        premium_paid = 0.0
        if previous_row is not None:
            start_spot = float(previous_row["QQQ"])
            start_volatility = float(previous_row["VXN"]) / 100.0
            if option_units > 0.0 and float(row["growth_exposure"]) >= 0.50:
                start_mid = black_scholes_call(
                    start_spot,
                    strike,
                    days_remaining / 252.0,
                    start_volatility,
                )
                overlay_cash += (
                    option_units
                    * start_mid
                    * (1.0 - half_spread_fraction)
                )
                option_units = 0.0
                strike = float("nan")
                days_remaining = 0
                early_exits += 1
                exit_trade = 1
            if (
                option_units == 0.0
                and float(row["growth_exposure"]) < 0.20
                and bool(row["recovery_quality"])
            ):
                strike = start_spot
                start_mid = black_scholes_call(
                    start_spot,
                    strike,
                    expiry_days / 252.0,
                    start_volatility,
                )
                ask = start_mid * (1.0 + half_spread_fraction)
                budget = premium_budget_share * max(
                    previous_combined_equity,
                    1e-6,
                )
                option_units = budget / ask
                overlay_cash -= budget
                days_remaining = expiry_days
                entries += 1
                entry = 1
                premium_paid = budget
                total_premium += budget

        overlay_cash *= 1.0 + float(row["cash_return"])
        base_equity *= 1.0 + float(row["base_return"])
        option_value = 0.0
        if option_units > 0.0:
            days_remaining -= 1
            if days_remaining <= 0:
                payoff = max(float(row["QQQ"]) - strike, 0.0)
                overlay_cash += option_units * payoff
                option_units = 0.0
                strike = float("nan")
                expiries += 1
            else:
                option_value = option_units * black_scholes_call(
                    float(row["QQQ"]),
                    strike,
                    days_remaining / 252.0,
                    float(row["VXN"]) / 100.0,
                )
        combined_equity = base_equity + overlay_cash + option_value
        net_return = combined_equity / previous_combined_equity - 1.0
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "equity": combined_equity,
                "base_equity": base_equity,
                "overlay_value": overlay_cash + option_value,
                "option_value": option_value,
                "option_entry": entry,
                "option_early_exit": exit_trade,
                "premium_paid": premium_paid,
                "growth_exposure": float(row["growth_exposure"]),
                "recovery_quality": int(bool(row["recovery_quality"])),
            }
        )
        previous_combined_equity = combined_equity
        previous_row = row

    result = pd.DataFrame(rows).set_index("date")
    return result, {
        "entries": entries,
        "early_exits": early_exits,
        "expiries": expiries,
        "total_premium_fraction_of_initial_capital": total_premium,
        "ending_overlay_value": float(result.iloc[-1]["overlay_value"]),
    }


def load_scenarios() -> dict[str, dict[str, pd.DataFrame]]:
    scenarios: dict[str, dict[str, pd.DataFrame]] = {
        "normal": {},
        "cost15": {},
        "extended": {},
    }
    for strategy, directory in STRATEGIES.items():
        scenarios["normal"][strategy] = load_frame(directory)
        scenarios["cost15"][strategy] = load_frame(directory + COST15_SUFFIX)
        scenarios["extended"][strategy] = load_frame(
            directory + EXTENDED_SUFFIX
        )
    return scenarios


def append_option_strategy(
    scenarios: dict[str, dict[str, pd.DataFrame]],
    prices: pd.DataFrame,
    premium_budget_share: float = 0.005,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    option_summary_rows = []
    for scenario, frames in scenarios.items():
        directory = STRATEGIES["step7_defensive_carry"]
        if scenario == "cost15":
            directory += COST15_SUFFIX
        elif scenario == "extended":
            directory += EXTENDED_SUFFIX
        weights = load_frame(directory, "weights.csv")
        option_daily, summary = simulate_call_overlay(
            frames["step7_defensive_carry"]["net_return"],
            weights,
            prices,
            premium_budget_share,
        )
        frames["step8_call_convexity"] = option_daily
        option_daily.to_csv(
            DESTINATION / f"step8_option_daily_{scenario}.csv"
        )
        option_summary_rows.append({"scenario": scenario, **summary})
    return scenarios, pd.DataFrame(option_summary_rows).set_index("scenario")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_recovery_quality.csv",
        index_col=0,
        parse_dates=True,
    )
    scenarios = load_scenarios()
    scenarios, option_summary = append_option_strategy(scenarios, prices)
    option_summary.to_csv(DESTINATION / "option_summary.csv")
    benchmarks = benchmark_returns(prices)

    metric_rows: list[dict[str, float | str]] = []
    for scenario, frames in scenarios.items():
        for strategy, frame in frames.items():
            periods = (
                {"extended_2012_2025": ("2012-01-01", "2025-12-31")}
                if scenario == "extended"
                else PERIODS
            )
            for period, (start, end) in periods.items():
                sample = frame.loc[start:end, "net_return"]
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "period": period,
                        **performance_metrics(sample),
                    }
                )
    metrics = pd.DataFrame(metric_rows).set_index(
        ["scenario", "strategy", "period"]
    )
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    capture_rows: list[dict[str, float | int | str]] = []
    for strategy, frame in scenarios["normal"].items():
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

    bootstrap_rows = []
    samples = {
        "normal": slice("2015-01-01", "2025-12-31"),
        "cost15": slice("2015-01-01", "2025-12-31"),
        "extended": slice("2012-01-01", "2025-12-31"),
    }
    for comparison, (candidate, baseline) in COMPARISONS.items():
        for scenario, sample in samples.items():
            result = circular_block_bootstrap(
                scenarios[scenario][candidate].loc[sample, "net_return"],
                scenarios[scenario][baseline].loc[sample, "net_return"],
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

    behavior_rows = []
    for strategy in (
        "step5_recovery_quality",
        "step6_episode_memory",
        "step7_defensive_carry",
    ):
        directory = STRATEGIES[strategy]
        for member_file in sorted(
            (OUTPUT_ROOT / directory / "members").glob(
                "seed_*/daily_returns.csv"
            )
        ):
            frame = pd.read_csv(member_file)
            quality_checked = frame[
                "daily_hmm_recovery_quality_enabled"
            ].eq(1)
            behavior_rows.append(
                {
                    "strategy": strategy,
                    "member": member_file.parent.name,
                    "daily_transitions": int(
                        frame["daily_hmm_transition_active"].sum()
                    ),
                    "daily_entries": int(
                        (
                            frame["daily_hmm_transition_active"].eq(1)
                            & frame["daily_hmm_risk_on_candidate"].eq(1)
                        ).sum()
                    ),
                    "quality_checks": int(quality_checked.sum()),
                    "quality_passes": int(
                        frame.loc[
                            quality_checked,
                            "daily_hmm_recovery_quality_pass",
                        ].sum()
                    ),
                    "memory_blocked_days": int(
                        frame["daily_hmm_episode_memory_blocked"].sum()
                    ),
                }
            )
    behavior = pd.DataFrame(behavior_rows).set_index(
        ["strategy", "member"]
    )
    behavior.to_csv(DESTINATION / "behavior_summary.csv")

    sensitivity_rows = []
    step7_directory = STRATEGIES["step7_defensive_carry"]
    step7_weights = load_frame(step7_directory, "weights.csv")
    for budget in (0.0025, 0.005, 0.01):
        option_daily, summary = simulate_call_overlay(
            scenarios["normal"]["step7_defensive_carry"]["net_return"],
            step7_weights,
            prices,
            budget,
        )
        sample = option_daily.loc["2015-01-01":"2025-12-31", "net_return"]
        sensitivity_rows.append(
            {
                "premium_budget_share": budget,
                **performance_metrics(sample),
                **summary,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).set_index(
        "premium_budget_share"
    )
    sensitivity.to_csv(DESTINATION / "option_budget_sensitivity.csv")

    acceptance_rows = []
    for comparison, (candidate, baseline) in COMPARISONS.items():
        primary_candidate = metrics.loc[
            ("normal", candidate, "complete_2015_2025")
        ]
        primary_baseline = metrics.loc[
            ("normal", baseline, "complete_2015_2025")
        ]
        development_candidate = metrics.loc[
            ("normal", candidate, "development_2015_2021")
        ]
        development_baseline = metrics.loc[
            ("normal", baseline, "development_2015_2021")
        ]
        holdout_candidate = metrics.loc[
            ("normal", candidate, "holdout_2022_2025")
        ]
        holdout_baseline = metrics.loc[
            ("normal", baseline, "holdout_2022_2025")
        ]
        cost_candidate = metrics.loc[
            ("cost15", candidate, "complete_2015_2025")
        ]
        cost_baseline = metrics.loc[
            ("cost15", baseline, "complete_2015_2025")
        ]
        extended_candidate = metrics.loc[
            ("extended", candidate, "extended_2012_2025")
        ]
        extended_baseline = metrics.loc[
            ("extended", baseline, "extended_2012_2025")
        ]
        candidate_capture = captures.loc[
            (candidate, "complete_2015_2025", "GROWTH_EQUAL")
        ]["capture_ratio"]
        baseline_capture = captures.loc[
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
            candidate_capture["up"] - baseline_capture["up"]
        )
        down_capture_delta = float(
            candidate_capture["down"] - baseline_capture["down"]
        )
        tolerance = 1e-8
        gates = {
            "complete_positive_pass": int(cagr_delta > tolerance),
            "development_nonnegative_pass": int(
                development_delta >= -tolerance
            ),
            "holdout_nonnegative_pass": int(holdout_delta >= -tolerance),
            "cost15_nonnegative_pass": int(cost15_delta >= -tolerance),
            "extended_nonnegative_pass": int(extended_delta >= -tolerance),
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
                "capture_spread_delta": up_capture_delta
                - down_capture_delta,
                "bootstrap_probability_positive": float(
                    bootstrap.loc[
                        (comparison, "normal"),
                        "probability_positive",
                    ]
                ),
                **gates,
                "all_evidence_gates_pass": int(
                    all(bool(value) for value in gates.values())
                ),
            }
        )
    acceptance = pd.DataFrame(acceptance_rows).set_index("comparison")
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    primary = metrics.xs(
        ("normal", "complete_2015_2025"),
        level=("scenario", "period"),
    )[["cagr", "annual_volatility", "sharpe", "max_drawdown"]]
    growth_capture = captures.xs(
        ("complete_2015_2025", "GROWTH_EQUAL"),
        level=("period", "benchmark"),
    )["capture_ratio"].unstack("direction")
    primary = primary.join(
        growth_capture.rename(
            columns={"up": "growth_up_capture", "down": "growth_down_capture"}
        )
    )
    primary["growth_capture_spread"] = (
        primary["growth_up_capture"] - primary["growth_down_capture"]
    )
    primary.to_csv(DESTINATION / "primary_comparison.csv")

    decision = """# 实验 5–8 结论

第 2 项净额执行仍是生产冠军。恢复质量门槛显著减少日度重入，但没有
产生稳定收益增量；episode memory 在样本中没有绑定；risk-off 防御趋势
增加了波动与回撤；合成看涨期权结果仅用于判断经济可行性，不能视为
可成交的历史期权回测。
"""
    (DESTINATION / "decision.md").write_text(decision, encoding="utf-8")

    print(primary.round(6).to_string())
    print("\nEvidence:")
    print(acceptance.round(6).to_string())
    print("\nOption sensitivity:")
    print(sensitivity.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
