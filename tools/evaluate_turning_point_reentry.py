from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "turning_point_reentry_validation"
STRATEGIES = {
    "baseline": "paper_core_growth_gold20_daily_risk_ensemble_rerun",
    "candidate": "paper_core_growth_gold20_turning_point_reentry_ensemble",
    "baseline_cost15": "paper_core_growth_gold20_daily_risk_ensemble_rerun_cost15",
    "candidate_cost15": (
        "paper_core_growth_gold20_turning_point_reentry_ensemble_cost15"
    ),
    "baseline_2012": "paper_core_growth_gold20_daily_risk_ensemble_2012_current",
    "candidate_2012": (
        "paper_core_growth_gold20_turning_point_reentry_ensemble_2012"
    ),
}
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def load_daily(label: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / STRATEGIES[label] / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def annualized_relative_log_return(
    candidate: pd.Series,
    baseline: pd.Series,
) -> float:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    return float(
        (np.log1p(aligned.iloc[:, 0]) - np.log1p(aligned.iloc[:, 1])).mean()
        * 252.0
    )


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
    statistics = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:count] % count
        statistics[sample] = float(relative[indices].mean() * 252.0)
    observed = float(relative.mean() * 252.0)
    return pd.Series(
        {
            "annualized_relative_log_return": observed,
            "lower_95": float(np.quantile(statistics, 0.025)),
            "upper_95": float(np.quantile(statistics, 0.975)),
            "probability_positive": float((statistics > 0.0).mean()),
            "block_days": block_days,
            "samples": samples,
        }
    )


def benchmark_returns() -> pd.DataFrame:
    prices = pd.read_csv(
        "data/adjusted_open_close_2011_present.csv",
        index_col=0,
        parse_dates=True,
    )
    returns = prices[["close_QQQ", "close_SEMIS"]].pct_change(fill_method=None)
    returns.columns = ["QQQ", "SMH"]
    returns["GROWTH_EQUAL"] = returns[["QQQ", "SMH"]].mean(axis=1)
    return returns


def capture_ratios(
    returns: pd.Series,
    benchmarks: pd.DataFrame,
    period: str,
    start: str,
    end: str | None,
    strategy: str,
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


def reentry_events(
    baseline: pd.Series,
    candidate: pd.Series,
    growth: pd.Series,
) -> pd.DataFrame:
    diagnostics = pd.read_csv(
        OUTPUT_ROOT
        / STRATEGIES["candidate"]
        / "members"
        / "seed_7"
        / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    event_dates = diagnostics.index[
        diagnostics["turning_point_reentry_executed"].eq(1)
    ]
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate.rename("candidate"),
            growth.rename("growth"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    rows = []
    for date in event_dates:
        if date not in aligned.index:
            continue
        start = int(aligned.index.get_loc(date))
        window = aligned.iloc[start : start + 21]
        rows.append(
            {
                "date": date,
                "sessions": len(window),
                "baseline_21d_return": float(
                    (1.0 + window["baseline"]).prod() - 1.0
                ),
                "candidate_21d_return": float(
                    (1.0 + window["candidate"]).prod() - 1.0
                ),
                "growth_21d_return": float(
                    (1.0 + window["growth"]).prod() - 1.0
                ),
                "candidate_minus_baseline_21d": float(
                    (1.0 + window["candidate"]).prod()
                    / (1.0 + window["baseline"]).prod()
                    - 1.0
                ),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {label: load_daily(label) for label in STRATEGIES}
    benchmarks = benchmark_returns()

    metric_rows: list[dict[str, float | str]] = []
    for strategy in ("baseline", "candidate", "baseline_cost15", "candidate_cost15"):
        for period, (start, end) in PERIODS.items():
            sample = daily[strategy].loc[start:end, "net_return"]
            metric_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(sample),
                }
            )
    for strategy in ("baseline_2012", "candidate_2012"):
        metric_rows.append(
            {
                "strategy": strategy,
                "period": "full_2012_present",
                **performance_metrics(daily[strategy]["net_return"]),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    capture_rows: list[dict[str, float | int | str]] = []
    for strategy in ("baseline", "candidate"):
        for period, (start, end) in PERIODS.items():
            capture_rows.extend(
                capture_ratios(
                    daily[strategy]["net_return"],
                    benchmarks,
                    period,
                    start,
                    end,
                    strategy,
                )
            )
    capture = pd.DataFrame(capture_rows).set_index(
        ["strategy", "period", "benchmark", "direction"]
    )
    capture.to_csv(DESTINATION / "capture_ratios.csv")

    exposure_rows = []
    for strategy in ("baseline", "candidate"):
        weights = pd.read_csv(
            OUTPUT_ROOT / STRATEGIES[strategy] / "weights.csv",
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
    exposure = pd.DataFrame(exposure_rows).set_index("strategy")
    exposure.to_csv(DESTINATION / "exposure_profile.csv")

    trigger_rows = []
    for member_file in sorted(
        (
            OUTPUT_ROOT / STRATEGIES["candidate"] / "members"
        ).glob("seed_*/daily_returns.csv")
    ):
        frame = pd.read_csv(member_file, index_col=0, parse_dates=True)
        trigger_rows.append(
            {
                "member": member_file.parent.name,
                "rebound_days": int(frame["turning_point_state"].eq(3).sum()),
                "binding_days": int(
                    frame["turning_point_reentry_active"].sum()
                ),
                "executed_days": int(
                    frame["turning_point_reentry_executed"].sum()
                ),
            }
        )
    triggers = pd.DataFrame(trigger_rows).set_index("member")
    triggers.to_csv(DESTINATION / "trigger_summary.csv")

    sample = slice("2015-01-01", "2025-12-31")
    annual = pd.concat(
        {
            "baseline": daily["baseline"].loc[sample, "net_return"],
            "candidate": daily["candidate"].loc[sample, "net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_baseline"] = (
        annual["candidate"] - annual["baseline"]
    )
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    events = reentry_events(
        daily["baseline"]["net_return"],
        daily["candidate"]["net_return"],
        benchmarks["GROWTH_EQUAL"],
    )
    events.to_csv(DESTINATION / "reentry_events.csv")

    bootstrap = pd.DataFrame(
        {
            "normal_cost_2015_2025": circular_block_bootstrap(
                daily["candidate"].loc[sample, "net_return"],
                daily["baseline"].loc[sample, "net_return"],
            ),
            "cost15_2015_2025": circular_block_bootstrap(
                daily["candidate_cost15"].loc[sample, "net_return"],
                daily["baseline_cost15"].loc[sample, "net_return"],
            ),
            "extended_2012_present": circular_block_bootstrap(
                daily["candidate_2012"]["net_return"],
                daily["baseline_2012"]["net_return"],
            ),
        }
    ).T
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    primary = ("complete_2015_2025",)
    baseline_primary = metrics.loc[("baseline", *primary)]
    candidate_primary = metrics.loc[("candidate", *primary)]
    baseline_dev = metrics.loc[("baseline", "development_2015_2021")]
    candidate_dev = metrics.loc[("candidate", "development_2015_2021")]
    baseline_holdout = metrics.loc[("baseline", "holdout_2022_2025")]
    candidate_holdout = metrics.loc[("candidate", "holdout_2022_2025")]
    baseline_cost = metrics.loc[("baseline_cost15", *primary)]
    candidate_cost = metrics.loc[("candidate_cost15", *primary)]
    up_capture_delta = float(
        capture.loc[
            ("candidate", *primary, "GROWTH_EQUAL", "up"),
            "capture_ratio",
        ]
        - capture.loc[
            ("baseline", *primary, "GROWTH_EQUAL", "up"),
            "capture_ratio",
        ]
    )
    down_capture_delta = float(
        capture.loc[
            ("candidate", *primary, "GROWTH_EQUAL", "down"),
            "capture_ratio",
        ]
        - capture.loc[
            ("baseline", *primary, "GROWTH_EQUAL", "down"),
            "capture_ratio",
        ]
    )
    cagr_delta = float(candidate_primary["cagr"] - baseline_primary["cagr"])
    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr_delta": cagr_delta,
                "development_cagr_delta": float(
                    candidate_dev["cagr"] - baseline_dev["cagr"]
                ),
                "holdout_cagr_delta": float(
                    candidate_holdout["cagr"] - baseline_holdout["cagr"]
                ),
                "cost15_cagr_delta": float(
                    candidate_cost["cagr"] - baseline_cost["cagr"]
                ),
                "max_drawdown_delta": float(
                    candidate_primary["max_drawdown"]
                    - baseline_primary["max_drawdown"]
                ),
                "growth_up_capture_delta": up_capture_delta,
                "growth_down_capture_delta": down_capture_delta,
                "average_growth_exposure_delta": float(
                    exposure.loc["candidate", "average_growth_exposure"]
                    - exposure.loc["baseline", "average_growth_exposure"]
                ),
                "incremental_annualized_cost_drag": float(
                    daily["candidate"]["cost"].mean() * 252.0
                    - daily["baseline"]["cost"].mean() * 252.0
                ),
                "normal_cost_positive_pass": int(cagr_delta > 0.0),
                "development_nonnegative_pass": int(
                    candidate_dev["cagr"] >= baseline_dev["cagr"]
                ),
                "holdout_nonnegative_pass": int(
                    candidate_holdout["cagr"] >= baseline_holdout["cagr"]
                ),
                "cost15_nonnegative_pass": int(
                    candidate_cost["cagr"] >= baseline_cost["cagr"]
                ),
                "drawdown_not_worse_50bp_pass": int(
                    candidate_primary["max_drawdown"]
                    >= baseline_primary["max_drawdown"] - 0.005
                ),
                "up_capture_improved_pass": int(up_capture_delta > 0.0),
                "production_cagr_plus_1pp_pass": int(cagr_delta >= 0.01),
            }
        ],
        index=["turning_point_reentry"],
    )
    engineering_passes = [
        "normal_cost_positive_pass",
        "development_nonnegative_pass",
        "holdout_nonnegative_pass",
        "cost15_nonnegative_pass",
        "drawdown_not_worse_50bp_pass",
        "up_capture_improved_pass",
    ]
    acceptance["engineering_pass"] = (
        acceptance[engineering_passes].all(axis=1).astype(int)
    )
    acceptance["production_upgrade_pass"] = (
        acceptance["engineering_pass"]
        & acceptance["production_cagr_plus_1pp_pass"]
    ).astype(int)
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    decision = f"""# Turning-point re-entry decision

The one-shot Rebound sleeve is an engineering improvement but does not clear the
production replacement hurdle.

- 2015-2025 CAGR delta: {cagr_delta:.2%}
- Development CAGR delta: {acceptance.iloc[0]["development_cagr_delta"]:.2%}
- Holdout CAGR delta: {acceptance.iloc[0]["holdout_cagr_delta"]:.2%}
- 15 bps CAGR delta: {acceptance.iloc[0]["cost15_cagr_delta"]:.2%}
- Growth up-capture delta: {up_capture_delta:.2%}
- Growth down-capture delta: {down_capture_delta:.2%}
- Maximum-drawdown delta: {acceptance.iloc[0]["max_drawdown_delta"]:.2%}
- 21-day block-bootstrap probability of positive relative return:
  {bootstrap.loc["normal_cost_2015_2025", "probability_positive"]:.1%}
- Engineering gate: {"PASS" if bool(acceptance.iloc[0]["engineering_pass"]) else "FAIL"}
- Production +1 percentage point gate:
  {"PASS" if bool(acceptance.iloc[0]["production_upgrade_pass"]) else "FAIL"}

Keep the production configuration unchanged. Retain this candidate as a shadow
strategy because it improves entry mechanics with bounded downside, but its
historical return increment is too small to distinguish confidently from noise.
"""
    (DESTINATION / "decision.md").write_text(decision, encoding="utf-8")

    print(acceptance.T.round(6).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
