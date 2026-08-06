from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_regime_walkforward import (
    capture_ratio,
    circular_block_bootstrap,
)
from evaluate_drawdown_uncertainty import (
    paired_circular_block_bootstrap,
    strategy_summary,
)
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "lowvol_rebound_candidate_validation"
BASELINE = "paper_core_growth_gold20_daily_risk_netted_ensemble"
CANDIDATE = (
    "paper_core_growth_gold20_lowvol_rebound_floor_guard_netted_ensemble"
)
BASELINE_COST15 = f"{BASELINE}_cost15"
CANDIDATE_COST15 = f"{CANDIDATE}_cost15"
BASELINE_2012 = f"{BASELINE}_2012"
CANDIDATE_2012 = f"{CANDIDATE}_2012"
CANDIDATE_KEY = "lowvol_rebound"
REPORT_TITLE = "Low-vol rebound floor guard"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
RECOVERY_FAMILY = {
    "rebound20": "paper_core_growth_gold20_recovery_rebound20_netted_ensemble",
    "positive20": "paper_core_growth_gold20_recovery_positive20_netted_ensemble",
    "rebound63": "paper_core_growth_gold20_recovery_rebound63_netted_ensemble",
    "positive63": "paper_core_growth_gold20_recovery_positive63_netted_ensemble",
    "positive20_guarded": (
        "paper_core_growth_gold20_recovery_positive20_guarded_netted_ensemble"
    ),
    "correction_veto20_guarded": (
        "paper_core_growth_gold20_correction_veto20_guarded_netted_ensemble"
    ),
    "correction_veto63_guarded": (
        "paper_core_growth_gold20_correction_veto63_guarded_netted_ensemble"
    ),
    "slow_negative_guarded": (
        "paper_core_growth_gold20_slow_negative_guarded_netted_ensemble"
    ),
    "correction_veto63_exact": (
        "paper_core_growth_gold20_correction_veto63_exact_floor_guard_"
        "netted_ensemble"
    ),
    CANDIDATE_KEY: CANDIDATE,
}


def load_daily(directory: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def family_reality_check(
    differential: pd.DataFrame,
    block_days: int,
    samples: int = 10_000,
    seed: int = 20_260_724,
) -> pd.Series:
    values = differential.to_numpy(dtype=float)
    observed = values.mean(axis=0) * 252.0
    centered = values - values.mean(axis=0, keepdims=True)
    count = len(centered)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    generator = np.random.default_rng(seed)
    null_maxima = np.empty(samples, dtype=float)
    for sample in range(samples):
        starts = generator.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        null_maxima[sample] = float(
            centered[indices].mean(axis=0).max() * 252.0
        )
    candidate_observed = float(
        observed[differential.columns.get_loc(CANDIDATE_KEY)]
    )
    return pd.Series(
        {
            "candidate_count": differential.shape[1],
            "family_best": differential.columns[int(np.argmax(observed))],
            "family_observed_max": float(observed.max()),
            "candidate_observed": candidate_observed,
            "candidate_selection_adjusted_p_value": float(
                np.mean(null_maxima >= candidate_observed)
            ),
            "family_best_p_value": float(
                np.mean(null_maxima >= observed.max())
            ),
            "null_maximum_95": float(np.quantile(null_maxima, 0.95)),
        }
    )


def write_decision(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    family: pd.DataFrame,
    capture: pd.DataFrame,
    seed_consistency: pd.DataFrame,
    tail_summary: pd.DataFrame,
    leave_one_year_out: pd.DataFrame,
) -> None:
    base = metrics.loc[("normal", "baseline", "complete_2015_2025")]
    candidate = metrics.loc[
        ("normal", "candidate", "complete_2015_2025")
    ]
    base_holdout = metrics.loc[
        ("normal", "baseline", "holdout_2022_2025")
    ]
    candidate_holdout = metrics.loc[
        ("normal", "candidate", "holdout_2022_2025")
    ]
    base_cost = metrics.loc[
        ("cost15", "baseline", "complete_2015_2025")
    ]
    candidate_cost = metrics.loc[
        ("cost15", "candidate", "complete_2015_2025")
    ]
    extended_base = metrics.loc[
        ("extended", "baseline", "extended_2012_2025")
    ]
    extended_candidate = metrics.loc[
        ("extended", "candidate", "extended_2012_2025")
    ]
    probability = float(
        bootstrap.loc[("normal", 21), "probability_positive"]
    )
    adjusted_p = float(
        family.loc[21, "candidate_selection_adjusted_p_value"]
    )
    baseline_tail = tail_summary.loc[
        ("complete_2015_2025", 21, "baseline")
    ]
    candidate_tail = tail_summary.loc[
        ("complete_2015_2025", 21, "candidate")
    ]
    point_gate = (
        candidate["cagr"] >= base["cagr"] + 0.005
        and candidate["sharpe"] > base["sharpe"]
        and candidate["max_drawdown"] >= base["max_drawdown"] - 0.01
        and candidate_holdout["cagr"] > base_holdout["cagr"]
        and candidate_cost["cagr"] > base_cost["cagr"]
        and extended_candidate["cagr"] > extended_base["cagr"]
        and bool(seed_consistency["cagr_delta"].gt(0.0).all())
    )
    production_gate = (
        point_gate
        and probability >= 0.90
        and adjusted_p <= 0.10
    )
    verdict = (
        "升级生产"
        if production_gate
        else (
            "进入影子执行，尚不升级生产"
            if point_gate
            else "拒绝"
        )
    )
    report = f"""# {REPORT_TITLE}

## 决策

**{verdict}**

| 场景 | 生产 CAGR | 候选 CAGR | 生产 Sharpe | 候选 Sharpe | 候选 MDD |
|---|---:|---:|---:|---:|---:|
| 2015-2025 | {base['cagr']:.3%} | {candidate['cagr']:.3%} | {base['sharpe']:.3f} | {candidate['sharpe']:.3f} | {candidate['max_drawdown']:.3%} |
| 2022-2025 | {base_holdout['cagr']:.3%} | {candidate_holdout['cagr']:.3%} | {base_holdout['sharpe']:.3f} | {candidate_holdout['sharpe']:.3f} | {candidate_holdout['max_drawdown']:.3%} |
| 15 bps, 2015-2025 | {base_cost['cagr']:.3%} | {candidate_cost['cagr']:.3%} | {base_cost['sharpe']:.3f} | {candidate_cost['sharpe']:.3f} | {candidate_cost['max_drawdown']:.3%} |
| 2012-2025 | {extended_base['cagr']:.3%} | {extended_candidate['cagr']:.3%} | {extended_base['sharpe']:.3f} | {extended_candidate['sharpe']:.3f} | {extended_candidate['max_drawdown']:.3%} |

- 21 日配对区块 bootstrap 正增量概率：{probability:.1%}。
- 加入 {int(family.loc[21, 'candidate_count'])} 个 recovery 家族后的
  selection-adjusted p 值：{adjusted_p:.4f}。
- 上涨捕获率变化：
  {capture.loc['candidate', 'up_capture'] - capture.loc['baseline', 'up_capture']:+.3%}；
  下跌捕获率变化：
  {capture.loc['candidate', 'down_capture'] - capture.loc['baseline', 'down_capture']:+.3%}。
- 三个 HMM 种子的 CAGR 变化全部为正：
  {bool(seed_consistency['cagr_delta'].gt(0.0).all())}。
- 逐年剔除后的年化相对对数收益最小值：
  {leave_one_year_out['annualized_relative_log_return'].min():.3%}。
- 21 日区块重排的 18% MDD 违规率：生产
  {baseline_tail['probability_breach_18pct']:.1%}，候选
  {candidate_tail['probability_breach_18pct']:.1%}；候选 5% 尾部 MDD 为
  {candidate_tail['max_drawdown_5pct']:.2%}。

点估计门槛与统计/多重检验门槛分开。只有两者同时通过才允许升级生产；
否则最多进入影子执行。
"""
    (DESTINATION / "decision.md").write_text(report, encoding="utf-8")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    normal = {
        "baseline": load_daily(BASELINE),
        "candidate": load_daily(CANDIDATE),
    }
    cost15 = {
        "baseline": load_daily(BASELINE_COST15),
        "candidate": load_daily(CANDIDATE_COST15),
    }
    extended = {
        "baseline": load_daily(BASELINE_2012),
        "candidate": load_daily(CANDIDATE_2012),
    }
    rows: list[dict[str, float | str]] = []
    for scenario, frames, periods in (
        ("normal", normal, PERIODS),
        ("cost15", cost15, PERIODS),
        (
            "extended",
            extended,
            {"extended_2012_2025": ("2012-01-01", "2025-12-31")},
        ),
    ):
        for strategy, frame in frames.items():
            for period, (start, end) in periods.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "period": period,
                        **performance_metrics(
                            frame.loc[start:end, "net_return"]
                        ),
                    }
                )
    metrics = pd.DataFrame(rows).set_index(
        ["scenario", "strategy", "period"]
    )

    annual = pd.concat(
        {
            name: frame.loc["2015":"2025", "net_return"]
            for name, frame in normal.items()
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_baseline"] = (
        annual["candidate"] - annual["baseline"]
    )
    relative_log = pd.concat(
        {
            name: frame.loc["2015":"2025", "net_return"]
            for name, frame in normal.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    relative_log = (
        np.log1p(relative_log["candidate"])
        - np.log1p(relative_log["baseline"])
    )
    leave_one_year_out_rows = []
    for excluded_year in sorted(set(relative_log.index.year)):
        sample = relative_log.loc[relative_log.index.year != excluded_year]
        leave_one_year_out_rows.append(
            {
                "excluded_year": excluded_year,
                "annualized_relative_log_return": float(
                    sample.mean() * 252.0
                ),
                "remaining_observations": len(sample),
            }
        )
    leave_one_year_out = pd.DataFrame(
        leave_one_year_out_rows
    ).set_index("excluded_year")

    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    )
    growth = prices[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1)
    capture = pd.DataFrame(
        {
            name: {
                "up_capture": capture_ratio(
                    frame.loc["2015":"2025", "net_return"],
                    growth.loc["2015":"2025"],
                    "up",
                ),
                "down_capture": capture_ratio(
                    frame.loc["2015":"2025", "net_return"],
                    growth.loc["2015":"2025"],
                    "down",
                ),
            }
            for name, frame in normal.items()
        }
    ).T

    bootstrap_rows = []
    for scenario, frames, period in (
        ("normal", normal, slice("2015", "2025")),
        ("cost15", cost15, slice("2015", "2025")),
        ("extended", extended, slice("2012", "2025")),
    ):
        for block in (21, 63, 126):
            bootstrap_rows.append(
                {
                    "scenario": scenario,
                    "block_days": block,
                    **circular_block_bootstrap(
                        frames["candidate"].loc[period, "net_return"],
                        frames["baseline"].loc[period, "net_return"],
                        block_days=block,
                    ),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index(
        ["scenario", "block_days"]
    )

    seed_rows = []
    for seed in (7, 42, 123):
        baseline_member = pd.read_csv(
            OUTPUT / BASELINE / "members" / f"seed_{seed}" / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        ).loc["2015":"2025", "net_return"]
        candidate_member = pd.read_csv(
            OUTPUT
            / CANDIDATE
            / "members"
            / f"seed_{seed}"
            / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        ).loc["2015":"2025", "net_return"]
        baseline_metrics = performance_metrics(baseline_member)
        candidate_metrics = performance_metrics(candidate_member)
        seed_rows.append(
            {
                "seed": seed,
                "baseline_cagr": baseline_metrics["cagr"],
                "candidate_cagr": candidate_metrics["cagr"],
                "cagr_delta": (
                    candidate_metrics["cagr"] - baseline_metrics["cagr"]
                ),
                "baseline_sharpe": baseline_metrics["sharpe"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "baseline_mdd": baseline_metrics["max_drawdown"],
                "candidate_mdd": candidate_metrics["max_drawdown"],
            }
        )
    seed_consistency = pd.DataFrame(seed_rows).set_index("seed")

    baseline_development = normal["baseline"].loc[
        "2015":"2021", "net_return"
    ]
    differential = pd.concat(
        {
            name: (
                np.log1p(
                    load_daily(directory).loc[
                        "2015":"2021", "net_return"
                    ]
                )
                - np.log1p(baseline_development)
            )
            for name, directory in RECOVERY_FAMILY.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    family = pd.DataFrame(
        {
            block: family_reality_check(differential, block)
            for block in (21, 63, 126)
        }
    ).T
    family.index.name = "block_days"

    tail_rows = []
    relative_tail_rows = []
    for sample_name, frames, period in (
        ("complete_2015_2025", normal, slice("2015", "2025")),
        ("extended_2012_2025", extended, slice("2012", "2025")),
    ):
        aligned = pd.concat(
            {
                name: frame.loc[period, "net_return"]
                for name, frame in frames.items()
            },
            axis=1,
            join="inner",
        ).dropna()
        for block in (21, 63, 126):
            simulations = paired_circular_block_bootstrap(
                aligned["candidate"].to_numpy(),
                aligned["baseline"].to_numpy(),
                block,
                simulations=5_000,
                seed=20_260_724 + block,
            )
            for strategy in ("baseline", "candidate"):
                tail_rows.append(
                    {
                        "sample": sample_name,
                        **strategy_summary(
                            simulations,
                            block,
                            strategy,
                        ),
                    }
                )
            relative_tail_rows.append(
                {
                    "sample": sample_name,
                    "block_length": block,
                    "probability_candidate_cagr_higher": float(
                        simulations["cagr_delta"].gt(0.0).mean()
                    ),
                    "probability_candidate_drawdown_better": float(
                        simulations["max_drawdown_delta"].gt(0.0).mean()
                    ),
                    "median_cagr_delta": float(
                        simulations["cagr_delta"].median()
                    ),
                    "median_max_drawdown_delta": float(
                        simulations["max_drawdown_delta"].median()
                    ),
                }
            )
    tail_summary = pd.DataFrame(tail_rows).set_index(
        ["sample", "block_length", "strategy"]
    )
    relative_tail = pd.DataFrame(relative_tail_rows).set_index(
        ["sample", "block_length"]
    )

    metrics.to_csv(DESTINATION / "metrics.csv")
    annual.to_csv(DESTINATION / "annual_returns.csv")
    capture.to_csv(DESTINATION / "capture.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")
    seed_consistency.to_csv(DESTINATION / "seed_consistency.csv")
    leave_one_year_out.to_csv(DESTINATION / "leave_one_year_out.csv")
    family.to_csv(DESTINATION / "family_reality_check.csv")
    tail_summary.to_csv(DESTINATION / "tail_summary.csv")
    relative_tail.to_csv(DESTINATION / "relative_tail_summary.csv")
    write_decision(
        metrics,
        bootstrap,
        family,
        capture,
        seed_consistency,
        tail_summary,
        leave_one_year_out,
    )

    print(metrics[["cagr", "sharpe", "max_drawdown"]].round(6))
    print("\nSeed consistency:")
    print(seed_consistency.round(6).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string())
    print("\nFamily reality check:")
    print(family.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
