from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_hierarchical_regime_portfolios import (
    ASSETS,
    STATE_ORDER,
    build_causal_signals,
    build_factorized_daily_states,
    development_overlay_selection,
    load_regime_schedule,
    simulate_anchor_tilt,
)
from regime_strategy.report import performance_metrics


PRODUCTION = Path(
    "output/paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
)
DESTINATION = Path("output/regime_walkforward")
PRICE_PATH = Path("data/prices_vix_hedge.csv")
TRAINING_START = "2015-01-01"
TEST_YEARS = range(2019, 2026)
BLENDS = (0.0, 0.125, 0.25, 0.50)


def select_regularized_blend(
    development_metrics: pd.DataFrame,
    retention_fraction: float = 0.80,
) -> float:
    baseline = development_metrics.loc[0.0]
    eligible = development_metrics[
        (
            development_metrics["max_drawdown"]
            >= baseline["max_drawdown"] - 0.01
        )
        & (development_metrics["sharpe"] >= baseline["sharpe"])
    ].copy()
    eligible["cagr_delta"] = eligible["cagr"] - baseline["cagr"]
    if eligible.empty:
        return 0.0
    best_delta = float(eligible["cagr_delta"].max())
    if best_delta <= 0.0:
        return 0.0
    near_best = eligible[
        eligible["cagr_delta"] >= retention_fraction * best_delta
    ]
    return float(near_best.index.min())


def circular_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_724,
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
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for sample in range(samples):
        starts = generator.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[sample] = float(relative[indices].mean() * 252.0)
    observed = float(relative.mean() * 252.0)
    return pd.Series(
        {
            "annualized_relative_log_return": observed,
            "lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)),
            "probability_positive": float(np.mean(estimates > 0.0)),
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


def write_decision(
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    bootstrap: pd.DataFrame,
    capture: pd.DataFrame,
) -> None:
    baseline = metrics.loc["production"]
    candidate = metrics.loc["walkforward"]
    stressed = metrics.loc["walkforward_overlay_cost15"]
    accepted = int(selections["accepted_states"].sum())
    active_years = int(selections["selected_blend"].gt(0.0).sum())
    probability = float(bootstrap.loc[21, "probability_positive"])
    production_gate = (
        candidate["cagr"] >= baseline["cagr"] + 0.005
        and candidate["sharpe"] > baseline["sharpe"]
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.01
        and stressed["cagr"] > baseline["cagr"]
        and probability >= 0.90
    )
    verdict = (
        "通过生产候选门槛，进入影子执行"
        if production_gate
        else "未通过生产门槛，继续研究且不修改生产"
    )
    report = f"""# Regime overlay 逐年 Walk-forward

## 决策

**{verdict}**

每个测试年份只使用此前完整年份。2019 年使用 2015-2018，之后逐年扩展，
直至 2025 年使用 2015-2024。状态映射和 blend 每年重新选择；测试年份本身
不参与选择。

| 策略 | CAGR | 波动 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| 生产锚 | {baseline['cagr']:.3%} | {baseline['annual_volatility']:.3%} | {baseline['sharpe']:.3f} | {baseline['max_drawdown']:.3%} |
| Walk-forward overlay | {candidate['cagr']:.3%} | {candidate['annual_volatility']:.3%} | {candidate['sharpe']:.3f} | {candidate['max_drawdown']:.3%} |
| Overlay 成本翻倍 | {stressed['cagr']:.3%} | {stressed['annual_volatility']:.3%} | {stressed['sharpe']:.3f} | {stressed['max_drawdown']:.3%} |

- {len(TEST_YEARS)} 个测试年份中，{active_years} 年启用了 overlay。
- 所有年度合计接受 {accepted} 个 state-year 组合，其余均保持生产锚。
- 21 日区块 bootstrap 的正增量概率为 {probability:.1%}。
- 成长上涨捕获率变化为
  {capture.loc['walkforward', 'up_capture'] - capture.loc['production', 'up_capture']:+.3%}；
  下跌捕获率变化为
  {capture.loc['walkforward', 'down_capture'] - capture.loc['production', 'down_capture']:+.3%}。

生产门槛要求 CAGR 至少提高 0.5 个百分点、Sharpe 提高、最大回撤不恶化
超过 1 个百分点、额外 overlay 成本翻倍后仍跑赢，且区块 bootstrap
正增量概率至少 90%。
"""
    (DESTINATION / "decision.md").write_text(report, encoding="utf-8")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True)
    asset_returns = prices[ASSETS].pct_change(fill_method=None).dropna(how="any")
    production_daily = pd.read_csv(
        PRODUCTION / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    anchor_weights = pd.read_csv(
        PRODUCTION / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    schedule = load_regime_schedule(PRODUCTION)
    signals = build_causal_signals(prices)
    scheduled_states, daily_state = build_factorized_daily_states(
        schedule,
        signals,
        asset_returns.index,
    )

    candidate_parts: list[pd.Series] = []
    stressed_parts: list[pd.Series] = []
    baseline_parts: list[pd.Series] = []
    selection_rows: list[dict[str, float | int | str]] = []
    state_selection_rows: list[dict[str, float | int | str]] = []
    blend_rows: list[dict[str, float | int]] = []

    for test_year in TEST_YEARS:
        training_period = (TRAINING_START, f"{test_year - 1}-12-31")
        mapping, state_selection = development_overlay_selection(
            daily_state,
            asset_returns,
            anchor_weights,
            training_period=training_period,
        )
        candidate_runs: dict[
            float, tuple[pd.DataFrame, pd.DataFrame]
        ] = {}
        development_rows: list[dict[str, float]] = []
        for blend in BLENDS:
            daily, weights = simulate_anchor_tilt(
                daily_state,
                asset_returns,
                anchor_weights,
                production_daily,
                mapping,
                blend,
            )
            candidate_runs[blend] = (daily, weights)
            development_rows.append(
                {
                    "blend": blend,
                    **performance_metrics(
                        daily.loc[slice(*training_period), "net_return"]
                    ),
                }
            )
        development_metrics = pd.DataFrame(development_rows).set_index(
            "blend"
        )
        selected_blend = select_regularized_blend(development_metrics)
        selected_daily = candidate_runs[selected_blend][0]
        stressed_daily, _ = simulate_anchor_tilt(
            daily_state,
            asset_returns,
            anchor_weights,
            production_daily,
            mapping,
            selected_blend,
            cost_bps=15.0,
        )
        test_slice = slice(f"{test_year}-01-01", f"{test_year}-12-31")
        candidate_parts.append(selected_daily.loc[test_slice, "net_return"])
        stressed_parts.append(stressed_daily.loc[test_slice, "net_return"])
        baseline_parts.append(production_daily.loc[test_slice, "net_return"])
        accepted_states = state_selection[
            state_selection["selection_accepted"].eq(1)
        ]
        selection_rows.append(
            {
                "test_year": test_year,
                "training_start": training_period[0],
                "training_end": training_period[1],
                "selected_blend": selected_blend,
                "accepted_states": len(accepted_states),
                "accepted_state_names": "|".join(
                    accepted_states.index.tolist()
                ),
            }
        )
        for state in STATE_ORDER:
            state_selection_rows.append(
                {
                    "test_year": test_year,
                    "state": state,
                    **state_selection.loc[state].to_dict(),
                }
            )
        for blend, row in development_metrics.iterrows():
            blend_rows.append(
                {
                    "test_year": test_year,
                    "blend": blend,
                    **row.to_dict(),
                }
            )

    returns = pd.concat(
        {
            "production": pd.concat(baseline_parts).sort_index(),
            "walkforward": pd.concat(candidate_parts).sort_index(),
            "walkforward_overlay_cost15": (
                pd.concat(stressed_parts).sort_index()
            ),
        },
        axis=1,
        join="inner",
    ).dropna()
    metrics = pd.DataFrame(
        {
            strategy: performance_metrics(returns[strategy])
            for strategy in returns
        }
    ).T
    annual = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    annual["walkforward_minus_production"] = (
        annual["walkforward"] - annual["production"]
    )
    growth = (
        prices[["QQQ", "SEMIS"]]
        .pct_change(fill_method=None)
        .mean(axis=1)
        .reindex(returns.index)
    )
    capture = pd.DataFrame(
        {
            strategy: {
                "up_capture": capture_ratio(
                    returns[strategy], growth, "up"
                ),
                "down_capture": capture_ratio(
                    returns[strategy], growth, "down"
                ),
            }
            for strategy in returns
        }
    ).T
    bootstrap = pd.DataFrame(
        {
            block: circular_block_bootstrap(
                returns["walkforward"],
                returns["production"],
                block_days=block,
            )
            for block in (21, 63, 126)
        }
    ).T
    bootstrap.index.name = "block_days"
    selections = pd.DataFrame(selection_rows).set_index("test_year")
    state_selections = pd.DataFrame(state_selection_rows).set_index(
        ["test_year", "state"]
    )
    blend_history = pd.DataFrame(blend_rows).set_index(
        ["test_year", "blend"]
    )

    returns.to_csv(DESTINATION / "daily_returns.csv")
    metrics.to_csv(DESTINATION / "metrics.csv")
    annual.to_csv(DESTINATION / "annual_returns.csv")
    capture.to_csv(DESTINATION / "capture.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")
    selections.to_csv(DESTINATION / "annual_selections.csv")
    state_selections.to_csv(DESTINATION / "state_selections.csv")
    blend_history.to_csv(DESTINATION / "blend_history.csv")
    scheduled_states.to_csv(DESTINATION / "scheduled_states.csv")
    daily_state.rename("state").to_csv(DESTINATION / "daily_states.csv")
    write_decision(metrics, selections, bootstrap, capture)

    print(metrics[["cagr", "annual_volatility", "sharpe", "max_drawdown"]])
    print("\nAnnual selections:")
    print(selections.to_string())
    print("\nBootstrap:")
    print(bootstrap.to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
