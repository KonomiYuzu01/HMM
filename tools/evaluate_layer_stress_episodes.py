from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE = Path("output/open_execution_validation")
DESTINATION = Path("output/layer_stress_episodes")
STRATEGIES = {
    "qqq_smh_hmm_core": "growth_primary_daily.csv",
    "add_20pct_gold": "gold_20_unlevered_daily.csv",
    "production_daily_cap": "gold_20_daily_asset_cap_daily.csv",
    "jump_aware_shadow": "gold_20_jump_aware_daily_cap_daily.csv",
}
EPISODES = {
    "2018_q4_selloff": ("2018-09-20", "2018-12-24"),
    "2020_pandemic_crash": ("2020-02-19", "2020-03-23"),
    "2022_tightening_bear": ("2022-01-03", "2022-10-14"),
}
HORIZONS = (21, 63, 126)
MATERIAL_DRAWDOWN_IMPROVEMENT = 0.001


def path_metrics(returns: pd.Series) -> tuple[float, float]:
    equity = pd.concat(
        [pd.Series([1.0]), (1.0 + returns).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity / equity.cummax() - 1.0
    return float(equity.iloc[-1] - 1.0), float(drawdown.min())


def main() -> None:
    returns = {}
    for name, filename in STRATEGIES.items():
        frame = pd.read_csv(SOURCE / filename, index_col=0, parse_dates=True)
        returns[name] = frame["net_return"].astype(float)
    daily = pd.concat(returns, axis=1, join="inner").sort_index()
    if daily.isna().any().any():
        raise RuntimeError("Aligned stress-test returns contain missing values")

    episode_rows = []
    for episode, (start, end) in EPISODES.items():
        sample = daily.loc[start:end]
        if sample.empty:
            raise RuntimeError(f"Missing observations for {episode}")
        for strategy in daily.columns:
            total_return, max_drawdown = path_metrics(sample[strategy])
            episode_rows.append(
                {
                    "episode": episode,
                    "strategy": strategy,
                    "observations": len(sample),
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                }
            )
    episodes = pd.DataFrame(episode_rows).set_index(["episode", "strategy"])

    rolling_rows = []
    for horizon in HORIZONS:
        compounded = (1.0 + daily).rolling(horizon).apply(
            lambda values: values.prod() - 1.0,
            raw=True,
        )
        for strategy in daily.columns:
            series = compounded[strategy].dropna()
            end_date = series.idxmin()
            rolling_rows.append(
                {
                    "horizon_sessions": horizon,
                    "strategy": strategy,
                    "worst_return": float(series.loc[end_date]),
                    "window_end": end_date.date().isoformat(),
                }
            )
    rolling = pd.DataFrame(rolling_rows).set_index(
        ["horizon_sessions", "strategy"]
    )

    production = "production_daily_cap"
    core = "qqq_smh_hmm_core"
    gold = "add_20pct_gold"
    lines = [
        "# 跨危机层级压力测试",
        "",
        "口径与生产消融一致：2015–2025、隔日开盘执行代理、7.5 bps。危机区间事先固定；另用固定长度最差滚动收益减少日期边界依赖。",
        "",
        "## 固定危机区间",
        "",
        "| 区间 | 核心回报/MDD | +GLD 回报/MDD | 生产回报/MDD | 跳跃感知回报/MDD |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for episode in EPISODES:
        values = []
        for strategy in STRATEGIES:
            row = episodes.loc[(episode, strategy)]
            values.append(f"{row['total_return']:.2%} / {row['max_drawdown']:.2%}")
        lines.append(f"| {episode} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 最差固定长度累计收益",
            "",
            "| 交易日 | 核心 | +GLD | 生产 | 跳跃感知 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in HORIZONS:
        values = [
            f"{rolling.loc[(horizon, strategy), 'worst_return']:.2%}"
            for strategy in STRATEGIES
        ]
        lines.append(f"| {horizon} | " + " | ".join(values) + " |")

    production_wins = 0
    for episode in EPISODES:
        if (
            episodes.loc[(episode, production), "max_drawdown"]
            - episodes.loc[(episode, gold), "max_drawdown"]
            >= MATERIAL_DRAWDOWN_IMPROVEMENT
        ):
            production_wins += 1
    worst_63_core = rolling.loc[(63, core), "worst_return"]
    worst_63_production = rolling.loc[(63, production), "worst_return"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"每日压力波控在 {production_wins}/{len(EPISODES)} 个预设危机区间把 GLD 版的区间内最大回撤实质改善至少 {MATERIAL_DRAWDOWN_IMPROVEMENT:.1%}；2018 Q4 基本没有额外作用。最差 63 日累计收益从核心的 {worst_63_core:.2%} 改善到生产版的 {worst_63_production:.2%}。这支持它在急剧或持续高波动冲击中发挥作用，而不只是为全样本最大回撤日期定制；它不是所有下跌环境都会触发的保险，任何单一历史路径也不能把未来回撤锁在 18%。",
        ]
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(DESTINATION / "episode_metrics.csv")
    rolling.to_csv(DESTINATION / "worst_rolling_returns.csv")
    (DESTINATION / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(episodes.to_string())
    print(rolling.to_string())
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
