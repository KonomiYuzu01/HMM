from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


SOURCE = Path("output/open_execution_validation/metrics.csv")
DESTINATION = Path("output/layer_ablation")
EXECUTION = "open_execution_proxy"
LAYERS = {
    "qqq_smh_hmm_core": "growth_primary",
    "add_20pct_gold": "gold_20_unlevered",
    "add_daily_stress_cap": "gold_20_daily_asset_cap",
    "use_jump_aware_slow_vol": "gold_20_jump_aware_daily_cap",
}
DAILY_FILES = {
    "qqq_smh_hmm_core": "growth_primary_daily.csv",
    "add_20pct_gold": "gold_20_unlevered_daily.csv",
    "add_daily_stress_cap": "gold_20_daily_asset_cap_daily.csv",
    "use_jump_aware_slow_vol": "gold_20_jump_aware_daily_cap_daily.csv",
}
METRICS = [
    "cagr",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
]
PERIODS = {
    "development_2015_2020": ("2015-01-01", "2020-12-31"),
    "holdout_2021_2025": ("2021-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}
START_YEARS = range(2015, 2022)


def percentage_points(value: float) -> float:
    return 100.0 * value


def main() -> None:
    frame = pd.read_csv(SOURCE)
    selected = frame.loc[
        (frame["execution"] == EXECUTION)
        & frame["strategy"].isin(LAYERS.values())
    ].copy()
    selected = selected.set_index("strategy")
    missing = set(LAYERS.values()) - set(selected.index)
    if missing:
        raise RuntimeError(f"Missing ablation strategies: {sorted(missing)}")

    rows = []
    for layer, strategy in LAYERS.items():
        source = selected.loc[strategy]
        rows.append(
            {
                "layer": layer,
                "strategy": strategy,
                **{metric: float(source[metric]) for metric in METRICS},
            }
        )
    metrics = pd.DataFrame(rows).set_index("layer")

    effects = []
    layer_names = list(metrics.index)
    for before, after in zip(layer_names, layer_names[1:]):
        effects.append(
            {
                "change": f"{before} -> {after}",
                "cagr_delta_pp": percentage_points(
                    metrics.loc[after, "cagr"] - metrics.loc[before, "cagr"]
                ),
                "volatility_delta_pp": percentage_points(
                    metrics.loc[after, "annual_volatility"]
                    - metrics.loc[before, "annual_volatility"]
                ),
                "sharpe_delta": (
                    metrics.loc[after, "sharpe"] - metrics.loc[before, "sharpe"]
                ),
                "max_drawdown_improvement_pp": percentage_points(
                    metrics.loc[after, "max_drawdown"]
                    - metrics.loc[before, "max_drawdown"]
                ),
                "calmar_delta": (
                    metrics.loc[after, "calmar"] - metrics.loc[before, "calmar"]
                ),
            }
        )
    effects_frame = pd.DataFrame(effects).set_index("change")

    implementation_rows = []
    period_rows = []
    start_year_rows = []
    for layer, filename in DAILY_FILES.items():
        daily = pd.read_csv(
            SOURCE.parent / filename,
            index_col=0,
            parse_dates=True,
        )
        traded = daily["traded_at_open"].eq(1)
        implementation_rows.append(
            {
                "layer": layer,
                "annualized_one_way_turnover": float(
                    daily["open_turnover"].mean() * 252.0
                ),
                "trade_day_share": float(traded.mean()),
                "trade_days": int(traded.sum()),
                "average_turnover_when_traded": float(
                    daily.loc[traded, "open_turnover"].mean()
                ),
                "annualized_cost": float(daily["cost"].mean() * 252.0),
            }
        )
        for period, (start, end) in PERIODS.items():
            sample = daily.loc[start:end, "net_return"]
            if sample.empty:
                raise RuntimeError(f"Missing observations for {layer}/{period}")
            period_rows.append(
                {
                    "period": period,
                    "layer": layer,
                    **{
                        metric: value
                        for metric, value in performance_metrics(sample).items()
                        if metric in METRICS
                    },
                }
            )
        for start_year in START_YEARS:
            sample = daily.loc[f"{start_year}-01-01":"2025-12-31", "net_return"]
            start_year_rows.append(
                {
                    "start_year": start_year,
                    "layer": layer,
                    **{
                        metric: value
                        for metric, value in performance_metrics(sample).items()
                        if metric in METRICS
                    },
                }
            )
    implementation = pd.DataFrame(implementation_rows).set_index("layer")
    period_metrics = pd.DataFrame(period_rows).set_index(["period", "layer"])
    start_year_metrics = pd.DataFrame(start_year_rows).set_index(
        ["start_year", "layer"]
    )

    core = metrics.loc["qqq_smh_hmm_core"]
    production = metrics.loc["add_daily_stress_cap"]
    gold = metrics.loc["add_20pct_gold"]
    jump = metrics.loc["use_jump_aware_slow_vol"]
    development = period_metrics.loc["development_2015_2020"]
    holdout = period_metrics.loc["holdout_2021_2025"]
    start_year_checks = []
    for start_year in START_YEARS:
        cohort = start_year_metrics.loc[start_year]
        start_year_checks.append(
            {
                "start_year": start_year,
                "gold_cagr_improves": bool(
                    cohort.loc["add_20pct_gold", "cagr"]
                    > cohort.loc["qqq_smh_hmm_core", "cagr"]
                ),
                "gold_mdd_improves": bool(
                    cohort.loc["add_20pct_gold", "max_drawdown"]
                    > cohort.loc["qqq_smh_hmm_core", "max_drawdown"]
                ),
                "daily_cap_mdd_improves": bool(
                    cohort.loc["add_daily_stress_cap", "max_drawdown"]
                    > cohort.loc["add_20pct_gold", "max_drawdown"]
                ),
                "production_mdd_within_18pct": bool(
                    cohort.loc["add_daily_stress_cap", "max_drawdown"] >= -0.18
                ),
                "jump_cagr_improves": bool(
                    cohort.loc["use_jump_aware_slow_vol", "cagr"]
                    > cohort.loc["add_daily_stress_cap", "cagr"]
                ),
            }
        )
    start_year_check_frame = pd.DataFrame(start_year_checks).set_index("start_year")
    check_counts = start_year_check_frame.sum().astype(int)
    total_mdd_improvement = percentage_points(
        production["max_drawdown"] - core["max_drawdown"]
    )
    gold_mdd_improvement = percentage_points(
        gold["max_drawdown"] - core["max_drawdown"]
    )
    cap_mdd_improvement = percentage_points(
        production["max_drawdown"] - gold["max_drawdown"]
    )
    report = f"""# 生产路径结构消融

口径：2015–2025、隔日开盘执行代理、单边换仓成本 7.5 bps。四个配置逐层只改变一个主要结构，因此比跨策略横向排名更接近机制消融；但 HMM、GLD 与波控仍会经路径依赖相互作用，不能解释为严格因果分解。

| 层级 | CAGR | 年化波动 | Sharpe | 最大回撤 | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: |
| QQQ/SMH HMM 核心 | {core['cagr']:.2%} | {core['annual_volatility']:.2%} | {core['sharpe']:.3f} | {core['max_drawdown']:.2%} | {core['calmar']:.3f} |
| 加入 20% GLD | {gold['cagr']:.2%} | {gold['annual_volatility']:.2%} | {gold['sharpe']:.3f} | {gold['max_drawdown']:.2%} | {gold['calmar']:.3f} |
| 加入每日压力波控（生产） | {production['cagr']:.2%} | {production['annual_volatility']:.2%} | {production['sharpe']:.3f} | {production['max_drawdown']:.2%} | {production['calmar']:.3f} |
| 慢波动改为跳跃感知（影子） | {jump['cagr']:.2%} | {jump['annual_volatility']:.2%} | {jump['sharpe']:.3f} | {jump['max_drawdown']:.2%} | {jump['calmar']:.3f} |

## 边际作用

- GLD 层：CAGR 变化 {percentage_points(gold['cagr'] - core['cagr']):+.2f} 个百分点，最大回撤改善 {gold_mdd_improvement:+.2f} 个百分点。
- 每日压力波控层：CAGR 变化 {percentage_points(production['cagr'] - gold['cagr']):+.2f} 个百分点，最大回撤改善 {cap_mdd_improvement:+.2f} 个百分点。
- 跳跃感知层：CAGR 变化 {percentage_points(jump['cagr'] - production['cagr']):+.2f} 个百分点，最大回撤改善 {percentage_points(jump['max_drawdown'] - production['max_drawdown']):+.4f} 个百分点。
- 从 QQQ/SMH HMM 核心到生产版，最大回撤共改善 {total_mdd_improvement:.2f} 个百分点；其中 GLD 路径贡献约 {gold_mdd_improvement / total_mdd_improvement:.1%}，每日压力波控路径贡献约 {cap_mdd_improvement / total_mdd_improvement:.1%}。这是路径归因，不是统计因果贡献。

## 决策含义

低回撤不是单独由 HMM 产生：在同一 HMM 核心之上，GLD 和每日压力波控各承担了约一半的历史回撤改善。GLD 同时改善 CAGR 与回撤，是结构上最有价值的一层；每日波控用约 {abs(percentage_points(production['cagr'] - gold['cagr'])):.2f} 个百分点 CAGR 换取约 {cap_mdd_improvement:.2f} 个百分点回撤缓冲，是满足 18% 历史约束所付的明确保险费。跳跃感知只回收约 {percentage_points(jump['cagr'] - production['cagr']):.2f} 个百分点 CAGR，尚不足以证明应替换生产估计器。

每日波控把交易日从 {implementation.loc['add_20pct_gold', 'trade_days']:.0f} 增加到 {implementation.loc['add_daily_stress_cap', 'trade_days']:.0f}，但年化单边换手只从 {implementation.loc['add_20pct_gold', 'annualized_one_way_turnover']:.2f} 倍升到 {implementation.loc['add_daily_stress_cap', 'annualized_one_way_turnover']:.2f} 倍，7.5 bps 口径年化成本仅增加约 {percentage_points(implementation.loc['add_daily_stress_cap', 'annualized_cost'] - implementation.loc['add_20pct_gold', 'annualized_cost']):.02f} 个百分点。因此 CAGR 牺牲主要来自高波动期降低风险暴露，而不是额外换手成本。

## 开发期与留出期

| 分段 | 核心 CAGR/MDD | +GLD CAGR/MDD | 生产 CAGR/MDD | 跳跃感知 CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2015–2020 开发 | {development.loc['qqq_smh_hmm_core', 'cagr']:.2%} / {development.loc['qqq_smh_hmm_core', 'max_drawdown']:.2%} | {development.loc['add_20pct_gold', 'cagr']:.2%} / {development.loc['add_20pct_gold', 'max_drawdown']:.2%} | {development.loc['add_daily_stress_cap', 'cagr']:.2%} / {development.loc['add_daily_stress_cap', 'max_drawdown']:.2%} | {development.loc['use_jump_aware_slow_vol', 'cagr']:.2%} / {development.loc['use_jump_aware_slow_vol', 'max_drawdown']:.2%} |
| 2021–2025 留出 | {holdout.loc['qqq_smh_hmm_core', 'cagr']:.2%} / {holdout.loc['qqq_smh_hmm_core', 'max_drawdown']:.2%} | {holdout.loc['add_20pct_gold', 'cagr']:.2%} / {holdout.loc['add_20pct_gold', 'max_drawdown']:.2%} | {holdout.loc['add_daily_stress_cap', 'cagr']:.2%} / {holdout.loc['add_daily_stress_cap', 'max_drawdown']:.2%} | {holdout.loc['use_jump_aware_slow_vol', 'cagr']:.2%} / {holdout.loc['use_jump_aware_slow_vol', 'max_drawdown']:.2%} |

## 起点扰动

固定 2025-12-31 终点，在 2015–2021 七个起点中：GLD 的 CAGR 改善通过 {check_counts['gold_cagr_improves']}/7，GLD 的 MDD 改善通过 {check_counts['gold_mdd_improves']}/7，每日波控相对 GLD 的 MDD 改善通过 {check_counts['daily_cap_mdd_improves']}/7，生产版 MDD 保持在 18% 内通过 {check_counts['production_mdd_within_18pct']}/7，跳跃感知 CAGR 改善通过 {check_counts['jump_cagr_improves']}/7。起点通过率只检验方向稳定性，不提供独立样本或显著性。
"""

    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DESTINATION / "metrics.csv")
    effects_frame.to_csv(DESTINATION / "incremental_effects.csv")
    implementation.to_csv(DESTINATION / "implementation.csv")
    period_metrics.to_csv(DESTINATION / "period_metrics.csv")
    start_year_metrics.to_csv(DESTINATION / "start_year_metrics.csv")
    start_year_check_frame.to_csv(DESTINATION / "start_year_checks.csv")
    (DESTINATION / "decision.md").write_text(report, encoding="utf-8")
    print(metrics.to_string())
    print(effects_frame.to_string())
    print(implementation.to_string())
    print(period_metrics.to_string())
    print(start_year_check_frame.to_string())
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
