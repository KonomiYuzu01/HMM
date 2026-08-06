from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_strategy.operations import (
    dollar_order_plan,
    growth_risk_snapshot,
    next_session_execution_status,
    onboarding_targets,
)
from regime_strategy.data import completed_us_daily_prices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a position-aware QQQ/SMH operational panel"
    )
    parser.add_argument(
        "--strategy-output",
        default="output/paper_core_growth_gold20_daily_risk_ensemble",
    )
    parser.add_argument("--account-value", type=float, default=700_000.0)
    parser.add_argument("--current-qqq", type=float, default=0.0)
    parser.add_argument("--current-smh", type=float, default=0.0)
    parser.add_argument("--current-gld", type=float, default=0.0)
    parser.add_argument("--current-bil", type=float, default=0.0)
    parser.add_argument("--current-vixy", type=float, default=0.0)
    parser.add_argument(
        "--validation-output",
        default="output/lev110_trendlev_industrymom_vixhedge4_validation",
    )
    parser.add_argument(
        "--output-dir", default="output/current_operational_panel"
    )
    args = parser.parse_args()

    strategy_output = Path(args.strategy_output)
    validation_output = Path(args.validation_output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = strategy_output / "run_metadata.json"
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_as_of = pd.Timestamp(model_metadata["price_as_of"])
    strategy_config = yaml.safe_load(
        Path(model_metadata["config_path"]).read_text(encoding="utf-8")
    )
    base_config = yaml.safe_load(
        Path(strategy_config["base_config"]).read_text(encoding="utf-8")
    )
    price_cache = Path(base_config["data"]["cache"])
    assets = list(base_config["data"]["tickers"])
    prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
    prices = completed_us_daily_prices(
        prices,
        source_modified_at=pd.Timestamp.fromtimestamp(
            price_cache.stat().st_mtime,
            tz="America/New_York",
        ),
    )
    prices = prices.reindex(columns=assets)
    execution_status, execution_date = next_session_execution_status(
        prices.index[-1]
    )
    if model_as_of.normalize() != prices.index[-1].normalize():
        execution_status = "STALE_MODEL"
    growth_assets = ["QQQ", "SEMIS"]
    cash_index = assets.index("CASH")
    historical_returns = prices.pct_change(fill_method=None).dropna(how="any")
    pair_returns = historical_returns[growth_assets]
    fast_volatility = pair_returns.rolling(20).std(ddof=1) * np.sqrt(252.0)
    slow_volatility = pair_returns.rolling(60).std(ddof=1) * np.sqrt(252.0)
    effective_volatility = fast_volatility.combine(slow_volatility, np.maximum)
    rolling_correlation = pair_returns["QQQ"].rolling(60).corr(
        pair_returns["SEMIS"]
    )
    rolling_beta = (
        pair_returns["QQQ"].rolling(60).cov(pair_returns["SEMIS"])
        / pair_returns["QQQ"].rolling(60).var()
    )
    risk_history = pd.DataFrame(
        {
            "qqq_effective_volatility": effective_volatility["QQQ"],
            "smh_effective_volatility": effective_volatility["SEMIS"],
            "smh_qqq_volatility_ratio": (
                effective_volatility["SEMIS"] / effective_volatility["QQQ"]
            ),
            "rolling_60d_correlation": rolling_correlation,
            "rolling_60d_smh_beta": rolling_beta,
        }
    ).dropna()
    risk_percentile_rows = []
    current_risk_history = risk_history.iloc[-1]
    for metric, value in current_risk_history.items():
        risk_percentile_rows.append(
            {
                "metric": metric,
                "current_value": value,
                "historical_percentile": float(
                    risk_history[metric].le(value).mean()
                ),
            }
        )
    risk_percentiles = pd.DataFrame(risk_percentile_rows).set_index("metric")
    risk_percentiles.to_csv(output_dir / "risk_percentiles.csv")

    target_frame = pd.read_csv(
        strategy_output / "next_target_weights.csv", index_col=0
    )
    model_series = target_frame["ensemble_current_sleeve_weight"].reindex(assets)
    model_weights = model_series.to_numpy(dtype=float)
    scenarios = onboarding_targets(
        model_weights,
        historical_returns,
        assets,
        growth_assets,
        cash_index,
        fast_days=20,
        slow_days=60,
        target_volatility=0.20,
    )

    current_dollars = np.zeros(len(assets), dtype=float)
    current_dollars[assets.index("QQQ")] = args.current_qqq
    current_dollars[assets.index("SEMIS")] = args.current_smh
    current_dollars[assets.index("GOLD")] = args.current_gld
    current_dollars[cash_index] = args.current_bil
    if "VIX_HEDGE" in assets:
        current_dollars[assets.index("VIX_HEDGE")] = args.current_vixy
    current_risky = float(
        current_dollars[
            [index for index, asset in enumerate(assets) if asset != "CASH"]
        ].sum()
    )
    selected_scenario = (
        "account_volatility_capped"
        if current_risky <= 1.0
        else "model_replication"
    )
    selected_target = scenarios[selected_scenario]

    latest_prices = prices.iloc[-1].to_numpy(dtype=float)
    order_plan = dollar_order_plan(
        selected_target,
        current_dollars,
        latest_prices,
        assets,
        args.account_value,
    )
    executable = execution_status in {"UPCOMING", "READY_FOR_OPEN"}
    order_plan["execution_status"] = execution_status
    order_plan["executable"] = executable
    if not executable:
        order_plan["action"] = "DRAFT_" + order_plan["action"].astype(str)
    order_plan.to_csv(output_dir / "order_plan.csv", index=False)

    scenario_frame = pd.DataFrame(scenarios, index=assets).T
    scenario_frame.to_csv(output_dir / "scenario_weights.csv")
    risk_rows = []
    for scenario, weights in scenarios.items():
        scenario_risk_assets = [
            asset
            for index, asset in enumerate(assets)
            if asset != "CASH" and weights[index] > 1e-8
        ]
        snapshot = growth_risk_snapshot(
            weights,
            historical_returns,
            assets,
            growth_assets,
            fast_days=20,
            slow_days=60,
            risk_assets=scenario_risk_assets,
        )
        risk_rows.append(
            {
                "scenario": scenario,
                "growth_exposure": float(
                    weights[[assets.index(asset) for asset in growth_assets]].sum()
                ),
                **snapshot,
                "one_week_one_sigma_dollars": (
                    snapshot["account_realized_volatility"]
                    / np.sqrt(52.0)
                    * args.account_value
                ),
            }
        )
    risk_frame = pd.DataFrame(risk_rows).set_index("scenario")
    risk_frame.to_csv(output_dir / "risk_scenarios.csv")

    diagnostics = pd.read_csv(
        strategy_output / "next_signal_diagnostics.csv", index_col=0
    )
    model_actions = diagnostics.loc["action"].astype(str)
    review_date = str(diagnostics.loc["estimated_rebalance_date"].iloc[0])
    member_risk_on = 0
    member_rows = []
    for member_dir in sorted((strategy_output / "members").glob("seed_*")):
        member_regimes = pd.read_csv(
            member_dir / "regimes.csv", index_col=0, parse_dates=True
        )
        latest = member_regimes.iloc[-1]
        member_risk_on += int(latest["risk_on_leverage"])
        member_rows.append(
            {
                "member": member_dir.name,
                "signal_date": member_regimes.index[-1],
                "risk_on": int(latest["risk_on_leverage"]),
                "zero_entry_overlay_active": int(
                    latest["asset_risk_overlay_active"]
                ),
                "current_growth_exposure": latest[
                    "previous_asset_risk_growth_exposure"
                ],
            }
        )
    pd.DataFrame(member_rows).set_index("member").to_csv(
        output_dir / "member_status.csv"
    )

    selected = pd.Series(selected_target, index=assets)
    selected_risk = risk_frame.loc[selected_scenario]
    selected_orders = order_plan[order_plan["trade_dollars"].abs() > 0.01]
    total_rounding_residual = float(
        selected_orders["rounding_residual_dollars"].sum()
    )
    order_lines = "\n".join(
        f"- {row.asset}: {row.action} ${abs(row.trade_dollars):,.0f} "
        f"（参考 {row.reference_trade_shares:,.2f} 份；整股 {row.whole_trade_shares:,.0f} 份；价格 ${row.reference_price:,.2f}）"
        for row in selected_orders.itertuples()
    )
    open_metrics = pd.read_csv(
        validation_output / "metrics_by_period_and_cost.csv"
    )
    primary_metrics = open_metrics[
        (open_metrics["period"] == "complete_2015_2025")
        & (open_metrics["strategy"] == "candidate")
        & np.isclose(open_metrics["cost_bps"], 7.5)
    ].iloc[0]
    tail_metrics = pd.read_csv(validation_output / "tail_bootstrap.csv")
    primary_tail = tail_metrics[
        (tail_metrics["sample"] == "complete_2015_2025_open")
        & (tail_metrics["strategy"] == "candidate")
    ].sort_values("block_days")
    tail_breach_summary = " / ".join(
        f"{int(row.block_days)}日 {row.probability_breach_18pct:.1%}"
        for row in primary_tail.itertuples()
    )
    short_tail = primary_tail.iloc[0]
    reality_check = pd.read_csv(
        validation_output / "family_reality_check.csv", index_col=0
    ).iloc[:, 0]
    hedge_active = int(
        pd.to_numeric(
            diagnostics.loc["conditional_vix_hedge_active"], errors="coerce"
        )
        .fillna(0.0)
        .max()
    )
    hedge_term_ratio = float(
        pd.to_numeric(
            diagnostics.loc["conditional_vix_hedge_term_ratio"], errors="coerce"
        ).mean()
    )
    model_target = target_frame["ensemble_current_sleeve_weight"]
    panel = f"""# 当前操作 Panel

- 数据截止：{prices.index[-1].date().isoformat()}
- 模型截止：{model_as_of.date().isoformat()}
- 理论执行日：{execution_date.date().isoformat()}
- 执行窗口状态：{execution_status}
- 账户规模：${args.account_value:,.0f}
- 模型动作：{', '.join(model_actions.unique())}
- 三成员 risk-on：{member_risk_on}/3
- 下次计划评估：{review_date}
- 当前持有风险资产（QQQ/SMH/GLD/VIXY）：${current_risky:,.0f}
- VIX/VIX3M：{hedge_term_ratio:.3f}；条件性 VIXY 对冲：{'开启' if hedge_active else '关闭'}
- 面板动作：{'模型或执行窗口失效，等待完整收盘并重算，当前禁止追单' if not executable else ('零仓位全账户 20% 压力波动接入' if selected_scenario == 'account_volatility_capped' else '向模型目标再平衡')}

## 长期模型目标

- QQQ：{model_target.get('QQQ', 0.0):.2%}
- SMH：{model_target.get('SEMIS', 0.0):.2%}
- GLD：{model_target.get('GOLD', 0.0):.2%}
- BIL/融资现金：{model_target.get('CASH', 0.0):.2%}
- VIXY：{model_target.get('VIX_HEDGE', 0.0):.2%}

## 建议目标（{selected_scenario}）

- QQQ：{selected['QQQ']:.2%}
- SMH：{selected['SEMIS']:.2%}
- GLD：{selected['GOLD']:.2%}
- BIL：{selected['CASH']:.2%}
- VIXY：{selected.get('VIX_HEDGE', 0.0):.2%}
- 账户级压力实现波动：{selected_risk['account_realized_volatility']:.2%}
- SMH 风险贡献占比：{selected_risk['smh_risk_share']:.1%}
- GLD 风险贡献占比：{selected_risk.get('gold_risk_share', 0.0):.1%}
- 一周 1σ 波动金额（不是最大损失）：${selected_risk['one_week_one_sigma_dollars']:,.0f}
- 冻结窗开盘执行 CAGR / Sharpe / 最大回撤：{primary_metrics['cagr']:.2%} / {primary_metrics['sharpe']:.2f} / {primary_metrics['max_drawdown']:.2%}
- 区块重采样突破 18% 回撤频率：{tail_breach_summary}
- 21 日区块重采样 5% 尾部 MDD：约 {short_tail['max_drawdown_5pct']:.1%}（按当前账户约 ${abs(args.account_value * short_tail['max_drawdown_5pct']):,.0f}）
- 18 候选 family-wise Reality Check p 值：{reality_check['familywise_reality_check_p_value']:.2%}

## 当前风险结构

- QQQ 压力实现波动：{risk_percentiles.loc['qqq_effective_volatility', 'current_value']:.2%}（历史百分位 {risk_percentiles.loc['qqq_effective_volatility', 'historical_percentile']:.1%}）
- SMH 压力实现波动：{risk_percentiles.loc['smh_effective_volatility', 'current_value']:.2%}（历史百分位 {risk_percentiles.loc['smh_effective_volatility', 'historical_percentile']:.1%}）
- SMH/QQQ 波动比：{risk_percentiles.loc['smh_qqq_volatility_ratio', 'current_value']:.2f}（历史百分位 {risk_percentiles.loc['smh_qqq_volatility_ratio', 'historical_percentile']:.1%}）
- 60 日相关系数：{risk_percentiles.loc['rolling_60d_correlation', 'current_value']:.3f}（历史百分位 {risk_percentiles.loc['rolling_60d_correlation', 'historical_percentile']:.1%}）
- SMH 对 QQQ 的 60 日 beta：{risk_percentiles.loc['rolling_60d_smh_beta', 'current_value']:.2f}（历史百分位 {risk_percentiles.loc['rolling_60d_smh_beta', 'historical_percentile']:.1%}）

## {'失效订单草案（禁止执行）' if not executable else '参考订单'}

{order_lines}

- 全部使用整股时，未分配取整残差约 ${total_rounding_residual:,.0f}，应保留为券商现金，不为了凑满仓而额外增加 SMH。

## 执行清单

1. 仅在 {execution_date.date().isoformat()} 开盘窗口执行；若错过 09:30 ET，等待当日完整收盘重算，禁止盘中追价。
2. 以上一节的目标美元为准；参考份额按 {prices.index[-1].date().isoformat()} 收盘价估算，开盘价格变化后必须重新换算份额。
3. 先完成 QQQ、SMH、GLD 风险资产订单，再用剩余目标资金配置 BIL；整股取整残差留作券商现金。
4. 成交后立即保存实际成交价、份额和未成交数量，并用实际持仓重新生成下一版 Panel；部分成交不等于授权扩大 SMH。
5. `executable={str(executable)}` 只表示信号时间有效，不表示 18% 回撤保证，也不替代税务、账户类型与个人风险承受能力复核。

## 规则解释

模型的 HOLD 指“已有模型仓位无需交易”，不等于零仓位账户继续持币。零仓位接入先保留模型 GLD 配置、把部分 SMH 高 beta 风险转成 QQQ，再按 QQQ+SMH+GLD（对冲开启时包含 VIXY）的 20/60 日联合压力协方差同比例缩放全部风险资产到 20% 波动预算，释放部分进入 BIL；这是首次建仓保护，下次既有评估再决定是否向长期模型目标增仓。长期模型仅在 QQQ 与 SMH 均高于各自 252 日前水平时允许把总敞口提高到 110%，并在每个交易日执行只降不升的跳跃感知压力波控；当上一完整收盘 VIX 高于 VIX3M 时，从现金融资 4% VIXY 对冲，否则 VIXY 为零。信号必须在对应的下一交易日开盘窗口执行；冻结窗开盘执行代理的历史 MDD 为 {primary_metrics['max_drawdown']:.2%}，不构成未来上限，区块重采样路径可能显著更深。若面板显示 MISSED 或 STALE_MODEL，必须等当日完整收盘后刷新重算，不能执行草案或主观追单。参考份额未考虑盘中价格、税务、买卖价差和券商是否支持碎股。
"""
    (output_dir / "panel.md").write_text(panel, encoding="utf-8")
    print(panel)
    print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
