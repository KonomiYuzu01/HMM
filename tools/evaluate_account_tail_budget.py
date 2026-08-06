from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_drawdown_uncertainty import (
    paired_circular_block_bootstrap,
    strategy_summary,
)
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/account_tail_budget")
TARGET_ACCOUNT_TAIL = 0.18
ACCOUNT_VALUE = 700_000.0
MONTE_CARLO_SEEDS = (7, 42, 123, 1001, 2003, 3001, 4001, 5003, 6007, 7001)


def main() -> None:
    tails = pd.read_csv("output/risk_capacity_frontier/bootstrap_tail_summary.csv")
    source = tails.loc[
        (tails["growth_target_volatility"] == 0.20)
        & (tails["block_length"] == 21)
        & (tails["strategy"] == "candidate")
    ].iloc[0]
    source_tail = abs(float(source["max_drawdown_5pct"]))
    sleeve = min(TARGET_ACCOUNT_TAIL / source_tail, 1.0)

    strategy = pd.read_csv(
        "output/open_execution_validation/gold_20_daily_asset_cap_daily.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]
    prices = pd.read_csv(
        "data/adjusted_open_close_2011_present.csv",
        index_col=0,
        parse_dates=True,
    )
    aligned = pd.concat(
        [
            strategy.rename("strategy"),
            prices["close_CASH"].pct_change(fill_method=None).rename("BIL"),
            prices["close_SPX"].pct_change(fill_method=None).rename("SPY"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    mixed = sleeve * aligned["strategy"] + (1.0 - sleeve) * aligned["BIL"]
    metrics = performance_metrics(mixed)

    current = pd.read_csv(
        "output/current_operational_panel/scenario_weights.csv",
        index_col=0,
    ).loc["account_volatility_capped"]
    weights = (current * sleeve).copy()
    weights["CASH"] += 1.0 - sleeve
    weights = weights.rename("target_weight").to_frame()
    weights["target_dollars"] = weights["target_weight"] * ACCOUNT_VALUE

    bootstrap_rows = []
    for block_length in (21, 63, 126):
        simulations = paired_circular_block_bootstrap(
            mixed.to_numpy(),
            aligned["SPY"].to_numpy(),
            block_length=block_length,
            simulations=5_000,
            seed=20260723 + block_length,
        )
        bootstrap_rows.append(
            strategy_summary(simulations, block_length, "candidate")
        )
    bootstrap = pd.DataFrame(bootstrap_rows).set_index("block_length")
    stability_rows = []
    for seed in MONTE_CARLO_SEEDS:
        simulations = paired_circular_block_bootstrap(
            mixed.to_numpy(),
            aligned["SPY"].to_numpy(),
            block_length=21,
            simulations=5_000,
            seed=seed,
        )
        stability_rows.append(
            {
                "seed": seed,
                "probability_breach_18pct": float(
                    (simulations["candidate_max_drawdown"] < -0.18).mean()
                ),
                "max_drawdown_5pct": float(
                    simulations["candidate_max_drawdown"].quantile(0.05)
                ),
            }
        )
    stability = pd.DataFrame(stability_rows).set_index("seed")
    summary = pd.DataFrame(
        [
            {
                "target_account_tail": TARGET_ACCOUNT_TAIL,
                "source_21d_block_5pct_mdd": -source_tail,
                "strategy_sleeve": sleeve,
                "account_value": ACCOUNT_VALUE,
                "linearized_tail_dollars": TARGET_ACCOUNT_TAIL * ACCOUNT_VALUE,
                **metrics,
            }
        ]
    )
    decision = f"""# 账户尾部风险预算诊断

若把 18% 解释为账户可承受的重采样 5% 尾部，而不是历史 MDD 目标，则用 20% 生产策略的 21 日区块尾部 {source_tail:.2%} 进行线性缩放，策略袖套只能为 {sleeve:.2%}，其余持 BIL。

- 当前映射：QQQ {weights.loc['QQQ', 'target_weight']:.2%}（${weights.loc['QQQ', 'target_dollars']:,.0f}）、SMH {weights.loc['SEMIS', 'target_weight']:.2%}（${weights.loc['SEMIS', 'target_dollars']:,.0f}）、GLD {weights.loc['GOLD', 'target_weight']:.2%}（${weights.loc['GOLD', 'target_dollars']:,.0f}）、BIL {weights.loc['CASH', 'target_weight']:.2%}（${weights.loc['CASH', 'target_dollars']:,.0f}）。
- 历史开盘口径：CAGR {metrics['cagr']:.2%}、Sharpe {metrics['sharpe']:.3f}、MDD {metrics['max_drawdown']:.2%}。
- 21/63/126 日区块突破 18% 的频率：{bootstrap.loc[21, 'probability_breach_18pct']:.2%}/{bootstrap.loc[63, 'probability_breach_18pct']:.2%}/{bootstrap.loc[126, 'probability_breach_18pct']:.2%}。
- 十个固定 bootstrap 种子的 21 日区块违规频率范围为 {stability['probability_breach_18pct'].min():.2%}–{stability['probability_breach_18pct'].max():.2%}，5% MDD 分位范围为 {stability['max_drawdown_5pct'].min():.2%}–{stability['max_drawdown_5pct'].max():.2%}。
- 对 70 万美元，18% 是 126,000 美元；这仍不是损失保证。缩放后的 CAGR 明显低于收益目标，因此它是硬风险容量备选，不是生产 Panel 的自动替代。
"""

    DESTINATION.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DESTINATION / "summary.csv", index=False)
    weights.to_csv(DESTINATION / "weights.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap_summary.csv")
    stability.to_csv(DESTINATION / "monte_carlo_stability.csv")
    (DESTINATION / "decision.md").write_text(decision, encoding="utf-8")
    print(summary.to_string(index=False))
    print(weights.to_string())
    print(bootstrap.to_string())
    print(stability.to_string())
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
