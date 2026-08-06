from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_strategy.forward_monitoring import append_immutable
from regime_strategy.operations import next_session_execution_status


RELEASE = "2026-07-25-r8-v3"
PARENT_RELEASE = "2026-07-25-r8-v2"
OUTPUT = Path("output")
RESEARCH = OUTPUT / "regime_strategy_research_2026-07-25"
RELEASE_DIR = RESEARCH / "r8_release"
PANEL_DIR = OUTPUT / "current_operational_panel"
MONITORING_DIR = OUTPUT / "forward_monitoring"
PRODUCTION = "paper_core_growth_gold20_daily_risk_netted_ensemble"
DUAL = (
    "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
    "netted_ensemble"
)
CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
    "inverse_momentum_netted_ensemble"
)
CANDIDATE_CONFIG = Path(f"config/{CANDIDATE}.yaml")
ACCOUNT_VALUE = 700_000.0
MAX_TRANCHE_ONE_WAY_TURNOVER = 0.10
MAX_TOTAL_EXPLICIT_COST_NAV_BPS = 10.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_closure(path: Path) -> list[Path]:
    files: list[Path] = []
    current = path
    while True:
        files.append(current)
        config = yaml.safe_load(current.read_text(encoding="utf-8"))
        parent = config.get("parent_config")
        if parent is None:
            base = config.get("base_config")
            if base is not None:
                files.append(Path(str(base)))
            break
        current = Path(str(parent))
    return files


def load_target(directory: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "next_target_weights.csv",
        index_col=0,
    )["ensemble_current_sleeve_weight"]


def load_metadata(directory: str) -> dict[str, object]:
    return json.loads(
        (OUTPUT / directory / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )


def build_gate_table() -> pd.DataFrame:
    metrics = pd.read_csv(
        RESEARCH / "r8_statistical_validation" / "metrics.csv",
        index_col=[0, 1, 2],
    )
    proxy = pd.read_csv(
        RESEARCH / "r8_proxy_metrics.csv",
        index_col=0,
    )
    delays = pd.read_csv(
        RESEARCH / "r8_execution_delay.csv",
        index_col=[0, 1],
    )
    episodes = pd.read_csv(
        RESEARCH / "r8_episode_summary.csv",
        index_col=0,
    )
    family = pd.read_csv(
        RESEARCH / "r8_statistical_validation" / "family_reality_check.csv",
    )
    migration = json.loads(
        (RELEASE_DIR / "migration_summary.json").read_text(encoding="utf-8")
    )
    baseline = metrics.loc[
        ("normal", "baseline", "complete_2015_2025")
    ]
    candidate = metrics.loc[
        ("normal", "candidate", "complete_2015_2025")
    ]
    baseline_cost = metrics.loc[
        ("cost15", "baseline", "complete_2015_2025")
    ]
    candidate_cost = metrics.loc[
        ("cost15", "candidate", "complete_2015_2025")
    ]
    baseline_extended = metrics.loc[
        ("extended", "baseline", "extended_2012_2025")
    ]
    candidate_extended = metrics.loc[
        ("extended", "candidate", "extended_2012_2025")
    ]
    one_day = delays.loc[("r8_floor45_goodvol_bear20", 1)]
    selection_p_max = float(
        family["candidate_selection_adjusted_p_value"].max()
    )
    episode_min = float(
        episodes["annualized_after_largest"].min()
    )
    cost_bps = float(migration["estimated_total_explicit_cost_nav_bps"])
    rows = [
        {
            "gate": "frozen_cagr_and_sharpe",
            "pass": (
                candidate["cagr"] > baseline["cagr"]
                and candidate["sharpe"] > baseline["sharpe"]
            ),
            "evidence": (
                f"CAGR {candidate['cagr']:.4%} vs {baseline['cagr']:.4%}; "
                f"Sharpe {candidate['sharpe']:.4f} vs {baseline['sharpe']:.4f}"
            ),
        },
        {
            "gate": "historical_mdd",
            "pass": candidate["max_drawdown"] >= -0.18,
            "evidence": f"MDD {candidate['max_drawdown']:.4%}",
        },
        {
            "gate": "cost_15bps",
            "pass": candidate_cost["cagr"] > baseline_cost["cagr"],
            "evidence": (
                f"CAGR {candidate_cost['cagr']:.4%} vs "
                f"{baseline_cost['cagr']:.4%}"
            ),
        },
        {
            "gate": "extended_2012_and_proxy",
            "pass": (
                candidate_extended["cagr"] > baseline_extended["cagr"]
                and proxy.loc[
                    "r8_floor45_goodvol_bear20", "cagr"
                ]
                > proxy.loc["production", "cagr"]
            ),
            "evidence": (
                f"2012 CAGR {candidate_extended['cagr']:.4%} vs "
                f"{baseline_extended['cagr']:.4%}; proxy "
                f"{proxy.loc['r8_floor45_goodvol_bear20', 'cagr']:.4%} vs "
                f"{proxy.loc['production', 'cagr']:.4%}"
            ),
        },
        {
            "gate": "one_session_delay",
            "pass": one_day["cagr_delta_vs_production"] > 0.0,
            "evidence": (
                f"CAGR delta {one_day['cagr_delta_vs_production']:+.4%}; "
                f"MDD {one_day['max_drawdown']:.4%}"
            ),
        },
        {
            "gate": "episode_concentration",
            "pass": episode_min > 0.0,
            "evidence": (
                f"minimum annualized return after best episode "
                f"{episode_min:+.4%}"
            ),
        },
        {
            "gate": "family_selection_adjustment",
            "pass": selection_p_max <= 0.10,
            "evidence": (
                f"maximum selection-adjusted p={selection_p_max:.4f}"
            ),
        },
        {
            "gate": "staged_migration",
            "pass": (
                migration["maximum_tranche_one_way_turnover"]
                <= MAX_TRANCHE_ONE_WAY_TURNOVER
                and cost_bps <= MAX_TOTAL_EXPLICIT_COST_NAV_BPS
            ),
            "evidence": (
                f"max tranche turnover "
                f"{migration['maximum_tranche_one_way_turnover']:.4%}; "
                f"total explicit cost {cost_bps:.2f}bp NAV"
            ),
        },
    ]
    return pd.DataFrame(rows).set_index("gate")


def build_reference_orders(
    trades: pd.DataFrame,
    prices: pd.Series,
    execution_status: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tranche, changes in trades.iterrows():
        for asset in ("QQQ", "SEMIS", "GOLD", "CASH"):
            weight_change = float(changes[asset])
            trade_dollars = weight_change * ACCOUNT_VALUE
            price = float(prices[asset])
            rows.append(
                {
                    "tranche": int(tranche),
                    "asset": asset,
                    "action": (
                        "DRAFT_BUY" if trade_dollars > 0 else "DRAFT_SELL"
                    ),
                    "target_weight_change": weight_change,
                    "reference_trade_dollars": trade_dollars,
                    "reference_price": price,
                    "reference_trade_shares": trade_dollars / price,
                    "execution_status": execution_status,
                    "executable": False,
                    "blocker": "NEEDS_CURRENT_HOLDINGS_AND_FRESH_CLOSE",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    gates = build_gate_table()
    all_gates_pass = bool(gates["pass"].all())
    release_status = "APPROVED_STAGED" if all_gates_pass else "REJECTED"
    gates.to_csv(RELEASE_DIR / "production_admission_gates.csv")

    production_metadata = load_metadata(PRODUCTION)
    dual_metadata = load_metadata(DUAL)
    candidate_metadata = load_metadata(CANDIDATE)
    signal_dates = {
        str(production_metadata["price_as_of"]),
        str(dual_metadata["price_as_of"]),
        str(candidate_metadata["price_as_of"]),
    }
    if len(signal_dates) != 1:
        raise RuntimeError("Production, Dual, and R8 signal dates differ")
    signal_date = pd.Timestamp(signal_dates.pop())
    execution_status, execution_date = next_session_execution_status(
        signal_date
    )

    data_quality = candidate_metadata.get("data_quality", {})
    fallback_quality = dict(data_quality).get(
        "signal_history_fallbacks",
        {},
    )
    vix3m_quality = dict(fallback_quality).get("VIX3M", {})
    data_quality_pass = (
        dict(data_quality).get("price_as_of")
        == signal_date.date().isoformat()
        and dict(vix3m_quality).get("latest_source_date")
        == signal_date.date().isoformat()
    )
    if not data_quality_pass:
        execution_status = "STALE_SIGNAL"

    targets = pd.read_csv(
        RELEASE_DIR / "staged_target_weights.csv",
        index_col=0,
    )
    trades = pd.read_csv(
        RELEASE_DIR / "staged_trades.csv",
        index_col=0,
    )
    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    ).loc[signal_date]
    orders = build_reference_orders(trades, prices, execution_status)
    orders.to_csv(PANEL_DIR / "order_plan.csv", index=False)
    targets.to_csv(PANEL_DIR / "staged_target_weights.csv")
    trades.to_csv(PANEL_DIR / "staged_trades.csv")

    production_target = load_target(PRODUCTION)
    candidate_target = load_target(CANDIDATE)
    diagnostics = pd.read_csv(
        OUTPUT / CANDIDATE / "next_signal_diagnostics.csv",
        index_col=0,
    )
    migration = json.loads(
        (RELEASE_DIR / "migration_summary.json").read_text(encoding="utf-8")
    )
    activation = pd.read_csv(
        RELEASE_DIR / "historical_activation_summary.csv",
        index_col=0,
    ).loc["four_consecutive_sessions"]
    long_metrics = pd.read_csv(
        RESEARCH / "r8_long_history_robustness" / "fixed_metrics.csv",
        index_col=[0, 1],
    )
    rolling = pd.read_csv(
        RESEARCH / "r8_long_history_robustness" / "rolling_summary.csv",
        index_col=0,
    )
    proxy_normal = long_metrics.loc[("normal", "r8")]
    proxy_cost15 = long_metrics.loc[("cost15", "r8")]
    rolling_three = rolling.loc[3]

    gate_lines = "\n".join(
        f"| {gate} | {'PASS' if bool(row['pass']) else 'FAIL'} | "
        f"{row['evidence']} |"
        for gate, row in gates.iterrows()
    )
    target_lines = "\n".join(
        f"| {int(tranche)} | {row['QQQ']:.2%} | {row['SEMIS']:.2%} | "
        f"{row['GOLD']:.2%} | {row['CASH']:.2%} |"
        for tranche, row in targets.iterrows()
    )
    panel = f"""# 当前生产 Panel — R8 分段发布

- 发布：`{RELEASE}`
- 策略准入：**{release_status}**
- 数据截止：{signal_date.date().isoformat()}
- Cboe VIX3M 补源：PASS；本次补 {dict(vix3m_quality).get('filled_observations', 0)} 条，官方最新日期 {dict(vix3m_quality).get('latest_source_date')}
- 下一执行窗口：{execution_date.date().isoformat()} 09:30 ET（{execution_status}）
- 模型动作：{' / '.join(diagnostics.loc['action'].astype(str).unique())}
- 模型下次计划评估：{diagnostics.loc['estimated_rebalance_date'].iloc[0]}
- 订单状态：**BLOCKED_NEEDS_CURRENT_HOLDINGS**；下表是模型迁移参考，不是可直接提交的券商订单

## 当前目标

| 资产 | 旧生产 | R8 生产 |
|---|---:|---:|
| QQQ | {production_target['QQQ']:.2%} | {candidate_target['QQQ']:.2%} |
| SMH | {production_target['SEMIS']:.2%} | {candidate_target['SEMIS']:.2%} |
| GLD | {production_target['GOLD']:.2%} | {candidate_target['GOLD']:.2%} |
| BIL/现金 | {production_target['CASH']:.2%} | {candidate_target['CASH']:.2%} |
| VIXY | {production_target.get('VIX_HEDGE', 0.0):.2%} | {candidate_target.get('VIX_HEDGE', 0.0):.2%} |

当前 R8 与 Dual 目标一致；新 floor/bridge 此刻均未绑定。VIX/VIX3M={prices['VIX'] / prices['VIX3M']:.3f}，条件性 VIXY 对冲关闭。

## 四段累计目标

| 段 | QQQ | SMH | GLD | BIL/现金 |
|---:|---:|---:|---:|---:|
{target_lines}

- 每段模型单边换手：{migration['maximum_tranche_one_way_turnover']:.2%}（上限 10%）。
- 总模型单边换手：{migration['total_one_way_turnover']:.2%}。
- 15bps 假设总显性迁移成本：{migration['estimated_total_explicit_cost_nav_bps']:.2f}bp NAV（按 ${ACCOUNT_VALUE:,.0f} 约 ${ACCOUNT_VALUE * migration['estimated_total_explicit_cost_fraction']:,.0f}）。
- 历史任意上线日重放：连续四日方案平均相对立即切换 {activation['active_mean_relative_transition_return']:+.2%}，活跃路径 5% 分位 {activation['active_p05_relative_transition_return']:+.2%}，最差 {activation['worst_relative_transition_return']:+.2%}（{activation['worst_activation_date']}）。

## 生产准入

| 门槛 | 结果 | 证据 |
|---|---|---|
{gate_lines}

## 下一步调仓规则

1. 先读取真实账户 QQQ/SMH/GLD/BIL/VIXY 份额与可用现金；未读取前 `executable=False`。
2. 每个完整收盘后重新刷新 Yahoo 资产价格与 Cboe VIX3M，并重算 R8；不得沿用静态四段表追单。
3. 每次最多向当日 R8 目标移动总组合的 25%，且实际单边换手硬封顶 10%；目标漂移时以更小者为准。
4. 只在下一交易日 09:30 ET 窗口执行；错过开盘或数据源不新鲜时标记 `MISSED/STALE_SIGNAL`，等待下一完整收盘。
5. 每次成交后记录实际价格与剩余持仓，再生成下一段；四段不是预授权连续订单。

## 风险边界

20 年代理普通成本下 CAGR / Sharpe / MDD 为 {proxy_normal['cagr']:.2%} / {proxy_normal['sharpe']:.3f} / {proxy_normal['max_drawdown']:.2%}；15bps 下为 {proxy_cost15['cagr']:.2%} / {proxy_cost15['sharpe']:.3f} / {proxy_cost15['max_drawdown']:.2%}。高成本代理 MDD 已超过 18%，因此 18% 只能描述普通成本历史路径。

滚动 3 年跑赢旧生产的窗口占比为 {rolling_three['outperformance_share']:.1%}，最差年化相对收益 {rolling_three['worst_annualized_relative_return']:.2%}（窗口结束 {rolling_three['worst_window_end']}）。2008–09 和 2011 是已知相对失效区间；前瞻监控应报告滚动相对收益，但不得因短期落后临时调参。

历史 MDD 与 bootstrap 尾部都不是未来保证。连续四日迁移在 2025-04-09 式急反弹中曾相对立即切换落后约 2.78%；分段是控制执行冲击与操作错误，不是免费择时。参考份额未计税务、点差、碎股限制与账户约束。
"""
    (PANEL_DIR / "panel.md").write_text(panel, encoding="utf-8")

    decision = {
        "release": RELEASE,
        "parent_release": PARENT_RELEASE,
        "release_status": release_status,
        "signal_date": signal_date.date().isoformat(),
        "first_execution_date": execution_date.date().isoformat(),
        "signal_status": execution_status,
        "data_quality_pass": data_quality_pass,
        "orders_executable": False,
        "order_blocker": "NEEDS_CURRENT_HOLDINGS_AND_FRESH_CLOSE",
        "previous_production": PRODUCTION,
        "production_config": str(CANDIDATE_CONFIG),
        "all_admission_gates_pass": all_gates_pass,
        "migration_policy": {
            "tranches": 4,
            "maximum_tranche_one_way_turnover": (
                MAX_TRANCHE_ONE_WAY_TURNOVER
            ),
            "maximum_total_explicit_cost_nav_bps": (
                MAX_TOTAL_EXPLICIT_COST_NAV_BPS
            ),
        },
    }
    (RELEASE_DIR / "production_release_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    core_files = [
        Path("src/regime_strategy/model.py"),
        Path("src/regime_strategy/features.py"),
        Path("src/regime_strategy/portfolio.py"),
        Path("src/regime_strategy/backtest.py"),
        Path("src/regime_strategy/ensemble.py"),
        Path("src/regime_strategy/data.py"),
        Path("src/regime_strategy/ensemble_cli.py"),
        Path("src/regime_strategy/operations.py"),
    ]
    evidence_files = [
        RESEARCH / "preregistration.md",
        RESEARCH / "r8_statistical_validation" / "metrics.csv",
        RESEARCH / "r8_statistical_validation" / "bootstrap.csv",
        RESEARCH / "r8_statistical_validation" / "family_reality_check.csv",
        RESEARCH / "r8_statistical_validation" / "seed_consistency.csv",
        RESEARCH / "r8_proxy_metrics.csv",
        RESEARCH / "r8_execution_delay.csv",
        RESEARCH / "r8_episode_summary.csv",
        RESEARCH / "r8_neighborhood" / "metrics.csv",
        RESEARCH / "r8_neighborhood" / "decision.csv",
        RESEARCH / "r8_long_history_robustness" / "fixed_metrics.csv",
        RESEARCH / "r8_long_history_robustness" / "rolling_summary.csv",
        RESEARCH / "r8_long_history_robustness" / "stress_period_metrics.csv",
        RESEARCH / "r8_long_history_robustness" / "annual_returns.csv",
        RELEASE_DIR / "migration_summary.json",
        RELEASE_DIR / "historical_activation_summary.csv",
        RELEASE_DIR / "production_admission_gates.csv",
        RELEASE_DIR / "production_release_decision.json",
        Path("data/prices_vix_hedge.csv.metadata.json"),
        PANEL_DIR / "panel.md",
        PANEL_DIR / "order_plan.csv",
        PANEL_DIR / "staged_target_weights.csv",
        PANEL_DIR / "staged_trades.csv",
    ]
    files = [*core_files, *config_closure(CANDIDATE_CONFIG), *evidence_files]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing release files: {missing}")
    manifest = {
        **decision,
        "files": {str(path): sha256(path) for path in files},
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = MONITORING_DIR / f"freeze_manifest_{RELEASE}.json"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise RuntimeError(
                f"Frozen release {RELEASE} conflicts; use a new release name"
            )
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")

    snapshot = {
        **decision,
        "candidate_config_sha256": sha256(CANDIDATE_CONFIG),
        "candidate_qqq": float(candidate_target["QQQ"]),
        "candidate_smh": float(candidate_target["SEMIS"]),
        "candidate_gold": float(candidate_target["GOLD"]),
        "candidate_cash": float(candidate_target["CASH"]),
        "candidate_vix_hedge": float(
            candidate_target.get("VIX_HEDGE", 0.0)
        ),
    }
    append_immutable(
        MONITORING_DIR / "r8_production_signal_log.csv",
        snapshot,
        keys=["release", "signal_date"],
    )
    (
        MONITORING_DIR
        / f"r8_production_{signal_date.date()}_{RELEASE}.json"
    ).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(panel)
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
