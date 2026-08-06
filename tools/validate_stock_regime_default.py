from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_individual_stock_overlay import (
    DEFAULT_COST_RATE,
    EVALUATION_END,
    EVALUATION_START,
    FIXED_BASKETS,
    ActiveRiskCache,
    RiskCache,
    load_inputs,
    load_stock_prices,
    period_metrics,
    random_baskets,
    simulate_overlay,
)
from validate_stock_overlay_candidate import (
    PERIODS,
    break_even_alpha,
    leave_one_year_out_rows,
    moving_block_bootstrap,
    yearly_rows,
)


DESTINATION = Path("output/individual_stock_regime_final_validation")
BASELINE = "active_regime_te3_zero_exit_monthly_tight_caps"
CANDIDATE = "active_regime_stable_rs63_126_disp67"
POLICIES = (BASELINE, CANDIDATE)
BOOTSTRAP_SEED = 20_260_726


def state_rows(
    result: pd.DataFrame,
    states: pd.Series,
    basket: str,
    policy: str,
    period: str,
    start: str,
    end: str,
) -> list[dict[str, object]]:
    sample = result.loc[start:end].copy()
    sample["state"] = states.reindex(sample.index)
    sample["relative_log_return"] = np.log1p(sample["net_return"]) - np.log1p(
        sample["base_net_return"]
    )
    rows: list[dict[str, object]] = []
    for state, group in sample.groupby("state"):
        average_weight = float(group["stock_weight"].mean())
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "period": period,
                "state": state,
                "days": len(group),
                "average_stock_weight": average_weight,
                "annualized_account_relative_log_return": float(
                    group["relative_log_return"].mean() * 252.0
                ),
                "annualized_relative_return_per_stock_weight": (
                    float(
                        group["relative_log_return"].mean()
                        * 252.0
                        / average_weight
                    )
                    if average_weight > 1.0e-8
                    else np.nan
                ),
            }
        )
    return rows


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics.loc[metrics["basket"].str.startswith("random_")]
    return (
        random.groupby(["period", "policy"])
        .agg(
            median_cagr=("cagr", "median"),
            cagr_10pct=("cagr", lambda values: values.quantile(0.10)),
            median_max_drawdown=("max_drawdown", "median"),
            max_drawdown_10pct=(
                "max_drawdown",
                lambda values: values.quantile(0.10),
            ),
            median_relative_log_return=(
                "annualized_relative_log_return",
                "median",
            ),
            relative_log_return_10pct=(
                "annualized_relative_log_return",
                lambda values: values.quantile(0.10),
            ),
            median_stock_weight=("average_stock_weight", "median"),
            median_turnover=("annualized_overlay_turnover", "median"),
        )
        .reset_index()
    )


def write_summary(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    summary = summary_table(metrics)
    recent = summary.loc[summary["period"] == "holdout_2022_2025"].set_index(
        "policy"
    )
    complete = summary.loc[
        summary["period"] == "complete_2015_2025"
    ].set_index("policy")
    random_thresholds = thresholds.loc[
        thresholds["basket"].str.startswith("random_")
    ]
    candidate_recent_thresholds = random_thresholds.loc[
        (random_thresholds["policy"] == CANDIDATE)
        & (random_thresholds["period"] == "recent_2022_2025"),
        "break_even_alpha",
    ]
    baseline_recent_thresholds = random_thresholds.loc[
        (random_thresholds["policy"] == BASELINE)
        & (random_thresholds["period"] == "recent_2022_2025"),
        "break_even_alpha",
    ]
    candidate_bootstrap = bootstrap.loc[
        bootstrap["basket"].str.startswith("random_")
        & (bootstrap["policy"] == CANDIDATE)
        & (bootstrap["period"] == "complete_2015_2025")
    ]
    candidate_yearly = yearly.loc[
        yearly["basket"].str.startswith("random_")
        & (yearly["policy"] == CANDIDATE)
    ]
    yearly_summary = (
        candidate_yearly.groupby("year")["relative_log_return"]
        .agg(
            median="median",
            relative_10pct=lambda values: values.quantile(0.10),
        )
    )
    random_leave_one_out = leave_one_out.loc[
        leave_one_out["basket"].str.startswith("random_")
        & (leave_one_out["policy"] == CANDIDATE)
    ]
    leave_one_out_summary = random_leave_one_out.groupby("excluded_year")[
        "annualized_relative_log_return"
    ].agg(
        median="median",
        relative_10pct=lambda values: values.quantile(0.10),
    )

    text = f"""# 个股与 ETF 环境切换最终验证

## 候选规则

只有同时满足以下条件，个股才能等额替代 QQQ 或 SMH：

1. R9 成长资产目标高于 35%；
2. 市场状态为安静上涨或正常上涨；
3. 用户所选股票篮子过去 63 日和 126 日都跑赢对应 ETF；
4. 过去 63 日股票相对 ETF 收益的横截面分化，不处于过去三年的最高三分之一。

其余时间使用 ETF。单股 5%、个股合计 20%、年化差异波动预算 3%、月末增加和
风险归零后下一交易日退出等规则保持不变。

## 主要结果

2015—2025 完整区间，候选随机篮子的中位年化复合收益率为
{complete.loc[CANDIDATE, 'median_cagr']:.2%}，较差 10% 为
{complete.loc[CANDIDATE, 'cagr_10pct']:.2%}；中位最大回撤为
{complete.loc[CANDIDATE, 'median_max_drawdown']:.2%}。旧生产规则的对应数字为
{complete.loc[BASELINE, 'median_cagr']:.2%}、
{complete.loc[BASELINE, 'cagr_10pct']:.2%} 和
{complete.loc[BASELINE, 'median_max_drawdown']:.2%}。

2022—2025 近期区间，候选相对纯 R9 的中位年化对数收益为
{recent.loc[CANDIDATE, 'median_relative_log_return']:.2%}，较差 10% 为
{recent.loc[CANDIDATE, 'relative_log_return_10pct']:.2%}；旧生产规则分别为
{recent.loc[BASELINE, 'median_relative_log_return']:.2%} 和
{recent.loc[BASELINE, 'relative_log_return_10pct']:.2%}。

候选近期随机篮子无需额外选股收益即可追平 R9 的比例为
{(candidate_recent_thresholds == 0.0).mean():.1%}，旧生产规则为
{(baseline_recent_thresholds == 0.0).mean():.1%}。候选追平门槛的 90% 分位为
{candidate_recent_thresholds.quantile(.90):.2%}，旧生产规则为
{baseline_recent_thresholds.quantile(.90):.2%}。候选有
{candidate_recent_thresholds.isna().mean():.1%} 的篮子即使假设 50% 额外年化选股
收益也没有追平；原因是这些篮子被允许的个股仓位极小，而退出成本已经发生。

21 个交易日成块重复抽样后，完整区间 95% 下界的篮子中位数为
{candidate_bootstrap['ci_2_5'].median():.2%}，上界中位数为
{candidate_bootstrap['ci_97_5'].median():.2%}，相对收益为正的概率中位数为
{candidate_bootstrap['probability_positive'].median():.1%}。

逐年中位相对收益有 {(yearly_summary['median'] > 0.0).sum()} 年为正、
{(yearly_summary['median'] < 0.0).sum()} 年为负，其余年份为零或近似零。逐一剔除
任何单一年份后，候选中位年化相对收益的最低值仍为
{leave_one_out_summary['median'].min():.2%}；较差 10% 的最低值为
{leave_one_out_summary['relative_10pct'].min():.2%}。这些检验不能消除当前股票池的
幸存者偏差。

## 决策边界

候选适合作为没有经过实盘证明的默认模式，因为它显著减少错误选股拖累。旧生产
规则仍适合具有可验证长期选股优势、并愿意承担更大个股路径风险的高捕获模式。
两者不能用当前股票池回看结果证明未来收益。
"""
    (DESTINATION / "validation_summary.md").write_text(
        text,
        encoding="utf-8",
    )
    summary.to_csv(DESTINATION / "summary.csv", index=False)


def main() -> None:
    stock_prices = load_stock_prices(False)
    base, weights, all_returns, states = load_inputs(stock_prices)
    baskets = {**FIXED_BASKETS, **random_baskets(10)}
    risk_cache: RiskCache = {}
    active_risk_cache: ActiveRiskCache = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    yearly: list[dict[str, object]] = []
    leave_one_out: list[dict[str, object]] = []
    states_by_period: list[dict[str, object]] = []

    for basket_name, basket in baskets.items():
        for policy in POLICIES:
            result, _ = simulate_overlay(
                basket_name,
                basket,
                policy,
                base,
                weights,
                all_returns,
                states,
                stock_prices,
                risk_cache,
                active_risk_cache,
                cost_rate=DEFAULT_COST_RATE,
            )
            metric_rows.extend(
                period_metrics(
                    result,
                    basket_name,
                    policy,
                    0.0,
                    DEFAULT_COST_RATE,
                )
            )
            yearly.extend(yearly_rows(result, basket_name, policy))
            leave_one_out.extend(
                leave_one_year_out_rows(result, basket_name, policy)
            )
            for period, (start, end) in PERIODS.items():
                sample = result.loc[start:end]
                relative_log_return = (
                    np.log1p(sample["net_return"])
                    - np.log1p(sample["base_net_return"])
                )
                low, high, probability = moving_block_bootstrap(
                    relative_log_return,
                    rng,
                )
                threshold_rows.append(
                    {
                        "basket": basket_name,
                        "policy": policy,
                        "period": period,
                        "break_even_alpha": break_even_alpha(
                            result,
                            start,
                            end,
                        ),
                    }
                )
                bootstrap_rows.append(
                    {
                        "basket": basket_name,
                        "policy": policy,
                        "period": period,
                        "ci_2_5": low,
                        "ci_97_5": high,
                        "probability_positive": probability,
                    }
                )
                states_by_period.extend(
                    state_rows(
                        result,
                        states,
                        basket_name,
                        policy,
                        period,
                        start,
                        end,
                    )
                )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    yearly_frame = pd.DataFrame(yearly)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    thresholds.to_csv(DESTINATION / "alpha_thresholds.csv", index=False)
    bootstrap.to_csv(DESTINATION / "block_bootstrap.csv", index=False)
    yearly_frame.to_csv(DESTINATION / "yearly_results.csv", index=False)
    pd.DataFrame(leave_one_out).to_csv(
        DESTINATION / "leave_one_year_out.csv",
        index=False,
    )
    pd.DataFrame(states_by_period).to_csv(
        DESTINATION / "state_attribution.csv",
        index=False,
    )
    leave_one_out_frame = pd.DataFrame(leave_one_out)
    write_summary(
        metrics,
        thresholds,
        bootstrap,
        yearly_frame,
        leave_one_out_frame,
    )
    manifest = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "cost_rate": DEFAULT_COST_RATE,
        "evaluation_start": EVALUATION_START,
        "evaluation_end": EVALUATION_END,
        "fixed_baskets": len(FIXED_BASKETS),
        "random_baskets": len(random_baskets(10)),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print((DESTINATION / "validation_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
