from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_individual_stock_overlay import (
    ALPHA_SCENARIOS,
    DEFAULT_COST_RATE,
    EVALUATION_END,
    EVALUATION_START,
    FIXED_BASKETS,
    ActiveRiskCache,
    RiskCache,
    event_metrics,
    load_inputs,
    load_stock_prices,
    period_metrics,
    random_baskets,
    simulate_overlay,
)
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/individual_stock_overlay_candidate_validation")
CANDIDATE = "active_regime_te3_zero_exit_monthly_tight_caps"
COMPARATOR = "active_regime_te3_monthly_tight_caps"
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "recent_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": (EVALUATION_START, EVALUATION_END),
}
BLOCK_LENGTH = 21
BOOTSTRAP_REPETITIONS = 1_000
BOOTSTRAP_SEED = 20_260_726


def annualized_relative_log_return(
    result: pd.DataFrame,
    start: str,
    end: str,
    alpha: float,
) -> float:
    sample = result.loc[start:end]
    alpha_daily = (1.0 + alpha) ** (1.0 / 252.0) - 1.0
    adjusted = sample["net_return"] + sample["stock_weight"] * alpha_daily
    relative = np.log1p(adjusted) - np.log1p(sample["base_net_return"])
    return float(relative.mean() * 252.0)


def break_even_alpha(
    result: pd.DataFrame,
    start: str,
    end: str,
) -> float:
    if annualized_relative_log_return(result, start, end, 0.0) >= 0.0:
        return 0.0
    upper = 0.50
    if annualized_relative_log_return(result, start, end, upper) < 0.0:
        return np.nan
    lower = 0.0
    for _ in range(50):
        midpoint = (lower + upper) / 2.0
        if annualized_relative_log_return(
            result,
            start,
            end,
            midpoint,
        ) >= 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def moving_block_bootstrap(
    relative_log_return: pd.Series,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = relative_log_return.dropna().to_numpy(dtype=float)
    if len(values) < BLOCK_LENGTH:
        raise ValueError("Insufficient observations for block bootstrap")
    block_sums = np.convolve(
        values,
        np.ones(BLOCK_LENGTH, dtype=float),
        mode="valid",
    )
    blocks_per_sample = int(np.ceil(len(values) / BLOCK_LENGTH))
    draws = rng.integers(
        0,
        len(block_sums),
        size=(BOOTSTRAP_REPETITIONS, blocks_per_sample),
    )
    annualized = (
        block_sums[draws].mean(axis=1) / BLOCK_LENGTH * 252.0
    )
    return (
        float(np.quantile(annualized, 0.025)),
        float(np.quantile(annualized, 0.975)),
        float((annualized > 0.0).mean()),
    )


def yearly_rows(
    result: pd.DataFrame,
    basket: str,
    policy: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, sample in result.groupby(result.index.year):
        if year < 2015 or year > 2025:
            continue
        relative_log = (
            np.log1p(sample["net_return"])
            - np.log1p(sample["base_net_return"])
        )
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "year": year,
                "strategy_return": float(
                    (1.0 + sample["net_return"]).prod() - 1.0
                ),
                "base_return": float(
                    (1.0 + sample["base_net_return"]).prod() - 1.0
                ),
                "relative_log_return": float(relative_log.sum()),
                "average_stock_weight": float(sample["stock_weight"].mean()),
                "realized_tracking_error": float(
                    (
                        sample["net_return"]
                        - sample["base_net_return"]
                    ).std(ddof=1)
                    * np.sqrt(252.0)
                ),
                "overlay_turnover": float(sample["overlay_turnover"].sum()),
            }
        )
    return rows


def leave_one_year_out_rows(
    result: pd.DataFrame,
    basket: str,
    policy: str,
) -> list[dict[str, object]]:
    relative_log = (
        np.log1p(result["net_return"])
        - np.log1p(result["base_net_return"])
    )
    rows: list[dict[str, object]] = []
    for excluded_year in range(2015, 2026):
        sample = relative_log.loc[relative_log.index.year != excluded_year]
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "excluded_year": excluded_year,
                "annualized_relative_log_return": float(
                    sample.mean() * 252.0
                ),
            }
        )
    return rows


def write_summary(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    thresholds: pd.DataFrame,
    yearly: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    random_mask = metrics["basket"].str.startswith("random_")
    complete = metrics.loc[
        random_mask
        & (metrics["period"] == "complete_2015_2025")
        & (metrics["alpha_assumption"] == 0.0)
    ]
    candidate = complete.loc[complete["policy"] == CANDIDATE]
    comparator = complete.loc[complete["policy"] == COMPARATOR]
    recent_thresholds = thresholds.loc[
        (thresholds["policy"] == CANDIDATE)
        & thresholds["basket"].str.startswith("random_")
        & (thresholds["period"] == "recent_2022_2025"),
        "break_even_alpha",
    ]
    bootstrap_candidate = bootstrap.loc[
        (bootstrap["policy"] == CANDIDATE)
        & bootstrap["basket"].str.startswith("random_")
        & (bootstrap["period"] == "complete_2015_2025")
    ]
    tracking = yearly.loc[yearly["policy"] == CANDIDATE]
    tracking = tracking.loc[tracking["basket"].str.startswith("random_")]
    event_candidate = events.loc[
        (events["policy"] == CANDIDATE)
        & events["basket"].str.startswith("random_")
    ]
    q4 = event_candidate.loc[
        event_candidate["event"] == "2018_q4_selloff"
    ]
    text = f"""# 个股叠加层生产候选统计验证

## 候选

- 只有 R9 成长仓超过 35% 后才允许个股。
- 安静/正常上涨环境最多替换对应 ETF 的 50%；脆弱上涨或反弹最多 25%。
- 用“个股篮子收益减去被替换 ETF 收益”的 252 日和 60 日差异波动控制仓位，
  两者取较高值；账户年化差异波动预算为 3%。
- 每只股票最多 5%，全部受管理个股最多 20%，按月调整。
- 若 R9 将对应成长 ETF 的允许替代预算降到 0，受管理个股下一交易日同步归零。
- 个股与 ETF 等金额替换，不为风险倍数额外留现金。

## 随机篮子结果

| 指标 | 归零同步退出 | 纯月度 |
|---|---:|---:|
| 中位 CAGR | {candidate['cagr'].median():.2%} | {comparator['cagr'].median():.2%} |
| 10%较差 CAGR | {candidate['cagr'].quantile(.1):.2%} | {comparator['cagr'].quantile(.1):.2%} |
| 中位最大回撤 | {candidate['max_drawdown'].median():.2%} | {comparator['max_drawdown'].median():.2%} |
| 10%较差最大回撤 | {candidate['max_drawdown'].quantile(.1):.2%} | {comparator['max_drawdown'].quantile(.1):.2%} |
| 中位年换手率 | {candidate['annualized_overlay_turnover'].median():.2%} | {comparator['annualized_overlay_turnover'].median():.2%} |

21 个交易日成块重采样后，候选相对 R9 年化收益的 95% 区间下界在不同随机篮子
中的中位数为 {bootstrap_candidate['ci_2_5'].median():.2%}，区间上界中位数为
{bootstrap_candidate['ci_97_5'].median():.2%}。重采样结果为正的概率中位数为
{bootstrap_candidate['probability_positive'].median():.1%}。这说明历史优势存在，
但统计上不足以证明不依赖选股。

2022-2025 子区间，中位随机篮子要追平 R9 所需的额外选股年化收益为
{recent_thresholds.median():.2%}，90% 较差篮子的门槛为
{recent_thresholds.quantile(.9):.2%}。

候选逐年实际差异波动的中位数为
{tracking['realized_tracking_error'].median():.2%}，90% 分位为
{tracking['realized_tracking_error'].quantile(.9):.2%}。

2018 年四季度压力窗口中，相对 R9 的中位累计对数收益为
{q4['relative_log_return'].median():.2%}，10% 较差为
{q4['relative_log_return'].quantile(.1):.2%}。归零同步退出减少了 R9 已结束对应
成长风险预算后继续持有个股的暴露，但在快速反转中也可能少赚。

## 解释边界

- 股票池用当前名单回看，含幸存者偏差与选择偏差；不能据此证明任何股票有效。
- 人工额外收益门槛是假设，不是对用户选股能力的估计。
- 重采样只衡量时间路径不确定性，不能消除制度变化、数据源或股票池偏差。
- 本候选是“怎样限制选股风险”的规则，不是自动选股模型。
"""
    (DESTINATION / "validation_summary.md").write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    stock_prices = load_stock_prices(refresh=False)
    base, weights, all_returns, states = load_inputs(stock_prices)
    baskets = {**FIXED_BASKETS, **random_baskets(10)}
    policies = (CANDIDATE, COMPARATOR)
    risk_cache: RiskCache = {}
    active_risk_cache: ActiveRiskCache = {}
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    yearly: list[dict[str, object]] = []
    leave_one_out: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for basket_name, basket in baskets.items():
        for policy in policies:
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
            events.extend(event_metrics(result, basket_name, policy))
            for period, (start, end) in PERIODS.items():
                sample = result.loc[start:end]
                relative_log = (
                    np.log1p(sample["net_return"])
                    - np.log1p(sample["base_net_return"])
                )
                low, high, probability = moving_block_bootstrap(
                    relative_log,
                    rng,
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

    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    thresholds = pd.DataFrame(threshold_rows)
    yearly_frame = pd.DataFrame(yearly)
    leave_one_out_frame = pd.DataFrame(leave_one_out)
    event_frame = pd.DataFrame(events)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    bootstrap.to_csv(DESTINATION / "block_bootstrap.csv", index=False)
    thresholds.to_csv(DESTINATION / "alpha_thresholds.csv", index=False)
    yearly_frame.to_csv(DESTINATION / "yearly_results.csv", index=False)
    leave_one_out_frame.to_csv(
        DESTINATION / "leave_one_year_out.csv",
        index=False,
    )
    event_frame.to_csv(DESTINATION / "event_metrics.csv", index=False)
    write_summary(metrics, bootstrap, thresholds, yearly_frame, event_frame)
    manifest = {
        "candidate": CANDIDATE,
        "comparator": COMPARATOR,
        "evaluation_start": EVALUATION_START,
        "evaluation_end": EVALUATION_END,
        "random_baskets": 40,
        "fixed_baskets": len(FIXED_BASKETS),
        "block_length": BLOCK_LENGTH,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stock_price_cache_sha256": hashlib.sha256(
            Path("data/individual_stock_overlay_prices.csv").read_bytes()
        ).hexdigest(),
        "alpha_grid_retained_for_comparability": list(ALPHA_SCENARIOS),
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_metrics = metrics.loc[
        (metrics["policy"] == CANDIDATE)
        & (metrics["basket"].str.startswith("random_"))
        & (metrics["period"] == "complete_2015_2025")
    ]
    print(
        candidate_metrics[
            ["cagr", "max_drawdown", "annualized_overlay_turnover"]
        ].quantile([0.1, 0.5, 0.9]).round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
