from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OPEN_RESULTS = Path("output/open_execution_validation")
PRICE_CACHE = Path("data/adjusted_open_close_2011_present.csv")
DESTINATION = Path("output/rolling_benchmark_consistency")
STRATEGIES = {
    "production_baseline": "gold_20_daily_asset_cap_daily.csv",
    "jump_aware_candidate": "gold_20_jump_aware_daily_cap_daily.csv",
}
WINDOWS = {"1_year": 252, "3_year": 756, "5_year": 1260}


def rolling_annualized_return(returns: pd.Series, sessions: int) -> pd.Series:
    log_returns = np.log1p(returns)
    return np.expm1(log_returns.rolling(sessions).sum() * 252.0 / sessions)


def main() -> None:
    prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    spy_returns = prices["close_SPX"].pct_change(fill_method=None).rename("SPY")
    strategy_returns = {}
    for name, filename in STRATEGIES.items():
        daily = pd.read_csv(
            OPEN_RESULTS / filename,
            index_col=0,
            parse_dates=True,
        )
        strategy_returns[name] = daily["net_return"].astype(float)
    aligned = pd.concat(
        {**strategy_returns, "SPY": spy_returns},
        axis=1,
        join="inner",
    ).dropna()

    detail_rows = []
    summary_rows = []
    for window, sessions in WINDOWS.items():
        annualized = aligned.apply(rolling_annualized_return, sessions=sessions)
        annualized = annualized.dropna()
        if annualized.empty:
            raise RuntimeError(f"No complete observations for {window}")
        for strategy in STRATEGIES:
            relative = annualized[strategy] - annualized["SPY"]
            for date, value in relative.items():
                detail_rows.append(
                    {
                        "window": window,
                        "window_sessions": sessions,
                        "window_end": date,
                        "strategy": strategy,
                        "strategy_annualized_return": annualized.loc[date, strategy],
                        "spy_annualized_return": annualized.loc[date, "SPY"],
                        "annualized_return_delta": value,
                    }
                )
            summary_rows.append(
                {
                    "window": window,
                    "window_sessions": sessions,
                    "strategy": strategy,
                    "observations": len(relative),
                    "outperformance_share": float((relative > 0.0).mean()),
                    "median_delta": float(relative.median()),
                    "p10_delta": float(relative.quantile(0.10)),
                    "p90_delta": float(relative.quantile(0.90)),
                    "worst_delta": float(relative.min()),
                    "best_delta": float(relative.max()),
                }
            )
    detail = pd.DataFrame(detail_rows).set_index(
        ["window", "strategy", "window_end"]
    )
    summary = pd.DataFrame(summary_rows).set_index(["window", "strategy"])

    lines = [
        "# 滚动基准一致性",
        "",
        "比较生产版和跳跃感知影子版的滚动年化收益与 SPY。窗口彼此重叠，胜率用于描述路径一致性，不是独立试验的显著性。",
        "",
        "| 窗口 | 策略 | 跑赢比例 | 中位超额 | 10%分位 | 90%分位 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (window, strategy), row in summary.iterrows():
        lines.append(
            f"| {window} | {strategy} | {row['outperformance_share']:.1%} | "
            f"{row['median_delta']:+.2%} | {row['p10_delta']:+.2%} | "
            f"{row['p90_delta']:+.2%} |"
        )
    production_three_year = summary.loc[("3_year", "production_baseline")]
    lines.extend(
        [
            "",
            "## 决策含义",
            "",
            f"生产版在重叠三年窗口中跑赢 SPY 的比例为 {production_three_year['outperformance_share']:.1%}，中位年化超额 {production_three_year['median_delta']:+.2%}，10% 分位 {production_three_year['p10_delta']:+.2%}。若中低分位仍为负，就不能把单一起点的正超额宣传为稳定 alpha；策略价值应主要定义为回撤和波动控制。",
        ]
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DESTINATION / "rolling_detail.csv")
    summary.to_csv(DESTINATION / "summary.csv")
    (DESTINATION / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string())
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
