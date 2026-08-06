from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_lowvol_rebound_candidate as evaluation
import evaluate_relief5_regime_candidate as relief_evaluation
from evaluate_dual_reentry_execution_delay import delayed_execution
from evaluate_zero_entry_bridge_episodes import episode_rows
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "regime_strategy_research_2026-07-25"
PRODUCTION = "paper_core_growth_gold20_daily_risk_netted_ensemble"
DUAL = (
    "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
    "netted_ensemble"
)
CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_floor45_inverse_momentum_"
    "netted_ensemble"
)
PRODUCTION_PROXY = "past_20y_drawdown_backtest_netted"
DUAL_PROXY = f"{DUAL}_20y_proxy"
CANDIDATE_PROXY = f"{CANDIDATE}_20y_proxy"
CANDIDATE_KEY = "r7_floor45"
ARTIFACT_PREFIX = "r7"


def candidate_family() -> dict[str, str]:
    additions = {
        "r5_confirm3": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_jump_"
            "confirm3_inverse_momentum_netted_ensemble"
        ),
        "r5_stage3": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_jump_"
            "stage3_inverse_momentum_netted_ensemble"
        ),
        "r6_relative_bridge": (
            "paper_core_growth_gold20_dual_reentry_relative_bridge_"
            "inverse_momentum_netted_ensemble"
        ),
        CANDIDATE_KEY: CANDIDATE,
    }
    return {**relief_evaluation.extended_family(), **additions}


def write_execution_delay() -> pd.DataFrame:
    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    )
    asset_returns = prices.pct_change(fill_method=None)
    strategies = {
        "production": PRODUCTION,
        "dual": DUAL,
        CANDIDATE_KEY: CANDIDATE,
    }
    simulations = {
        (strategy, delay): delayed_execution(
            directory,
            asset_returns,
            delay,
        )
        for strategy, directory in strategies.items()
        for delay in (0, 1, 2)
    }
    rows = []
    for delay in (0, 1, 2):
        metrics = {
            strategy: performance_metrics(
                simulations[(strategy, delay)].loc[
                    "2015-01-01":"2025-12-31"
                ]
            )
            for strategy in strategies
        }
        for strategy, values in metrics.items():
            rows.append(
                {
                    "strategy": strategy,
                    "delay_sessions": delay,
                    **values,
                    "cagr_delta_vs_production": (
                        values["cagr"] - metrics["production"]["cagr"]
                    ),
                    "cagr_delta_vs_dual": (
                        values["cagr"] - metrics["dual"]["cagr"]
                    ),
                }
            )
    frame = pd.DataFrame(rows).set_index(["strategy", "delay_sessions"])
    frame.to_csv(DESTINATION / f"{ARTIFACT_PREFIX}_execution_delay.csv")
    return frame


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, sample in episodes.groupby("scenario"):
        contributions = sample["relative_log_return"].sort_values(
            ascending=False
        )
        years = (
            pd.Timestamp(sample["end"].max())
            - pd.Timestamp(sample["start"].min())
        ).days / 365.25
        total = float(contributions.sum())
        rows.append(
            {
                "scenario": scenario,
                "episodes": len(sample),
                "positive_episodes": int(
                    sample["relative_log_return"].gt(0.0).sum()
                ),
                "win_rate": float(
                    sample["relative_log_return"].gt(0.0).mean()
                ),
                "total_relative_log_return": total,
                "largest_positive_share": (
                    float(contributions.iloc[0] / total)
                    if total > 0.0
                    else float("nan")
                ),
                "annualized_after_largest": float(
                    contributions.iloc[1:].sum() / years
                ),
                "annualized_after_top_two": float(
                    contributions.iloc[2:].sum() / years
                ),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def write_episodes() -> pd.DataFrame:
    episodes = pd.DataFrame(
        [
            *episode_rows(
                "normal_2015_2025",
                DUAL,
                CANDIDATE,
                "2015-01-01",
                "2025-12-31",
            ),
            *episode_rows(
                "proxy_2006_2025",
                DUAL_PROXY,
                CANDIDATE_PROXY,
                "2006-01-01",
                "2025-12-31",
            ),
        ]
    )
    episodes.to_csv(
        DESTINATION / f"{ARTIFACT_PREFIX}_episodes.csv",
        index=False,
    )
    summary = summarize_episodes(episodes)
    summary.to_csv(
        DESTINATION / f"{ARTIFACT_PREFIX}_episode_summary.csv"
    )
    return summary


def write_proxy_metrics() -> pd.DataFrame:
    rows = []
    for strategy, directory in {
        "production": PRODUCTION_PROXY,
        "dual": DUAL_PROXY,
        CANDIDATE_KEY: CANDIDATE_PROXY,
    }.items():
        returns = pd.read_csv(
            OUTPUT / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        ).loc["2006-08-01":"2025-12-31", "net_return"]
        rows.append({"strategy": strategy, **performance_metrics(returns)})
    frame = pd.DataFrame(rows).set_index("strategy")
    frame.to_csv(DESTINATION / f"{ARTIFACT_PREFIX}_proxy_metrics.csv")
    return frame


def write_migration() -> pd.DataFrame:
    targets = {
        strategy: pd.read_csv(
            OUTPUT / directory / "next_target_weights.csv",
            index_col=0,
        )["ensemble_current_sleeve_weight"]
        for strategy, directory in {
            "production": PRODUCTION,
            "dual": DUAL,
            CANDIDATE_KEY: CANDIDATE,
        }.items()
    }
    frame = pd.DataFrame(targets).fillna(0.0)
    frame.to_csv(
        DESTINATION / f"{ARTIFACT_PREFIX}_current_target_migration.csv"
    )
    summary = pd.DataFrame(
        [
            {
                "one_way_turnover_vs_production": float(
                    0.5
                    * (
                        frame[CANDIDATE_KEY] - frame["production"]
                    ).abs().sum()
                ),
                "one_way_turnover_vs_dual": float(
                    0.5
                    * (
                        frame[CANDIDATE_KEY] - frame["dual"]
                    ).abs().sum()
                ),
            }
        ],
        index=[CANDIDATE_KEY],
    )
    summary.to_csv(
        DESTINATION
        / f"{ARTIFACT_PREFIX}_current_target_migration_summary.csv"
    )
    return summary


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    evaluation.DESTINATION = (
        DESTINATION / f"{ARTIFACT_PREFIX}_statistical_validation"
    )
    evaluation.CANDIDATE = CANDIDATE
    evaluation.CANDIDATE_COST15 = f"{CANDIDATE}_cost15"
    evaluation.CANDIDATE_2012 = f"{CANDIDATE}_2012"
    evaluation.CANDIDATE_KEY = CANDIDATE_KEY
    evaluation.REPORT_TITLE = "Dual Reentry 45% floor compatibility"
    evaluation.RECOVERY_FAMILY = candidate_family()
    evaluation.main()

    delay = write_execution_delay()
    episodes = write_episodes()
    proxy = write_proxy_metrics()
    migration = write_migration()

    print(f"\n{CANDIDATE_KEY} execution delay:")
    print(delay.loc[CANDIDATE_KEY].round(6).to_string())
    print(f"\n{CANDIDATE_KEY} episode concentration:")
    print(episodes.round(6).to_string())
    print(f"\n{CANDIDATE_KEY} proxy:")
    print(proxy[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print(f"\n{CANDIDATE_KEY} migration:")
    print(migration.round(6).to_string())


if __name__ == "__main__":
    main()
