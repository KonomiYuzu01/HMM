from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_lowvol_rebound_candidate as evaluation
from evaluate_dual_reentry_execution_delay import delayed_execution
from evaluate_zero_entry_bridge_episodes import episode_rows
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "regime_strategy_research_2026-07-25"
PRODUCTION = "paper_core_growth_gold20_daily_risk_netted_ensemble"
INVERSE_GATE = (
    "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
    "netted_ensemble"
)
DUAL = (
    "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
    "netted_ensemble"
)
CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_goodvol_bear20_relief5_"
    "inverse_momentum_netted_ensemble"
)
JUMP_CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_goodvol_bear20_jump_"
    "inverse_momentum_netted_ensemble"
)
PRODUCTION_PROXY = "past_20y_drawdown_backtest_netted"
DUAL_PROXY = f"{DUAL}_20y_proxy"
CANDIDATE_PROXY = f"{CANDIDATE}_20y_proxy"


def load_daily(directory: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def extended_family() -> dict[str, str]:
    additions = {
        "panic_veto": (
            "paper_core_growth_gold20_panic_veto_floor_guard_"
            "netted_ensemble"
        ),
        "crisis_veto10": (
            "paper_core_growth_gold20_crisis_veto10_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_combined": (
            "paper_core_growth_gold20_downside_semivol_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_rebound": (
            "paper_core_growth_gold20_downside_semivol_rebound_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_low_rebound": (
            "paper_core_growth_gold20_downside_semivol_low_rebound_"
            "floor_guard_netted_ensemble"
        ),
        "momentum_regime_gate": (
            "paper_core_growth_gold20_lowvol_momentum_floor_guard_"
            "netted_ensemble"
        ),
        "inverse_momentum_regime_gate": INVERSE_GATE,
        "zero_bridge_qqq20": (
            "paper_core_growth_gold20_zero_bridge_qqq20_momentum_gate_"
            "netted_ensemble"
        ),
        "dual_reentry": DUAL,
        "dual_fast10": (
            "paper_core_growth_gold20_dual_reentry_fast10_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_fast40": (
            "paper_core_growth_gold20_dual_reentry_fast40_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_q85": (
            "paper_core_growth_gold20_dual_reentry_q85_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_q95": (
            "paper_core_growth_gold20_dual_reentry_q95_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_equal_bridge": (
            "paper_core_growth_gold20_dual_reentry_equal_bridge_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_hysteresis19": (
            "paper_core_growth_gold20_dual_reentry_hysteresis19_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol": (
            "paper_core_growth_gold20_dual_reentry_goodvol_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol_bear20": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol_bear20_contango": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_"
            "contango_inverse_momentum_netted_ensemble"
        ),
        "dual_parent_correction_veto": (
            "paper_core_growth_gold20_dual_reentry_parent_correction_veto_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_plain_momentum": (
            "paper_core_growth_gold20_dual_reentry_plain_momentum_"
            "netted_ensemble"
        ),
        "r4_jump": JUMP_CANDIDATE,
        "r4_relief5": CANDIDATE,
    }
    return {**evaluation.RECOVERY_FAMILY, **additions}


def write_execution_delay() -> pd.DataFrame:
    prices = pd.read_csv(
        "data/prices_vix_hedge.csv",
        index_col=0,
        parse_dates=True,
    )
    asset_returns = prices.pct_change(fill_method=None)
    strategies = {
        "production": PRODUCTION,
        "inverse_gate": INVERSE_GATE,
        "dual": DUAL,
        "r4_relief5": CANDIDATE,
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
                    "cagr_delta_vs_inverse_gate": (
                        values["cagr"] - metrics["inverse_gate"]["cagr"]
                    ),
                    "cagr_delta_vs_dual": (
                        values["cagr"] - metrics["dual"]["cagr"]
                    ),
                }
            )
    frame = pd.DataFrame(rows).set_index(["strategy", "delay_sessions"])
    frame.to_csv(DESTINATION / "execution_delay.csv")
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
                "median_episode_relative_log_return": float(
                    contributions.median()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def write_episodes() -> pd.DataFrame:
    rows = [
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
    episodes = pd.DataFrame(rows)
    episodes.to_csv(DESTINATION / "relief5_episodes.csv", index=False)
    summary = summarize_episodes(episodes)
    summary.to_csv(DESTINATION / "relief5_episode_summary.csv")
    return summary


def write_proxy_metrics() -> pd.DataFrame:
    rows = []
    for strategy, directory in {
        "production": PRODUCTION_PROXY,
        "dual": DUAL_PROXY,
        "r4_relief5": CANDIDATE_PROXY,
    }.items():
        returns = load_daily(directory).loc[
            "2006-08-01":"2025-12-31",
            "net_return",
        ]
        rows.append(
            {
                "strategy": strategy,
                **performance_metrics(returns),
            }
        )
    frame = pd.DataFrame(rows).set_index("strategy")
    frame.to_csv(DESTINATION / "proxy_metrics.csv")
    return frame


def write_migration() -> pd.DataFrame:
    targets = {}
    for strategy, directory in {
        "production": PRODUCTION,
        "dual": DUAL,
        "r4_relief5": CANDIDATE,
    }.items():
        targets[strategy] = pd.read_csv(
            OUTPUT / directory / "next_target_weights.csv",
            index_col=0,
        )["ensemble_current_sleeve_weight"]
    frame = pd.DataFrame(targets).fillna(0.0)
    frame["candidate_minus_production"] = (
        frame["r4_relief5"] - frame["production"]
    )
    frame.to_csv(DESTINATION / "current_target_migration.csv")
    pd.DataFrame(
        [
            {
                "one_way_turnover_vs_production": float(
                    0.5
                    * (
                        frame["r4_relief5"] - frame["production"]
                    ).abs().sum()
                ),
                "one_way_turnover_vs_dual": float(
                    0.5
                    * (
                        frame["r4_relief5"] - frame["dual"]
                    ).abs().sum()
                ),
            }
        ],
        index=["r4_relief5"],
    ).to_csv(DESTINATION / "current_target_migration_summary.csv")
    return frame


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    evaluation.DESTINATION = DESTINATION
    evaluation.CANDIDATE = CANDIDATE
    evaluation.CANDIDATE_COST15 = f"{CANDIDATE}_cost15"
    evaluation.CANDIDATE_2012 = f"{CANDIDATE}_2012"
    evaluation.CANDIDATE_KEY = "r4_relief5"
    evaluation.REPORT_TITLE = "Five-session volatility-relief regime bridge"
    evaluation.RECOVERY_FAMILY = extended_family()
    evaluation.main()

    delay = write_execution_delay()
    episodes = write_episodes()
    proxy = write_proxy_metrics()
    migration = write_migration()

    print("\nExecution delay:")
    print(delay.loc["r4_relief5"].round(6).to_string())
    print("\nEpisode concentration:")
    print(episodes.round(6).to_string())
    print("\nProxy:")
    print(proxy[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print(
        "\nCurrent one-way migration vs production:",
        round(
            float(
                0.5
                * (
                    migration["r4_relief5"] - migration["production"]
                ).abs().sum()
            ),
            6,
        ),
    )


if __name__ == "__main__":
    main()
