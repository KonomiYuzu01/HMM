from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
SCENARIOS = {
    "normal_2015_2025": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
        "2015-01-01",
        "2025-12-31",
    ),
    "proxy_2006_2026": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble_20y_proxy",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy",
        "2006-01-01",
        "2026-12-31",
    ),
    "hysteresis19_2015_2025": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_hysteresis19_inverse_"
        "momentum_netted_ensemble",
        "2015-01-01",
        "2025-12-31",
    ),
    "hysteresis19_proxy_2006_2026": (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble_20y_proxy",
        "paper_core_growth_gold20_dual_reentry_hysteresis19_inverse_"
        "momentum_netted_ensemble_20y_proxy",
        "2006-01-01",
        "2026-12-31",
    ),
    "goodvol_2015_2025": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_goodvol_inverse_momentum_"
        "netted_ensemble",
        "2015-01-01",
        "2025-12-31",
    ),
    "goodvol_proxy_2006_2026": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy",
        "paper_core_growth_gold20_dual_reentry_goodvol_inverse_momentum_"
        "netted_ensemble_20y_proxy",
        "2006-01-01",
        "2026-12-31",
    ),
    "goodvol_bear20_2015_2025": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_goodvol_bear20_inverse_"
        "momentum_netted_ensemble",
        "2015-01-01",
        "2025-12-31",
    ),
    "goodvol_bear20_proxy_2006_2026": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy",
        "paper_core_growth_gold20_dual_reentry_goodvol_bear20_inverse_"
        "momentum_netted_ensemble_20y_proxy",
        "2006-01-01",
        "2026-12-31",
    ),
}


def load_frame(directory: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def episode_rows(
    scenario: str,
    baseline_directory: str,
    candidate_directory: str,
    start: str,
    end: str,
) -> list[dict[str, object]]:
    baseline_weights = load_frame(baseline_directory, "weights.csv").loc[
        start:end
    ]
    candidate_weights = load_frame(candidate_directory, "weights.csv").loc[
        start:end
    ]
    returns = pd.concat(
        {
            "baseline": load_frame(
                baseline_directory,
                "daily_returns.csv",
            )["net_return"],
            "candidate": load_frame(
                candidate_directory,
                "daily_returns.csv",
            )["net_return"],
        },
        axis=1,
        join="inner",
    ).loc[start:end].dropna()
    dates = (
        returns.index.intersection(baseline_weights.index)
        .intersection(candidate_weights.index)
    )
    assets = baseline_weights.columns.intersection(candidate_weights.columns)
    weight_difference = (
        candidate_weights.loc[dates, assets]
        - baseline_weights.loc[dates, assets]
    )
    active = weight_difference.abs().max(axis=1) > 1e-10
    group = (active != active.shift(fill_value=False)).cumsum()
    rows = []
    episode_number = 0
    for _, mask in active.groupby(group):
        episode_dates = mask.index[mask]
        if len(episode_dates) == 0:
            continue
        episode_number += 1
        episode_returns = returns.loc[episode_dates]
        relative_log = (
            np.log1p(episode_returns["candidate"])
            - np.log1p(episode_returns["baseline"])
        )
        candidate_growth = candidate_weights.loc[
            episode_dates,
            ["QQQ", "SEMIS"],
        ].sum(axis=1)
        baseline_growth = baseline_weights.loc[
            episode_dates,
            ["QQQ", "SEMIS"],
        ].sum(axis=1)
        rows.append(
            {
                "scenario": scenario,
                "episode": episode_number,
                "start": episode_dates[0].date().isoformat(),
                "end": episode_dates[-1].date().isoformat(),
                "sessions": len(episode_dates),
                "candidate_return": float(
                    (1.0 + episode_returns["candidate"]).prod() - 1.0
                ),
                "baseline_return": float(
                    (1.0 + episode_returns["baseline"]).prod() - 1.0
                ),
                "relative_log_return": float(relative_log.sum()),
                "mean_growth_exposure_delta": float(
                    (candidate_growth - baseline_growth).mean()
                ),
                "maximum_growth_exposure_delta": float(
                    (candidate_growth - baseline_growth).max()
                ),
            }
        )
    return rows


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, arguments in SCENARIOS.items():
        rows.extend(episode_rows(scenario, *arguments))
    episodes = pd.DataFrame(rows)
    episodes.to_csv(DESTINATION / "zero_entry_episodes.csv", index=False)

    summary_rows = []
    for scenario, sample in episodes.groupby("scenario"):
        contributions = sample["relative_log_return"].sort_values(
            ascending=False
        )
        years = (
            pd.Timestamp(sample["end"].max())
            - pd.Timestamp(sample["start"].min())
        ).days / 365.25
        total = float(contributions.sum())
        summary_rows.append(
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
                    sample["relative_log_return"].median()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("scenario")
    summary.to_csv(DESTINATION / "zero_entry_episode_summary.csv")
    print(summary.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
