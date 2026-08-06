from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.audit_hmm_robust_scaler_mechanism import aligned_weight_l1
from tools.evaluate_hmm_order_ensemble import DEFINITIONS, with_strategy_inputs
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_robust_scaler_episode_concentration")
DIRECTORIES = {
    "normal": (
        Path("output/research_hmm_standard_restarts5"),
        Path("output/research_hmm_robust_restarts5"),
    ),
    "proxy": (
        Path("output/research_hmm_standard_restarts5_20y_proxy"),
        Path("output/research_hmm_robust_restarts5_20y_proxy"),
    ),
}
ACTIVE_WEIGHT_L1 = 0.05


def contiguous_episode_ids(active: pd.Series) -> pd.Series:
    flags = active.fillna(False).astype(bool)
    starts = flags & ~flags.shift(1, fill_value=False)
    return starts.cumsum().where(flags).astype("Int64").rename("episode")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    scenario = next(item for item in COST_SCENARIOS if item.name == "current_liquidity")
    definitions = deepcopy(DEFINITIONS)
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for sample, (baseline_directory, candidate_directory) in DIRECTORIES.items():
        sample_name = str(definitions[sample]["sample"])
        baseline_settings = with_strategy_inputs(samples[sample_name], baseline_directory)
        candidate_settings = with_strategy_inputs(samples[sample_name], candidate_directory)
        baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
        candidate, _ = r39.simulate_candidate(candidate_settings, scenario)
        baseline_weights = pd.read_csv(
            baseline_directory / "weights.csv", index_col=0, parse_dates=True
        )
        candidate_weights = pd.read_csv(
            candidate_directory / "weights.csv", index_col=0, parse_dates=True
        )
        weight_l1 = aligned_weight_l1(baseline_weights, candidate_weights)
        common = baseline.index.intersection(candidate.index).intersection(weight_l1.index)
        relative_log = (
            np.log1p(candidate.loc[common, "net_return"])
            - np.log1p(baseline.loc[common, "net_return"])
        )
        episode_ids = contiguous_episode_ids(weight_l1.loc[common].gt(ACTIVE_WEIGHT_L1))
        for episode, selected in episode_ids.dropna().groupby(episode_ids.dropna()).groups.items():
            dates = pd.DatetimeIndex(selected)
            rows.append(
                {
                    "sample": sample,
                    "episode": int(episode),
                    "start": dates[0].date().isoformat(),
                    "end": dates[-1].date().isoformat(),
                    "observations": len(dates),
                    "relative_log_return": float(relative_log.loc[dates].sum()),
                    "mean_weight_l1": float(weight_l1.loc[dates].mean()),
                    "maximum_weight_l1": float(weight_l1.loc[dates].max()),
                }
            )

    episodes = pd.DataFrame(rows)
    episodes.to_csv(OUTPUT / "episodes.csv", index=False)
    for sample, frame in episodes.groupby("sample"):
        contribution = frame["relative_log_return"].astype(float)
        positive = contribution[contribution > 0.0].sort_values(ascending=False)
        positive_total = float(positive.sum())
        summaries[sample] = {
            "episodes": len(frame),
            "positive_episode_fraction": float(contribution.gt(0.0).mean()),
            "total_episode_relative_log_return": float(contribution.sum()),
            "largest_positive_episode": float(positive.iloc[0]) if len(positive) else 0.0,
            "worst_episode": float(contribution.min()),
            "largest_positive_share": (
                float(positive.iloc[0] / positive_total) if positive_total > 0.0 else 0.0
            ),
            "top_five_positive_share": (
                float(positive.head(5).sum() / positive_total)
                if positive_total > 0.0
                else 0.0
            ),
        }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "active_weight_l1_threshold": ACTIVE_WEIGHT_L1,
        "samples": summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(episodes.sort_values("relative_log_return").round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
