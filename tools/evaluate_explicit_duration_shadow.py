from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output/explicit_duration_shadow")
SAMPLES = {
    "normal": Path("output/research_hmm_gaussian_diagnostics"),
    "proxy": Path("output/research_hmm_gaussian_diagnostics_20y_proxy"),
}


def empirical_survival_probability(
    completed_lengths: list[int],
    current_length: int,
    alpha: float = 1.0,
) -> float:
    at_risk = sum(length >= current_length for length in completed_lengths)
    survived = sum(length >= current_length + 1 for length in completed_lengths)
    return float((survived + alpha) / (at_risk + 2.0 * alpha))


def causal_duration_predictions(states: pd.Series) -> pd.DataFrame:
    values = states.astype(int).to_numpy()
    if len(values) < 2:
        raise ValueError("Duration shadow requires at least two states")
    completed: dict[int, list[int]] = {0: [], 1: []}
    self_transitions = {0: 0, 1: 0}
    total_transitions = {0: 0, 1: 0}
    current_state = int(values[0])
    current_length = 1
    rows: list[dict[str, object]] = []
    for position in range(len(values) - 1):
        state = int(values[position])
        if state != current_state:
            raise AssertionError("Run state tracking became inconsistent")
        duration_probability = empirical_survival_probability(
            completed[state], current_length
        )
        markov_probability = float(
            (self_transitions[state] + 1.0)
            / (total_transitions[state] + 2.0)
        )
        next_state = int(values[position + 1])
        persisted = int(next_state == state)
        rows.append(
            {
                "date": states.index[position],
                "state": state,
                "run_length": current_length,
                "completed_state_episodes": len(completed[state]),
                "duration_persistence_probability": duration_probability,
                "markov_persistence_probability": markov_probability,
                "persisted_next_evaluation": persisted,
            }
        )
        total_transitions[state] += 1
        self_transitions[state] += persisted
        if persisted:
            current_length += 1
        else:
            completed[state].append(current_length)
            current_state = next_state
            current_length = 1
    return pd.DataFrame(rows).set_index("date")


def brier(actual: pd.Series, probability: pd.Series) -> float:
    return float(np.square(actual.astype(float) - probability.astype(float)).mean())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    for sample, directory in SAMPLES.items():
        for path in sorted((directory / "members").glob("seed_*/regimes.csv")):
            regimes = pd.read_csv(path, index_col=0, parse_dates=True)
            predictions = causal_duration_predictions(
                regimes["paper_risk_on_candidate"]
            )
            predictions["sample"] = sample
            predictions["member"] = path.parent.name
            prediction_frames.append(predictions.reset_index())
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)

    periods = {
        "normal_development": ("normal", "2015-01-01", "2021-12-31"),
        "normal_holdout": ("normal", "2022-01-01", "2025-12-31"),
        "proxy_early": ("proxy", "2006-08-01", "2014-12-31"),
        "proxy_late": ("proxy", "2015-01-01", "2025-12-31"),
    }
    metric_rows: list[dict[str, object]] = []
    for period, (sample, start, end) in periods.items():
        selected = predictions.loc[
            predictions["sample"].eq(sample)
            & predictions["date"].between(start, end)
        ]
        duration_brier = brier(
            selected["persisted_next_evaluation"],
            selected["duration_persistence_probability"],
        )
        markov_brier = brier(
            selected["persisted_next_evaluation"],
            selected["markov_persistence_probability"],
        )
        metric_rows.append(
            {
                "period": period,
                "sample": sample,
                "observations": len(selected),
                "duration_brier": duration_brier,
                "markov_brier": markov_brier,
                "brier_improvement": markov_brier - duration_brier,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    episode_rows: list[dict[str, object]] = []
    for (sample, member, state), group in predictions.groupby(
        ["sample", "member", "state"]
    ):
        episode_rows.append(
            {
                "sample": sample,
                "member": member,
                "state": int(state),
                "mature_completed_episodes": int(
                    group["completed_state_episodes"].max()
                ),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(OUTPUT / "episode_counts.csv", index=False)

    improvement = metrics.set_index("period")["brier_improvement"]
    gates = {
        "normal_development_brier_improved": bool(
            improvement["normal_development"] > 0.0
        ),
        "normal_holdout_brier_improved": bool(
            improvement["normal_holdout"] > 0.0
        ),
        "proxy_early_brier_improved": bool(improvement["proxy_early"] > 0.0),
        "proxy_late_brier_improved": bool(improvement["proxy_late"] > 0.0),
        "minimum_30_episodes_per_member_state": bool(
            episodes["mature_completed_episodes"].min() >= 30
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_shadow",
        "production_changed": False,
        "orders_generated": False,
        "gates": gates,
        "minimum_mature_episodes": int(
            episodes["mature_completed_episodes"].min()
        ),
        "research_pass": bool(all(gates.values())),
        "decision": (
            "continue_hsmm_research" if all(gates.values()) else "reject_duration_as_action_input"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

