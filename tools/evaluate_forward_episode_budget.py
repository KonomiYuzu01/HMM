from __future__ import annotations

from math import ceil, comb
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


DESTINATION = Path("output/dual_reentry_candidate_validation")


def one_sided_sign_p_value(wins: int, observations: int) -> float:
    return sum(
        comb(observations, value)
        for value in range(wins, observations + 1)
    ) / 2**observations


def optimistic_wins_required(
    wins: int,
    observations: int,
    alpha: float,
) -> int:
    additional = 0
    while (
        one_sided_sign_p_value(
            wins + additional,
            observations + additional,
        )
        > alpha
    ):
        additional += 1
    return additional


def normal_power_sample_size(
    values: np.ndarray,
    alpha: float = 0.10,
    power: float = 0.80,
) -> float:
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if mean <= 0.0 or standard_deviation <= 0.0:
        return float("inf")
    effect_size = mean / standard_deviation
    critical = NormalDist().inv_cdf(1.0 - alpha)
    target = NormalDist().inv_cdf(power)
    return float(ceil(((critical + target) / effect_size) ** 2))


def main() -> None:
    episodes = pd.read_csv(DESTINATION / "zero_entry_episodes.csv")
    sample = episodes.loc[
        episodes["scenario"].eq("normal_2015_2025"),
        "relative_log_return",
    ].to_numpy(dtype=float)
    observations = len(sample)
    wins = int((sample > 0.0).sum())
    event_years = 11.0
    annual_event_rate = observations / event_years
    without_largest = np.delete(sample, int(np.argmax(sample)))
    winsorized = np.minimum(sample, np.quantile(sample, 0.95))

    rows = []
    for alpha in (0.10, 0.05):
        additional_wins = optimistic_wins_required(
            wins,
            observations,
            alpha,
        )
        rows.append(
            {
                "test": "exact_sign_optimistic_all_future_events_win",
                "alpha": alpha,
                "current_observations": observations,
                "current_wins": wins,
                "current_p_value": one_sided_sign_p_value(
                    wins,
                    observations,
                ),
                "additional_events_required": additional_wins,
                "minimum_calendar_years_at_historical_event_rate": (
                    additional_wins / annual_event_rate
                ),
            }
        )
    for name, values in (
        ("observed_mean_normal_approximation", sample),
        ("winsorized_95_normal_approximation", winsorized),
        ("leave_largest_out_normal_approximation", without_largest),
    ):
        required = normal_power_sample_size(values)
        rows.append(
            {
                "test": name,
                "alpha": 0.10,
                "current_observations": len(values),
                "current_wins": int((values > 0.0).sum()),
                "current_p_value": float("nan"),
                "additional_events_required": (
                    required - len(values)
                    if np.isfinite(required)
                    else float("inf")
                ),
                "minimum_calendar_years_at_historical_event_rate": (
                    (required - len(values)) / annual_event_rate
                    if np.isfinite(required)
                    else float("inf")
                ),
            }
        )
    result = pd.DataFrame(rows).set_index(["test", "alpha"])
    result.to_csv(DESTINATION / "forward_episode_evidence_budget.csv")
    print(result.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
