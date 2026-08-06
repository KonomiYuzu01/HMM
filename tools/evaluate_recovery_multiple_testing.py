from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
DESTINATION = OUTPUT / "correction_veto_validation"
DEVELOPMENT = slice("2015-01-01", "2021-12-31")
BASELINE = "paper_core_growth_gold20_daily_risk_netted_ensemble"
FINAL_CANDIDATE = "correction_veto63_exact_floor_guard"
CANDIDATES = {
    "rebound20": (
        "paper_core_growth_gold20_recovery_rebound20_netted_ensemble"
    ),
    "positive20": (
        "paper_core_growth_gold20_recovery_positive20_netted_ensemble"
    ),
    "rebound63": (
        "paper_core_growth_gold20_recovery_rebound63_netted_ensemble"
    ),
    "positive63": (
        "paper_core_growth_gold20_recovery_positive63_netted_ensemble"
    ),
    "positive20_guarded": (
        "paper_core_growth_gold20_recovery_positive20_"
        "guarded_netted_ensemble"
    ),
    "correction_veto20_guarded": (
        "paper_core_growth_gold20_correction_veto20_"
        "guarded_netted_ensemble"
    ),
    "correction_veto63_guarded": (
        "paper_core_growth_gold20_correction_veto63_"
        "guarded_netted_ensemble"
    ),
    "slow_negative_guarded": (
        "paper_core_growth_gold20_slow_negative_guarded_netted_ensemble"
    ),
    FINAL_CANDIDATE: (
        "paper_core_growth_gold20_correction_veto63_exact_"
        "floor_guard_netted_ensemble"
    ),
}


def load(directory: str) -> pd.Series:
    frame = pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return frame.loc[DEVELOPMENT, "net_return"]


def circular_reality_check(
    differential: pd.DataFrame,
    block_days: int,
    samples: int = 10_000,
    seed: int = 20_260_724,
) -> tuple[np.ndarray, np.ndarray]:
    values = differential.to_numpy(dtype=float)
    observed = values.mean(axis=0) * 252.0
    centered = values - values.mean(axis=0, keepdims=True)
    count = len(centered)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    bootstrapped_maxima = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        bootstrapped_maxima[sample] = float(
            centered[indices].mean(axis=0).max() * 252.0
        )
    return observed, bootstrapped_maxima


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    baseline = load(BASELINE)
    differential = pd.concat(
        {
            name: np.log1p(load(directory)) - np.log1p(baseline)
            for name, directory in CANDIDATES.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    observed = differential.mean(axis=0) * 252.0
    observed.rename("annualized_relative_log_return").to_csv(
        DESTINATION / "multiple_testing_observed.csv"
    )

    rows: list[dict[str, float | int | str]] = []
    for block_days in (21, 63, 126):
        block_observed, null_maxima = circular_reality_check(
            differential,
            block_days,
        )
        best_index = int(np.argmax(block_observed))
        final_observed = float(observed.loc[FINAL_CANDIDATE])
        rows.append(
            {
                "block_days": block_days,
                "candidate_count": len(CANDIDATES),
                "family_best_strategy": differential.columns[best_index],
                "family_observed_max": float(block_observed[best_index]),
                "family_reality_check_p_value": float(
                    (null_maxima >= block_observed[best_index]).mean()
                ),
                "final_candidate_observed": final_observed,
                "final_candidate_selection_adjusted_p_value": float(
                    (null_maxima >= final_observed).mean()
                ),
                "null_maximum_95": float(
                    np.quantile(null_maxima, 0.95)
                ),
            }
        )
    result = pd.DataFrame(rows).set_index("block_days")
    result.to_csv(DESTINATION / "multiple_testing_reality_check.csv")
    print(result.round(6).to_string())


if __name__ == "__main__":
    main()
