from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r10_combined_family_reality_check import (
    relative_log_path,
)
from tools.evaluate_smh_causal_family_reality_check import (
    circular_family_reality_check,
)


OUTPUT = Path("output/r10_frontier_family_reality_check")
FRONTIER = Path("output/r10_high_fraction_band_frontier")
BASELINE = Path("output/r10_combined_tail_capital")
FRACTIONS = (0.40, 0.50, 0.60)
BANDS = (0.0025, 0.0100, 0.0200)
SELECTED = (0.50, 0.0200)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    rankings: list[pd.DataFrame] = []
    for sample, sample_prefix in {
        "normal": "normal_synthetic",
        "proxy": "proxy_synthetic",
    }.items():
        baseline = pd.read_csv(
            BASELINE / f"{sample_prefix}_baseline_daily.csv",
            index_col=0,
            parse_dates=True,
        )["net_return"]
        arrays: list[np.ndarray] = []
        names: list[str] = []
        selected_index: int | None = None
        for fraction in FRACTIONS:
            for band in BANDS:
                fraction_label = int(round(fraction * 100))
                band_label = int(round(band * 10_000))
                candidate = pd.read_csv(
                    FRONTIER
                    / (
                        f"{sample_prefix}_fraction{fraction_label}"
                        f"_band{band_label}bp_daily.csv"
                    ),
                    index_col=0,
                    parse_dates=True,
                )["net_return"]
                arrays.append(relative_log_path(candidate, baseline))
                names.append(
                    f"fraction{fraction_label}_band{band_label}bp"
                )
                if (fraction, band) == SELECTED:
                    selected_index = len(arrays) - 1
        if selected_index is None:
            raise RuntimeError("Selected frontier candidate is missing")
        matrix = np.column_stack(arrays)
        observed = matrix.mean(axis=0) * 252.0
        ranking = (
            pd.DataFrame(
                {
                    "sample": sample,
                    "candidate": names,
                    "annualized_relative_log_return": observed,
                    "includes_selected": [
                        int(index == selected_index)
                        for index in range(len(names))
                    ],
                }
            )
            .sort_values(
                "annualized_relative_log_return",
                ascending=False,
            )
            .reset_index(drop=True)
        )
        ranking["return_rank"] = np.arange(1, len(ranking) + 1)
        rankings.append(ranking)
        selected_rank = int(
            ranking.loc[
                ranking["includes_selected"].eq(1),
                "return_rank",
            ].iloc[0]
        )
        for block_days in (21, 63, 126):
            rows.append(
                {
                    "sample": sample,
                    **circular_family_reality_check(
                        matrix,
                        selected_index,
                        block_days,
                    ),
                    "declared_candidate_paths": len(names),
                    "selected_rank_by_return": selected_rank,
                    "selection_method": (
                        "highest_fraction_then_smallest_band_passing_"
                        "declared_risk_constraints"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    ranking_result = pd.concat(rankings, ignore_index=True)
    result.to_csv(OUTPUT / "family_reality_check.csv", index=False)
    ranking_result.to_csv(
        OUTPUT / "candidate_ranking.csv",
        index=False,
    )
    print("Frontier family reality check:")
    print(result.round(6).to_string(index=False))
    print("\nCandidate ranking:")
    print(ranking_result.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
