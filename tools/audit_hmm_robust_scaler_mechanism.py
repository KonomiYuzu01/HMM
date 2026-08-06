from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output/hmm_robust_scaler_mechanism")
SAMPLES = {
    "normal": (
        Path("output/research_hmm_standard_restarts5"),
        Path("output/research_hmm_robust_restarts5"),
    ),
    "proxy": (
        Path("output/research_hmm_standard_restarts5_20y_proxy"),
        Path("output/research_hmm_robust_restarts5_20y_proxy"),
    ),
}
TEMPLATE_COLUMNS = [f"template_{index}_probability" for index in range(4)]


def aligned_weight_l1(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.Series:
    common_index = baseline.index.intersection(candidate.index)
    common_columns = baseline.columns.intersection(candidate.columns)
    if common_index.empty or common_columns.empty:
        raise ValueError("Weight frames must have common dates and assets")
    return (
        candidate.loc[common_index, common_columns]
        .sub(baseline.loc[common_index, common_columns])
        .abs()
        .sum(axis=1)
        .rename("weight_l1")
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    member_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    sample_summaries: dict[str, dict[str, float | int]] = {}
    for sample, (baseline_directory, candidate_directory) in SAMPLES.items():
        for baseline_path in sorted(
            (baseline_directory / "members").glob("seed_*/regimes.csv")
        ):
            candidate_path = (
                candidate_directory / "members" / baseline_path.parent.name / "regimes.csv"
            )
            baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
            candidate = pd.read_csv(candidate_path, index_col=0, parse_dates=True)
            common = baseline.index.intersection(candidate.index)
            probability_l1 = (
                candidate.loc[common, TEMPLATE_COLUMNS]
                .sub(baseline.loc[common, TEMPLATE_COLUMNS])
                .abs()
                .sum(axis=1)
            )
            member_rows.append(
                {
                    "sample": sample,
                    "member": baseline_path.parent.name,
                    "observations": len(common),
                    "order_agreement": float(
                        candidate.loc[common, "hmm_order"]
                        .eq(baseline.loc[common, "hmm_order"])
                        .mean()
                    ),
                    "dominant_template_agreement": float(
                        candidate.loc[common, "dominant_template"]
                        .eq(baseline.loc[common, "dominant_template"])
                        .mean()
                    ),
                    "risk_on_agreement": float(
                        candidate.loc[common, "paper_risk_on_candidate"]
                        .eq(baseline.loc[common, "paper_risk_on_candidate"])
                        .mean()
                    ),
                    "median_template_probability_l1": float(probability_l1.median()),
                    "maximum_template_probability_l1": float(probability_l1.max()),
                    "median_gross_leverage_abs_difference": float(
                        candidate.loc[common, "gross_leverage"]
                        .sub(baseline.loc[common, "gross_leverage"])
                        .abs()
                        .median()
                    ),
                    "maximum_gross_leverage_abs_difference": float(
                        candidate.loc[common, "gross_leverage"]
                        .sub(baseline.loc[common, "gross_leverage"])
                        .abs()
                        .max()
                    ),
                }
            )

        baseline_weights = pd.read_csv(
            baseline_directory / "weights.csv", index_col=0, parse_dates=True
        )
        candidate_weights = pd.read_csv(
            candidate_directory / "weights.csv", index_col=0, parse_dates=True
        )
        weight_l1 = aligned_weight_l1(baseline_weights, candidate_weights)
        sample_summaries[sample] = {
            "weight_observations": len(weight_l1),
            "median_weight_l1": float(weight_l1.median()),
            "mean_weight_l1": float(weight_l1.mean()),
            "maximum_weight_l1": float(weight_l1.max()),
            "fraction_weight_l1_above_5pct": float(weight_l1.gt(0.05).mean()),
        }
        for date, value in weight_l1.nlargest(10).items():
            top_rows.append(
                {
                    "sample": sample,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "weight_l1": float(value),
                }
            )

    members = pd.DataFrame(member_rows)
    top = pd.DataFrame(top_rows)
    members.to_csv(OUTPUT / "member_comparison.csv", index=False)
    top.to_csv(OUTPUT / "largest_weight_differences.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "samples": sample_summaries,
        "member_medians": {
            sample: {
                column: float(frame[column].median())
                for column in (
                    "order_agreement",
                    "dominant_template_agreement",
                    "risk_on_agreement",
                    "median_template_probability_l1",
                    "median_gross_leverage_abs_difference",
                )
            }
            for sample, frame in members.groupby("sample")
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(members.round(6).to_string(index=False))
    print("\nLargest final-weight differences:")
    print(top.round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
