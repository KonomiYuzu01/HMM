from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = Path("output/research_hmm_gaussian_diagnostics")
CANDIDATE = Path("output/research_hmm_postchange_identity")
OUTPUT = Path("output/hmm_postchange_identity")
TOLERANCE = 1e-10


def maximum_common_numeric_difference(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> float:
    common_index = baseline.index.intersection(candidate.index)
    common_columns = baseline.select_dtypes(include=[np.number]).columns.intersection(
        candidate.select_dtypes(include=[np.number]).columns
    )
    if common_index.empty or common_columns.empty:
        raise ValueError("Frames must share dates and numeric columns")
    return float(
        candidate.loc[common_index, common_columns]
        .sub(baseline.loc[common_index, common_columns])
        .abs()
        .max()
        .max()
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily_baseline = pd.read_csv(
        BASELINE / "daily_returns.csv", index_col=0, parse_dates=True
    )
    daily_candidate = pd.read_csv(
        CANDIDATE / "daily_returns.csv", index_col=0, parse_dates=True
    )
    weights_baseline = pd.read_csv(BASELINE / "weights.csv", index_col=0, parse_dates=True)
    weights_candidate = pd.read_csv(CANDIDATE / "weights.csv", index_col=0, parse_dates=True)
    daily_difference = maximum_common_numeric_difference(
        daily_baseline[["net_return"]], daily_candidate[["net_return"]]
    )
    weight_difference = maximum_common_numeric_difference(
        weights_baseline, weights_candidate
    )

    member_rows: list[dict[str, object]] = []
    for baseline_path in sorted((BASELINE / "members").glob("seed_*/regimes.csv")):
        candidate_path = CANDIDATE / "members" / baseline_path.parent.name / "regimes.csv"
        baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
        candidate = pd.read_csv(candidate_path, index_col=0, parse_dates=True)
        common = baseline.index.intersection(candidate.index)
        member_rows.append(
            {
                "member": baseline_path.parent.name,
                "observations": len(common),
                "order_agreement": float(
                    baseline.loc[common, "hmm_order"]
                    .eq(candidate.loc[common, "hmm_order"])
                    .mean()
                ),
                "risk_on_agreement": float(
                    baseline.loc[common, "paper_risk_on_candidate"]
                    .eq(candidate.loc[common, "paper_risk_on_candidate"])
                    .mean()
                ),
            }
        )
    members = pd.DataFrame(member_rows)
    members.to_csv(OUTPUT / "member_identity.csv", index=False)
    gates = {
        "daily_net_return_within_tolerance": daily_difference <= TOLERANCE,
        "weights_within_tolerance": weight_difference <= TOLERANCE,
        "orders_identical": bool(members["order_agreement"].eq(1.0).all()),
        "risk_on_identical": bool(members["risk_on_agreement"].eq(1.0).all()),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_identity",
        "production_changed": False,
        "orders_generated": False,
        "tolerance": TOLERANCE,
        "maximum_daily_net_return_difference": daily_difference,
        "maximum_weight_difference": weight_difference,
        "gates": gates,
        "identity_pass": bool(all(gates.values())),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(members.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["identity_pass"]:
        raise SystemExit("Default-path identity audit failed")


if __name__ == "__main__":
    main()
