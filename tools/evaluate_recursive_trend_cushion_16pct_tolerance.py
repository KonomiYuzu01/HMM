from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import tools.evaluate_recursive_cushion_budget as cushion
import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/recursive_trend_cushion_16pct_tolerance")
CENTRAL = cushion.CushionCandidate(
    "target_16pct_tolerance",
    floor_drawdown=-0.1475,
    bull_multiplier=100.0,
    bear_multiplier=8.5,
    tier_size=0.05,
)
NEIGHBORS = (
    replace(CENTRAL, name="bear_0825", bear_multiplier=8.25),
    replace(CENTRAL, name="bear_0875", bear_multiplier=8.75),
    replace(CENTRAL, name="floor_145", floor_drawdown=-0.145),
    replace(CENTRAL, name="floor_150", floor_drawdown=-0.15),
    replace(CENTRAL, name="tier_04", tier_size=0.04),
    replace(CENTRAL, name="tier_06", tier_size=0.06),
)


def write_acceptance() -> bool:
    metrics = pd.read_csv(OUTPUT / "metrics.csv")
    events = pd.read_csv(OUTPUT / "pre2008_events.csv")
    neighborhood = pd.read_csv(OUTPUT / "parameter_neighborhood.csv")
    pre = metrics.loc[metrics["sample"].eq("pre2008")].iloc[0]
    modern = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    proxy = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    neighbor_pre = neighborhood.loc[neighborhood["sample"].eq("pre2008")]
    gates = [
        (
            "all_named_events_mdd_at_or_above_minus_16_5pct",
            events["candidate_max_drawdown"].ge(-0.165).all(),
        ),
        (
            "complete_pre2008_mdd_at_or_above_minus_16_5pct",
            pre["candidate_max_drawdown"] >= -0.165,
        ),
        ("modern_cagr_cost_at_most_2pct", modern["cagr_delta"] >= -0.02),
        ("proxy_cagr_cost_at_most_2pct", proxy["cagr_delta"] >= -0.02),
        (
            "neighbor_pre2008_mdd_at_or_above_minus_17pct",
            neighbor_pre["candidate_max_drawdown"].ge(-0.17).all(),
        ),
    ]
    acceptance = pd.DataFrame(gates, columns=["gate", "passed"])
    research_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = ["research_pass", research_pass]
    acceptance.to_csv(OUTPUT / "acceptance_16pct_tolerance.csv", index=False)
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "accepted_max_drawdown": -0.165,
            "capital_line_drawdown": CENTRAL.floor_drawdown,
            "research_pass_16_5pct_and_2pct": research_pass,
            "production_eligible": False,
            "production_changed": False,
            "orders_generated": False,
            "complete_pre2008_candidate_max_drawdown": float(
                pre["candidate_max_drawdown"]
            ),
            "modern_candidate_cagr": float(modern["candidate_cagr"]),
            "modern_cagr_delta": float(modern["cagr_delta"]),
            "proxy_candidate_cagr": float(proxy["candidate_cagr"]),
            "proxy_cagr_delta": float(proxy["cagr_delta"]),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return research_pass


def main() -> None:
    original_output = reference.OUTPUT
    original_central = reference.CENTRAL
    original_neighbors = reference.NEIGHBORS
    original_simulate = reference.simulate_candidate
    try:
        reference.OUTPUT = OUTPUT
        reference.CENTRAL = CENTRAL  # type: ignore[assignment]
        reference.NEIGHBORS = NEIGHBORS  # type: ignore[assignment]
        reference.simulate_candidate = cushion.simulate_cushion_candidate  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.simulate_candidate = original_simulate
    passed = write_acceptance()
    print((OUTPUT / "summary.json").read_text())
    print("\n16% tolerance acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance_16pct_tolerance.csv").to_string(
            index=False
        )
    )
    if not passed:
        print("\nThe tolerance specification failed combined qualification.")


if __name__ == "__main__":
    main()
