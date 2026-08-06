from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import tools.evaluate_recursive_cushion_budget as cushion
import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/recursive_trend_cushion_16pct")
CENTRAL = cushion.CushionCandidate(
    "target_16pct_with_1_25pct_reserve",
    floor_drawdown=-0.1475,
    bull_multiplier=100.0,
    bear_multiplier=9.5,
    tier_size=0.05,
)
NEIGHBORS = (
    replace(CENTRAL, name="floor_145", floor_drawdown=-0.145),
    replace(CENTRAL, name="floor_150", floor_drawdown=-0.15),
    replace(CENTRAL, name="literal_floor_160", floor_drawdown=-0.16),
    replace(CENTRAL, name="bear_0925", bear_multiplier=9.25),
    replace(CENTRAL, name="bear_0975", bear_multiplier=9.75),
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
    neighbor_modern = neighborhood.loc[
        neighborhood["sample"].eq("normal_synthetic")
    ]
    gates = [
        (
            "all_named_events_mdd_at_or_above_minus_16pct",
            events["candidate_max_drawdown"].ge(-0.16).all(),
        ),
        (
            "complete_pre2008_mdd_at_or_above_minus_16pct",
            pre["candidate_max_drawdown"] >= -0.16,
        ),
        ("modern_cagr_cost_at_most_2pct", modern["cagr_delta"] >= -0.02),
        ("proxy_cagr_cost_at_most_2pct", proxy["cagr_delta"] >= -0.02),
        ("modern_drawdown_not_worse", modern["max_drawdown_delta"] >= 0.0),
        ("proxy_drawdown_not_worse", proxy["max_drawdown_delta"] >= 0.0),
        (
            "nearby_pre2008_mdd_at_or_above_minus_18pct",
            neighbor_pre.loc[
                ~neighbor_pre["candidate"].eq("literal_floor_160"),
                "candidate_max_drawdown",
            ].ge(-0.18).all(),
        ),
        (
            "nearby_modern_cagr_cost_at_most_2_5pct",
            neighbor_modern.loc[
                ~neighbor_modern["candidate"].eq("literal_floor_160"),
                "cagr_delta",
            ].ge(-0.025).all(),
        ),
    ]
    acceptance = pd.DataFrame(gates, columns=["gate", "passed"])
    research_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = ["research_pass", research_pass]
    acceptance.to_csv(OUTPUT / "acceptance_16pct.csv", index=False)
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    literal = neighborhood.loc[
        neighborhood["candidate"].eq("literal_floor_160")
    ]
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_max_drawdown": -0.16,
            "capital_line_drawdown": CENTRAL.floor_drawdown,
            "research_pass_16pct_and_2pct": research_pass,
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
            "literal_floor_16_comparison": literal.to_dict(orient="records"),
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
    print("\n16% acceptance:")
    print(pd.read_csv(OUTPUT / "acceptance_16pct.csv").to_string(index=False))
    if not passed:
        print("\nThe 16% specification remains research-only and failed qualification.")


if __name__ == "__main__":
    main()
