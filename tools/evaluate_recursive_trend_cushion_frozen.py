from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

import tools.evaluate_recursive_cushion_budget as cushion
import tools.evaluate_recursive_equity_risk_budget as recursive
import tools.evaluate_reference_drawdown_governor as reference


OUTPUT = Path("output/recursive_trend_cushion_frozen")
CENTRAL = cushion.CushionCandidate(
    "central",
    floor_drawdown=-0.1875,
    bull_multiplier=100.0,
    bear_multiplier=9.25,
    tier_size=0.05,
)
NEIGHBORS = (
    replace(CENTRAL, name="bear_090", bear_multiplier=9.0),
    replace(CENTRAL, name="bear_095", bear_multiplier=9.5),
    replace(CENTRAL, name="floor_185", floor_drawdown=-0.185),
    replace(CENTRAL, name="floor_190", floor_drawdown=-0.19),
    replace(CENTRAL, name="tier_04", tier_size=0.04),
    replace(CENTRAL, name="tier_06", tier_size=0.06),
)


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
    passed = recursive.write_strict_acceptance(OUTPUT)
    print((OUTPUT / "summary.json").read_text())
    print("\nStrict acceptance:")
    print(pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False))
    if not passed:
        print("\nCandidate remains research-only and failed strict qualification.")


if __name__ == "__main__":
    main()
