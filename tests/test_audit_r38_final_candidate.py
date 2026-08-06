from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from tools.audit_r38_final_candidate import (
    research_gate_pass,
    spa_gate_pass,
    validate_market_frame,
)


def test_spa_requires_all_three_fixed_block_lengths() -> None:
    spa = pd.DataFrame(
        {
            "target": ["r38", "r38", "r38"],
            "block_days": [21, 63, 126],
            "familywise_spa_p_value": [0.02, 0.03, 0.04],
        }
    )
    assert spa_gate_pass(spa)
    assert not spa_gate_pass(spa.iloc[:2])
    failed = spa.copy()
    failed.loc[2, "familywise_spa_p_value"] = 0.05
    assert not spa_gate_pass(failed)


def test_research_gate_cannot_ignore_regular_failure() -> None:
    acceptance = pd.DataFrame(
        {
            "gate": [
                "complete_cagr_at_least_25pct",
                "multiple_testing_pass",
                "production_pass",
            ],
            "passed": [False, False, False],
        }
    )
    assert not research_gate_pass(acceptance)
    acceptance.loc[0, "passed"] = True
    assert research_gate_pass(acceptance)


def test_market_frame_rejects_duplicates_and_nonpositive_prices() -> None:
    frame = pd.DataFrame(
        {"QQQ": [100.0, 0.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-02"]),
    )
    audit = validate_market_frame(frame, ("QQQ",))
    assert audit["duplicate_dates"] == 1
    assert audit["nonpositive_values"] == 1
