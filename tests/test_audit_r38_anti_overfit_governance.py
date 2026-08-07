from __future__ import annotations

import pandas as pd
import pytest

from tools.audit_r38_anti_overfit_governance import (
    completed_forward_sessions,
    spa_evidence_passes,
)


def test_forward_sessions_count_only_post_freeze_completed_dates() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2026-07-28",
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
                "2026-08-03",
            ]
        )
    )
    assert (
        completed_forward_sessions(
            dates,
            freeze_price_as_of="2026-07-28",
            current_price_as_of="2026-08-03",
        )
        == 4
    )


def test_forward_sessions_reject_duplicate_dates() -> None:
    dates = pd.Series(pd.to_datetime(["2026-07-29", "2026-07-29"]))
    with pytest.raises(ValueError, match="ordered and unique"):
        completed_forward_sessions(
            dates,
            freeze_price_as_of="2026-07-28",
            current_price_as_of="2026-07-29",
        )


def test_spa_evidence_requires_all_blocks_paths_and_adjusted_significance() -> None:
    frame = pd.DataFrame(
        {
            "target": ["r38_accel1375"] * 3,
            "block_days": [21, 63, 126],
            "unique_candidate_paths": [900, 900, 900],
            "familywise_spa_p_value": [0.02, 0.03, 0.04],
        }
    )
    arguments = {
        "target": "r38_accel1375",
        "required_block_days": {21, 63, 126},
        "minimum_unique_candidate_paths": 900,
        "maximum_p_value": 0.05,
    }
    assert spa_evidence_passes(frame, **arguments)
    frame.loc[2, "familywise_spa_p_value"] = 0.051
    assert not spa_evidence_passes(frame, **arguments)
