from __future__ import annotations

import pandas as pd

from tools.export_r38_panel_snapshot import (
    effective_execution_window_status,
    r40_eligibility,
)


def test_execution_window_remains_upcoming_before_open() -> None:
    next_open = pd.Timestamp("2026-08-06 09:30", tz="America/New_York")
    now = pd.Timestamp("2026-08-06 13:29", tz="UTC")
    assert effective_execution_window_status("UPCOMING", next_open, now_utc=now) == "UPCOMING"


def test_execution_window_is_missed_at_or_after_open() -> None:
    next_open = pd.Timestamp("2026-08-06 09:30", tz="America/New_York")
    now = pd.Timestamp("2026-08-06 13:30", tz="UTC")
    assert effective_execution_window_status("UPCOMING", next_open, now_utc=now) == "MISSED"


def test_r40_same_date_qualification_allows_draft_during_freeze() -> None:
    status = r40_eligibility(
        qualification_pass=True,
        r40_price_as_of="2026-08-07",
        active_price_as_of="2026-08-07",
        successor_promotion_allowed=False,
    )

    assert status == {
        "productionQualificationPass": True,
        "promotionAllowed": False,
        "draftEligible": True,
        "productionEligible": False,
        "sameDate": True,
    }


def test_r40_stale_qualification_blocks_draft() -> None:
    status = r40_eligibility(
        qualification_pass=True,
        r40_price_as_of="2026-08-06",
        active_price_as_of="2026-08-07",
        successor_promotion_allowed=False,
    )

    assert status["draftEligible"] is False
