from __future__ import annotations

import pandas as pd

from tools.export_r38_panel_snapshot import effective_execution_window_status


def test_execution_window_remains_upcoming_before_open() -> None:
    next_open = pd.Timestamp("2026-08-06 09:30", tz="America/New_York")
    now = pd.Timestamp("2026-08-06 13:29", tz="UTC")
    assert effective_execution_window_status("UPCOMING", next_open, now_utc=now) == "UPCOMING"


def test_execution_window_is_missed_at_or_after_open() -> None:
    next_open = pd.Timestamp("2026-08-06 09:30", tz="America/New_York")
    now = pd.Timestamp("2026-08-06 13:30", tz="UTC")
    assert effective_execution_window_status("UPCOMING", next_open, now_utc=now) == "MISSED"
