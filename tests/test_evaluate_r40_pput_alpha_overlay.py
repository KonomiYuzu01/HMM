from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.evaluate_r40_pput_alpha_overlay import (
    PPUT_HISTORY,
    PputOverlayCandidate,
    load_pput_history,
    pput_alpha_overlay_return,
)


def test_official_pput_history_is_repository_frozen() -> None:
    assert PPUT_HISTORY == Path("data/reference/cboe_pput_history.csv")


def test_load_pput_history_parses_us_dates(tmp_path: Path) -> None:
    path = tmp_path / "pput.csv"
    path.write_text("DATE,PPUT\n06/30/1986,100\n07/01/1986,101\n")
    result = load_pput_history(path)
    assert result.index[0] == pd.Timestamp("1986-06-30")
    assert result.iloc[-1] == pytest.approx(101.0)


def test_overlay_is_zero_before_official_history_and_charges_no_fee() -> None:
    dates = pd.to_datetime(["1986-06-27", "1986-06-30"])
    closes = pd.DataFrame({"SPX": [100.0, 101.0]}, index=dates)
    pput = pd.Series([100.0], index=[pd.Timestamp("1986-06-30")])
    overlay, available = pput_alpha_overlay_return(
        closes,
        pput,
        PputOverlayCandidate("test", 0.03),
    )
    assert not available.any()
    assert overlay.eq(0.0).all()


def test_overlay_uses_pput_minus_spx_and_adds_implementation_drag() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    closes = pd.DataFrame({"SPX": [100.0, 90.0]}, index=dates)
    pput = pd.Series([100.0, 95.0], index=dates)
    candidate = PputOverlayCandidate(
        "test",
        overlay_notional=0.04,
        annual_implementation_drag=0.005,
    )
    overlay, available = pput_alpha_overlay_return(closes, pput, candidate)
    expected = 0.04 * (0.05) - 0.04 * 0.005 / 252.0
    assert available.iloc[-1]
    assert overlay.iloc[-1] == pytest.approx(expected)
