from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_20y_proxy_open_close import splice_ohlc  # noqa: E402


def test_splice_ohlc_uses_one_scale_for_open_and_close() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    proxy_open = pd.Series([9.0, 10.0, 11.0, 12.0], index=dates)
    proxy_close = pd.Series([10.0, 11.0, 12.0, 13.0], index=dates)
    actual_open = pd.Series([None, None, 22.0, 24.0], index=dates)
    actual_close = pd.Series([None, None, 24.0, 26.0], index=dates)

    combined, first_actual = splice_ohlc(
        proxy_open,
        proxy_close,
        actual_open,
        actual_close,
        dates,
    )

    assert first_actual == dates[2]
    assert combined.loc[dates[0], "open"] == pytest.approx(18.0)
    assert combined.loc[dates[0], "close"] == pytest.approx(20.0)
    assert combined.loc[dates[2], "open"] == pytest.approx(22.0)
    assert combined.loc[dates[2], "close"] == pytest.approx(24.0)
