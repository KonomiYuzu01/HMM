import pandas as pd
import pytest

from tools.export_stock_overlay_panel_data import require_aligned_latest_date


def test_stock_overlay_export_rejects_a_stale_market_state() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
    )
    states = pd.Series(
        ["quiet_bull"],
        index=pd.to_datetime(["2026-07-27"]),
    )

    with pytest.raises(ValueError, match="market state is stale"):
        require_aligned_latest_date(states, prices)


def test_stock_overlay_export_accepts_aligned_inputs() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
    )
    states = pd.Series(
        ["quiet_bull", "quiet_bull"],
        index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
    )

    require_aligned_latest_date(states, prices)
