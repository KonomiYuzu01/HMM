from __future__ import annotations

import pandas as pd

from scripts.build_20y_proxy_prices import (
    clip_to_completion_reference,
)


def test_proxy_tail_is_clipped_to_complete_production_date(
    tmp_path,
) -> None:
    reference = tmp_path / "open_close.csv"
    pd.DataFrame(
        {"close_SPX": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-23", "2026-07-24"]),
    ).to_csv(reference, index_label="date")
    proxy = pd.DataFrame(
        {"SPX": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(
            ["2026-07-23", "2026-07-24", "2026-07-27"]
        ),
    )

    clipped, as_of = clip_to_completion_reference(proxy, reference)

    assert as_of == pd.Timestamp("2026-07-24")
    assert clipped.index.max() == pd.Timestamp("2026-07-24")
