from __future__ import annotations

import hashlib
import json

import pandas as pd

from tools.evaluate_retail_alternatives import write_cache_metadata


def test_write_cache_metadata_records_exact_file_provenance(
    tmp_path,
) -> None:
    path = tmp_path / "prices.csv"
    frame = pd.DataFrame(
        {"GDE": [50.0, 51.0]},
        index=pd.to_datetime(["2026-07-24", "2026-07-27"]),
    )
    frame.to_csv(path, index_label="date")

    write_cache_metadata(
        path,
        frame,
        adjustment="test adjustment",
        generator="test generator",
        purpose="test purpose",
        extra={"ticker": "GDE"},
    )

    metadata = json.loads(
        path.with_suffix(".csv.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["cache_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert metadata["first_date"] == "2026-07-24"
    assert metadata["price_as_of"] == "2026-07-27"
    assert metadata["row_count"] == 2
    assert metadata["ticker"] == "GDE"
