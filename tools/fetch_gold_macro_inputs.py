from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import ssl
from urllib.request import urlopen

import certifi
import pandas as pd


SERIES = {
    "DFII10": (
        "10-Year Treasury Inflation-Indexed Security, Constant Maturity",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10",
    ),
    "DTWEXBGS": (
        "Nominal Broad U.S. Dollar Index",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS",
    ),
    "T10YIE": (
        "10-Year Breakeven Inflation Rate",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE",
    ),
}
DEFAULT_OUTPUT = Path("data/gold_macro_fred.csv")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_fred_csv(content: bytes, series_id: str) -> pd.Series:
    frame = pd.read_csv(
        BytesIO(content),
        index_col="observation_date",
        parse_dates=True,
        na_values=".",
    )
    if list(frame.columns) != [series_id]:
        raise ValueError(
            f"Unexpected FRED columns for {series_id}: {list(frame.columns)}"
        )
    series = pd.to_numeric(frame[series_id], errors="coerce")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{series_id} dates are not increasing")
    if series.index.has_duplicates:
        raise ValueError(f"{series_id} contains duplicate dates")
    if series.dropna().empty:
        raise ValueError(f"{series_id} contains no numeric observations")
    return series.rename(series_id)


def download_series(url: str) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=30, context=context) as response:
        if response.status != 200:
            raise RuntimeError(f"FRED download failed with HTTP {response.status}")
        return response.read()


def build_cache() -> tuple[pd.DataFrame, dict[str, object]]:
    columns: list[pd.Series] = []
    sources: dict[str, object] = {}
    for series_id, (description, url) in SERIES.items():
        content = download_series(url)
        series = parse_fred_csv(content, series_id)
        columns.append(series)
        valid = series.dropna()
        sources[series_id] = {
            "description": description,
            "url": url,
            "source": "Federal Reserve Bank of St. Louis FRED",
            "raw_sha256": sha256_bytes(content),
            "first_observation": valid.index.min().date().isoformat(),
            "last_observation": valid.index.max().date().isoformat(),
            "valid_observations": int(valid.shape[0]),
            "missing_observations": int(series.isna().sum()),
        }
    cache = pd.concat(columns, axis=1).sort_index()
    cache.index.name = "date"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal gold-regime research. Every trading signal must lag these "
            "end-of-day observations by at least one trading day."
        ),
        "series": sources,
        "rows": int(cache.shape[0]),
        "first_date": cache.index.min().date().isoformat(),
        "last_date": cache.index.max().date().isoformat(),
    }
    return cache, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download audited gold macro inputs from official FRED CSVs"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cache, metadata = build_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(args.output)
    output_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata["output_sha256"] = output_hash
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    print(metadata_path.resolve())


if __name__ == "__main__":
    main()
