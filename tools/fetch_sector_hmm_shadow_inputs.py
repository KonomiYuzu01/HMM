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
import yaml
import yfinance as yf


DEFAULT_CONFIG = Path("config/research_sector_hmm_shadow.yaml")
DEFAULT_PRICE_OUTPUT = Path("data/sector_hmm_shadow_prices.csv")
DEFAULT_MACRO_OUTPUT = Path("data/sector_hmm_shadow_macro.csv")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


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
    series = pd.to_numeric(frame[series_id], errors="coerce").rename(series_id)
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"{series_id} has invalid dates")
    if series.dropna().empty:
        raise ValueError(f"{series_id} contains no observations")
    return series


def download_fred_series(series_id: str) -> tuple[pd.Series, dict[str, object]]:
    url = FRED_URL.format(series_id=series_id)
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=30, context=context) as response:
        if response.status != 200:
            raise RuntimeError(
                f"FRED {series_id} download failed with HTTP {response.status}"
            )
        content = response.read()
    series = parse_fred_csv(content, series_id)
    valid = series.dropna()
    return series, {
        "url": url,
        "source": "Federal Reserve Bank of St. Louis FRED",
        "raw_sha256": sha256_bytes(content),
        "first_observation": valid.index.min().date().isoformat(),
        "last_observation": valid.index.max().date().isoformat(),
        "valid_observations": int(valid.shape[0]),
    }


def download_prices(
    tickers: list[str],
    start: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no prices")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response does not contain Close")
        prices = raw["Close"].copy()
    elif len(tickers) == 1 and "Close" in raw:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise ValueError("Unexpected Yahoo Finance response columns")
    prices = prices.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.index.name = "date"
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price cache has invalid dates")
    missing = [ticker for ticker in tickers if prices[ticker].dropna().empty]
    if missing:
        raise ValueError(f"Yahoo Finance returned no prices for {missing}")
    sources: dict[str, object] = {}
    for ticker in tickers:
        valid = prices[ticker].dropna()
        sources[ticker] = {
            "source": "Yahoo Finance adjusted close via yfinance",
            "first_observation": valid.index.min().date().isoformat(),
            "last_observation": valid.index.max().date().isoformat(),
            "valid_observations": int(valid.shape[0]),
        }
    return prices, sources


def write_csv_with_metadata(
    frame: pd.DataFrame,
    output: Path,
    metadata: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    metadata["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata["rows"] = int(frame.shape[0])
    metadata["first_date"] = frame.index.min().date().isoformat()
    metadata["last_date"] = frame.index.max().date().isoformat()
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch isolated inputs for the sector-HMM shadow experiment"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--price-output", type=Path, default=DEFAULT_PRICE_OUTPUT)
    parser.add_argument("--macro-output", type=Path, default=DEFAULT_MACRO_OUTPUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assets = config["assets"]
    tickers = list(
        dict.fromkeys(
            [
                assets["benchmark"],
                assets["defensive_bond"],
                assets["credit"],
                assets["cash_proxy"],
                *assets["sectors"],
            ]
        )
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    prices, price_sources = download_prices(
        tickers,
        str(config["research"]["start"]),
    )
    write_csv_with_metadata(
        prices,
        args.price_output,
        {
            "generated_at_utc": generated_at,
            "purpose": config["research"]["purpose"],
            "adjustment": "auto_adjust=True",
            "sources": price_sources,
        },
    )

    macro_series: list[pd.Series] = []
    macro_sources: dict[str, object] = {}
    for series_id in config["macro"]["series"]:
        series, source = download_fred_series(str(series_id))
        macro_series.append(series)
        macro_sources[str(series_id)] = source
    macro = pd.concat(macro_series, axis=1, sort=True).sort_index()
    macro.index.name = "date"
    write_csv_with_metadata(
        macro,
        args.macro_output,
        {
            "generated_at_utc": generated_at,
            "purpose": config["research"]["purpose"],
            "sources": macro_sources,
        },
    )
    print(args.price_output.resolve())
    print(args.macro_output.resolve())


if __name__ == "__main__":
    main()
