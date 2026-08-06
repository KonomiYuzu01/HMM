from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from regime_strategy.data import completed_us_daily_prices


OUTPUT = Path("data/r11_cross_asset_shock_open_close.csv")
ASSETS = (
    "XLK",
    "IGV",
    "XBI",
    "XLY",
    "IWM",
    "TAN",
    "KWEB",
    "ARKK",
)
REFERENCE_ASSETS = ("QQQ", "BIL")
START = "2005-01-01"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_panel(frame: pd.DataFrame) -> None:
    expected = {
        f"{field}_{asset}"
        for field in ("open", "close")
        for asset in (*ASSETS, *REFERENCE_ASSETS)
    }
    if set(frame.columns) != expected:
        raise ValueError("Cross-asset panel columns do not match registration")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Cross-asset panel dates are not sorted")
    if frame.index.duplicated().any():
        raise ValueError("Cross-asset panel contains duplicate dates")
    if frame.isna().any().any():
        raise ValueError("Cross-asset panel contains missing prices")
    if frame.le(0.0).any().any():
        raise ValueError("Cross-asset panel contains non-positive prices")


def download_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    downloaded_at = pd.Timestamp.now(tz="America/New_York")
    end = (downloaded_at.normalize() + pd.Timedelta(days=2)).date().isoformat()
    symbols = list((*ASSETS, *REFERENCE_ASSETS))
    raw = yf.download(
        symbols,
        start=START,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo did not return multi-asset Open/Close data")
    open_prices = raw["Open"].reindex(columns=symbols)
    close_prices = raw["Close"].reindex(columns=symbols)
    first_valid = {
        symbol: {
            "open": (
                open_prices[symbol].first_valid_index().date().isoformat()
                if open_prices[symbol].first_valid_index() is not None
                else None
            ),
            "close": (
                close_prices[symbol].first_valid_index().date().isoformat()
                if close_prices[symbol].first_valid_index() is not None
                else None
            ),
        }
        for symbol in symbols
    }
    panel = pd.concat(
        {"open": open_prices, "close": close_prices},
        axis=1,
    )
    panel.columns = [
        f"{field}_{asset}" for field, asset in panel.columns
    ]
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel = panel.sort_index().dropna(how="any")
    panel = completed_us_daily_prices(
        panel.astype(float),
        now=downloaded_at,
        source_modified_at=downloaded_at,
    )
    validate_panel(panel)
    metadata: dict[str, object] = {
        "purpose": (
            "Pre-registered cross-asset validation of the R11 structural "
            "shock reentry-brake mechanism; not a production quote source."
        ),
        "preregistration": (
            "research/r11_cross_asset_shock_preregistration_2026-07-29.md"
        ),
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": "auto_adjust=True",
        "requested_start": START,
        "requested_end_exclusive": end,
        "assets": list(ASSETS),
        "reference_assets": list(REFERENCE_ASSETS),
        "first_valid_date_by_asset": first_valid,
        "common_first_date": panel.index.min().date().isoformat(),
        "price_as_of": panel.index.max().date().isoformat(),
        "row_count": len(panel),
        "column_count": len(panel.columns),
        "downloaded_at": downloaded_at.isoformat(),
        "known_limitations": [
            "Yahoo Finance is a research data source, not an execution venue.",
            "ETF compositions change through time as part of the investable return series.",
            "The common window begins at the latest inception among registered ETFs.",
        ],
    }
    return panel, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    metadata_path = OUTPUT.with_suffix(OUTPUT.suffix + ".metadata.json")
    if OUTPUT.exists() and metadata_path.exists() and not args.refresh:
        frame = pd.read_csv(OUTPUT, index_col=0, parse_dates=True)
        validate_panel(frame)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_sha256(OUTPUT) != metadata["cache_sha256"]:
            raise ValueError("Cross-asset cache fingerprint does not match")
        print(f"Verified cached panel: {OUTPUT.resolve()}")
        return

    frame, metadata = download_panel()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index_label="date")
    metadata["cache_sha256"] = file_sha256(OUTPUT)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(frame)} rows from {frame.index.min().date()} "
        f"through {frame.index.max().date()}"
    )
    print(f"Panel: {OUTPUT.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
