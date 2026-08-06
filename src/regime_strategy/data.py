from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Callable, Mapping
from datetime import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def validate_price_frame(
    prices: pd.DataFrame,
    *,
    source: str,
) -> None:
    """Fail closed when a price table is not safe for strategy calculations."""
    if prices.empty:
        raise ValueError(f"{source} contains no price observations")
    if prices.index.hasnans:
        raise ValueError(f"{source} contains invalid dates")
    duplicate_count = int(prices.index.duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(f"{source} contains {duplicate_count} duplicate dates")
    if not prices.index.is_monotonic_increasing:
        raise ValueError(f"{source} dates are not sorted in increasing order")
    missing_count = int(prices.isna().sum().sum())
    if missing_count:
        raise ValueError(f"{source} contains {missing_count} missing values")
    finite = np.isfinite(prices.to_numpy(dtype=float))
    if not bool(finite.all()):
        raise ValueError(f"{source} contains non-finite values")
    non_positive_count = int((prices <= 0.0).sum().sum())
    if non_positive_count:
        raise ValueError(f"{source} contains {non_positive_count} non-positive values")


def completed_us_daily_prices(
    prices: pd.DataFrame,
    now: pd.Timestamp | None = None,
    completion_time: time = time(16, 15),
    source_modified_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Exclude a current U.S. session until its daily bar is safely complete.

    A cache written before the completion buffer remains incomplete even when it
    is read later that evening. This prevents an intraday snapshot from becoming
    a seemingly valid close merely because wall-clock time advanced.
    """
    current = now or pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    current_date = current.date()
    current_session_complete = (
        current.weekday() < 5 and current.time().replace(tzinfo=None) >= completion_time
    )
    if source_modified_at is not None:
        modified = source_modified_at
        if modified.tzinfo is None:
            modified = modified.tz_localize("America/New_York")
        else:
            modified = modified.tz_convert("America/New_York")
        source_has_completed_session = (
            modified.date() > current_date
            or (
                modified.date() == current_date
                and modified.time().replace(tzinfo=None) >= completion_time
            )
        )
        current_session_complete = (
            current_session_complete and source_has_completed_session
        )
    cutoff = current_date if current_session_complete else current_date - pd.Timedelta(days=1)
    completed = prices.loc[prices.index.date <= cutoff].copy()
    if completed.empty:
        raise RuntimeError("No completed U.S. daily price observations are available")
    return completed


def download_adjusted_prices(
    tickers: Mapping[str, str],
    start: str,
    end: str | None = None,
    leading_backfill_assets: list[str] | None = None,
    signal_history_fallbacks: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Download adjusted closes and return columns named by economic sleeve."""
    symbols = list(tickers.values())
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})

    symbol_to_sleeve = {symbol: sleeve for sleeve, symbol in tickers.items()}
    prices = close.rename(columns=symbol_to_sleeve).reindex(columns=list(tickers))
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = fill_leading_asset_prices(
        prices.sort_index(),
        leading_backfill_assets or [],
    )
    prices = fill_signal_history_fallbacks(
        prices,
        signal_history_fallbacks or {},
    )
    signal_quality = prices.attrs.get("signal_history_fallbacks", {})
    prices = prices.ffill(limit=3).dropna(how="any")
    if prices.empty:
        raise RuntimeError("No common adjusted-price history exists for the configured universe")
    prices = completed_us_daily_prices(prices.astype(float))
    validate_price_frame(prices, source="Downloaded prices")
    for signal, raw_quality in signal_quality.items():
        quality = dict(raw_quality)
        quality["remaining_missing_observations"] = int(
            prices[signal].isna().sum()
        )
        quality["output_first_date"] = prices.index[0].date().isoformat()
        quality["output_last_date"] = prices.index[-1].date().isoformat()
        signal_quality[signal] = quality
    prices.attrs["signal_history_fallbacks"] = signal_quality
    return prices


def fill_leading_asset_prices(
    prices: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """Fill only pre-inception rows for explicitly excluded overlay assets."""
    unknown = sorted(set(assets).difference(prices))
    if unknown:
        raise ValueError(f"Unknown leading-backfill assets: {unknown}")
    filled = prices.copy()
    for asset in assets:
        first_valid = filled[asset].first_valid_index()
        if first_valid is None:
            raise ValueError(f"Leading-backfill asset has no prices: {asset}")
        leading = filled.index < first_valid
        filled.loc[leading, asset] = float(filled.loc[first_valid, asset])
    return filled


def download_csv_history(url: str) -> pd.DataFrame:
    """Download a public CSV using the verified requests/certifi trust store."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def fill_signal_history_fallbacks(
    prices: pd.DataFrame,
    fallbacks: Mapping[str, object],
    loader: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fill or replace signal observations from explicit date-indexed sources."""
    filled = prices.copy()
    quality: dict[str, object] = {}
    for signal, raw_config in fallbacks.items():
        if signal not in filled:
            raise ValueError(f"Signal fallback is not a downloaded column: {signal}")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Signal fallback configuration must be a mapping: {signal}")
        url = str(raw_config["url"])
        date_column = str(raw_config.get("date_column", "DATE"))
        value_column = str(raw_config.get("value_column", "CLOSE"))
        authoritative = bool(raw_config.get("authoritative", False))
        history = (loader or download_csv_history)(url)
        missing_columns = [
            column
            for column in (date_column, value_column)
            if column not in history
        ]
        if missing_columns:
            raise ValueError(
                f"Signal fallback for {signal} is missing columns: "
                f"{missing_columns}"
            )
        dates = pd.to_datetime(history[date_column], errors="coerce")
        values = pd.to_numeric(history[value_column], errors="coerce")
        source = pd.Series(values.to_numpy(), index=dates).dropna()
        source = source[source.index.notna()]
        source = source[~source.index.duplicated(keep="last")].sort_index()
        existing = filled[signal].copy()
        missing_before = filled[signal].isna()
        replacements = source.reindex(filled.index)
        overlap = existing.notna() & replacements.notna()
        absolute_conflict = (existing - replacements).abs().loc[overlap]
        conflict_count = int((absolute_conflict > 1.0e-9).sum())
        fill_mask = missing_before & replacements.notna()
        if authoritative:
            filled[signal] = replacements
        else:
            filled.loc[fill_mask, signal] = replacements.loc[fill_mask]
        quality[signal] = {
            "url": url,
            "authoritative": authoritative,
            "filled_observations": int(fill_mask.sum()),
            "overlapping_observations": int(overlap.sum()),
            "conflicting_observations": conflict_count,
            "maximum_absolute_conflict": (
                float(absolute_conflict.max())
                if not absolute_conflict.empty
                else 0.0
            ),
            "first_source_date": (
                source.index[0].date().isoformat() if not source.empty else None
            ),
            "latest_source_date": (
                source.index[-1].date().isoformat() if not source.empty else None
            ),
            "remaining_missing_observations": int(filled[signal].isna().sum()),
        }
    filled.attrs["signal_history_fallbacks"] = quality
    return filled


def load_prices(config: Mapping[str, object], refresh: bool = False) -> pd.DataFrame:
    cache = Path(str(config["cache"]))
    tickers = dict(config["tickers"])  # type: ignore[arg-type]
    signal_tickers = dict(config.get("signal_tickers", {}))  # type: ignore[arg-type]
    cache_signals = [str(name) for name in config.get("cache_signals", [])]  # type: ignore[arg-type]
    overlap = set(tickers).intersection(set(signal_tickers).union(cache_signals))
    if overlap:
        raise ValueError(f"Tradable and signal names overlap: {sorted(overlap)}")
    duplicate_signals = set(signal_tickers).intersection(cache_signals)
    if duplicate_signals:
        raise ValueError(f"Downloaded and cache-only signals overlap: {sorted(duplicate_signals)}")
    requested_columns = [*tickers, *signal_tickers, *cache_signals]
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [name for name in requested_columns if name not in prices]
        if missing:
            raise ValueError(
                f"Cached prices are missing configured columns: {missing}; rerun with --refresh"
            )
        prices = prices.reindex(columns=requested_columns).astype(float)
        metadata_path = cache.with_suffix(cache.suffix + ".metadata.json")
        if metadata_path.exists():
            data_quality = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            expected_sha256 = data_quality.get("cache_sha256")
            actual_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Cached prices SHA-256 does not match {metadata_path}; "
                    "rebuild or refresh the cache"
                )
            prices.attrs["data_quality"] = data_quality
        validate_price_frame(prices, source=f"Cached prices {cache}")
        cache_modified_at = pd.Timestamp.fromtimestamp(
            cache.stat().st_mtime,
            tz="America/New_York",
        )
        return completed_us_daily_prices(
            prices,
            source_modified_at=cache_modified_at,
        )

    if cache_signals:
        raise ValueError(
            "Cache-only signals cannot be refreshed by Yahoo; rebuild the configured cache first"
        )

    prices = download_adjusted_prices(
        tickers={**tickers, **signal_tickers},
        start=str(config["start"]),
        end=config.get("end") and str(config["end"]),
        leading_backfill_assets=[
            str(asset)
            for asset in config.get("leading_backfill_assets", [])  # type: ignore[arg-type]
        ],
        signal_history_fallbacks=dict(
            config.get("signal_history_fallbacks", {})  # type: ignore[arg-type]
        ),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache, index_label="date")
    cache_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
    data_quality = {
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": "auto_adjust=True",
        "ticker_map": {**tickers, **signal_tickers},
        "requested_start": str(config["start"]),
        "requested_end": config.get("end") and str(config["end"]),
        "first_date": prices.index[0].date().isoformat(),
        "price_as_of": prices.index[-1].date().isoformat(),
        "row_count": len(prices),
        "cache_sha256": cache_sha256,
        "signal_history_fallbacks": prices.attrs.get(
            "signal_history_fallbacks",
            {},
        ),
    }
    cache.with_suffix(cache.suffix + ".metadata.json").write_text(
        json.dumps(data_quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prices.attrs["data_quality"] = data_quality
    return prices
