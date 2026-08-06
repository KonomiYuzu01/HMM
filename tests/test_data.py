import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from regime_strategy.data import (
    completed_us_daily_prices,
    fill_leading_asset_prices,
    fill_signal_history_fallbacks,
    load_prices,
)


def test_current_us_daily_bar_is_excluded_before_completion() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-21", "2026-07-22"]),
    )
    intraday = pd.Timestamp("2026-07-22 14:45", tz="America/New_York")
    completed = completed_us_daily_prices(prices, intraday)
    assert completed.index[-1] == pd.Timestamp("2026-07-21")


def test_current_us_daily_bar_is_allowed_after_completion_buffer() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-21", "2026-07-22"]),
    )
    after_close = pd.Timestamp("2026-07-22 16:16", tz="America/New_York")
    completed = completed_us_daily_prices(prices, after_close)
    assert completed.index[-1] == pd.Timestamp("2026-07-22")


def test_intraday_cache_remains_incomplete_when_read_after_close() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-21", "2026-07-22"]),
    )
    after_close = pd.Timestamp("2026-07-22 17:00", tz="America/New_York")
    intraday_write = pd.Timestamp("2026-07-22 14:00", tz="America/New_York")
    completed = completed_us_daily_prices(
        prices,
        after_close,
        source_modified_at=intraday_write,
    )
    assert completed.index[-1] == pd.Timestamp("2026-07-21")


def test_post_close_cache_allows_current_daily_bar() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-21", "2026-07-22"]),
    )
    after_close = pd.Timestamp("2026-07-22 17:00", tz="America/New_York")
    post_close_write = pd.Timestamp("2026-07-22 16:20", tz="America/New_York")
    completed = completed_us_daily_prices(
        prices,
        after_close,
        source_modified_at=post_close_write,
    )
    assert completed.index[-1] == pd.Timestamp("2026-07-22")


def test_leading_asset_backfill_does_not_fill_internal_gaps() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "CORE": [10.0, 10.1, 10.2, 10.3, 10.4],
            "HEDGE": [float("nan"), float("nan"), 5.0, float("nan"), 5.2],
        },
        index=dates,
    )
    filled = fill_leading_asset_prices(prices, ["HEDGE"])
    assert filled["HEDGE"].iloc[:3].tolist() == [5.0, 5.0, 5.0]
    assert pd.isna(filled["HEDGE"].iloc[3])
    assert filled["CORE"].equals(prices["CORE"])


def test_signal_history_fallback_fills_only_missing_dates() -> None:
    dates = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"])
    prices = pd.DataFrame(
        {"VIX3M": [20.40, float("nan"), float("nan")]},
        index=dates,
    )
    official = pd.DataFrame(
        {
            "DATE": ["07/20/2026", "07/21/2026", "07/22/2026"],
            "CLOSE": [99.0, 19.59, 19.54],
        }
    )

    filled = fill_signal_history_fallbacks(
        prices,
        {
            "VIX3M": {
                "url": "official.csv",
                "date_column": "DATE",
                "value_column": "CLOSE",
            }
        },
        loader=lambda _: official,
    )

    assert filled["VIX3M"].tolist() == [20.40, 19.59, 19.54]
    quality = filled.attrs["signal_history_fallbacks"]["VIX3M"]
    assert quality["filled_observations"] == 2
    assert quality["latest_source_date"] == "2026-07-22"


def test_authoritative_signal_history_replaces_conflicts_and_limits_dates() -> None:
    dates = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"])
    prices = pd.DataFrame({"VIX": [99.0, 18.1, 18.2]}, index=dates)
    official = pd.DataFrame(
        {
            "DATE": ["07/20/2026", "07/21/2026"],
            "CLOSE": [18.0, 18.1],
        }
    )

    filled = fill_signal_history_fallbacks(
        prices,
        {
            "VIX": {
                "url": "official.csv",
                "authoritative": True,
            }
        },
        loader=lambda _: official,
    )

    assert filled["VIX"].iloc[:2].tolist() == [18.0, 18.1]
    assert pd.isna(filled["VIX"].iloc[2])
    quality = filled.attrs["signal_history_fallbacks"]["VIX"]
    assert quality["authoritative"] is True
    assert quality["conflicting_observations"] == 1
    assert quality["maximum_absolute_conflict"] == 81.0
    assert quality["first_source_date"] == "2026-07-20"


def test_signal_history_fallback_requires_declared_columns() -> None:
    prices = pd.DataFrame(
        {"VIX3M": [float("nan")]},
        index=pd.to_datetime(["2026-07-20"]),
    )
    with pytest.raises(ValueError, match="missing columns"):
        fill_signal_history_fallbacks(
            prices,
            {"VIX3M": {"url": "official.csv"}},
            loader=lambda _: pd.DataFrame({"DATE": ["07/20/2026"]}),
        )


def test_cached_signal_columns_are_loaded_but_kept_distinct(tmp_path: Path) -> None:
    cache = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "SPX": [100.0, 101.0],
            "CASH": [90.0, 90.01],
            "VIX": [18.0, 17.0],
            "VIX3M": [20.0, 20.5],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    ).to_csv(cache)
    config = {
        "tickers": {"SPX": "SPY", "CASH": "BIL"},
        "signal_tickers": {"VIX": "^VIX", "VIX3M": "^VIX3M"},
        "start": "2024-01-01",
        "end": None,
        "cache": str(cache),
    }
    loaded = load_prices(config)
    assert list(loaded.columns) == ["SPX", "CASH", "VIX", "VIX3M"]


def test_cached_signal_columns_must_be_complete(tmp_path: Path) -> None:
    cache = tmp_path / "prices.csv"
    pd.DataFrame(
        {"SPX": [100.0], "CASH": [90.0]},
        index=pd.bdate_range("2024-01-01", periods=1),
    ).to_csv(cache)
    config = {
        "tickers": {"SPX": "SPY", "CASH": "BIL"},
        "signal_tickers": {"VIX": "^VIX"},
        "start": "2024-01-01",
        "end": None,
        "cache": str(cache),
    }
    with pytest.raises(ValueError, match="rerun with --refresh"):
        load_prices(config)


@pytest.mark.parametrize(
    ("index", "values", "message"),
    [
        (
            pd.to_datetime(["2024-01-02", "2024-01-02"]),
            [100.0, 101.0],
            "duplicate dates",
        ),
        (
            pd.to_datetime(["2024-01-02", "2024-01-01"]),
            [100.0, 101.0],
            "not sorted",
        ),
        (
            pd.to_datetime(["2024-01-01", "2024-01-02"]),
            [100.0, 0.0],
            "non-positive",
        ),
    ],
)
def test_invalid_cached_prices_fail_closed(
    tmp_path: Path,
    index: pd.DatetimeIndex,
    values: list[float],
    message: str,
) -> None:
    cache = tmp_path / "prices.csv"
    pd.DataFrame({"SPX": values}, index=index).to_csv(cache)
    config = {
        "tickers": {"SPX": "SPY"},
        "start": "2024-01-01",
        "end": None,
        "cache": str(cache),
    }
    with pytest.raises(ValueError, match=message):
        load_prices(config)


def test_cached_price_hash_must_match_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "prices.csv"
    pd.DataFrame(
        {"SPX": [100.0, 101.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    ).to_csv(cache)
    metadata_path = cache.with_suffix(cache.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps({"cache_sha256": hashlib.sha256(b"other").hexdigest()}),
        encoding="utf-8",
    )
    config = {
        "tickers": {"SPX": "SPY"},
        "start": "2024-01-01",
        "end": None,
        "cache": str(cache),
    }
    with pytest.raises(ValueError, match="SHA-256"):
        load_prices(config)


def test_cache_only_signals_cannot_be_refreshed_through_yahoo(tmp_path: Path) -> None:
    cache = tmp_path / "prices.csv"
    pd.DataFrame(
        {"SPX": [100.0], "CASH": [90.0], "VX1": [18.0], "VX2": [20.0]},
        index=pd.bdate_range("2024-01-01", periods=1),
    ).to_csv(cache)
    config = {
        "tickers": {"SPX": "SPY", "CASH": "BIL"},
        "cache_signals": ["VX1", "VX2"],
        "start": "2024-01-01",
        "end": None,
        "cache": str(cache),
    }
    loaded = load_prices(config)
    assert list(loaded.columns) == ["SPX", "CASH", "VX1", "VX2"]
    with pytest.raises(ValueError, match="Cache-only signals"):
        load_prices(config, refresh=True)
