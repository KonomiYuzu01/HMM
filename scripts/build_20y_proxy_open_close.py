from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from build_20y_proxy_prices import synthetic_cash_index


ASSETS = [
    "SPX",
    "QQQ",
    "SEMIS",
    "BOND",
    "GOLD",
    "OIL",
    "USD",
    "CASH",
    "VIX_HEDGE",
]
TICKERS = [
    "SPY",
    "QQQ",
    "SOXX",
    "SMH",
    "IEF",
    "GC=F",
    "GLD",
    "^SPGSCI",
    "DBC",
    "DX-Y.NYB",
    "UUP",
    "^IRX",
    "BIL",
    "VIXY",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build long-proxy Open/Close data with real overnight gaps"
    )
    parser.add_argument(
        "--proxy-close",
        default="data/prices_20y_proxy.csv",
    )
    parser.add_argument(
        "--output",
        default="data/adjusted_open_close_20y_proxy.csv",
    )
    parser.add_argument("--end")
    return parser.parse_args()


def splice_ohlc(
    proxy_open: pd.Series,
    proxy_close: pd.Series,
    actual_open: pd.Series,
    actual_close: pd.Series,
    index: pd.DatetimeIndex,
    actual_not_before: str | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    proxy = pd.DataFrame(
        {
            "open": proxy_open.reindex(index).ffill(limit=3),
            "close": proxy_close.reindex(index).ffill(limit=3),
        }
    )
    actual = pd.DataFrame(
        {
            "open": actual_open.reindex(index).ffill(limit=3),
            "close": actual_close.reindex(index).ffill(limit=3),
        }
    )
    if actual_not_before is not None:
        actual.loc[actual.index < actual_not_before] = np.nan
    first_actual = actual.dropna().first_valid_index()
    if first_actual is None:
        raise ValueError("Actual Open/Close history is unavailable")
    proxy_close_at_start = float(proxy.loc[first_actual, "close"])
    if not np.isfinite(proxy_close_at_start) or proxy_close_at_start <= 0.0:
        raise ValueError("Proxy close is unavailable at the splice date")
    scale = float(actual.loc[first_actual, "close"]) / proxy_close_at_start
    combined = actual.copy()
    before_actual = combined.index < first_actual
    combined.loc[before_actual] = proxy.loc[before_actual] * scale
    return combined, pd.Timestamp(first_actual)


def main() -> None:
    args = parse_args()
    proxy_path = Path(args.proxy_close)
    proxy_close = pd.read_csv(
        proxy_path,
        index_col=0,
        parse_dates=True,
    )[ASSETS].astype(float)
    start = (
        proxy_close.index.min() - pd.Timedelta(days=10)
    ).date().isoformat()
    raw = yf.download(
        TICKERS,
        start=start,
        end=args.end,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo did not return multi-asset Open/Close data")
    open_price = raw["Open"].copy()
    close_price = raw["Close"].copy()
    open_price.index = pd.to_datetime(open_price.index).tz_localize(None)
    close_price.index = pd.to_datetime(close_price.index).tz_localize(None)
    index = proxy_close.index

    semis, smh_start = splice_ohlc(
        open_price["SOXX"],
        close_price["SOXX"],
        open_price["SMH"],
        close_price["SMH"],
        index,
        actual_not_before="2011-12-20",
    )
    gold, gld_start = splice_ohlc(
        open_price["GC=F"],
        close_price["GC=F"],
        open_price["GLD"],
        close_price["GLD"],
        index,
    )
    oil, dbc_start = splice_ohlc(
        open_price["^SPGSCI"],
        close_price["^SPGSCI"],
        open_price["DBC"],
        close_price["DBC"],
        index,
    )
    usd, uup_start = splice_ohlc(
        open_price["DX-Y.NYB"],
        close_price["DX-Y.NYB"],
        open_price["UUP"],
        close_price["UUP"],
        index,
    )
    synthetic_cash = synthetic_cash_index(close_price["^IRX"], index)
    synthetic_cash_open = synthetic_cash.shift(1).fillna(
        synthetic_cash.iloc[0]
    )
    cash, bil_start = splice_ohlc(
        synthetic_cash_open,
        synthetic_cash,
        open_price["BIL"],
        close_price["BIL"],
        index,
    )

    vixy = pd.DataFrame(
        {
            "open": open_price["VIXY"].reindex(index).ffill(limit=3),
            "close": close_price["VIXY"].reindex(index).ffill(limit=3),
        }
    )
    vixy_start = vixy.dropna().first_valid_index()
    if vixy_start is None:
        raise ValueError("VIXY Open/Close history is unavailable")
    vixy.loc[vixy.index < vixy_start] = float(
        vixy.loc[vixy_start, "close"]
    )

    source_ohlc = {
        "SPX": pd.DataFrame(
            {
                "open": open_price["SPY"].reindex(index),
                "close": close_price["SPY"].reindex(index),
            }
        ),
        "QQQ": pd.DataFrame(
            {
                "open": open_price["QQQ"].reindex(index),
                "close": close_price["QQQ"].reindex(index),
            }
        ),
        "SEMIS": semis,
        "BOND": pd.DataFrame(
            {
                "open": open_price["IEF"].reindex(index),
                "close": close_price["IEF"].reindex(index),
            }
        ),
        "GOLD": gold,
        "OIL": oil,
        "USD": usd,
        "CASH": cash,
        "VIX_HEDGE": vixy,
    }
    reconstructed_open = pd.DataFrame(index=index, columns=ASSETS, dtype=float)
    for asset, source in source_ohlc.items():
        source = source.ffill(limit=3)
        overnight_gap = source["open"] / source["close"].shift(1) - 1.0
        reconstructed_open[asset] = (
            proxy_close[asset].shift(1) * (1.0 + overnight_gap)
        )
    reconstructed_open.iloc[0] = proxy_close.iloc[0]

    frame = pd.concat(
        {
            "open": reconstructed_open,
            "close": proxy_close,
        },
        axis=1,
    ).dropna(how="any")
    frame.columns = [
        f"{field}_{asset}" for field, asset in frame.columns
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index_label="date")

    reconstructed_closes = frame[
        [f"close_{asset}" for asset in ASSETS]
    ].copy()
    reconstructed_closes.columns = ASSETS
    close_error = float(
        (
            reconstructed_closes
            - proxy_close.reindex(reconstructed_closes.index)
        )
        .abs()
        .to_numpy()
        .max()
    )
    metadata = {
        "purpose": (
            "Historical execution stress proxy; not a production execution "
            "record."
        ),
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": "auto_adjust=True",
        "proxy_close_source": str(proxy_path),
        "proxy_close_sha256": hashlib.sha256(
            proxy_path.read_bytes()
        ).hexdigest(),
        "first_date": frame.index.min().date().isoformat(),
        "price_as_of": frame.index.max().date().isoformat(),
        "row_count": len(frame),
        "maximum_proxy_close_reconstruction_error": close_error,
        "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "splices": {
            "SEMIS": {
                "proxy": "SOXX",
                "actual": "SMH",
                "actual_start": smh_start.date().isoformat(),
            },
            "GOLD": {
                "proxy": "GC=F",
                "actual": "GLD",
                "actual_start": gld_start.date().isoformat(),
            },
            "OIL": {
                "proxy": "^SPGSCI",
                "actual": "DBC",
                "actual_start": dbc_start.date().isoformat(),
            },
            "USD": {
                "proxy": "DX-Y.NYB",
                "actual": "UUP",
                "actual_start": uup_start.date().isoformat(),
            },
            "CASH": {
                "proxy": "synthetic index from ^IRX yield",
                "actual": "BIL",
                "actual_start": bil_start.date().isoformat(),
            },
            "VIX_HEDGE": {
                "proxy": "flat before inception",
                "actual": "VIXY",
                "actual_start": pd.Timestamp(vixy_start).date().isoformat(),
            },
        },
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Proxy Open/Close cache:", output.resolve())


if __name__ == "__main__":
    main()
