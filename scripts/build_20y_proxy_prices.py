from __future__ import annotations

import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


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

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX3M_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clearly labeled long-history proxy cache."
    )
    parser.add_argument("--start", default="2002-07-01")
    parser.add_argument("--end")
    parser.add_argument("--output", default="data/prices_20y_proxy.csv")
    parser.add_argument(
        "--completion-reference",
        default="data/adjusted_open_close_2011_present.csv",
        help=(
            "Clip the proxy tail to the latest date with a complete "
            "production open/close row."
        ),
    )
    parser.add_argument(
        "--metadata",
        default="output/past_20y_drawdown_backtest/proxy_metadata.json",
    )
    return parser.parse_args()


def splice_history(
    proxy: pd.Series,
    actual: pd.Series,
    index: pd.DatetimeIndex,
    actual_not_before: str | None = None,
) -> tuple[pd.Series, pd.Timestamp]:
    proxy_aligned = proxy.reindex(index).ffill(limit=3)
    actual_aligned = actual.reindex(index).ffill(limit=3)
    if actual_not_before is not None:
        actual_aligned.loc[actual_aligned.index < actual_not_before] = np.nan
    first_actual = actual_aligned.first_valid_index()
    if first_actual is None:
        raise ValueError(f"Actual series {actual.name} has no valid prices")
    proxy_at_start = proxy_aligned.loc[first_actual]
    if pd.isna(proxy_at_start) or float(proxy_at_start) <= 0.0:
        raise ValueError(f"Proxy series {proxy.name} is unavailable at splice date")
    scaled_proxy = proxy_aligned * (
        float(actual_aligned.loc[first_actual]) / float(proxy_at_start)
    )
    combined = actual_aligned.copy()
    combined.loc[combined.index < first_actual] = scaled_proxy.loc[
        scaled_proxy.index < first_actual
    ]
    return combined, first_actual


def synthetic_cash_index(
    annualized_yield_percent: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    annual_yield = (
        annualized_yield_percent.reindex(index).ffill(limit=5).clip(lower=0.0)
        / 100.0
    )
    daily_return = np.power(1.0 + annual_yield, 1.0 / 252.0) - 1.0
    daily_return = daily_return.fillna(0.0)
    return (100.0 * (1.0 + daily_return).cumprod()).rename("SYNTHETIC_CASH")


def download_cboe_close(url: str) -> pd.Series:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    history = pd.read_csv(StringIO(response.text))
    missing = sorted({"DATE", "CLOSE"}.difference(history))
    if missing:
        raise ValueError(f"Cboe history is missing columns: {missing}")
    dates = pd.to_datetime(history["DATE"], errors="coerce")
    values = pd.to_numeric(history["CLOSE"], errors="coerce")
    close = pd.Series(values.to_numpy(), index=dates, name="CLOSE").dropna()
    close = close[close.index.notna()]
    return close[~close.index.duplicated(keep="last")].sort_index()


def clip_to_completion_reference(
    prices: pd.DataFrame,
    completion_reference: Path,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not completion_reference.exists():
        raise FileNotFoundError(
            f"Missing completion reference: {completion_reference}"
        )
    completion_dates = pd.read_csv(
        completion_reference,
        index_col=0,
        parse_dates=True,
        usecols=[0],
    ).index
    completion_as_of = completion_dates.max()
    return (
        prices.loc[prices.index <= completion_as_of].copy(),
        pd.Timestamp(completion_as_of),
    )


def main() -> None:
    args = parse_args()
    raw = yf.download(
        TICKERS,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo did not return multi-asset adjusted prices")
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    market_index = close["SPY"].dropna().index

    semis, smh_start = splice_history(
        close["SOXX"],
        close["SMH"],
        market_index,
        actual_not_before="2011-12-20",
    )
    gold, gld_start = splice_history(close["GC=F"], close["GLD"], market_index)
    oil, dbc_start = splice_history(
        close["^SPGSCI"], close["DBC"], market_index
    )
    usd, uup_start = splice_history(close["DX-Y.NYB"], close["UUP"], market_index)
    cash_proxy = synthetic_cash_index(close["^IRX"], market_index)
    cash, bil_start = splice_history(cash_proxy, close["BIL"], market_index)

    vix_hedge = close["VIXY"].reindex(market_index).ffill(limit=3)
    vixy_start = vix_hedge.first_valid_index()
    if vixy_start is None:
        raise ValueError("VIXY has no valid prices")
    vix_hedge.loc[vix_hedge.index < vixy_start] = float(
        vix_hedge.loc[vixy_start]
    )

    vix_source = download_cboe_close(VIX_URL)
    vix3m_source = download_cboe_close(VIX3M_URL)
    vix = vix_source.reindex(market_index).ffill(limit=3)
    vix3m = vix3m_source.reindex(market_index).ffill(limit=3)
    vix3m_start = vix3m_source.first_valid_index()
    if vix3m_start is None:
        raise ValueError("VIX3M has no valid prices")
    vix3m.loc[vix3m.index < vix3m_start] = vix.loc[
        vix3m.index < vix3m_start
    ]

    prices = pd.DataFrame(
        {
            "SPX": close["SPY"].reindex(market_index),
            "QQQ": close["QQQ"].reindex(market_index),
            "SEMIS": semis,
            "BOND": close["IEF"].reindex(market_index),
            "GOLD": gold,
            "OIL": oil,
            "USD": usd,
            "CASH": cash,
            "VIX_HEDGE": vix_hedge,
            "VIX": vix,
            "VIX3M": vix3m,
        },
        index=market_index,
    ).ffill(limit=3)
    prices = prices.dropna(how="any").astype(float)
    completion_reference = Path(args.completion_reference)
    prices, completion_as_of = clip_to_completion_reference(
        prices,
        completion_reference,
    )
    if prices.empty:
        raise RuntimeError("No common proxy-price history is available")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output, index_label="date")
    cache_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    metadata = {
        "purpose": "Historical stress proxy only; not a production execution record.",
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": "auto_adjust=True",
        "official_signal_sources": {
            "VIX": VIX_URL,
            "VIX3M": VIX3M_URL,
        },
        "start": prices.index.min().date().isoformat(),
        "end": prices.index.max().date().isoformat(),
        "rows": len(prices),
        "cache_sha256": cache_sha256,
        "completion_reference": str(completion_reference),
        "completion_reference_as_of": (
            completion_as_of.date().isoformat()
        ),
        "completion_reference_sha256": hashlib.sha256(
            completion_reference.read_bytes()
        ).hexdigest(),
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
                "proxy": "flat before inception; no hypothetical hedge payoff",
                "actual": "VIXY",
                "actual_start": vixy_start.date().isoformat(),
            },
            "VIX3M": {
                "proxy": "VIX before index history; neutral term ratio",
                "actual": "Cboe VIX3M official daily history",
                "actual_start": vix3m_start.date().isoformat(),
            },
        },
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        metadata_text,
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"Proxy cache: {output.resolve()}")


if __name__ == "__main__":
    main()
