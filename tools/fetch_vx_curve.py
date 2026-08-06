from __future__ import annotations

import argparse
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import certifi

from regime_strategy.vx import build_front_month_curve, monthly_vix_expiry


CBOE_URL = (
    "https://cdn.cboe.com/data/us/futures/market_statistics/"
    "historical_data/VX/VX_{expiry}.csv"
)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def contract_candidates(year: int, month: int) -> list[date]:
    theoretical = monthly_vix_expiry(year, month)
    return [
        theoretical,
        theoretical - timedelta(days=1),
        theoretical + timedelta(days=1),
    ]


def fetch_contract(
    year: int,
    month: int,
    raw_directory: Path,
    timeout: float,
) -> tuple[pd.Timestamp, pd.DataFrame] | None:
    raw_directory.mkdir(parents=True, exist_ok=True)
    for expiry in contract_candidates(year, month):
        cache = raw_directory / f"VX_{expiry.isoformat()}.csv"
        if cache.exists():
            payload = cache.read_bytes()
        else:
            request = Request(
                CBOE_URL.format(expiry=expiry.isoformat()),
                headers={"User-Agent": "Mozilla/5.0 (compatible; causal-vx-research/1.0)"},
            )
            try:
                with urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
                    payload = response.read()
            except HTTPError as error:
                if error.code in {403, 404}:
                    continue
                raise
            except URLError as error:
                raise RuntimeError(
                    f"Cboe download failed for {expiry.isoformat()}: {error.reason}"
                ) from error
            if not payload.startswith(b"Trade Date,"):
                continue
            cache.write_bytes(payload)
        frame = pd.read_csv(BytesIO(payload))
        if {"Trade Date", "Settle"}.issubset(frame.columns):
            return pd.Timestamp(expiry), frame
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a causal front/second-month VX settlement curve from Cboe CSVs."
    )
    parser.add_argument("--base-cache", default="data/prices_high_cagr.csv")
    parser.add_argument("--output", default="data/prices_vx_curve.csv")
    parser.add_argument("--raw-dir", default="data/vx_contracts")
    parser.add_argument("--first-year", type=int, default=2014)
    parser.add_argument("--last-year", type=int, default=date.today().year + 1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_directory = Path(args.raw_dir)
    contracts = [
        (year, month)
        for year in range(args.first_year, args.last_year + 1)
        for month in range(1, 13)
    ]
    records: list[pd.DataFrame] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_contract,
                year,
                month,
                raw_directory,
                args.timeout,
            ): (year, month)
            for year, month in contracts
        }
        for future in as_completed(futures):
            year, month = futures[future]
            result = future.result()
            if result is None:
                failures.append(f"{year:04d}-{month:02d}")
                continue
            expiry, frame = result
            contract = frame.copy()
            contract["Expiry"] = expiry
            records.append(contract)

    if not records:
        raise RuntimeError("No Cboe VX contract histories were downloaded")
    curve = build_front_month_curve(pd.concat(records, ignore_index=True))
    base = pd.read_csv(args.base_cache, index_col=0, parse_dates=True).sort_index()
    aligned = curve[["VX1", "VX2"]].reindex(base.index).ffill(limit=3)
    if aligned.loc[aligned.index >= "2015-01-01"].dropna().empty:
        raise RuntimeError("The VX curve does not cover the required out-of-sample period")
    combined = base.join(aligned)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index_label="date")
    covered = aligned.dropna().index
    print(
        f"saved={output} contracts={len(records)} failures={len(failures)} "
        f"curve_start={covered.min().date()} curve_end={covered.max().date()}"
    )
    if failures:
        print("missing_contract_months=" + ",".join(sorted(failures)))


if __name__ == "__main__":
    main()
