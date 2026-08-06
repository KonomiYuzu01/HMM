from __future__ import annotations

import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests


FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF_FACTORS = f"{FF_BASE}/F-F_Research_Data_Factors_daily_CSV.zip"
FF_INDUSTRY_30 = f"{FF_BASE}/30_Industry_Portfolios_Daily_CSV.zip"
FF_INDUSTRY_49 = f"{FF_BASE}/49_Industry_Portfolios_Daily_CSV.zip"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
OUTPUT = Path("data/prices_pre2008_crash_proxy.csv")
END = pd.Timestamp("2007-12-31")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def parse_french_zip(content: bytes, header_prefix: str) -> pd.DataFrame:
    archive = zipfile.ZipFile(BytesIO(content))
    names = archive.namelist()
    if len(names) != 1:
        raise ValueError("Expected one Fama/French CSV per archive")
    lines = archive.read(names[0]).decode("latin1").splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith(header_prefix)
    )
    end = start + 1
    while end < len(lines):
        token = lines[end].split(",", 1)[0].strip()
        if len(token) != 8 or not token.isdigit():
            break
        end += 1
    frame = pd.read_csv(StringIO("\n".join(lines[start:end])))
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(
        frame["date"].astype(str), format="%Y%m%d"
    )
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce")
    return frame.mask(frame <= -90.0) / 100.0


def parse_fred(content: bytes, series_id: str) -> pd.Series:
    frame = pd.read_csv(BytesIO(content), na_values=".")
    date_column = "observation_date" if "observation_date" in frame else "DATE"
    if series_id not in frame:
        raise ValueError(f"FRED response missing {series_id}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=series_id).dropna()
    series = series[series.index.notna()]
    return series[~series.index.duplicated(keep="last")].sort_index()


def return_index(returns: pd.Series, name: str) -> pd.Series:
    clean = returns.fillna(0.0).astype(float)
    if clean.le(-1.0).any():
        raise ValueError(f"{name} contains a total-loss return")
    return (100.0 * (1.0 + clean).cumprod()).rename(name)


def price_proxy(
    source: pd.Series,
    index: pd.DatetimeIndex,
    *,
    name: str,
) -> pd.Series:
    aligned = source.reindex(index).ffill(limit=5)
    first = aligned.first_valid_index()
    if first is None:
        raise ValueError(f"{name} has no overlapping observations")
    aligned.loc[aligned.index < first] = float(aligned.loc[first])
    aligned = aligned.ffill(limit=5)
    returns = aligned.pct_change(fill_method=None).fillna(0.0)
    return return_index(returns, name)


def approximate_treasury_returns(
    yield_percent: pd.Series,
    risk_free_returns: pd.Series,
    index: pd.DatetimeIndex,
    *,
    duration: float = 7.0,
) -> pd.Series:
    yields = yield_percent.reindex(index).ffill(limit=5)
    available = yields.notna()
    decimal_yield = yields / 100.0
    approximation = decimal_yield / 252.0 - duration * decimal_yield.diff()
    approximation = approximation.clip(lower=-0.12, upper=0.12)
    result = risk_free_returns.reindex(index).fillna(0.0)
    result.loc[available] = approximation.loc[available].fillna(
        decimal_yield.loc[available] / 252.0
    )
    return result.rename("BOND")


def build_prices(raw: dict[str, bytes]) -> pd.DataFrame:
    factors = parse_french_zip(raw["ff_factors"], ",Mkt-RF")
    industry30 = parse_french_zip(raw["ff_industry30"], ",Food")
    industry49 = parse_french_zip(raw["ff_industry49"], ",Agric")
    index = factors.loc[:END].index
    required = pd.concat(
        [
            factors[["Mkt-RF", "RF"]],
            industry30[["BusEq"]],
            industry49[["Chips", "Gold", "Oil"]],
        ],
        axis=1,
        join="inner",
    ).reindex(index)
    if required.drop(columns=["Gold"]).isna().any().any():
        raise ValueError("Fama/French proxy returns are incomplete")

    market_return = required["Mkt-RF"] + required["RF"]
    cash_return = required["RF"]
    dgs10 = parse_fred(raw["fred_dgs10"], "DGS10")
    bond_return = approximate_treasury_returns(dgs10, cash_return, index)
    gold_return = required["Gold"].fillna(0.0)
    gold = return_index(gold_return, "GOLD")
    usd = price_proxy(
        parse_fred(raw["fred_usd"], "DTWEXM"),
        index,
        name="USD",
    )

    vix_frame = pd.read_csv(BytesIO(raw["cboe_vix"]))
    vix_dates = pd.to_datetime(vix_frame["DATE"], errors="coerce")
    vix_values = pd.to_numeric(vix_frame["CLOSE"], errors="coerce")
    official_vix = pd.Series(vix_values.to_numpy(), index=vix_dates).dropna()
    official_vix = official_vix[~official_vix.index.duplicated(keep="last")]
    realized_vix = (
        market_return.rolling(20, min_periods=20).std(ddof=1)
        * np.sqrt(252.0)
        * 100.0
    ).clip(lower=9.0)
    vix = official_vix.reindex(index).ffill(limit=3).combine_first(realized_vix)
    vix = vix.bfill().clip(lower=1.0).rename("VIX")

    prices = pd.DataFrame(
        {
            "SPX": return_index(market_return, "SPX"),
            "QQQ": return_index(required["BusEq"], "QQQ"),
            "SEMIS": return_index(required["Chips"], "SEMIS"),
            "BOND": return_index(bond_return, "BOND"),
            "GOLD": gold,
            "OIL": return_index(required["Oil"], "OIL"),
            "USD": usd,
            "CASH": return_index(cash_return, "CASH"),
            "VIX_HEDGE": 100.0,
            "VIX": vix,
            "VIX3M": vix,
        },
        index=index,
    )
    if prices.isna().any().any() or not np.isfinite(prices).all().all():
        raise ValueError("Pre-2008 proxy contains invalid values")
    if prices.le(0.0).any().any():
        raise ValueError("Pre-2008 proxy contains non-positive values")
    prices.index.name = "date"
    return prices.astype(float)


def main() -> None:
    urls = {
        "ff_factors": FF_FACTORS,
        "ff_industry30": FF_INDUSTRY_30,
        "ff_industry49": FF_INDUSTRY_49,
        "fred_dgs10": FRED.format("DGS10"),
        "fred_usd": FRED.format("DTWEXM"),
        "cboe_vix": CBOE_VIX,
    }
    raw = {name: download(url) for name, url in urls.items()}
    prices = build_prices(raw)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT)
    metadata = {
        "purpose": (
            "Counterfactual pre-2008 crash research only; never production "
            "data or an execution record."
        ),
        "first_date": prices.index.min().date().isoformat(),
        "last_date": prices.index.max().date().isoformat(),
        "rows": len(prices),
        "cache_sha256": sha256(OUTPUT.read_bytes()),
        "sources": {
            name: {"url": url, "raw_sha256": sha256(raw[name])}
            for name, url in urls.items()
        },
        "proxy_map": {
            "SPX": "Fama/French market excess return plus RF",
            "QQQ": "Fama/French 30-industry BusEq value-weight return",
            "SEMIS": "Fama/French 49-industry Chips value-weight return",
            "BOND": (
                "RF before DGS10; then duration-7 total-return approximation "
                "from DGS10"
            ),
            "GOLD": (
                "flat before the Fama/French Gold industry exists in 1963; "
                "then Gold mining equity return, a high-beta proxy rather "
                "than bullion"
            ),
            "OIL": "Fama/French 49-industry Oil equity return; regime input only",
            "USD": "flat before 1973; then FRED major-currency dollar index",
            "CASH": "Fama/French daily RF",
            "VIX": "20-day realized market volatility before Cboe VIX history",
            "VIX3M": "set equal to VIX; term structure deliberately neutral",
            "VIX_HEDGE": "flat; no hypothetical pre-inception hedge payoff",
        },
        "known_limitations": [
            "No historical open prices; crash replay assigns daily proxy returns after the synthetic open.",
            "1929 first-leg crash cannot be tested because the 1008-session HMM warm-up is incomplete.",
            "Pre-inception assets are economic proxies, not tradable ETF returns.",
            "The GOLD sleeve uses gold-mining equities after 1963 and can behave very differently from bullion.",
        ],
    }
    metadata_path = OUTPUT.with_suffix(OUTPUT.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
