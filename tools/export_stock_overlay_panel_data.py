from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


STOCK_PATH = Path("data/individual_stock_overlay_prices.csv")
MARKET_PATH = Path("data/prices_vix_hedge.csv")
DESTINATION = Path("strategy-panel/app/stock-risk-data.ts")
STATE_PATH = Path("output/hierarchical_regime_portfolios/daily_states.csv")
WINDOWS = (60, 252)
RELATIVE_STRENGTH_WINDOWS = (63, 126)
SEMIS = {
    "NVDA",
    "AMD",
    "AVGO",
    "QCOM",
    "TXN",
    "AMAT",
    "LRCX",
    "KLAC",
    "MU",
    "INTC",
    "MCHP",
    "ADI",
    "MRVL",
    "ON",
    "TSM",
    "ASML",
}


def require_aligned_latest_date(
    states: pd.Series,
    prices: pd.DataFrame,
) -> None:
    if states.index[-1] != prices.index[-1]:
        raise ValueError(
            "Stock overlay market state is stale: "
            f"prices end {prices.index[-1].date()}, "
            f"states end {states.index[-1].date()}"
        )


def rounded_matrix(frame: pd.DataFrame, window: int) -> list[list[float]]:
    covariance = LedoitWolf().fit(
        frame.iloc[-window:].to_numpy(dtype=float)
    ).covariance_
    return [
        [round(float(value), 12) for value in row]
        for row in covariance
    ]


def main() -> None:
    stocks = pd.read_csv(STOCK_PATH, index_col=0, parse_dates=True)
    market = pd.read_csv(MARKET_PATH, index_col=0, parse_dates=True)[
        ["QQQ", "SEMIS"]
    ].rename(columns={"SEMIS": "SMH"})
    prices = pd.concat([stocks, market], axis=1, join="inner").dropna()
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < max(WINDOWS):
        raise ValueError("Insufficient common history for Panel risk data")
    assets = [*stocks.columns, "QQQ", "SMH"]
    returns = returns[assets]
    log_returns = np.log1p(returns)
    active_log_returns = pd.DataFrame(index=returns.index)
    benchmark_by_stock = {
        ticker: ("SMH" if ticker in SEMIS else "QQQ")
        for ticker in stocks.columns
    }
    for ticker, benchmark in benchmark_by_stock.items():
        active_log_returns[ticker] = (
            log_returns[ticker] - log_returns[benchmark]
        )
    trailing_active = active_log_returns.rolling(
        RELATIVE_STRENGTH_WINDOWS[0],
        min_periods=RELATIVE_STRENGTH_WINDOWS[0],
    ).sum()
    active_dispersion = trailing_active.std(axis=1)
    active_dispersion_percentile = active_dispersion.rolling(
        756,
        min_periods=252,
    ).rank(pct=True)
    states = pd.read_csv(
        STATE_PATH,
        index_col=0,
        parse_dates=True,
    )["state"]
    require_aligned_latest_date(states, prices)
    current_state = str(
        states.reindex(states.index.union([prices.index[-1]]))
        .sort_index()
        .ffill()
        .loc[prices.index[-1]]
    )
    payload = {
        "asOf": prices.index[-1].date().isoformat(),
        "annualization": 252,
        "assets": assets,
        "supportedStocks": list(stocks.columns),
        "latestPrices": {
            ticker: round(float(prices.iloc[-1][ticker]), 6)
            for ticker in stocks.columns
        },
        "benchmarkByStock": benchmark_by_stock,
        "activeLogReturn63": {
            ticker: round(float(active_log_returns[ticker].iloc[-63:].sum()), 8)
            for ticker in stocks.columns
        },
        "activeLogReturn126": {
            ticker: round(float(active_log_returns[ticker].iloc[-126:].sum()), 8)
            for ticker in stocks.columns
        },
        "activeDispersionPercentile": round(
            float(active_dispersion_percentile.iloc[-1]),
            8,
        ),
        "activeDispersionMaximumPercentile": 0.67,
        "marketState": current_state,
        "marketStateAsOf": states.index[-1].date().isoformat(),
        "stockFriendlyMarketStates": ["quiet_bull", "normal_bull"],
        "covariance60": rounded_matrix(returns, 60),
        "covariance252": rounded_matrix(returns, 252),
        "source": {
            "stockPriceCacheSha256": hashlib.sha256(
                STOCK_PATH.read_bytes()
            ).hexdigest(),
            "marketPriceCacheSha256": hashlib.sha256(
                MARKET_PATH.read_bytes()
            ).hexdigest(),
            "commonObservations": len(returns),
            "firstCommonDate": returns.index[0].date().isoformat(),
            "lastCommonDate": returns.index[-1].date().isoformat(),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    DESTINATION.write_text(
        "export const stockRiskData = "
        + serialized
        + " as const;\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "as_of": payload["asOf"],
                "assets": len(assets),
                "common_observations": len(returns),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
