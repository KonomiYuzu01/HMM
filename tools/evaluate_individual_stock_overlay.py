from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from regime_strategy.data import completed_us_daily_prices, validate_price_frame
from regime_strategy.report import performance_metrics
from regime_strategy.stock_overlay import (
    ActiveRiskEstimate,
    BasketRiskEstimate,
    ReplacementAllocation,
    estimate_active_risk,
    estimate_basket_risk,
    regime_replacement_share,
)


PRODUCTION = Path(
    "output/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
PRICE_PATH = Path("data/prices_vix_hedge.csv")
STATE_PATH = Path("output/hierarchical_regime_portfolios/daily_states.csv")
STOCK_PRICE_PATH = Path("data/individual_stock_overlay_prices.csv")
DESTINATION = Path("output/individual_stock_overlay_validation")
START = "2014-01-01"
END = (
    pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
).date().isoformat()
EVALUATION_START = "2015-01-01"
EVALUATION_END = "2025-12-31"
DEFAULT_COST_RATE = 0.0015
ALPHA_SCENARIOS = (0.00, 0.03, 0.06, 0.10, 0.15, 0.20)

SEMIS = (
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
)
GROWTH = (
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "CRM",
    "ADBE",
)
ALL_STOCKS = (*SEMIS, *GROWTH)
RiskCache = dict[
    tuple[int, str, tuple[str, ...]],
    BasketRiskEstimate,
]
ActiveRiskCache = dict[
    tuple[int, str, tuple[str, ...]],
    ActiveRiskEstimate,
]
HistoryCache = dict[
    tuple[int, str, tuple[str, ...]],
    tuple[pd.DataFrame, pd.Series],
]
ActiveSignalCache = dict[
    tuple[int, str, tuple[str, ...], tuple[int, ...]],
    tuple[dict[int, float], float],
]

FIXED_BASKETS = {
    "single_leaders": {
        "SEMIS": ("NVDA",),
        "QQQ": ("MSFT",),
    },
    "single_stress": {
        "SEMIS": ("INTC",),
        "QQQ": ("CRM",),
    },
    "three_compute_platform": {
        "SEMIS": ("NVDA", "AMD", "AVGO"),
        "QQQ": ("MSFT", "AMZN", "GOOGL"),
    },
    "three_legacy_cyclical": {
        "SEMIS": ("INTC", "MU", "QCOM"),
        "QQQ": ("CRM", "ADBE", "NFLX"),
    },
    "five_diversified": {
        "SEMIS": ("NVDA", "AVGO", "TXN", "AMAT", "MU"),
        "QQQ": ("MSFT", "AAPL", "AMZN", "GOOGL", "META"),
    },
    "eight_broad": {
        "SEMIS": ("NVDA", "AMD", "AVGO", "QCOM", "TXN", "AMAT", "LRCX", "KLAC"),
        "QQQ": GROWTH,
    },
}

POLICIES = {
    "naive_always": {
        "method": "naive",
        "regime_conditioned": False,
        "stock_trend_filter": False,
    },
    "standalone_always": {
        "method": "standalone",
        "regime_conditioned": False,
        "stock_trend_filter": False,
    },
    "joint_always": {
        "method": "joint",
        "regime_conditioned": False,
        "stock_trend_filter": False,
    },
    "joint_regime": {
        "method": "joint",
        "regime_conditioned": True,
        "stock_trend_filter": False,
    },
    "joint_regime_trend": {
        "method": "joint",
        "regime_conditioned": True,
        "stock_trend_filter": True,
    },
    "active_always_te4": {
        "method": "active",
        "regime_conditioned": False,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.04,
    },
    "active_regime_te3": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
    },
    "active_regime_te4": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.04,
    },
    "active_regime_te5": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.05,
    },
    "active_regime_trend_te4": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": True,
        "tracking_error_budget": 0.04,
    },
    "active_regime_te2": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.02,
    },
    "active_regime_te2_5": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.025,
    },
    "active_regime_te3_5": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.035,
    },
    "active_regime_te3_tight_caps": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
    },
    "active_regime_te3_loose_caps": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.10,
        "maximum_basket_weight": 0.30,
    },
    "active_regime_te3_gate20": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "confirmed_growth_threshold": 0.20,
    },
    "active_regime_te3_gate50": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "confirmed_growth_threshold": 0.50,
    },
    "active_regime_te3_share75": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "replacement_share_scale": 0.75,
    },
    "active_regime_te3_share125": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "replacement_share_scale": 1.25,
    },
    "active_regime_te3_biweekly": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "rebalance_period": "2W-FRI",
    },
    "active_regime_te3_monthly": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "rebalance_period": "M",
    },
    "active_regime_te3_monthly_tight_caps": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
    },
    "active_regime_te3_fast_exit_monthly": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 0.20,
    },
    "active_regime_te3_fast_exit_monthly_tight_caps": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 0.20,
    },
    "active_regime_te3_zero_exit_monthly_tight_caps": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
    },
    "active_regime_stable_only": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
    },
    "active_regime_quiet_focus": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.25,
        },
    },
    "active_regime_stable_rs63": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (63,),
    },
    "active_regime_stable_rs126": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (126,),
    },
    "active_regime_stable_rs252": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (252,),
    },
    "active_regime_stable_rs63_126": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (63, 126),
    },
    "active_regime_stable_rs63_126_disp67": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (63, 126),
        "maximum_active_dispersion_percentile": 0.67,
    },
    "active_regime_stable_rs63_126_disp50": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (63, 126),
        "maximum_active_dispersion_percentile": 0.50,
    },
    "active_regime_stable_rs63_126_disp33": {
        "method": "active",
        "regime_conditioned": True,
        "stock_trend_filter": False,
        "tracking_error_budget": 0.03,
        "maximum_name_weight": 0.05,
        "maximum_basket_weight": 0.20,
        "rebalance_period": "M",
        "daily_risk_exit": True,
        "exit_budget_drop_threshold": 1.0,
        "replacement_shares": {
            "quiet_bull": 0.50,
            "normal_bull": 0.50,
        },
        "relative_strength_windows": (63, 126),
        "maximum_active_dispersion_percentile": 0.33,
    },
}

for _label, _threshold in (
    ("disp60", 0.60),
    ("disp70", 0.70),
    ("disp75", 0.75),
    ("disp80", 0.80),
):
    _policy = POLICIES["active_regime_stable_rs63_126_disp67"].copy()
    _policy["maximum_active_dispersion_percentile"] = _threshold
    POLICIES[f"active_regime_stable_rs63_126_{_label}"] = _policy

EVENTS = {
    "2015_growth_selloff": ("2015-08-17", "2015-09-30"),
    "2018_q4_selloff": ("2018-10-01", "2018-12-24"),
    "2020_covid_drawdown": ("2020-02-19", "2020-03-23"),
    "2020_rebound": ("2020-03-24", "2020-08-31"),
    "2022_growth_bear": ("2022-01-03", "2022-10-14"),
}


def cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def load_stock_prices(refresh: bool) -> pd.DataFrame:
    metadata_path = cache_metadata_path(STOCK_PRICE_PATH)
    if STOCK_PRICE_PATH.exists() and not refresh:
        prices = pd.read_csv(
            STOCK_PRICE_PATH,
            index_col=0,
            parse_dates=True,
        ).astype(float)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_sha256 = hashlib.sha256(STOCK_PRICE_PATH.read_bytes()).hexdigest()
        if actual_sha256 != metadata["cache_sha256"]:
            raise ValueError("Individual-stock cache fingerprint does not match")
        validate_price_frame(prices, source=str(STOCK_PRICE_PATH))
        return prices

    raw = yf.download(
        list(ALL_STOCKS),
        start=START,
        end=END,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no individual-stock prices")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = close.reindex(columns=list(ALL_STOCKS)).copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    missing_by_ticker = prices.isna().sum().astype(int)
    first_valid = {
        ticker: (
            prices[ticker].first_valid_index().date().isoformat()
            if prices[ticker].first_valid_index() is not None
            else None
        )
        for ticker in prices
    }
    prices = prices.sort_index().ffill(limit=1).dropna(how="any")
    prices = completed_us_daily_prices(prices.astype(float))
    validate_price_frame(prices, source="Downloaded individual-stock prices")
    STOCK_PRICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(STOCK_PRICE_PATH, index_label="date")
    cache_sha256 = hashlib.sha256(STOCK_PRICE_PATH.read_bytes()).hexdigest()
    metadata = {
        "purpose": (
            "Risk calibration and counterfactual overlay stress tests only; "
            "not a point-in-time stock-selection universe"
        ),
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": "auto_adjust=True",
        "requested_start": START,
        "requested_end_exclusive": END,
        "first_date": prices.index[0].date().isoformat(),
        "price_as_of": prices.index[-1].date().isoformat(),
        "row_count": len(prices),
        "tickers": list(prices.columns),
        "missing_observations_before_common_window": missing_by_ticker.to_dict(),
        "first_valid_date_by_ticker": first_valid,
        "cache_sha256": cache_sha256,
        "known_limitations": [
            "The panel is selected with current knowledge and has survivorship bias.",
            "Delisted securities are absent.",
            "Results may calibrate risk but cannot validate a stock-selection alpha.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prices


def load_inputs(
    stock_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    base = pd.read_csv(
        PRODUCTION / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    weights = pd.read_csv(
        PRODUCTION / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    market_prices = pd.read_csv(
        PRICE_PATH,
        index_col=0,
        parse_dates=True,
    )
    states = pd.read_csv(
        STATE_PATH,
        index_col=0,
        parse_dates=True,
    )["state"]

    stock_returns = stock_prices.pct_change(fill_method=None)
    market_returns = market_prices[["QQQ", "SEMIS", "CASH"]].pct_change(
        fill_method=None
    )
    common_index = (
        base.loc[EVALUATION_START:EVALUATION_END]
        .index.intersection(weights.index)
        .intersection(market_returns.index)
        .intersection(stock_returns.index)
    )
    return (
        base.reindex(common_index),
        weights.reindex(common_index),
        pd.concat(
            [
                market_returns.reindex(common_index),
                stock_returns.reindex(common_index),
            ],
            axis=1,
        ),
        states.reindex(common_index).ffill(),
    )


def rebalance_dates(
    index: pd.DatetimeIndex,
    period: str = "W-FRI",
) -> pd.DatetimeIndex:
    if period == "2W-FRI":
        return rebalance_dates(index, "W-FRI")[::2]
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(
        series.groupby(index.to_period(period)).max().to_numpy()
    )


def naive_allocation(
    risk_budget: float,
    history: pd.DataFrame,
    benchmark: pd.Series,
    risk: BasketRiskEstimate,
) -> ReplacementAllocation:
    mix = pd.Series(1.0 / history.shape[1], index=history.columns)
    actual = min(risk_budget, 0.25, 0.075 / float(mix.max()))
    return ReplacementAllocation(
        stock_weights=mix * actual,
        risk_budget=risk_budget,
        actual_stock_weight=actual,
        cash_reserve_weight=max(0.0, risk_budget - actual),
        risk=risk,
    )


def cached_risk_estimate(
    cache: RiskCache,
    date: pd.Timestamp,
    benchmark_name: str,
    history: pd.DataFrame,
    benchmark: pd.Series,
) -> BasketRiskEstimate:
    tickers = tuple(history.columns)
    key = (date.value, benchmark_name, tickers)
    if key not in cache:
        cache[key] = estimate_basket_risk(history, benchmark)
    return cache[key]


def cached_active_risk_estimate(
    cache: ActiveRiskCache,
    date: pd.Timestamp,
    benchmark_name: str,
    history: pd.DataFrame,
    benchmark: pd.Series,
) -> ActiveRiskEstimate:
    tickers = tuple(history.columns)
    key = (date.value, benchmark_name, tickers)
    if key not in cache:
        cache[key] = estimate_active_risk(history, benchmark)
    return cache[key]


def cached_joint_allocation(
    risk_budget: float,
    history: pd.DataFrame,
    risk: BasketRiskEstimate,
) -> ReplacementAllocation:
    mix = pd.Series(1.0 / history.shape[1], index=history.columns)
    actual = min(
        risk_budget / risk.risk_multiple,
        0.25,
        0.075 / float(mix.max()),
    )
    return ReplacementAllocation(
        stock_weights=mix * actual,
        risk_budget=risk_budget,
        actual_stock_weight=actual,
        cash_reserve_weight=max(0.0, risk_budget - actual),
        risk=risk,
    )


def cached_standalone_allocation(
    risk_budget: float,
    history: pd.DataFrame,
    benchmark: pd.Series,
    date: pd.Timestamp,
    benchmark_name: str,
    cache: RiskCache,
) -> ReplacementAllocation:
    mix = pd.Series(1.0 / history.shape[1], index=history.columns)
    risks = {
        ticker: cached_risk_estimate(
            cache,
            date,
            benchmark_name,
            history[[ticker]],
            benchmark,
        )
        for ticker in history
    }
    weights = pd.Series(
        {
            ticker: min(
                risk_budget * float(mix[ticker])
                / risks[ticker].risk_multiple,
                0.075,
            )
            for ticker in history
        },
        dtype=float,
    )
    if float(weights.sum()) > 0.25:
        weights *= 0.25 / float(weights.sum())
    conservative = BasketRiskEstimate(
        risk_multiple=max(risk.risk_multiple for risk in risks.values()),
        long_volatility_ratio=max(
            risk.long_volatility_ratio for risk in risks.values()
        ),
        short_volatility_ratio=max(
            risk.short_volatility_ratio for risk in risks.values()
        ),
        tail_loss_ratio=max(
            risk.tail_loss_ratio for risk in risks.values()
        ),
        beta=max(risk.beta for risk in risks.values()),
        observations=min(risk.observations for risk in risks.values()),
    )
    return ReplacementAllocation(
        stock_weights=weights,
        risk_budget=risk_budget,
        actual_stock_weight=float(weights.sum()),
        cash_reserve_weight=max(0.0, risk_budget - float(weights.sum())),
        risk=conservative,
    )


def active_tickers(
    tickers: tuple[str, ...],
    stock_prices: pd.DataFrame,
    date: pd.Timestamp,
    trend_filter: bool,
) -> tuple[str, ...]:
    if not trend_filter:
        return tickers
    history = stock_prices.loc[:date, list(tickers)]
    if len(history) < 200:
        return ()
    latest = history.iloc[-1]
    average = history.iloc[-200:].mean()
    return tuple(ticker for ticker in tickers if latest[ticker] > average[ticker])


def active_dispersion_percentile(all_returns: pd.DataFrame) -> pd.Series:
    """Measure stock-versus-ETF return dispersion using only trailing data."""

    log_returns = np.log1p(all_returns)
    active_returns = pd.DataFrame(index=all_returns.index)
    for ticker in SEMIS:
        active_returns[ticker] = (
            log_returns[ticker] - log_returns["SEMIS"]
        )
    for ticker in GROWTH:
        active_returns[ticker] = (
            log_returns[ticker] - log_returns["QQQ"]
        )
    trailing_active_returns = active_returns.rolling(
        63,
        min_periods=63,
    ).sum()
    dispersion = trailing_active_returns.std(axis=1)
    return dispersion.rolling(756, min_periods=252).rank(pct=True)


def active_dispersion_percentiles_by_benchmark(
    all_returns: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Measure dispersion separately inside the QQQ and semiconductor groups."""

    log_returns = np.log1p(all_returns)
    result: dict[str, pd.Series] = {}
    for benchmark, tickers in (("QQQ", GROWTH), ("SEMIS", SEMIS)):
        active_returns = log_returns[list(tickers)].sub(
            log_returns[benchmark],
            axis=0,
        )
        trailing_active_returns = active_returns.rolling(
            63,
            min_periods=63,
        ).sum()
        dispersion = trailing_active_returns.std(axis=1)
        result[benchmark] = dispersion.rolling(
            756,
            min_periods=252,
        ).rank(pct=True)
    return result


def active_signal_statistics(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    windows: tuple[int, ...],
) -> tuple[dict[int, float], float]:
    """Return basket active log returns and the weakest per-stock breadth."""

    aligned = pd.concat(
        [stock_returns, benchmark_returns.rename("__benchmark__")],
        axis=1,
        join="inner",
    ).dropna()
    if not windows or len(aligned) < max(windows):
        return {}, 0.0
    log_returns = np.log1p(aligned)
    individual_active = log_returns[stock_returns.columns].sub(
        log_returns["__benchmark__"],
        axis=0,
    )
    basket_returns = aligned[stock_returns.columns].mean(axis=1)
    basket_active = np.log1p(basket_returns) - log_returns["__benchmark__"]
    active_log_returns = {
        window: float(basket_active.iloc[-window:].sum())
        for window in windows
    }
    weakest_breadth = min(
        float(
            (
                individual_active.iloc[-window:].sum(axis=0)
                > 0.0
            ).mean()
        )
        for window in windows
    )
    return active_log_returns, weakest_breadth


def simulate_overlay(
    name: str,
    baskets: dict[str, tuple[str, ...]],
    policy_name: str,
    base: pd.DataFrame,
    production_weights: pd.DataFrame,
    all_returns: pd.DataFrame,
    states: pd.Series,
    stock_prices: pd.DataFrame,
    risk_cache: RiskCache,
    active_risk_cache: ActiveRiskCache,
    *,
    cost_rate: float = DEFAULT_COST_RATE,
    precomputed_dispersion_percentile: pd.Series | None = None,
    precomputed_benchmark_dispersion_percentiles: (
        dict[str, pd.Series] | None
    ) = None,
    history_cache: HistoryCache | None = None,
    active_signal_cache: ActiveSignalCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = POLICIES[policy_name]
    index = base.index
    delta_columns = [*ALL_STOCKS, "QQQ", "SEMIS", "CASH"]
    target_deltas = pd.DataFrame(
        np.nan,
        index=index,
        columns=delta_columns,
    )
    diagnostic_rows: list[dict[str, object]] = []
    known_weights = production_weights
    known_states = states
    sizing_dates = rebalance_dates(
        index,
        str(policy.get("rebalance_period", "W-FRI")),
    )
    sizing_date_set = set(sizing_dates)
    decision_dates = (
        index
        if bool(policy.get("daily_risk_exit", False))
        else sizing_dates
    )
    previous_target = pd.Series(0.0, index=delta_columns)
    previous_desired_budgets = {"QQQ": 0.0, "SEMIS": 0.0}
    use_benchmark_dispersion = bool(
        policy.get("benchmark_specific_dispersion", False)
    )
    dispersion_percentile = (
        (
            precomputed_dispersion_percentile
            if precomputed_dispersion_percentile is not None
            else active_dispersion_percentile(all_returns)
        )
        if "maximum_active_dispersion_percentile" in policy
        and not use_benchmark_dispersion
        else None
    )
    benchmark_dispersion_percentiles = (
        (
            precomputed_benchmark_dispersion_percentiles
            if precomputed_benchmark_dispersion_percentiles is not None
            else active_dispersion_percentiles_by_benchmark(all_returns)
        )
        if "maximum_active_dispersion_percentile" in policy
        and use_benchmark_dispersion
        else None
    )

    for date in decision_dates:
        growth_exposure = float(
            known_weights.loc[date, ["QQQ", "SEMIS"]].sum()
        )
        state = str(known_states.loc[date])
        replacement_share = (
            regime_replacement_share(
                state,
                growth_exposure,
                confirmed_growth_threshold=float(
                    policy.get("confirmed_growth_threshold", 0.35)
                ),
                replacement_shares=policy.get("replacement_shares"),
            )
            if bool(policy["regime_conditioned"])
            else 0.50
        )
        replacement_share *= float(
            policy.get("replacement_share_scale", 1.0)
        )
        current_dispersion_percentile = (
            float(dispersion_percentile.loc[date])
            if dispersion_percentile is not None
            and pd.notna(dispersion_percentile.loc[date])
            else np.nan
        )
        maximum_dispersion_percentile = float(
            policy.get("maximum_active_dispersion_percentile", 1.0)
        )
        dispersion_scale = 1.0
        if (
            pd.notna(current_dispersion_percentile)
            and current_dispersion_percentile
            > maximum_dispersion_percentile
        ):
            replacement_share = 0.0
            dispersion_scale = 0.0
        elif (
            pd.notna(current_dispersion_percentile)
            and "active_dispersion_soft_start_percentile" in policy
        ):
            soft_start = float(
                policy["active_dispersion_soft_start_percentile"]
            )
            if not 0.0 <= soft_start < maximum_dispersion_percentile:
                raise ValueError(
                    "Dispersion soft start must be below the hard maximum"
                )
            if current_dispersion_percentile > soft_start:
                dispersion_scale = (
                    maximum_dispersion_percentile
                    - current_dispersion_percentile
                ) / (maximum_dispersion_percentile - soft_start)
        current_benchmark_dispersion = {
            benchmark: (
                float(benchmark_dispersion_percentiles[benchmark].loc[date])
                if benchmark_dispersion_percentiles is not None
                and pd.notna(
                    benchmark_dispersion_percentiles[benchmark].loc[date]
                )
                else np.nan
            )
            for benchmark in ("QQQ", "SEMIS")
        }
        dispersion_scales = {
            "QQQ": dispersion_scale,
            "SEMIS": dispersion_scale,
        }
        benchmark_replacement_shares = {
            "QQQ": replacement_share,
            "SEMIS": replacement_share,
        }
        if benchmark_dispersion_percentiles is not None:
            for benchmark in ("QQQ", "SEMIS"):
                percentile = current_benchmark_dispersion[benchmark]
                if (
                    pd.notna(percentile)
                    and percentile > maximum_dispersion_percentile
                ):
                    benchmark_replacement_shares[benchmark] = 0.0
                    dispersion_scales[benchmark] = 0.0
                elif (
                    pd.notna(percentile)
                    and "active_dispersion_soft_start_percentile" in policy
                ):
                    soft_start = float(
                        policy["active_dispersion_soft_start_percentile"]
                    )
                    if not 0.0 <= soft_start < maximum_dispersion_percentile:
                        raise ValueError(
                            "Dispersion soft start must be below the hard maximum"
                        )
                    if percentile > soft_start:
                        dispersion_scales[benchmark] = (
                            maximum_dispersion_percentile - percentile
                        ) / (
                            maximum_dispersion_percentile - soft_start
                        )
        stock_target = pd.Series(0.0, index=ALL_STOCKS)
        removed_budget = {"QQQ": 0.0, "SEMIS": 0.0}
        bucket_diagnostics: list[BasketRiskEstimate] = []
        active_tracking_errors: list[float] = []
        relative_strength_confirmed_buckets = 0
        desired_budgets = {
            benchmark: max(
                0.0,
                float(known_weights.loc[date, benchmark])
                * benchmark_replacement_shares[benchmark],
            )
            for benchmark in ("QQQ", "SEMIS")
        }
        desired_budget_total = sum(desired_budgets.values())
        if date not in sizing_date_set:
            row = previous_target.copy()
            drop_threshold = float(
                policy.get("exit_budget_drop_threshold", 0.20)
            )
            reduction_scales: dict[str, float] = {}
            for benchmark in ("QQQ", "SEMIS"):
                previous_budget = previous_desired_budgets[benchmark]
                current_budget = desired_budgets[benchmark]
                should_reduce = (
                    current_budget <= 0.0
                    or (
                        previous_budget > 0.0
                        and current_budget
                        < previous_budget * (1.0 - drop_threshold)
                    )
                )
                if not should_reduce:
                    continue
                reduction_scales[benchmark] = (
                    min(1.0, current_budget / previous_budget)
                    if previous_budget > 0.0
                    else 0.0
                )
            previous_desired_budgets = desired_budgets
            if not reduction_scales:
                continue
            for benchmark, scale in reduction_scales.items():
                bucket_names = list(baskets[benchmark])
                row.loc[bucket_names] *= scale
                row[benchmark] = -float(row.loc[bucket_names].sum())
            row["CASH"] = 0.0
            if not np.isclose(float(row.sum()), 0.0, atol=1.0e-10):
                raise AssertionError("Fast-exit target must be self-financing")
            target_deltas.loc[date] = row
            previous_target = row
            diagnostic_rows.append(
                {
                    "basket": name,
                    "policy": policy_name,
                    "date": date,
                    "state": state,
                    "growth_exposure": growth_exposure,
                    "replacement_share": replacement_share,
                    "removed_etf_risk_budget": float(
                        -row[["QQQ", "SEMIS"]].sum()
                    ),
                    "actual_stock_weight": float(
                        row[list(ALL_STOCKS)].sum()
                    ),
                    "cash_reserve_weight": 0.0,
                    "maximum_name_weight": float(
                        row[list(ALL_STOCKS)].max()
                    ),
                    "maximum_risk_multiple": 0.0,
                    "maximum_active_tracking_error": 0.0,
                    "active_dispersion_percentile": (
                        current_dispersion_percentile
                    ),
                }
            )
            continue

        for benchmark in ("QQQ", "SEMIS"):
            tickers = active_tickers(
                baskets[benchmark],
                stock_prices,
                date,
                bool(policy["stock_trend_filter"]),
            )
            risk_budget = desired_budgets[benchmark]
            if not tickers or risk_budget <= 0.0:
                continue
            history_key = (date.value, benchmark, tuple(tickers))
            cached_history = (
                history_cache.get(history_key)
                if history_cache is not None
                else None
            )
            if cached_history is None:
                stock_history = all_returns.loc[
                    :date, list(tickers)
                ].dropna()
                benchmark_history = all_returns.loc[
                    :date, benchmark
                ].dropna()
                history = pd.concat(
                    [
                        stock_history,
                        benchmark_history.rename("__benchmark__"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                stock_history = history[list(tickers)]
                benchmark_history = history["__benchmark__"]
                if history_cache is not None:
                    history_cache[history_key] = (
                        stock_history,
                        benchmark_history,
                    )
            else:
                stock_history, benchmark_history = cached_history
            if len(stock_history) < 252:
                continue
            relative_strength_windows = tuple(
                int(window)
                for window in policy.get("relative_strength_windows", ())
            )
            signal_key = (
                date.value,
                benchmark,
                tuple(tickers),
                relative_strength_windows,
            )
            cached_signal = (
                active_signal_cache.get(signal_key)
                if active_signal_cache is not None
                else None
            )
            if cached_signal is None:
                cached_signal = active_signal_statistics(
                    stock_history,
                    benchmark_history,
                    relative_strength_windows,
                )
                if active_signal_cache is not None:
                    active_signal_cache[signal_key] = cached_signal
            active_log_returns, active_breadth = cached_signal
            if relative_strength_windows and (
                not active_log_returns
                or any(
                    active_log_returns[window] <= 0.0
                    for window in relative_strength_windows
                )
            ):
                continue
            if active_breadth + 1.0e-12 < float(
                policy.get("minimum_active_breadth", 0.0)
            ):
                continue
            relative_strength_confirmed_buckets += 1
            method = str(policy["method"])
            joint_risk = None
            if method in {"naive", "joint"}:
                joint_risk = cached_risk_estimate(
                    risk_cache,
                    date,
                    benchmark,
                    stock_history,
                    benchmark_history,
                )
            if method == "naive":
                allocation = naive_allocation(
                    risk_budget,
                    stock_history,
                    benchmark_history,
                    joint_risk,
                )
            elif method == "standalone":
                allocation = cached_standalone_allocation(
                    risk_budget,
                    stock_history,
                    benchmark_history,
                    date,
                    benchmark,
                    risk_cache,
                )
            elif method == "joint":
                allocation = cached_joint_allocation(
                    risk_budget,
                    stock_history,
                    joint_risk,
                )
            elif method == "active":
                active_risk = cached_active_risk_estimate(
                    active_risk_cache,
                    date,
                    benchmark,
                    stock_history,
                    benchmark_history,
                )
                worst_tracking_error = max(
                    active_risk.long_tracking_error,
                    active_risk.short_tracking_error,
                    1.0e-12,
                )
                bucket_tracking_error_budget = (
                    float(policy["tracking_error_budget"])
                    * risk_budget
                    / desired_budget_total
                )
                mix = pd.Series(
                    1.0 / stock_history.shape[1],
                    index=stock_history.columns,
                )
                actual = min(
                    risk_budget,
                    bucket_tracking_error_budget / worst_tracking_error,
                    float(policy.get("maximum_basket_weight", 0.25)),
                    float(policy.get("maximum_name_weight", 0.075))
                    / float(mix.max()),
                )
                signal_scale = 1.0
                if "active_signal_full_strength_z" in policy:
                    full_strength_z = float(
                        policy["active_signal_full_strength_z"]
                    )
                    if full_strength_z <= 0.0:
                        raise ValueError(
                            "Full-strength active signal must be positive"
                        )
                    weakest_signal_z = min(
                        active_log_returns[window]
                        / (
                            worst_tracking_error
                            * np.sqrt(window / 252.0)
                        )
                        for window in relative_strength_windows
                    )
                    signal_scale = min(
                        1.0,
                        max(0.0, weakest_signal_z / full_strength_z),
                    )
                actual *= dispersion_scales[benchmark] * signal_scale
                if actual + 1.0e-12 < float(
                    policy.get("minimum_bucket_stock_weight", 0.0)
                ):
                    actual = 0.0
                allocation = ReplacementAllocation(
                    stock_weights=mix * actual,
                    risk_budget=risk_budget,
                    actual_stock_weight=actual,
                    cash_reserve_weight=0.0,
                    risk=BasketRiskEstimate(
                        risk_multiple=1.0,
                        long_volatility_ratio=0.0,
                        short_volatility_ratio=0.0,
                        tail_loss_ratio=0.0,
                        beta=active_risk.basket_beta,
                        observations=active_risk.observations,
                    ),
                )
                active_tracking_errors.append(worst_tracking_error)
            else:
                raise ValueError(f"Unknown policy method: {method}")
            stock_target.loc[allocation.stock_weights.index] += (
                allocation.stock_weights
            )
            removed_budget[benchmark] = (
                allocation.actual_stock_weight
                if method == "active"
                else risk_budget
            )
            bucket_diagnostics.append(allocation.risk)

        maximum_basket_weight = float(
            policy.get("maximum_basket_weight", 0.25)
        )
        if float(stock_target.sum()) > maximum_basket_weight:
            scale = maximum_basket_weight / float(stock_target.sum())
            stock_target *= scale
            if str(policy["method"]) == "active":
                removed_budget = {
                    benchmark: budget * scale
                    for benchmark, budget in removed_budget.items()
                }
        row = pd.Series(0.0, index=delta_columns)
        row.loc[stock_target.index] = stock_target
        row["QQQ"] = -removed_budget["QQQ"]
        row["SEMIS"] = -removed_budget["SEMIS"]
        row["CASH"] = (
            removed_budget["QQQ"]
            + removed_budget["SEMIS"]
            - float(stock_target.sum())
        )
        if not np.isclose(float(row.sum()), 0.0, atol=1.0e-10):
            raise AssertionError("Overlay target must be self-financing")
        minimum_rebalance_turnover = float(
            policy.get("minimum_rebalance_turnover", 0.0)
        )
        if minimum_rebalance_turnover < 0.0:
            raise ValueError(
                "Minimum rebalance turnover must be non-negative"
            )
        proposed_turnover = float((row - previous_target).abs().sum() / 2.0)
        if (
            float(stock_target.sum()) > 0.0
            and proposed_turnover + 1.0e-12
            < minimum_rebalance_turnover
        ):
            row = previous_target.copy()
        target_deltas.loc[date] = row
        previous_target = row
        previous_desired_budgets = desired_budgets
        diagnostic_rows.append(
            {
                "basket": name,
                "policy": policy_name,
                "date": date,
                "state": state,
                "growth_exposure": growth_exposure,
                "replacement_share": replacement_share,
                "removed_etf_risk_budget": sum(removed_budget.values()),
                "actual_stock_weight": float(stock_target.sum()),
                "cash_reserve_weight": float(row["CASH"]),
                "maximum_name_weight": float(stock_target.max()),
                "maximum_risk_multiple": (
                    max(risk.risk_multiple for risk in bucket_diagnostics)
                    if bucket_diagnostics
                    else 0.0
                ),
                "maximum_active_tracking_error": (
                    max(active_tracking_errors)
                    if active_tracking_errors
                    else 0.0
                ),
                "relative_strength_confirmed_buckets": (
                    relative_strength_confirmed_buckets
                ),
                "active_dispersion_percentile": (
                    current_dispersion_percentile
                ),
                "qqq_active_dispersion_percentile": (
                    current_benchmark_dispersion["QQQ"]
                ),
                "semis_active_dispersion_percentile": (
                    current_benchmark_dispersion["SEMIS"]
                ),
            }
        )

    applied_delta = target_deltas.ffill().fillna(0.0).shift(1).fillna(0.0)
    relative_gross_return = (
        applied_delta[all_returns.columns.intersection(applied_delta.columns)]
        * all_returns[
            all_returns.columns.intersection(applied_delta.columns)
        ]
    ).sum(axis=1)
    overlay_turnover = (
        applied_delta.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    )
    overlay_cost = overlay_turnover * cost_rate
    result = pd.DataFrame(index=index)
    result["base_net_return"] = base["net_return"]
    result["relative_gross_return"] = relative_gross_return
    result["overlay_turnover"] = overlay_turnover
    result["overlay_cost"] = overlay_cost
    result["net_return"] = (
        result["base_net_return"]
        + result["relative_gross_return"]
        - result["overlay_cost"]
    )
    result["stock_weight"] = applied_delta[list(ALL_STOCKS)].sum(axis=1)
    result["removed_etf_risk_budget"] = -applied_delta[
        ["QQQ", "SEMIS"]
    ].sum(axis=1)
    result["cash_reserve_weight"] = applied_delta["CASH"]
    result["growth_proxy_return"] = all_returns[["QQQ", "SEMIS"]].mean(axis=1)
    return result, pd.DataFrame(diagnostic_rows)


def random_baskets(draws_per_size: int = 10) -> dict[str, dict[str, tuple[str, ...]]]:
    rng = np.random.default_rng(20_260_726)
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for size in (1, 3, 5, 8):
        for draw in range(draws_per_size):
            semis = tuple(sorted(rng.choice(SEMIS, size=size, replace=False)))
            growth = tuple(sorted(rng.choice(GROWTH, size=size, replace=False)))
            result[f"random_{size}_{draw + 1:02d}"] = {
                "SEMIS": semis,
                "QQQ": growth,
            }
    return result


def period_metrics(
    result: pd.DataFrame,
    basket: str,
    policy: str,
    alpha: float,
    cost_rate: float,
) -> list[dict[str, object]]:
    alpha_daily = (1.0 + alpha) ** (1.0 / 252.0) - 1.0
    adjusted = (
        result["net_return"] + result["stock_weight"] * alpha_daily
    )
    rows: list[dict[str, object]] = []
    periods = {
        "development_2015_2021": ("2015-01-01", "2021-12-31"),
        "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
        "complete_2015_2025": (EVALUATION_START, EVALUATION_END),
    }
    for period, (start, end) in periods.items():
        sample = adjusted.loc[start:end]
        base_sample = result["base_net_return"].loc[start:end]
        growth = result["growth_proxy_return"].loc[start:end]
        relative_log = np.log1p(sample) - np.log1p(base_sample)
        up = growth > 0.0
        down = growth < 0.0
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "alpha_assumption": alpha,
                "cost_rate": cost_rate,
                "period": period,
                **performance_metrics(sample),
                "annualized_relative_log_return": float(
                    relative_log.mean() * 252.0
                ),
                "annualized_up_day_relative_log_return": float(
                    relative_log.loc[up].sum() / len(relative_log) * 252.0
                ),
                "annualized_down_day_relative_log_return": float(
                    relative_log.loc[down].sum() / len(relative_log) * 252.0
                ),
                "average_stock_weight": float(
                    result["stock_weight"].loc[start:end].mean()
                ),
                "average_reserved_cash_weight": float(
                    result["cash_reserve_weight"].loc[start:end].mean()
                ),
                "annualized_overlay_turnover": float(
                    result["overlay_turnover"].loc[start:end].mean() * 252.0
                ),
                "annualized_overlay_cost": float(
                    result["overlay_cost"].loc[start:end].mean() * 252.0
                ),
            }
        )
    return rows


def cost_stress_metrics(
    result: pd.DataFrame,
    basket: str,
    policy: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample = result.loc[EVALUATION_START:EVALUATION_END]
    for rate in (0.00075, 0.0015, 0.0030):
        returns = (
            sample["base_net_return"]
            + sample["relative_gross_return"]
            - sample["overlay_turnover"] * rate
        )
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "cost_rate": rate,
                **performance_metrics(returns),
                "annualized_relative_log_return": float(
                    (
                        np.log1p(returns)
                        - np.log1p(sample["base_net_return"])
                    ).mean()
                    * 252.0
                ),
            }
        )
    return rows


def event_metrics(
    result: pd.DataFrame,
    basket: str,
    policy: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event, (start, end) in EVENTS.items():
        sample = result.loc[start:end]
        if sample.empty:
            continue
        strategy_equity = (1.0 + sample["net_return"]).cumprod()
        base_equity = (1.0 + sample["base_net_return"]).cumprod()
        rows.append(
            {
                "basket": basket,
                "policy": policy,
                "event": event,
                "start": sample.index[0],
                "end": sample.index[-1],
                "strategy_return": float(strategy_equity.iloc[-1] - 1.0),
                "base_return": float(base_equity.iloc[-1] - 1.0),
                "relative_log_return": float(
                    (
                        np.log1p(sample["net_return"])
                        - np.log1p(sample["base_net_return"])
                    ).sum()
                ),
                "strategy_max_drawdown": float(
                    (strategy_equity / strategy_equity.cummax() - 1.0).min()
                ),
                "base_max_drawdown": float(
                    (base_equity / base_equity.cummax() - 1.0).min()
                ),
                "average_stock_weight": float(
                    sample["stock_weight"].mean()
                ),
                "overlay_turnover": float(sample["overlay_turnover"].sum()),
            }
        )
    return rows


def summary_by_policy(metrics: pd.DataFrame) -> pd.DataFrame:
    complete = metrics.loc[
        (metrics["period"] == "complete_2015_2025")
        & (metrics["alpha_assumption"] == 0.0)
        & metrics["basket"].str.startswith("random_")
    ]
    return (
        complete.groupby("policy")
        .agg(
            median_cagr=("cagr", "median"),
            cagr_10pct=("cagr", lambda values: values.quantile(0.10)),
            median_max_drawdown=("max_drawdown", "median"),
            max_drawdown_10pct=(
                "max_drawdown",
                lambda values: values.quantile(0.10),
            ),
            median_relative_log_return=(
                "annualized_relative_log_return",
                "median",
            ),
            relative_log_return_10pct=(
                "annualized_relative_log_return",
                lambda values: values.quantile(0.10),
            ),
            median_stock_weight=("average_stock_weight", "median"),
            median_reserved_cash=("average_reserved_cash_weight", "median"),
            median_turnover=("annualized_overlay_turnover", "median"),
        )
        .sort_values("max_drawdown_10pct", ascending=False)
    )


def alpha_break_even(metrics: pd.DataFrame) -> pd.DataFrame:
    complete = metrics.loc[
        (metrics["period"] == "complete_2015_2025")
        & metrics["basket"].str.startswith("random_")
    ]
    rows: list[dict[str, object]] = []
    for policy, frame in complete.groupby("policy"):
        medians = frame.groupby("alpha_assumption")[
            "annualized_relative_log_return"
        ].median()
        positive = medians.loc[medians >= 0.0]
        rows.append(
            {
                "policy": policy,
                "break_even_alpha_grid": (
                    float(positive.index.min()) if not positive.empty else np.nan
                ),
                **{
                    f"median_relative_log_return_alpha_{int(alpha * 100)}": float(
                        medians.loc[alpha]
                    )
                    for alpha in ALPHA_SCENARIOS
                },
            }
        )
    return pd.DataFrame(rows).set_index("policy")


def write_decision(
    policy_summary: pd.DataFrame,
    break_even: pd.DataFrame,
    metrics: pd.DataFrame,
    destination: Path,
) -> None:
    joint = policy_summary.loc["joint_regime"]
    standalone = policy_summary.loc["standalone_always"]
    always = policy_summary.loc["joint_always"]
    trend = policy_summary.loc["joint_regime_trend"]
    fixed = metrics.loc[
        (metrics["period"] == "complete_2015_2025")
        & (metrics["alpha_assumption"] == 0.0)
        & metrics["basket"].isin(FIXED_BASKETS)
    ]
    text = f"""# R9 个股叠加层第一轮验证

## 决策

**联合风险预算与分阶段准入进入生产实现；历史收益结果不用于宣称选股有效。**

本实验把用户选出的个股视为 QQQ 或 SMH 的替代品，不尝试从历史中挑选赢家。
代表性股票面板只用于校准风险，含当前知识选择和幸存者偏差，因此不能证明未来
个股选择会产生超额收益。

## 机制结论

1. 单只股票的风险倍数主要就是“个股波动率 ÷ ETF 波动率”；再与普通 Beta
   取最大值几乎没有新增信息。
2. 多只股票逐只扣减会忽略个股剩余风险之间的分散，系统性地少配个股。
   联合版本使用收缩协方差，并取 252 日、60 日和最差 5% 日均损失三个风险比
   的最大值。
3. R9 成长仓不高于 35% 时，个股替代比例为 0；首次恢复的 20% QQQ 和确认后的
   35% QQQ 都保持为 ETF。成长恢复充分后，安静/正常上涨环境最多替代对应 ETF
   风险额度的 50%，脆弱上涨或反弹环境最多替代 25%，其他环境为 0。
4. 单只个股最多占账户 7.5%，全部受管理个股合计最多占 25%。未使用的风险额度
   留在 BIL/现金，而不是重新加到 ETF。

## 随机组合压力结果

随机组合不是选股回测；它反复抽取 1、3、5、8 只现存成长/半导体股票，观察规则
对不同实现路径的敏感度。所有数字均含 15 个基点额外调仓成本。

| 规则 | 中位股票仓位 | 中位预留现金 | 10%较差组合 CAGR | 10%较差组合最大回撤 |
|---|---:|---:|---:|---:|
| 逐只风险扣减、始终启用 | {standalone['median_stock_weight']:.2%} | {standalone['median_reserved_cash']:.2%} | {standalone['cagr_10pct']:.2%} | {standalone['max_drawdown_10pct']:.2%} |
| 联合风险、始终启用 | {always['median_stock_weight']:.2%} | {always['median_reserved_cash']:.2%} | {always['cagr_10pct']:.2%} | {always['max_drawdown_10pct']:.2%} |
| 联合风险、按市场环境启用 | {joint['median_stock_weight']:.2%} | {joint['median_reserved_cash']:.2%} | {joint['cagr_10pct']:.2%} | {joint['max_drawdown_10pct']:.2%} |
| 再加个股 200 日趋势过滤 | {trend['median_stock_weight']:.2%} | {trend['median_reserved_cash']:.2%} | {trend['cagr_10pct']:.2%} | {trend['max_drawdown_10pct']:.2%} |

联合、按环境启用相对“联合但始终启用”的 10% 较差组合最大回撤变化为
{joint['max_drawdown_10pct'] - always['max_drawdown_10pct']:+.2%}；相对逐只扣减的
中位股票仓位变化为 {joint['median_stock_weight'] - standalone['median_stock_weight']:+.2%}。

在测试的人工选股年化超额收益网格中，联合、按环境启用的中位组合达到相对 R9
不拖累所需的最小输入为
{break_even.loc['joint_regime', 'break_even_alpha_grid']:.0%}。
这个数只是敏感度门槛，不是对用户能力的估计。

固定压力组合共 {fixed['basket'].nunique()} 组，包含集中赢家、老牌周期股和较宽组合。
完整明细见 `metrics_by_period.csv`、`policy_summary.csv`、`cost_stress.csv` 和
`rebalance_diagnostics.csv`。

## 不能从本实验得出的结论

- 不能据此选择 NVDA、MSFT 或任何其他股票。
- 不能把 2022-2025 称为纯样本外，因为股票面板和市场环境叠加层都已使用当前知识。
- 不能证明 7.5% 或 25% 是历史最优参数；它们是事前风险上限，没有做参数搜索。
- Yahoo 调整后收盘用于风险校准，不是可成交价格；真实执行仍需最新价格、税务和
  账户限制。
"""
    (destination / "decision.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--draws-per-size", type=int, default=10)
    parser.add_argument(
        "--policies",
        help="Comma-separated policy names; defaults to every policy.",
    )
    parser.add_argument(
        "--destination",
        default=str(DESTINATION),
    )
    parser.add_argument("--skip-decision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_prices = load_stock_prices(args.refresh)
    base, weights, all_returns, states = load_inputs(stock_prices)
    baskets = {**FIXED_BASKETS, **random_baskets(args.draws_per_size)}
    selected_policies = (
        [name.strip() for name in args.policies.split(",") if name.strip()]
        if args.policies
        else list(POLICIES)
    )
    unknown_policies = sorted(set(selected_policies) - set(POLICIES))
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}")
    destination = Path(args.destination)
    metric_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    diagnostics: list[pd.DataFrame] = []
    daily_examples: dict[str, pd.DataFrame] = {}
    risk_cache: RiskCache = {}
    active_risk_cache: ActiveRiskCache = {}

    for basket_name, basket in baskets.items():
        for policy_name in selected_policies:
            result, diagnostic = simulate_overlay(
                basket_name,
                basket,
                policy_name,
                base,
                weights,
                all_returns,
                states,
                stock_prices,
                risk_cache,
                active_risk_cache,
            )
            diagnostics.append(diagnostic)
            for alpha in ALPHA_SCENARIOS:
                metric_rows.extend(
                    period_metrics(
                        result,
                        basket_name,
                        policy_name,
                        alpha,
                        DEFAULT_COST_RATE,
                    )
                )
            cost_rows.extend(
                cost_stress_metrics(result, basket_name, policy_name)
            )
            event_rows.extend(event_metrics(result, basket_name, policy_name))
            if (
                basket_name == "five_diversified"
                and policy_name
                in {
                    "standalone_always",
                    "joint_always",
                    "joint_regime",
                    "active_regime_te4",
                }
            ):
                daily_examples[policy_name] = result

    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    costs = pd.DataFrame(cost_rows)
    events = pd.DataFrame(event_rows)
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)
    policy_summary = summary_by_policy(metrics)
    break_even = alpha_break_even(metrics)
    metrics.to_csv(destination / "metrics_by_period.csv", index=False)
    costs.to_csv(destination / "cost_stress.csv", index=False)
    events.to_csv(destination / "event_metrics.csv", index=False)
    diagnostics_frame.to_csv(
        destination / "rebalance_diagnostics.csv",
        index=False,
    )
    policy_summary.to_csv(destination / "policy_summary.csv")
    break_even.to_csv(destination / "alpha_break_even.csv")
    for policy, frame in daily_examples.items():
        frame.to_csv(destination / f"five_diversified_{policy}_daily.csv")
    if not args.skip_decision:
        write_decision(policy_summary, break_even, metrics, destination)

    audit = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "first_date": stock_prices[ticker].first_valid_index(),
                "last_date": stock_prices[ticker].last_valid_index(),
                "observations": int(stock_prices[ticker].notna().sum()),
                "minimum_price": float(stock_prices[ticker].min()),
                "maximum_price": float(stock_prices[ticker].max()),
            }
            for ticker in stock_prices
        ]
    )
    audit.to_csv(destination / "stock_data_audit.csv", index=False)
    print(policy_summary.round(6).to_string())
    print("\nAlpha sensitivity:")
    print(break_even.round(6).to_string())
    print(f"\nArtifacts: {destination.resolve()}")


if __name__ == "__main__":
    main()
