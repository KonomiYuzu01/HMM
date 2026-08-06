#!/usr/bin/env python3
"""Evaluate retail-accessible diversifiers without tuning allocation parameters.

The production strategy is treated as a net-of-cost synthetic component.  Each
alternative sleeve is funded in one of three pre-specified ways and is tested at
fixed 5%/10% sizes.  The script intentionally does not optimize thresholds,
lookbacks, product weights, or stress-period parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from regime_strategy.data import completed_us_daily_prices


ROOT = Path(__file__).resolve().parents[1]
BASE_RETURNS_PATH = (
    ROOT
    / "output"
    / "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
    / "daily_returns.csv"
)
BASE_WEIGHTS_PATH = (
    ROOT
    / "output"
    / "paper_core_growth_gold20_daily_risk_netted_ensemble_2012"
    / "weights.csv"
)
PRICE_CACHE_PATH = ROOT / "data" / "retail_alternatives_adjusted_close.csv"
OPEN_CLOSE_CACHE_PATH = ROOT / "data" / "retail_alternatives_open_close.csv"
BASE_OPEN_CLOSE_PATH = ROOT / "data" / "adjusted_open_close_2011_present.csv"
OUTPUT_DIR = ROOT / "output" / "retail_alternatives_validation"

TRADING_DAYS = 252
TICKERS = [
    "BIL",
    "SPY",
    "QQQ",
    "SMH",
    "CAOS",
    "CTA",
    "KMLM",
    "DBMF",
    "WTMF",
    "FMF",
    "FLSP",
    "BTAL",
    "QAI",
    "ILS",
    "GLD",
    "GDE",
    "NTSX",
    "RSST",
    "RSBT",
]


@dataclass(frozen=True)
class Product:
    ticker: str
    fee_pct: float
    spread_pct: float | None
    aum_usd_m: float | None
    inception: str
    mechanism: str
    issue: str


PRODUCTS = {
    "CAOS": Product(
        "CAOS",
        0.63,
        0.10,
        719.0,
        "2023-03-06; predecessor AVOLX from 2013-08-14",
        "Protective puts, put spreads, and box-spread collateral",
        "ETF-only daily history misses the 2018 and 2020 predecessor-fund crises",
    ),
    "CTA": Product(
        "CTA",
        0.75,
        0.11,
        1580.0,
        "2022-03-07",
        "Trend, mean reversion, carry, and risk-off across commodities/rates/FX",
        "Short live record; no 2020 or earlier ETF evidence",
    ),
    "KMLM": Product(
        "KMLM",
        0.90,
        0.07,
        380.0,
        "2020-12-01",
        "Rules-based trends across commodities, currencies, and bonds",
        "Concentrated managed-futures model and modest live-history length",
    ),
    "DBMF": Product(
        "DBMF",
        0.85,
        None,
        None,
        "2019-05-08",
        "Replicates aggregate managed-futures positioning across liquid futures",
        "Includes equity-index exposure and model-replication risk",
    ),
    "WTMF": Product(
        "WTMF",
        0.65,
        0.44,
        244.0,
        "2011-01-05",
        "Multi-asset managed futures",
        "Wide spread and current mandate can include bitcoin exposure",
    ),
    "FMF": Product(
        "FMF",
        0.98,
        0.50,
        None,
        "2013-08-02",
        "Active managed futures",
        "High fee/spread and prior tests did not improve drawdown",
    ),
    "FLSP": Product(
        "FLSP",
        0.65,
        0.73,
        None,
        "2019-12-18",
        "Long/short value, quality, momentum, and carry style premia",
        "Very wide spread and short history",
    ),
    "BTAL": Product(
        "BTAL",
        1.40,
        None,
        296.0,
        "2011-09-13",
        "Market-neutral long low-beta / short high-beta equities",
        "Negative-beta hedge but weak long-run net return and strategy change in 2022",
    ),
    "QAI": Product(
        "QAI",
        0.88,
        None,
        1020.0,
        "2009-03-25",
        "Hedge-fund index replication",
        "Low historical return and residual credit/equity exposure",
    ),
    "ILS": Product(
        "ILS",
        2.00,
        None,
        34.9,
        "2025-04-01",
        "Catastrophe-bond and insurance-linked securities",
        "Too new, expensive, and small for validation",
    ),
    "GDE": Product(
        "GDE",
        0.20,
        0.55,
        445.8,
        "2022-03-17",
        "Roughly 90% U.S. large-cap equities plus 90% gold-futures exposure",
        "Leveraged structure, wide current spread, and short live record",
    ),
    "NTSX": Product(
        "NTSX",
        0.20,
        0.24,
        1355.9,
        "2018-08-02",
        "Roughly 90% U.S. large-cap equities plus 60% Treasury-futures exposure",
        "Equity/bond diversification failed during the 2022 inflation shock",
    ),
    "RSST": Product(
        "RSST",
        0.99,
        None,
        490.5,
        "2023-09-05",
        "100% U.S. equity plus 100% managed-futures trend exposure",
        "Less than three years of live history and nearly 1% annual fee",
    ),
    "RSBT": Product(
        "RSBT",
        1.01,
        None,
        138.7,
        "2023-02-07",
        "100% bonds plus 100% managed-futures trend exposure",
        "Short history, fee above 1%, and substantial bond-duration exposure",
    ),
}


STRESS_WINDOWS = {
    "covid_fast_crash": ("2020-02-19", "2020-03-23"),
    "inflation_bear_2022": ("2022-01-03", "2022-10-14"),
    "yen_unwind_2024": ("2024-08-01", "2024-08-05"),
    "tariff_shock_2025": ("2025-02-19", "2025-04-08"),
    "recent_2026": ("2026-01-02", "2026-07-22"),
}


def funding_targets(
    cash_weight: float,
    sleeve_weight: float,
    mode: str,
) -> tuple[float, float, float]:
    """Return target weights for risky model, cash, and alternative sleeve."""
    c = float(cash_weight)
    s = float(sleeve_weight)
    if not 0.0 <= s < 1.0:
        raise ValueError("sleeve_weight must be in [0, 1)")

    if mode == "cash_capped":
        alternative = min(s, max(c, 0.0))
        risky = 1.0 - c
        cash = c - alternative
    elif mode == "cash_first":
        alternative = s
        if c >= s:
            risky = 1.0 - c
            cash = c - s
        else:
            risky = 1.0 - s
            cash = 0.0
    elif mode == "pro_rata":
        alternative = s
        risky = (1.0 - s) * (1.0 - c)
        cash = (1.0 - s) * c
    else:
        raise ValueError(f"unknown funding mode: {mode}")

    if not np.isclose(risky + cash + alternative, 1.0):
        raise AssertionError("funding weights do not sum to one")
    return risky, cash, alternative


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "total_return": np.nan,
            "years": 0.0,
        }
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = drawdown.min()
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "total_return": wealth.iloc[-1] - 1.0,
        "years": years,
    }


def conditional_beta(asset: pd.Series, growth: pd.Series, quantile: float = 0.10) -> dict[str, float]:
    frame = pd.concat([asset.rename("asset"), growth.rename("growth")], axis=1).dropna()
    if len(frame) < 30:
        return {"beta": np.nan, "downside_beta": np.nan, "correlation": np.nan}

    def beta(sample: pd.DataFrame) -> float:
        variance = sample["growth"].var(ddof=1)
        return sample["asset"].cov(sample["growth"]) / variance if variance > 0 else np.nan

    downside = frame.loc[frame["growth"] <= frame["growth"].quantile(quantile)]
    return {
        "beta": beta(frame),
        "downside_beta": beta(downside),
        "correlation": frame["asset"].corr(frame["growth"]),
    }


def write_cache_metadata(
    path: Path,
    frame: pd.DataFrame,
    *,
    adjustment: str,
    generator: str,
    purpose: str,
    extra: dict[str, object] | None = None,
) -> None:
    metadata = {
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "first_date": frame.index.min().date().isoformat(),
        "generator": generator,
        "market_data_adjustment": adjustment,
        "market_data_source": "Yahoo Finance via yfinance",
        "price_as_of": frame.index.max().date().isoformat(),
        "purpose": purpose,
        "row_count": len(frame),
        **(extra or {}),
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def download_adjusted_close(tickers: Iterable[str]) -> pd.DataFrame:
    requested = list(tickers)
    try:
        raw = yf.download(
            requested,
            start="2008-01-01",
            end=None,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise RuntimeError("empty yfinance response")
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Adj Close"].copy()
        else:
            prices = raw[["Adj Close"]].rename(columns={"Adj Close": requested[0]})
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.sort_index().dropna(how="all")
        downloaded_at = pd.Timestamp.now(tz="America/New_York")
        prices = completed_us_daily_prices(
            prices,
            now=downloaded_at,
            source_modified_at=downloaded_at,
        )
        PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(PRICE_CACHE_PATH, index_label="date")
        write_cache_metadata(
            PRICE_CACHE_PATH,
            prices,
            adjustment="auto_adjust=False; Adj Close field selected",
            generator=(
                "tools/evaluate_retail_alternatives.py::"
                "download_adjusted_close"
            ),
            purpose=(
                "Retail alternative and live GDE tracking-error research."
            ),
            extra={
                "selected_tracking_columns": [
                    "GDE",
                    "SPY",
                    "GLD",
                    "BIL",
                ]
            },
        )
        return prices
    except Exception:
        if not PRICE_CACHE_PATH.exists():
            raise
        cached = pd.read_csv(PRICE_CACHE_PATH, index_col="date", parse_dates=True)
        missing = sorted(set(requested) - set(cached.columns))
        if missing:
            raise RuntimeError(f"price cache is missing: {missing}")
        return cached


def download_gde_open_close() -> pd.DataFrame:
    try:
        raw = yf.download(
            "GDE",
            start="2022-03-17",
            end=None,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError("empty GDE open/close response")
        if isinstance(raw.columns, pd.MultiIndex):
            open_prices = raw["Open"].iloc[:, 0]
            close_prices = raw["Close"].iloc[:, 0]
        else:
            open_prices = raw["Open"]
            close_prices = raw["Close"]
        result = pd.DataFrame(
            {
                "open_GDE": open_prices,
                "close_GDE": close_prices,
            }
        )
        result.index = pd.to_datetime(result.index).tz_localize(None)
        result = result.dropna()
        downloaded_at = pd.Timestamp.now(tz="America/New_York")
        result = completed_us_daily_prices(
            result,
            now=downloaded_at,
            source_modified_at=downloaded_at,
        )
        result.to_csv(OPEN_CLOSE_CACHE_PATH, index_label="date")
        write_cache_metadata(
            OPEN_CLOSE_CACHE_PATH,
            result,
            adjustment="auto_adjust=True",
            generator=(
                "tools/evaluate_retail_alternatives.py::"
                "download_gde_open_close"
            ),
            purpose=(
                "Live GDE open/close execution validation; "
                "not a real-time quote source."
            ),
            extra={"ticker": "GDE"},
        )
        return result
    except Exception:
        if not OPEN_CLOSE_CACHE_PATH.exists():
            raise
        return pd.read_csv(
            OPEN_CLOSE_CACHE_PATH,
            index_col="date",
            parse_dates=True,
        )


def load_open_close_with_gde() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(BASE_OPEN_CLOSE_PATH, index_col=0, parse_dates=True)
    gde = download_gde_open_close()
    frame = base.join(gde, how="inner").dropna()
    assets = ["SPX", "QQQ", "SEMIS", "BOND", "GOLD", "OIL", "USD", "CASH", "VIX_HEDGE"]
    opens = frame[[f"open_{asset}" for asset in assets] + ["open_GDE"]].copy()
    closes = frame[[f"close_{asset}" for asset in assets] + ["close_GDE"]].copy()
    opens.columns = assets + ["GDE"]
    closes.columns = assets + ["GDE"]
    return opens.astype(float), closes.astype(float)


def load_baseline() -> pd.DataFrame:
    returns = pd.read_csv(BASE_RETURNS_PATH, index_col=0, parse_dates=True)
    weights = pd.read_csv(BASE_WEIGHTS_PATH, index_col=0, parse_dates=True)
    returns.index = pd.to_datetime(returns.index).tz_localize(None)
    weights.index = pd.to_datetime(weights.index).tz_localize(None)

    gross_col = "gross_return" if "gross_return" in returns.columns else "strategy_return"
    net_col = "net_return" if "net_return" in returns.columns else "strategy_return_net"
    required = {gross_col, net_col}
    if not required.issubset(returns.columns):
        raise ValueError(f"baseline return file lacks columns: {required - set(returns.columns)}")
    if "CASH" not in weights.columns:
        raise ValueError("baseline weights file lacks CASH")

    result = pd.DataFrame(
        {
            "gross_return": returns[gross_col],
            "net_return": returns[net_col],
            "cash_weight": weights["CASH"],
            "gold_weight": weights["GOLD"],
            "growth_weight": weights[["SPX", "QQQ", "SEMIS"]].sum(axis=1),
        }
    ).dropna()
    return result


def gde_synthetic_returns(price_returns: pd.DataFrame) -> pd.Series:
    """Conservative 90/90 proxy built from net ETF returns.

    GLD already carries a higher fee than gold futures, so no favorable
    fee calibration is applied.  The live GDE comparison quantifies the
    resulting basis and implementation error.
    """
    required = {"SPY", "GLD", "BIL"}
    missing = required - set(price_returns.columns)
    if missing:
        raise ValueError(f"GDE proxy lacks return columns: {sorted(missing)}")
    # A gold future earns approximately gold spot/ETF return minus the cash
    # financing rate.  The fund's 10% collateral earns cash, so the net cash
    # term is +0.10 BIL - 0.90 BIL = -0.80 BIL.
    return (
        0.90 * price_returns["SPY"]
        + 0.90 * price_returns["GLD"]
        - 0.80 * price_returns["BIL"]
    ).rename("GDE_SYNTHETIC")


def gde_tracking_diagnostics(
    actual: pd.Series,
    synthetic: pd.Series,
) -> pd.DataFrame:
    aligned = pd.concat(
        [actual.rename("actual"), synthetic.rename("synthetic")],
        axis=1,
    ).dropna()
    residual = aligned["actual"] - aligned["synthetic"]
    actual_metrics = performance_metrics(aligned["actual"])
    synthetic_metrics = performance_metrics(aligned["synthetic"])
    variance = aligned["synthetic"].var(ddof=1)
    return pd.DataFrame(
        [
            {
                "start": aligned.index.min().date().isoformat(),
                "end": aligned.index.max().date().isoformat(),
                "observations": len(aligned),
                "years": len(aligned) / TRADING_DAYS,
                "actual_cagr": actual_metrics["cagr"],
                "synthetic_cagr": synthetic_metrics["cagr"],
                "annualized_cagr_difference": (
                    actual_metrics["cagr"] - synthetic_metrics["cagr"]
                ),
                "tracking_error": residual.std(ddof=1) * np.sqrt(TRADING_DAYS),
                "correlation": aligned["actual"].corr(aligned["synthetic"]),
                "beta": (
                    aligned["actual"].cov(aligned["synthetic"]) / variance
                    if variance > 0
                    else np.nan
                ),
                "residual_mean_annualized": residual.mean() * TRADING_DAYS,
            }
        ]
    )


def simulate_gold_substitution(
    baseline: pd.DataFrame,
    gde_returns: pd.Series,
    gld_returns: pd.Series,
    substitution_fraction: float,
    gate_mode: str = "always",
    extra_one_way_cost: float = 0.0030,
) -> pd.DataFrame:
    """Replace a fixed fraction of each day's GOLD target with GDE.

    One dollar of GDE replaces one dollar of GLD.  GDE therefore retains about
    90 cents of gold exposure and adds about 90 cents of S&P 500 exposure.
    """
    if not 0.0 <= substitution_fraction <= 1.0:
        raise ValueError("substitution_fraction must be in [0, 1]")
    aligned = pd.concat(
        [
            baseline,
            gde_returns.rename("gde_return"),
            gld_returns.rename("gld_return"),
        ],
        axis=1,
        sort=False,
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    if gate_mode == "always":
        gate = pd.Series(1.0, index=aligned.index)
    elif gate_mode == "growth_linked":
        # The 80% normalizer is the production strategic core, fixed before
        # this test: QQQ 40% + SEMIS 40%.
        gate = (aligned["growth_weight"] / 0.80).clip(lower=0.0, upper=1.0)
    else:
        raise ValueError(f"unknown gold-substitution gate: {gate_mode}")

    target = (
        substitution_fraction
        * aligned["gold_weight"].clip(lower=0.0)
        * gate
    )
    candidate = aligned["net_return"] + target * (
        aligned["gde_return"] - aligned["gld_return"]
    )

    costs: list[float] = []
    previous_post = 0.0
    for date, row in aligned.iterrows():
        current_target = float(target.loc[date])
        trade = abs(current_target - previous_post)
        cost = trade * extra_one_way_cost
        costs.append(cost)
        net_before_cost = float(candidate.loc[date])
        denominator = 1.0 + net_before_cost
        previous_post = (
            current_target * (1.0 + float(row["gde_return"])) / denominator
            if denominator > 0
            else current_target
        )

    result = aligned[["net_return", "gold_weight"]].rename(
        columns={"net_return": "baseline_net"}
    )
    result["candidate_net"] = candidate - np.array(costs)
    result["execution_cost"] = costs
    result["gde_weight"] = target
    result["gate_intensity"] = gate
    result["added_spy_notional"] = 0.90 * target
    result["gold_notional_reduction"] = 0.10 * target
    return result


def evaluate_gold_substitution(
    baseline: pd.DataFrame,
    actual_gde: pd.Series,
    synthetic_gde: pd.Series,
    gld_returns: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_returns = {
        "GDE_SYNTHETIC": synthetic_gde,
        "GDE_LIVE": actual_gde,
    }
    periods = {
        "full_available": (None, None),
        "2012_2019": ("2012-01-01", "2019-12-31"),
        "2020_2021": ("2020-01-01", "2021-12-31"),
        "inflation_2022": ("2022-01-01", "2022-12-31"),
        "2023_present": ("2023-01-01", None),
    }
    for source, returns in source_returns.items():
        for fraction in [0.25, 0.50, 1.00]:
            for gate_mode in ["always", "growth_linked"]:
                path = simulate_gold_substitution(
                    baseline,
                    returns,
                    gld_returns,
                    fraction,
                    gate_mode=gate_mode,
                )
                for period, (start, end) in periods.items():
                    segment = path
                    if start is not None:
                        segment = segment.loc[segment.index >= pd.Timestamp(start)]
                    if end is not None:
                        segment = segment.loc[segment.index <= pd.Timestamp(end)]
                    if len(segment) < 126:
                        continue
                    base = performance_metrics(segment["baseline_net"])
                    candidate = performance_metrics(segment["candidate_net"])
                    rows.append(
                        {
                            "source": source,
                            "substitution_fraction": fraction,
                            "gate_mode": gate_mode,
                            "period": period,
                            "start": segment.index.min().date().isoformat(),
                            "end": segment.index.max().date().isoformat(),
                            "observations": len(segment),
                            "baseline_cagr": base["cagr"],
                            "candidate_cagr": candidate["cagr"],
                            "delta_cagr": candidate["cagr"] - base["cagr"],
                            "baseline_sharpe": base["sharpe"],
                            "candidate_sharpe": candidate["sharpe"],
                            "delta_sharpe": candidate["sharpe"] - base["sharpe"],
                            "baseline_max_drawdown": base["max_drawdown"],
                            "candidate_max_drawdown": candidate["max_drawdown"],
                            "delta_max_drawdown": (
                                candidate["max_drawdown"] - base["max_drawdown"]
                            ),
                            "baseline_calmar": base["calmar"],
                            "candidate_calmar": candidate["calmar"],
                            "delta_calmar": candidate["calmar"] - base["calmar"],
                            "annual_execution_cost": (
                                segment["execution_cost"].mean() * TRADING_DAYS
                            ),
                            "average_gde_weight": segment["gde_weight"].mean(),
                            "average_gate_intensity": (
                                segment["gate_intensity"].mean()
                            ),
                            "average_added_spy_notional": (
                                segment["added_spy_notional"].mean()
                            ),
                            "average_gold_notional_reduction": (
                                segment["gold_notional_reduction"].mean()
                            ),
                            "drawdown_within_18pct": (
                                candidate["max_drawdown"] >= -0.18
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_gold_substitution_uncertainty(
    baseline: pd.DataFrame,
    actual_gde: pd.Series,
    synthetic_gde: pd.Series,
    gld_returns: pd.Series,
    simulations: int = 1000,
    block_days: int = 21,
    random_seed: int = 20260724,
) -> pd.DataFrame:
    """Inject live GDE implementation residuals into the long-history proxy."""
    live = pd.concat(
        [
            actual_gde.rename("actual"),
            synthetic_gde.rename("synthetic"),
        ],
        axis=1,
        sort=False,
    ).dropna()
    residual = (live["actual"] - live["synthetic"]).to_numpy(dtype=float)
    if len(residual) < block_days:
        raise ValueError("insufficient live residuals for block bootstrap")

    sample_path = simulate_gold_substitution(
        baseline,
        synthetic_gde,
        gld_returns,
        0.0,
    )
    path_length = len(sample_path)
    blocks_needed = int(np.ceil(path_length / block_days))
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, len(residual), size=(simulations, blocks_needed))
    offsets = np.arange(block_days)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(residual)
    bootstrapped = residual[indices].reshape(simulations, -1)[:, :path_length].T

    base = performance_metrics(sample_path["baseline_net"])
    years = path_length / TRADING_DAYS
    rows: list[dict[str, object]] = []
    for fraction in [0.25, 0.50, 1.00]:
        for gate_mode in ["always", "growth_linked"]:
            path = simulate_gold_substitution(
                baseline,
                synthetic_gde,
                gld_returns,
                fraction,
                gate_mode=gate_mode,
            )
            common = sample_path.index.intersection(path.index)
            deterministic = path.loc[common, "candidate_net"].to_numpy(dtype=float)
            target = path.loc[common, "gde_weight"].to_numpy(dtype=float)
            simulated_returns = deterministic[:, None] + target[:, None] * bootstrapped

            wealth = np.cumprod(1.0 + simulated_returns, axis=0)
            cagr = wealth[-1] ** (1.0 / years) - 1.0
            peaks = np.maximum.accumulate(wealth, axis=0)
            max_drawdown = np.min(wealth / peaks - 1.0, axis=0)
            standard_deviation = simulated_returns.std(axis=0, ddof=1)
            sharpe = (
                simulated_returns.mean(axis=0)
                / np.where(standard_deviation > 0, standard_deviation, np.nan)
                * np.sqrt(TRADING_DAYS)
            )

            cagr_delta = cagr - base["cagr"]
            sharpe_delta = sharpe - base["sharpe"]
            within_drawdown = max_drawdown >= -0.18
            rows.append(
                {
                    "substitution_fraction": fraction,
                    "gate_mode": gate_mode,
                    "simulations": simulations,
                    "block_days": block_days,
                    "random_seed": random_seed,
                    "residual_live_start": live.index.min().date().isoformat(),
                    "residual_live_end": live.index.max().date().isoformat(),
                    "residual_mean_annualized": residual.mean() * TRADING_DAYS,
                    "residual_tracking_error": residual.std(ddof=1)
                    * np.sqrt(TRADING_DAYS),
                    "delta_cagr_p05": np.quantile(cagr_delta, 0.05),
                    "delta_cagr_median": np.quantile(cagr_delta, 0.50),
                    "delta_cagr_p95": np.quantile(cagr_delta, 0.95),
                    "delta_sharpe_p05": np.quantile(sharpe_delta, 0.05),
                    "delta_sharpe_median": np.quantile(sharpe_delta, 0.50),
                    "delta_sharpe_p95": np.quantile(sharpe_delta, 0.95),
                    "max_drawdown_p05": np.quantile(max_drawdown, 0.05),
                    "max_drawdown_median": np.quantile(max_drawdown, 0.50),
                    "max_drawdown_p95": np.quantile(max_drawdown, 0.95),
                    "probability_positive_delta_cagr": np.mean(cagr_delta > 0.0),
                    "probability_drawdown_within_18pct": np.mean(within_drawdown),
                    "probability_positive_delta_sharpe": np.mean(
                        sharpe_delta > 0.0
                    ),
                    "probability_all_objectives": np.mean(
                        (cagr_delta > 0.0)
                        & within_drawdown
                        & (sharpe_delta > 0.0)
                    ),
                    "probability_meet_user_objectives": np.mean(
                        (cagr_delta > 0.0)
                        & within_drawdown
                        & (sharpe >= 1.0)
                    ),
                }
            )
            del simulated_returns, wealth, peaks
    return pd.DataFrame(rows)


def simulate_gold_substitution_open(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    substitution_fraction: float,
    gate_mode: str,
    base_cost_bps: float = 7.5,
    gde_one_way_cost_bps: float = 30.0,
    financing_spread_bps: float = 100.0,
) -> pd.DataFrame:
    """Execute model targets at the next available daily open using live GDE."""
    weights = pd.read_csv(BASE_WEIGHTS_PATH, index_col=0, parse_dates=True)
    daily = pd.read_csv(BASE_RETURNS_PATH, index_col=0, parse_dates=True)
    base_assets = list(weights.columns)
    assets = base_assets + ["GDE"]
    dates = (
        weights.index.intersection(daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    if dates.empty:
        return pd.DataFrame()

    current = np.zeros(len(assets), dtype=float)
    current[assets.index("CASH")] = 1.0
    rows: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None
    for date in dates:
        if previous_date is None:
            overnight_return = 0.0
        else:
            overnight_asset_returns = (
                opens.loc[date, assets].to_numpy(dtype=float)
                / closes.loc[previous_date, assets].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = _post_return_weights(
                current,
                overnight_asset_returns,
                overnight_return,
            )

        baseline_trade = float(daily.loc[date, "turnover"]) > 1e-14
        trade_now = previous_date is None or baseline_trade
        trading_cost = 0.0
        gde_target = float(current[assets.index("GDE")])
        if trade_now:
            base_target = weights.loc[date, base_assets].to_numpy(
                dtype=float
            ).copy()
            growth_weight = float(
                weights.loc[date, ["SPX", "QQQ", "SEMIS"]].sum()
            )
            if gate_mode == "always":
                gate = 1.0
            elif gate_mode == "growth_linked":
                gate = float(np.clip(growth_weight / 0.80, 0.0, 1.0))
            else:
                raise ValueError(
                    f"unknown gold-substitution gate: {gate_mode}"
                )
            gold_index = base_assets.index("GOLD")
            gde_target = (
                substitution_fraction
                * max(float(base_target[gold_index]), 0.0)
                * gate
            )
            base_target[gold_index] -= gde_target
            target = np.concatenate([base_target, [gde_target]])
            delta = np.abs(target - current)
            rates = np.full(len(assets), base_cost_bps / 10_000.0)
            rates[assets.index("GDE")] = gde_one_way_cost_bps / 10_000.0
            trading_cost = float(delta @ rates)
            current = target

        intraday_asset_returns = (
            closes.loc[date, assets].to_numpy(dtype=float)
            / opens.loc[date, assets].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        financing_cost = (
            max(-float(current[assets.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / TRADING_DAYS
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
        )
        current = _post_return_weights(
            current,
            intraday_asset_returns,
            intraday_return,
        )
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "overnight_return": overnight_return,
                "intraday_return": intraday_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "traded_at_open": int(trade_now),
                "gde_target": gde_target,
            }
        )
        previous_date = date
    return pd.DataFrame(rows).set_index("date")


def evaluate_gold_substitution_open(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
) -> pd.DataFrame:
    baseline_path = simulate_gold_substitution_open(
        opens,
        closes,
        0.0,
        "always",
    )
    baseline_metrics = performance_metrics(baseline_path["net_return"])
    rows: list[dict[str, object]] = []
    for fraction in [0.25, 0.50, 1.00]:
        for gate_mode in ["always", "growth_linked"]:
            path = simulate_gold_substitution_open(
                opens,
                closes,
                fraction,
                gate_mode,
            )
            metrics = performance_metrics(path["net_return"])
            rows.append(
                {
                    "substitution_fraction": fraction,
                    "gate_mode": gate_mode,
                    "start": path.index.min().date().isoformat(),
                    "end": path.index.max().date().isoformat(),
                    "observations": len(path),
                    "average_gde_target": path["gde_target"].mean(),
                    "annual_trading_cost": path["trading_cost"].mean()
                    * TRADING_DAYS,
                    "baseline_cagr": baseline_metrics["cagr"],
                    "candidate_cagr": metrics["cagr"],
                    "delta_cagr": metrics["cagr"] - baseline_metrics["cagr"],
                    "baseline_sharpe": baseline_metrics["sharpe"],
                    "candidate_sharpe": metrics["sharpe"],
                    "delta_sharpe": (
                        metrics["sharpe"] - baseline_metrics["sharpe"]
                    ),
                    "baseline_max_drawdown": baseline_metrics["max_drawdown"],
                    "candidate_max_drawdown": metrics["max_drawdown"],
                    "delta_max_drawdown": (
                        metrics["max_drawdown"]
                        - baseline_metrics["max_drawdown"]
                    ),
                    "candidate_calmar": metrics["calmar"],
                    "drawdown_within_18pct": (
                        metrics["max_drawdown"] >= -0.18
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_sources(price_returns: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    sources: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}

    for ticker in ["CAOS", "WTMF", "FMF", "FLSP", "BTAL", "QAI"]:
        if ticker in price_returns:
            sources[ticker] = (
                price_returns[[ticker]].dropna(),
                {ticker: 1.0},
            )

    no_equity = ["CTA", "KMLM"]
    if set(no_equity).issubset(price_returns.columns):
        frame = price_returns[no_equity].dropna()
        sources["MF_NO_EQUITY"] = (frame, {ticker: 0.5 for ticker in no_equity})

    broad = ["CTA", "KMLM", "DBMF"]
    if set(broad).issubset(price_returns.columns):
        frame = price_returns[broad].dropna()
        sources["MF_BROAD"] = (frame, {ticker: 1.0 / len(broad) for ticker in broad})

    barbell = ["CAOS", "CTA", "KMLM"]
    if set(barbell).issubset(price_returns.columns):
        frame = price_returns[barbell].dropna()
        sources["FAST_SLOW_BARBELL"] = (
            frame,
            {"CAOS": 0.5, "CTA": 0.25, "KMLM": 0.25},
        )
    return sources


def _post_return_weights(target: np.ndarray, component_returns: np.ndarray, portfolio_return: float) -> np.ndarray:
    denominator = 1.0 + portfolio_return
    if denominator <= 0:
        return target.copy()
    return target * (1.0 + component_returns) / denominator


def simulate_overlay(
    baseline: pd.DataFrame,
    bil_returns: pd.Series,
    alt_returns: pd.DataFrame,
    alt_mix: dict[str, float],
    sleeve_weight: float,
    funding_mode: str,
) -> pd.DataFrame:
    columns = list(alt_mix)
    aligned = pd.concat(
        [
            baseline,
            bil_returns.rename("bil_return"),
            alt_returns[columns],
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    mix = np.array([alt_mix[column] for column in columns], dtype=float)
    mix = mix / mix.sum()
    spread_rates = np.array(
        [
            ((PRODUCTS[column].spread_pct or 0.20) / 100.0)
            if column in PRODUCTS
            else 0.0020
            for column in columns
        ]
    )

    candidate_returns: list[float] = []
    gross_returns: list[float] = []
    overlay_costs: list[float] = []
    effective_sleeves: list[float] = []
    previous_candidate_post: np.ndarray | None = None
    previous_baseline_post: np.ndarray | None = None

    for _, row in aligned.iterrows():
        cash_weight = float(row["cash_weight"])
        risky_target, cash_target, effective_sleeve = funding_targets(
            cash_weight,
            sleeve_weight,
            funding_mode,
        )
        alt_target = effective_sleeve * mix
        target = np.concatenate(([risky_target, cash_target], alt_target))

        baseline_target = np.concatenate(
            ([1.0 - cash_weight, cash_weight], np.zeros(len(columns)))
        )
        denominator = 1.0 - cash_weight
        if abs(denominator) < 1e-8:
            risky_return = 0.0
        else:
            risky_return = (
                float(row["gross_return"]) - cash_weight * float(row["bil_return"])
            ) / denominator
        component_returns = np.concatenate(
            (
                [risky_return, float(row["bil_return"])],
                row[columns].to_numpy(dtype=float),
            )
        )
        candidate_gross = float(np.dot(target, component_returns))

        if previous_candidate_post is None:
            previous_candidate_post = baseline_target.copy()
            previous_baseline_post = baseline_target.copy()

        candidate_delta = target - previous_candidate_post
        baseline_delta = baseline_target - previous_baseline_post
        candidate_turnover = 0.5 * float(np.abs(candidate_delta).sum())
        baseline_turnover = 0.5 * float(np.abs(baseline_delta).sum())
        incremental_turnover = max(0.0, candidate_turnover - baseline_turnover)

        alt_abs_trade = np.abs(candidate_delta[2:])
        alt_execution_cost = float(
            np.sum(alt_abs_trade * (0.0005 + spread_rates / 2.0))
        )
        non_alt_incremental = max(
            0.0,
            2.0 * incremental_turnover - float(alt_abs_trade.sum()),
        )
        overlay_cost = alt_execution_cost + non_alt_incremental * 0.00075

        baseline_internal_cost = max(
            0.0,
            float(row["gross_return"]) - float(row["net_return"]),
        )
        baseline_cost_scale = risky_target / denominator if abs(denominator) >= 1e-8 else 0.0
        candidate_net = (
            candidate_gross
            - baseline_internal_cost * max(0.0, baseline_cost_scale)
            - overlay_cost
        )

        candidate_returns.append(candidate_net)
        gross_returns.append(candidate_gross)
        overlay_costs.append(overlay_cost)
        effective_sleeves.append(effective_sleeve)

        previous_candidate_post = _post_return_weights(
            target,
            component_returns,
            candidate_gross,
        )
        base_component_return = float(
            np.dot(baseline_target, component_returns)
        )
        previous_baseline_post = _post_return_weights(
            baseline_target,
            component_returns,
            base_component_return,
        )

    result = aligned[["net_return"]].rename(columns={"net_return": "baseline_net"}).copy()
    result["candidate_gross"] = gross_returns
    result["candidate_net"] = candidate_returns
    result["overlay_cost"] = overlay_costs
    result["effective_sleeve"] = effective_sleeves
    return result


def period_slices(index: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    periods = {"full_live": index}
    for year in [2022, 2023, 2024, 2025]:
        subset = index[index >= pd.Timestamp(f"{year}-01-01")]
        if len(subset) >= 126:
            periods[f"{year}_present"] = subset
    return periods


def product_screen(
    price_returns: pd.DataFrame,
    growth_returns: pd.Series,
    bil_returns: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, product in PRODUCTS.items():
        if ticker not in price_returns:
            continue
        aligned = pd.concat(
            [
                price_returns[ticker].rename("asset"),
                growth_returns.rename("growth"),
                bil_returns.rename("bil"),
            ],
            axis=1,
        ).dropna()
        if aligned.empty:
            continue
        metrics = performance_metrics(aligned["asset"])
        beta = conditional_beta(aligned["asset"], aligned["growth"])
        relative_wealth = (
            (1.0 + aligned["asset"]).cumprod()
            / (1.0 + aligned["bil"]).cumprod()
        )
        years = len(aligned) / TRADING_DAYS
        excess_cagr = relative_wealth.iloc[-1] ** (1.0 / years) - 1.0
        execution_pass = (
            product.fee_pct <= 1.0
            and (product.spread_pct is None or product.spread_pct <= 0.20)
            and (product.aum_usd_m is None or product.aum_usd_m >= 100.0)
        )
        evidence_pass = years >= 3.0
        return_pass = excess_cagr > 0.0
        diversification_pass = beta["downside_beta"] < 0.25
        rows.append(
            {
                "ticker": ticker,
                "start": aligned.index.min().date().isoformat(),
                "end": aligned.index.max().date().isoformat(),
                **metrics,
                **beta,
                "excess_cagr_vs_bil": excess_cagr,
                "fee_pct": product.fee_pct,
                "spread_pct": product.spread_pct,
                "aum_usd_m": product.aum_usd_m,
                "execution_pass": execution_pass,
                "evidence_pass": evidence_pass,
                "return_pass": return_pass,
                "diversification_pass": diversification_pass,
                "all_quantitative_gates": (
                    execution_pass
                    and evidence_pass
                    and return_pass
                    and diversification_pass
                ),
                "mechanism": product.mechanism,
                "issue": product.issue,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["all_quantitative_gates", "sharpe"],
        ascending=[False, False],
    )


def evaluate_overlays(
    baseline: pd.DataFrame,
    bil_returns: pd.Series,
    sources: dict[str, tuple[pd.DataFrame, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    paths: dict[str, pd.DataFrame] = {}

    for source_name, (returns, mix) in sources.items():
        for sleeve_weight in [0.05, 0.10]:
            for funding_mode in ["cash_capped", "cash_first", "pro_rata"]:
                path = simulate_overlay(
                    baseline,
                    bil_returns,
                    returns,
                    mix,
                    sleeve_weight,
                    funding_mode,
                )
                if path.empty:
                    continue
                path_key = f"{source_name}_{int(sleeve_weight * 100)}_{funding_mode}"
                paths[path_key] = path

                for period_name, period_index in period_slices(path.index).items():
                    segment = path.loc[period_index]
                    base = performance_metrics(segment["baseline_net"])
                    candidate = performance_metrics(segment["candidate_net"])
                    rows.append(
                        {
                            "source": source_name,
                            "sleeve_weight": sleeve_weight,
                            "funding_mode": funding_mode,
                            "period": period_name,
                            "start": segment.index.min().date().isoformat(),
                            "end": segment.index.max().date().isoformat(),
                            "observations": len(segment),
                            "effective_sleeve_mean": segment["effective_sleeve"].mean(),
                            "annual_overlay_cost": segment["overlay_cost"].mean() * TRADING_DAYS,
                            "baseline_cagr": base["cagr"],
                            "candidate_cagr": candidate["cagr"],
                            "delta_cagr": candidate["cagr"] - base["cagr"],
                            "baseline_sharpe": base["sharpe"],
                            "candidate_sharpe": candidate["sharpe"],
                            "delta_sharpe": candidate["sharpe"] - base["sharpe"],
                            "baseline_max_drawdown": base["max_drawdown"],
                            "candidate_max_drawdown": candidate["max_drawdown"],
                            "delta_max_drawdown": (
                                candidate["max_drawdown"] - base["max_drawdown"]
                            ),
                            "baseline_calmar": base["calmar"],
                            "candidate_calmar": candidate["calmar"],
                            "delta_calmar": candidate["calmar"] - base["calmar"],
                        }
                    )

                for stress_name, (start, end) in STRESS_WINDOWS.items():
                    segment = path.loc[start:end]
                    if len(segment) < 2:
                        continue
                    base = performance_metrics(segment["baseline_net"])
                    candidate = performance_metrics(segment["candidate_net"])
                    stress_rows.append(
                        {
                            "source": source_name,
                            "sleeve_weight": sleeve_weight,
                            "funding_mode": funding_mode,
                            "stress": stress_name,
                            "start": segment.index.min().date().isoformat(),
                            "end": segment.index.max().date().isoformat(),
                            "observations": len(segment),
                            "baseline_total_return": base["total_return"],
                            "candidate_total_return": candidate["total_return"],
                            "delta_total_return": (
                                candidate["total_return"] - base["total_return"]
                            ),
                            "baseline_max_drawdown": base["max_drawdown"],
                            "candidate_max_drawdown": candidate["max_drawdown"],
                            "delta_max_drawdown": (
                                candidate["max_drawdown"] - base["max_drawdown"]
                            ),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(stress_rows), paths


def robustness_summary(overlay: pd.DataFrame) -> pd.DataFrame:
    full = overlay.loc[overlay["period"] == "full_live"].copy()
    subperiod = overlay.loc[overlay["period"] != "full_live"].copy()
    group_columns = ["source", "sleeve_weight", "funding_mode"]
    positive = (
        subperiod.groupby(group_columns)
        .agg(
            subperiods=("period", "nunique"),
            cagr_wins=("delta_cagr", lambda values: int((values > 0).sum())),
            sharpe_wins=("delta_sharpe", lambda values: int((values > 0).sum())),
            drawdown_wins=("delta_max_drawdown", lambda values: int((values > 0).sum())),
            worst_subperiod_delta_cagr=("delta_cagr", "min"),
            worst_subperiod_delta_sharpe=("delta_sharpe", "min"),
        )
        .reset_index()
    )
    result = full.merge(positive, on=group_columns, how="left")
    result["all_three_full_live_improve"] = (
        (result["delta_cagr"] > 0)
        & (result["delta_sharpe"] > 0)
        & (result["delta_max_drawdown"] > 0)
    )
    result["subperiod_cagr_win_rate"] = result["cagr_wins"] / result["subperiods"]
    result["subperiod_sharpe_win_rate"] = result["sharpe_wins"] / result["subperiods"]
    return result.sort_values(
        ["all_three_full_live_improve", "delta_calmar", "delta_cagr"],
        ascending=[False, False, False],
    )


def _percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{100.0 * value:.2f}%"


def write_report(
    product_results: pd.DataFrame,
    robustness: pd.DataFrame,
    stress: pd.DataFrame,
    gde_tracking: pd.DataFrame,
    gold_substitution: pd.DataFrame,
    gde_uncertainty: pd.DataFrame,
    gde_open: pd.DataFrame,
) -> None:
    candidates = robustness.loc[
        robustness["source"].isin(["CAOS", "MF_NO_EQUITY", "FAST_SLOW_BARBELL"])
    ].head(12)
    lines = [
        "# 散户可操作的独立收益源验证",
        "",
        "本报告只使用美国上市、可在普通券商账户交易的 ETF。生产策略未改动。",
        "所有袖套权重事前固定为 5% 或 10%，资金来源规则也事前固定；没有按回测结果搜索参数。",
        "",
        "## 产品层筛选",
        "",
        "| 产品 | 实盘期 | CAGR | Sharpe | 最大回撤 | 相对 BIL CAGR | 成长资产下行 beta | 费用 | 点差 | 量化门槛 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in product_results.iterrows():
        lines.append(
            "| {ticker} | {years:.1f} 年 | {cagr} | {sharpe:.2f} | {mdd} | {excess} | "
            "{down_beta:.2f} | {fee:.2f}% | {spread} | {gate} |".format(
                ticker=row["ticker"],
                years=row["years"],
                cagr=_percent(row["cagr"]),
                sharpe=row["sharpe"],
                mdd=_percent(row["max_drawdown"]),
                excess=_percent(row["excess_cagr_vs_bil"]),
                down_beta=row["downside_beta"],
                fee=row["fee_pct"],
                spread=(
                    "—"
                    if pd.isna(row["spread_pct"])
                    else f"{row['spread_pct']:.2f}%"
                ),
                gate="通过" if row["all_quantitative_gates"] else "未通过",
            )
        )

    lines.extend(
        [
            "",
            "量化门槛：费用不高于 1%、点差不高于 0.20%（已知时）、AUM 不低于 1 亿美元（已知时）、",
            "至少 3 年实盘、相对 BIL 的几何超额为正、成长资产最差 10% 交易日的 beta 低于 0.25。",
            "未知的点差/AUM 不被自动判负，但会在最终实盘判断中保留为尽调项。",
            "",
            "## 与当前策略组合后的关键结果",
            "",
            "| 收益源 | 袖套 | 资金来源 | 实盘起点 | ΔCAGR | ΔSharpe | Δ最大回撤 | ΔCalmar | 子期 CAGR 胜率 | 全部改善 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in candidates.iterrows():
        lines.append(
            "| {source} | {weight:.0f}% | {mode} | {start} | {dcagr} | {dsharpe:+.2f} | "
            "{dmdd} | {dcalmar:+.2f} | {win:.0f}% | {all_improve} |".format(
                source=row["source"],
                weight=100.0 * row["sleeve_weight"],
                mode=row["funding_mode"],
                start=row["start"],
                dcagr=_percent(row["delta_cagr"]),
                dsharpe=row["delta_sharpe"],
                dmdd=_percent(row["delta_max_drawdown"]),
                dcalmar=row["delta_calmar"],
                win=100.0 * row["subperiod_cagr_win_rate"],
                all_improve="是" if row["all_three_full_live_improve"] else "否",
            )
        )

    best = robustness.iloc[0] if not robustness.empty else None
    lines.extend(["", "## 机械结论", ""])
    if best is None:
        lines.append("没有足够数据形成结论。")
    else:
        lines.append(
            "按完整实盘重叠期的 Calmar 改善排序，第一名是 "
            f"`{best['source']}` / {best['sleeve_weight']:.0%} / `{best['funding_mode']}`："
            f"ΔCAGR {_percent(best['delta_cagr'])}，"
            f"ΔSharpe {best['delta_sharpe']:+.2f}，"
            f"Δ最大回撤 {_percent(best['delta_max_drawdown'])}。"
        )
        if bool(best["all_three_full_live_improve"]):
            lines.append(
                "它在同一重叠期同时改善了 CAGR、Sharpe 和最大回撤；"
                "仍需通过产品历史长度与跨压力期检验，不能据此直接晋升生产。"
            )
        else:
            lines.append(
                "没有证据表明最优候选能同时改善 CAGR、Sharpe 和最大回撤，"
                "因此当前结果不足以改生产策略。"
            )

    tracking = gde_tracking.iloc[0]
    long_gold = gold_substitution.loc[
        (gold_substitution["source"] == "GDE_SYNTHETIC")
        & (gold_substitution["period"] == "full_available")
    ].sort_values(["gate_mode", "substitution_fraction"])
    live_gold = gold_substitution.loc[
        (gold_substitution["source"] == "GDE_LIVE")
        & (gold_substitution["period"] == "full_available")
    ].sort_values(["gate_mode", "substitution_fraction"])
    lines.extend(
        [
            "",
            "## 资本效率验证：用 GDE 替代部分 GLD",
            "",
            "长历史代理固定为 `0.9×SPY + 0.9×GLD − 0.8×BIL`；"
            "它没有使用回测拟合参数，也没有把真实 GDE 的优异近期收益倒灌到历史。",
            f"在真实 GDE 上市后的 {tracking['years']:.1f} 年里，代理与 GDE 的日收益相关性为 "
            f"{tracking['correlation']:.3f}，年化跟踪误差 {_percent(tracking['tracking_error'])}，"
            f"真实 GDE 相对代理的 CAGR 差为 {_percent(tracking['annualized_cagr_difference'])}。",
            "",
            "| 数据 | 门控 | 替代 GOLD 比例 | 平均 GDE 权重 | 平均新增 SPY 名义敞口 | ΔCAGR | ΔSharpe | Δ最大回撤 | 候选最大回撤 | ≤18% |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in pd.concat([long_gold, live_gold], ignore_index=True).iterrows():
        lines.append(
            "| {source} | {gate_mode} | {fraction:.0f}% | {weight} | {spy} | {dcagr} | {dsharpe:+.2f} | "
            "{dmdd} | {mdd} | {pass_18} |".format(
                source=row["source"],
                gate_mode=row["gate_mode"],
                fraction=100.0 * row["substitution_fraction"],
                weight=_percent(row["average_gde_weight"]),
                spy=_percent(row["average_added_spy_notional"]),
                dcagr=_percent(row["delta_cagr"]),
                dsharpe=row["delta_sharpe"],
                dmdd=_percent(row["delta_max_drawdown"]),
                mdd=_percent(row["candidate_max_drawdown"]),
                pass_18="是" if row["drawdown_within_18pct"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "这里的关键风险不是 ETF 本身的价格波动，而是 90/90 结构增加了组合总名义敞口。",
            "因此即使 CAGR 上升，也只有在长历史、分段样本和 18% 回撤约束同时通过时才值得进入影子组合。",
            "",
            "### 真实产品跟踪误差的区块自助法",
            "",
            "| 门控 | 替代比例 | ΔCAGR 5%/中位/95% | 最大回撤 5%/中位/95% | P(ΔCAGR>0) | P(MDD≤18%) | P(CAGR↑/Sharpe≥1/MDD≤18%) | P(三项都优于基线) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in gde_uncertainty.sort_values(
        ["gate_mode", "substitution_fraction"]
    ).iterrows():
        lines.append(
            "| {gate} | {fraction:.0f}% | {cagr05}/{cagr50}/{cagr95} | "
            "{mdd05}/{mdd50}/{mdd95} | {pcagr} | {pmdd} | {pgoal} | {pall} |".format(
                gate=row["gate_mode"],
                fraction=100.0 * row["substitution_fraction"],
                cagr05=_percent(row["delta_cagr_p05"]),
                cagr50=_percent(row["delta_cagr_median"]),
                cagr95=_percent(row["delta_cagr_p95"]),
                mdd05=_percent(row["max_drawdown_p05"]),
                mdd50=_percent(row["max_drawdown_median"]),
                mdd95=_percent(row["max_drawdown_p95"]),
                pcagr=_percent(row["probability_positive_delta_cagr"]),
                pmdd=_percent(row["probability_drawdown_within_18pct"]),
                pgoal=_percent(row["probability_meet_user_objectives"]),
                pall=_percent(row["probability_all_objectives"]),
            )
        )
    lines.extend(
        [
            "",
            "自助法使用真实 GDE 相对融资正确代理的 21 个交易日残差区块，固定 1,000 次模拟。",
            "它保留短期跟踪误差聚集，但仍受限于真实残差只覆盖 2022 年以来。",
            "",
            "### 真实 GDE 的下一交易日开盘执行",
            "",
            "| 门控 | 替代比例 | 平均 GDE 目标 | CAGR | ΔCAGR | Sharpe | ΔSharpe | 最大回撤 | Δ最大回撤 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in gde_open.sort_values(
        ["gate_mode", "substitution_fraction"]
    ).iterrows():
        lines.append(
            "| {gate} | {fraction:.0f}% | {weight} | {cagr} | {dcagr} | "
            "{sharpe:.2f} | {dsharpe:+.2f} | {mdd} | {dmdd} |".format(
                gate=row["gate_mode"],
                fraction=100.0 * row["substitution_fraction"],
                weight=_percent(row["average_gde_target"]),
                cagr=_percent(row["candidate_cagr"]),
                dcagr=_percent(row["delta_cagr"]),
                sharpe=row["candidate_sharpe"],
                dsharpe=row["delta_sharpe"],
                mdd=_percent(row["candidate_max_drawdown"]),
                dmdd=_percent(row["delta_max_drawdown"]),
            )
        )
    lines.extend(
        [
            "",
            "开盘代理只覆盖 GDE 上市后的真实产品期，并按 GDE 30 bps 单边执行成本、"
            "其他资产 7.5 bps 单边成本计费。每次只在原模型发生交易时调整 GDE，"
            "不做日内或每日额外再平衡。",
        ]
    )

    shadow_long = gold_substitution.loc[
        (gold_substitution["source"] == "GDE_SYNTHETIC")
        & (gold_substitution["period"] == "full_available")
        & (gold_substitution["substitution_fraction"] == 0.50)
        & (gold_substitution["gate_mode"] == "growth_linked")
    ].iloc[0]
    shadow_open = gde_open.loc[
        (gde_open["substitution_fraction"] == 0.50)
        & (gde_open["gate_mode"] == "growth_linked")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## 最终决策与影子规则",
            "",
            "**生产策略不变。** 普通低相关 ETF 袖套没有产生经济上足够大的改善；"
            "GDE 候选有实质 CAGR 改善，但本质是资本效率/杠杆包装，不是独立 alpha。",
            "",
            "唯一进入影子监控的规则是：",
            "",
            "`GDE目标 = GOLD目标 × 50% × min(1, (SPX+QQQ+SEMIS最终目标)/80%)`",
            "",
            "`GLD目标 = GOLD目标 − GDE目标`；其余生产权重不变。"
            "只在原模型本来就交易时调整，GDE 使用限价单，不为追求成交跨越宽点差。",
            "",
            f"长历史融资正确代理：ΔCAGR {_percent(shadow_long['delta_cagr'])}，"
            f"Sharpe {shadow_long['candidate_sharpe']:.2f}，"
            f"最大回撤 {_percent(shadow_long['candidate_max_drawdown'])}。"
            f"真实 GDE 次日开盘期：ΔCAGR {_percent(shadow_open['delta_cagr'])}，"
            f"Sharpe {shadow_open['candidate_sharpe']:.2f}，"
            f"最大回撤 {_percent(shadow_open['candidate_max_drawdown'])}。",
            "",
            "选择 50% 而不是回测 CAGR 最高的 100%，是对 GDE 仅约四年实盘、"
            "9%左右年化跟踪误差和当前约0.55%点差的产品风险折扣；"
            "选择成长联动而不是永久叠加，是为了避免在生存模型已经降风险时重新塞回股票 beta。",
            "",
            "晋升生产前至少需要：前瞻影子成交记录确认单边总成本不高于30 bps；"
            "真实 GDE 的滚动跟踪误差不持续高于10%；候选按次日开盘口径继续保持"
            "Sharpe≥1、最大回撤≤18%，且不能只靠黄金与股票同步上涨获利。",
        ]
    )

    lines.extend(
        [
            "",
            "### 解释边界",
            "",
            "- CAOS 的 ETF 日频历史从 2023 年开始；2018/2020 只能引用其同策略前身基金 AVOLX 的官方材料，不能与 ETF 回测拼接。",
            "- CTA/KMLM 组合只能从 2022 年开始；它覆盖通胀熊市，但没有覆盖 2008 或 2020 的完整实盘。",
            "- ETF 费用已反映在复权净值中；组合换仓另外按产品点差与保守滑点计费。",
            "- 压力期是事前定义的诊断窗口，不用于选择权重。",
            "",
            "## 官方资料",
            "",
            "- CAOS: https://funds.alphaarchitect.com/caos/",
            "- CAOS/AVOLX continuity: https://www.sec.gov/Archives/edgar/data/1592900/000159290024000992/caossummaryprospectus497k.htm",
            "- CTA: https://www.simplify.us/etfs/cta-simplify-managed-futures-strategy-etf",
            "- KMLM: https://kraneshares.com/etf/kmlm/",
            "- DBMF: https://imgpfunds.com/im-dbi-managed-futures-strategy-etf/",
            "- WTMF: https://www.wisdomtree.com/us/products/alternative/wtmf",
            "- FLSP: https://www.franklintempleton.com/investments/options/exchange-traded-funds/products/28388/SINGLCLASS/franklin-systematic-style-premia-etf/FLSP",
            "- GDE: https://www.wisdomtree.com/us/products/capital-efficient/gde",
            "- NTSX: https://www.wisdomtree.com/us/products/capital-efficient/ntsx",
            "- Return Stacked ETFs: https://www.returnstackedetfs.com/",
        ]
    )

    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = download_adjusted_close(TICKERS)
    price_returns = prices.pct_change(fill_method=None)
    baseline = load_baseline()

    growth_prices = prices[["QQQ", "SMH"]].dropna()
    growth_returns = growth_prices.pct_change(fill_method=None).mean(axis=1)
    bil_returns = price_returns["BIL"]

    products = product_screen(price_returns, growth_returns, bil_returns)
    sources = build_candidate_sources(price_returns)
    overlays, stress, paths = evaluate_overlays(baseline, bil_returns, sources)
    robustness = robustness_summary(overlays)
    synthetic_gde = gde_synthetic_returns(price_returns)
    gde_tracking = gde_tracking_diagnostics(
        price_returns["GDE"],
        synthetic_gde,
    )
    gold_substitution = evaluate_gold_substitution(
        baseline,
        price_returns["GDE"],
        synthetic_gde,
        price_returns["GLD"],
    )
    gde_uncertainty = bootstrap_gold_substitution_uncertainty(
        baseline,
        price_returns["GDE"],
        synthetic_gde,
        price_returns["GLD"],
    )
    opens, closes = load_open_close_with_gde()
    gde_open = evaluate_gold_substitution_open(opens, closes)

    products.to_csv(OUTPUT_DIR / "product_screen.csv", index=False)
    overlays.to_csv(OUTPUT_DIR / "overlay_periods.csv", index=False)
    stress.to_csv(OUTPUT_DIR / "stress_windows.csv", index=False)
    robustness.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
    gde_tracking.to_csv(OUTPUT_DIR / "gde_tracking.csv", index=False)
    gold_substitution.to_csv(OUTPUT_DIR / "gold_substitution.csv", index=False)
    gde_uncertainty.to_csv(
        OUTPUT_DIR / "gde_bootstrap_uncertainty.csv",
        index=False,
    )
    gde_open.to_csv(OUTPUT_DIR / "gde_open_execution.csv", index=False)
    if paths:
        wealth = pd.concat(
            {
                name: (1.0 + path["candidate_net"]).cumprod()
                for name, path in paths.items()
            },
            axis=1,
            sort=False,
        )
        wealth.to_csv(OUTPUT_DIR / "candidate_wealth.csv", index_label="date")
    write_report(
        products,
        robustness,
        stress,
        gde_tracking,
        gold_substitution,
        gde_uncertainty,
        gde_open,
    )

    best_columns = [
        "source",
        "sleeve_weight",
        "funding_mode",
        "start",
        "delta_cagr",
        "delta_sharpe",
        "delta_max_drawdown",
        "delta_calmar",
        "subperiod_cagr_win_rate",
        "all_three_full_live_improve",
    ]
    print(robustness[best_columns].head(15).to_string(index=False))
    print("\nGold substitution, full available periods:")
    gold_columns = [
        "source",
        "substitution_fraction",
        "gate_mode",
        "start",
        "delta_cagr",
        "delta_sharpe",
        "delta_max_drawdown",
        "candidate_max_drawdown",
        "average_added_spy_notional",
        "drawdown_within_18pct",
    ]
    print(
        gold_substitution.loc[
            gold_substitution["period"] == "full_available",
            gold_columns,
        ].to_string(index=False)
    )
    print("\nGDE next-open execution:")
    print(
        gde_open[
            [
                "substitution_fraction",
                "gate_mode",
                "average_gde_target",
                "candidate_cagr",
                "delta_cagr",
                "candidate_sharpe",
                "delta_sharpe",
                "candidate_max_drawdown",
                "delta_max_drawdown",
            ]
        ].to_string(index=False)
    )
    print("\nGDE live-residual bootstrap:")
    print(
        gde_uncertainty[
            [
                "substitution_fraction",
                "gate_mode",
                "delta_cagr_p05",
                "delta_cagr_median",
                "max_drawdown_p05",
                "max_drawdown_median",
                "probability_positive_delta_cagr",
                "probability_drawdown_within_18pct",
                "probability_meet_user_objectives",
                "probability_all_objectives",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
