from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.data import download_csv_history


ROOT = Path(__file__).resolve().parents[1]
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX3M_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
TRADABLES = {
    "SPX": "SPY",
    "QQQ": "QQQ",
    "SEMIS": "SMH",
    "BOND": "IEF",
    "GOLD": "GLD",
    "OIL": "DBC",
    "USD": "UUP",
    "CASH": "BIL",
    "VIX_HEDGE": "VIXY",
}
PERFORMANCE_STARTS = {
    "1_month": ("2026-05-29", 1),
    "3_month": ("2026-03-31", 1),
    "quarter_to_date": ("2026-03-31", 1),
    "year_to_date": ("2025-12-31", 1),
    "1_year": ("2025-06-30", 1),
    "3_year": ("2023-06-30", 3),
    "5_year": ("2021-06-30", 5),
    "10_year": ("2016-06-30", 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every R9 data dependency.")
    parser.add_argument(
        "--output-dir",
        default="output/r9_data_audit_2026-07-25",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.astype(float).sort_index()


def maximum_flat_run(values: pd.Series) -> int:
    same = values.eq(values.shift())
    groups = same.ne(same.shift()).cumsum()
    runs = same.groupby(groups).sum()
    return int(runs.max()) if not runs.empty else 0


def integrity_rows(label: str, path: Path, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame:
        returns = frame[column].pct_change(fill_method=None)
        largest_date = returns.abs().idxmax()
        rows.append(
            {
                "dataset": label,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "rows": len(frame),
                "first_date": frame.index[0].date().isoformat(),
                "last_date": frame.index[-1].date().isoformat(),
                "duplicate_dates": int(frame.index.duplicated().sum()),
                "monotonic_dates": bool(frame.index.is_monotonic_increasing),
                "column": column,
                "missing_values": int(frame[column].isna().sum()),
                "nonpositive_values": int((frame[column] <= 0.0).sum()),
                "maximum_absolute_daily_return": float(returns.abs().max()),
                "maximum_return_date": largest_date.date().isoformat(),
                "maximum_flat_run_days": maximum_flat_run(frame[column]),
            }
        )
    return rows


def cboe_close(url: str) -> pd.Series:
    history = download_csv_history(url)
    dates = pd.to_datetime(history["DATE"], errors="coerce")
    values = pd.to_numeric(history["CLOSE"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna()
    series = series[series.index.notna()]
    return series[~series.index.duplicated(keep="last")].sort_index()


def official_signal_rows(
    production: pd.DataFrame,
    official: dict[str, pd.Series],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for signal, source in official.items():
        common = production.index.intersection(source.index)
        difference = production.loc[common, signal] - source.loc[common]
        rows.append(
            {
                "signal": signal,
                "overlap_rows": len(common),
                "official_first_date": source.index[0].date().isoformat(),
                "official_last_date": source.index[-1].date().isoformat(),
                "cache_first_date": production.index[0].date().isoformat(),
                "cache_last_date": production.index[-1].date().isoformat(),
                "maximum_absolute_difference": float(difference.abs().max()),
                "differences_over_0_01": int((difference.abs() > 0.01).sum()),
                "exactly_equal_rows": int((difference.abs() <= 1.0e-12).sum()),
            }
        )
    return rows


def cross_cache_rows(
    production: pd.DataFrame,
    open_close: pd.DataFrame,
    proxy: pd.DataFrame,
    proxy_metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    close = open_close.filter(like="close_").rename(
        columns=lambda value: value.removeprefix("close_")
    )
    common = production.index.intersection(close.index)
    for asset in TRADABLES:
        relative = (
            production.loc[common, asset] / close.loc[common, asset] - 1.0
        ).abs()
        rows.append(
            {
                "comparison": "production_close_vs_open_close_cache",
                "asset": asset,
                "overlap_rows": len(common),
                "maximum_absolute_relative_difference": float(relative.max()),
                "differences_over_1bp": int((relative > 0.0001).sum()),
            }
        )

    splices = proxy_metadata["splices"]
    for asset in [*TRADABLES, "VIX", "VIX3M"]:
        common = production.index.intersection(proxy.index)
        splice = splices.get(asset) if isinstance(splices, dict) else None
        if isinstance(splice, dict) and splice.get("actual_start"):
            common = common[common >= pd.Timestamp(str(splice["actual_start"]))]
        relative = (
            production.loc[common, asset] / proxy.loc[common, asset] - 1.0
        ).abs()
        rows.append(
            {
                "comparison": "production_vs_long_history_after_actual_start",
                "asset": asset,
                "overlap_rows": len(common),
                "maximum_absolute_relative_difference": float(relative.max()),
                "differences_over_1bp": int((relative > 0.0001).sum()),
            }
        )
    return rows


def proxy_splice_rows(
    proxy: pd.DataFrame,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    returns = proxy.pct_change(fill_method=None)
    for asset, details in metadata["splices"].items():
        start = pd.Timestamp(str(details["actual_start"]))
        if start not in proxy.index:
            continue
        window = returns.loc[start - pd.Timedelta(days=60) : start + pd.Timedelta(days=60), asset]
        splice_return = float(returns.loc[start, asset])
        percentile = float((window.abs() <= abs(splice_return)).mean())
        rows.append(
            {
                "asset": asset,
                "actual_start": start.date().isoformat(),
                "splice_day_return": splice_return,
                "absolute_return_percentile_in_120_calendar_day_window": percentile,
                "proxy_description": details["proxy"],
                "actual_description": details["actual"],
            }
        )
    return rows


def result_impact_rows(
    label: str,
    before_dir: Path,
    after_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    before_metrics = pd.read_csv(before_dir / "metrics.csv", index_col=0).loc["ENSEMBLE"]
    after_metrics = pd.read_csv(after_dir / "metrics.csv", index_col=0).loc["ENSEMBLE"]
    for metric in (
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "average_daily_turnover",
        "annualized_cost_drag",
    ):
        rows.append(
            {
                "sample": label,
                "measure": metric,
                "before": float(before_metrics[metric]),
                "after": float(after_metrics[metric]),
                "change": float(after_metrics[metric] - before_metrics[metric]),
            }
        )

    before_weights = pd.read_csv(before_dir / "weights.csv", index_col=0, parse_dates=True)
    after_weights = pd.read_csv(after_dir / "weights.csv", index_col=0, parse_dates=True)
    common = before_weights.index.intersection(after_weights.index)
    weight_difference = (
        before_weights.loc[common] - after_weights.loc[common]
    ).abs()
    rows.append(
        {
            "sample": label,
            "measure": "maximum_absolute_weight_change",
            "before": np.nan,
            "after": np.nan,
            "change": float(weight_difference.to_numpy().max()),
        }
    )
    rows.append(
        {
            "sample": label,
            "measure": "dates_with_weight_change_over_1e_8",
            "before": np.nan,
            "after": np.nan,
            "change": int((weight_difference.max(axis=1) > 1.0e-8).sum()),
        }
    )
    return rows


def period_impact_rows(destination: Path) -> list[dict[str, object]]:
    variants = {
        "previous_proxy": destination / "stable_schedule_before_proxy",
        "signal_only_corrected": destination / "stable_schedule_signal_only_proxy",
        "all_data_corrected_common_end": (
            destination / "stable_schedule_corrected_proxy_common_end"
        ),
    }
    periods = {
        "pre_official_vix3m": ("2006-08-01", "2009-09-17"),
        "financial_crisis": ("2007-10-09", "2009-03-09"),
        "post_official_vix3m": ("2009-09-18", "2026-07-23"),
    }
    rows: list[dict[str, object]] = []
    for variant, directory in variants.items():
        daily = pd.read_csv(
            directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        for period, (start, end) in periods.items():
            returns = daily.loc[start:end, "net_return"].dropna()
            equity = (1.0 + returns).cumprod()
            years = len(returns) / 252.0
            rows.append(
                {
                    "variant": variant,
                    "period": period,
                    "start": returns.index[0].date().isoformat(),
                    "end": returns.index[-1].date().isoformat(),
                    "total_return": float(equity.iloc[-1] - 1.0),
                    "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
                    "max_drawdown": float(
                        (equity / equity.cummax() - 1.0).min()
                    ),
                }
            )
    return rows


def source_manifest() -> pd.DataFrame:
    rows = [
        {
            "dataset": "production",
            "internal_name": asset,
            "external_symbol": ticker,
            "source": "Yahoo Finance via yfinance",
            "adjustment": "auto_adjust=True",
            "role": "tradable return and/or market-regime input",
            "limitation": "",
        }
        for asset, ticker in TRADABLES.items()
    ]
    rows.extend(
        [
            {
                "dataset": "production",
                "internal_name": "VIX",
                "external_symbol": "VIX",
                "source": VIX_URL,
                "adjustment": "Cboe official close",
                "role": "defense signal only; not tradable",
                "limitation": "",
            },
            {
                "dataset": "production",
                "internal_name": "VIX3M",
                "external_symbol": "VIX3M",
                "source": VIX3M_URL,
                "adjustment": "Cboe official close",
                "role": "defense signal only; not tradable",
                "limitation": "official daily history begins 2009-09-18",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "SEMIS",
                "external_symbol": "SOXX -> SMH",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "level-scaled splice",
                "role": "historical stress test only",
                "limitation": "SOXX proxy before SMH fund inception",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "GOLD",
                "external_symbol": "GC=F -> GLD",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "level-scaled splice",
                "role": "historical stress test only",
                "limitation": "gold futures proxy before GLD inception",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "OIL",
                "external_symbol": "^SPGSCI -> DBC",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "level-scaled splice",
                "role": "historical stress test only",
                "limitation": "broad commodity index proxy before DBC inception",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "USD",
                "external_symbol": "DX-Y.NYB -> UUP",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "level-scaled splice",
                "role": "historical stress test only",
                "limitation": "dollar index proxy before UUP inception",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "CASH",
                "external_symbol": "^IRX -> BIL",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "synthetic yield index, then level-scaled splice",
                "role": "historical stress test only",
                "limitation": "synthetic cash return before BIL inception",
            },
            {
                "dataset": "long_history_proxy",
                "internal_name": "VIX_HEDGE",
                "external_symbol": "flat -> VIXY",
                "source": "Yahoo Finance via yfinance",
                "adjustment": "flat before fund inception",
                "role": "historical stress test only",
                "limitation": "no hypothetical hedge payoff before VIXY inception",
            },
            {
                "dataset": "open_execution",
                "internal_name": "all tradable assets",
                "external_symbol": ", ".join(TRADABLES.values()),
                "source": "Yahoo Finance via yfinance",
                "adjustment": "auto_adjust=True Open and Close",
                "role": "next-open execution validation only",
                "limitation": "vendor data; reconciled to production adjusted closes",
            },
        ]
    )
    return pd.DataFrame(rows)


def latest_close_reconciliation(production: pd.DataFrame) -> pd.DataFrame:
    references = [
        ("SPX", 738.18, "fund issuer", "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy"),
        ("QQQ", 691.96, "exchange-labeled market feed", "https://markets.financialcontent.com/chroniclejournal/quote/detailedquote?Symbol=NQ%3AQQQ"),
        ("SEMIS", 580.17, "exchange-labeled market feed", "https://markets.financialcontent.com/stocks/quote/detailedquote?Symbol=NQ%3ASMH"),
        ("BOND", 92.85, "fund issuer", "https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf"),
        ("GOLD", 371.52, "fund issuer", "https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld"),
        ("OIL", 30.31, "independent market feed", "https://www.macrotrends.net/stocks/charts/DBC/invesco-db-commodity-index-tracking-etf/stock-price-history"),
        ("USD", 28.56, "independent market feed", "https://www.etfcentral.com/fund/UUP"),
        ("CASH", 91.58, "fund issuer", "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil"),
        ("VIX_HEDGE", 21.78, "fund issuer", "https://prod.proshares.com/our-etfs/strategic/vixy"),
    ]
    as_of = pd.Timestamp("2026-07-23")
    rows = []
    for asset, reference, authority, url in references:
        local = float(production.loc[as_of, asset])
        rows.append(
            {
                "asset": asset,
                "as_of": as_of.date().isoformat(),
                "local_close": local,
                "reference_close": reference,
                "absolute_difference": abs(local - reference),
                "reference_type": authority,
                "source_url": url,
            }
        )
    return pd.DataFrame(rows)


def issuer_performance_reconciliation(production: pd.DataFrame) -> pd.DataFrame:
    sources = {
        "SPX": (
            "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy",
            {
                "1_month": -1.07,
                "quarter_to_date": 15.08,
                "year_to_date": 10.02,
                "1_year": 22.08,
                "3_year": 20.44,
                "5_year": 13.25,
                "10_year": 15.34,
            },
        ),
        "SEMIS": (
            "https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/performance/",
            {
                "1_month": 9.51,
                "3_month": 71.07,
                "year_to_date": 82.13,
                "1_year": 135.90,
                "3_year": 63.44,
                "5_year": 38.81,
                "10_year": 38.10,
            },
        ),
        "BOND": (
            "https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf",
            {
                "1_month": 0.25,
                "3_month": 0.08,
                "year_to_date": -0.05,
                "1_year": 2.60,
                "3_year": 2.93,
                "5_year": -1.14,
                "10_year": 0.51,
            },
        ),
        "GOLD": (
            "https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld",
            {
                "1_month": -11.68,
                "quarter_to_date": -14.39,
                "year_to_date": -7.05,
                "1_year": 20.85,
                "3_year": 27.34,
                "5_year": 17.33,
                "10_year": 11.27,
            },
        ),
        "CASH": (
            "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil",
            {
                "1_month": 0.28,
                "quarter_to_date": 0.88,
                "year_to_date": 1.74,
                "1_year": 3.82,
                "3_year": 4.58,
                "5_year": 3.46,
                "10_year": 2.20,
            },
        ),
        "VIX_HEDGE": (
            "https://prod.proshares.com/our-etfs/strategic/vixy",
            {
                "1_month": -8.59,
                "3_month": -38.02,
                "year_to_date": -16.97,
                "1_year": -54.59,
                "3_year": -40.26,
                "5_year": -46.41,
                "10_year": -47.98,
            },
        ),
    }
    as_of = pd.Timestamp("2026-06-30")
    rows = []
    for asset, (url, published) in sources.items():
        for period, published_percent in published.items():
            start_date, years = PERFORMANCE_STARTS[period]
            growth = float(
                production.loc[as_of, asset]
                / production.loc[pd.Timestamp(start_date), asset]
            )
            local_percent = (
                (growth ** (1.0 / years) - 1.0) * 100.0
                if years > 1
                else (growth - 1.0) * 100.0
            )
            rows.append(
                {
                    "asset": asset,
                    "as_of": as_of.date().isoformat(),
                    "period": period,
                    "local_adjusted_close_return_percent": local_percent,
                    "issuer_market_price_return_percent": published_percent,
                    "difference_percentage_points": (
                        local_percent - published_percent
                    ),
                    "source_url": url,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    destination = ROOT / args.output_dir
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "production": ROOT / "data/prices_vix_hedge.csv",
        "long_history_proxy": ROOT / "data/prices_20y_proxy.csv",
        "open_execution": ROOT / "data/adjusted_open_close_2011_present.csv",
    }
    frames = {label: load_prices(path) for label, path in paths.items()}
    proxy_metadata = json.loads(
        (ROOT / "output/past_20y_drawdown_backtest/proxy_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    source_manifest().to_csv(destination / "source_manifest.csv", index=False)
    latest_close_reconciliation(frames["production"]).to_csv(
        destination / "latest_close_reconciliation.csv",
        index=False,
    )
    issuer_performance_reconciliation(frames["production"]).to_csv(
        destination / "issuer_performance_reconciliation.csv",
        index=False,
    )
    integrity = pd.DataFrame(
        [
            row
            for label, frame in frames.items()
            for row in integrity_rows(label, paths[label], frame)
        ]
    )
    integrity.to_csv(destination / "integrity_checks.csv", index=False)

    official = {
        "VIX": cboe_close(VIX_URL),
        "VIX3M": cboe_close(VIX3M_URL),
    }
    official_comparison = pd.DataFrame(
        official_signal_rows(frames["production"], official)
    )
    official_comparison.to_csv(
        destination / "official_signal_comparison.csv",
        index=False,
    )

    old_production = load_prices(
        destination / "before/prices_vix_hedge.csv"
    )
    pd.DataFrame(
        official_signal_rows(old_production, official)
    ).to_csv(
        destination / "previous_signal_comparison.csv",
        index=False,
    )
    common = old_production.index.intersection(frames["production"].index)
    old_backwardation = (
        old_production.loc[common, "VIX"] / old_production.loc[common, "VIX3M"]
        > 1.0
    )
    new_backwardation = (
        frames["production"].loc[common, "VIX"]
        / frames["production"].loc[common, "VIX3M"]
        > 1.0
    )
    official_comparison["backwardation_state_changes_vs_previous_cache"] = [
        int((old_backwardation != new_backwardation).sum()),
        int((old_backwardation != new_backwardation).sum()),
    ]
    official_comparison.to_csv(
        destination / "official_signal_comparison.csv",
        index=False,
    )

    cross_cache = pd.DataFrame(
        cross_cache_rows(
            frames["production"],
            frames["open_execution"],
            frames["long_history_proxy"],
            proxy_metadata,
        )
    )
    cross_cache.to_csv(destination / "cross_cache_consistency.csv", index=False)

    splice_checks = pd.DataFrame(
        proxy_splice_rows(frames["long_history_proxy"], proxy_metadata)
    )
    splice_checks.to_csv(destination / "proxy_splice_checks.csv", index=False)

    vix3m_start = pd.Timestamp(
        proxy_metadata["splices"]["VIX3M"]["actual_start"]
    )
    neutral_history = frames["long_history_proxy"].loc[
        frames["long_history_proxy"].index < vix3m_start
    ]
    neutral_error = (
        neutral_history["VIX3M"] / neutral_history["VIX"] - 1.0
    ).abs()

    impact = pd.DataFrame(
        [
            *result_impact_rows(
                "production_2015_present",
                destination / "stable_schedule_before_data",
                destination / "stable_schedule_corrected",
            ),
            *result_impact_rows(
                "long_history_all_data_common_end",
                destination / "stable_schedule_before_proxy",
                destination / "stable_schedule_corrected_proxy_common_end",
            ),
            *result_impact_rows(
                "long_history_signal_only_common_end",
                destination / "stable_schedule_before_proxy",
                destination / "stable_schedule_signal_only_proxy",
            ),
        ]
    )
    impact.to_csv(destination / "strategy_impact.csv", index=False)
    pd.DataFrame(period_impact_rows(destination)).to_csv(
        destination / "historical_impact_by_period.csv",
        index=False,
    )

    summary = {
        "audit_as_of": pd.Timestamp.now(tz="UTC").isoformat(),
        "datasets": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "rows": len(frames[label]),
                "first_date": frames[label].index[0].date().isoformat(),
                "last_date": frames[label].index[-1].date().isoformat(),
            }
            for label, path in paths.items()
        },
        "integrity_failures": int(
            (
                (integrity["duplicate_dates"] > 0)
                | (~integrity["monotonic_dates"])
                | (integrity["missing_values"] > 0)
                | (integrity["nonpositive_values"] > 0)
            ).sum()
        ),
        "official_signal_differences_over_0_01": int(
            official_comparison["differences_over_0_01"].sum()
        ),
        "production_backwardation_state_changes_vs_previous_cache": int(
            (old_backwardation != new_backwardation).sum()
        ),
        "maximum_open_close_relative_difference": float(
            cross_cache.loc[
                cross_cache["comparison"].eq(
                    "production_close_vs_open_close_cache"
                ),
                "maximum_absolute_relative_difference",
            ].max()
        ),
        "maximum_pre_vix3m_official_neutral_error": float(neutral_error.max()),
    }
    (destination / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
