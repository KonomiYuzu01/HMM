from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from regime_strategy.data import completed_us_daily_prices
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/open_execution_validation")
CACHE = Path("data/adjusted_open_close_2011_present.csv")
TICKERS = {
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
STRATEGIES = {
    "robust_baseline": "paper_core_robust_vol_guarded_floor_ensemble",
    "growth_primary": "paper_core_zero_entry_growth_reallocation_ensemble",
    "gold_20_unlevered": "paper_core_zero_entry_growth_gold20_ensemble",
    "gold_leverage_candidate": "paper_core_zero_entry_growth_gold20_lev110_ensemble",
    "gold_leverage_entry_cap": "paper_core_zero_entry_growth_gold20_lev110_accountcap_ensemble",
    "full_asset_volatility_cap": "paper_core_asset_aware_vol_budget_ensemble",
    "tail_asset_volatility_cap": "paper_core_tail_only_asset_risk_ensemble",
    "scheduled_asset_volatility_cap": "paper_core_tail_only_scheduled_ensemble",
    "daily_asset_volatility_cap": "paper_core_daily_risk_only_ensemble",
    "gold_20_daily_asset_cap": "paper_core_growth_gold20_daily_risk_ensemble",
    "gold_20_daily_asset_cap_cost0": "paper_core_growth_gold20_daily_risk_ensemble",
    "gold_20_lev110_daily_asset_cap": "paper_core_growth_gold20_lev110_daily_risk_ensemble",
    "gold_20_daily_asset_cap_cost15": "paper_core_growth_gold20_daily_risk_ensemble_cost15",
    "gold_20_daily_asset_cap_cost18": "paper_core_growth_gold20_daily_risk_ensemble",
    "gold_20_daily_asset_cap_cost19": "paper_core_growth_gold20_daily_risk_ensemble",
    "gold_20_daily_asset_cap_cost25": "paper_core_growth_gold20_daily_risk_ensemble",
    "gold_20_daily_asset_cap_2012": "paper_core_growth_gold20_daily_risk_ensemble_2012",
    "gold_20_daily_asset_cap_vol18": "paper_core_growth_gold20_daily_risk_vol18_ensemble",
    "gold_20_daily_asset_cap_vol22": "paper_core_growth_gold20_daily_risk_vol22_ensemble",
    "gold_20_symmetric_daily_cap": "paper_core_growth_gold20_symmetric_daily_risk_ensemble",
    "gold_20_symmetric_daily_cap_cost15": "paper_core_growth_gold20_symmetric_daily_risk_ensemble",
    "gold_20_symmetric_daily_cap_cost25": "paper_core_growth_gold20_symmetric_daily_risk_ensemble",
    "gold_20_daily_reallocation": "paper_core_growth_gold20_daily_reallocation_ensemble",
    "gold_20_daily_reallocation_cost15": "paper_core_growth_gold20_daily_reallocation_ensemble",
    "gold_20_daily_member_seed7": "paper_core_growth_gold20_daily_risk_ensemble/members/seed_7",
    "gold_20_daily_member_seed42": "paper_core_growth_gold20_daily_risk_ensemble/members/seed_42",
    "gold_20_daily_member_seed123": "paper_core_growth_gold20_daily_risk_ensemble/members/seed_123",
    "gold_20_jump_aware_daily_cap": "paper_core_growth_gold20_jump_aware_daily_risk_ensemble",
    "gold_20_jump_aware_daily_cap_cost15": "paper_core_growth_gold20_jump_aware_daily_risk_ensemble",
    "gold_20_jump_aware_daily_cap_2012": "paper_core_growth_gold20_jump_aware_daily_risk_ensemble_2012",
    "dynamic_defensive_jump_daily_cap": "paper_core_growth_dynamic_defensive_jump_daily_risk_ensemble",
    "jump_aware_notrade2": "paper_core_growth_gold20_jump_daily_risk_notrade2_ensemble",
    "jump_aware_notrade3": "paper_core_growth_gold20_jump_daily_risk_notrade3_ensemble",
}


def scenario_cost_bps(label: str) -> float:
    match = re.search(r"_cost(\d+(?:p\d+)?)$", label)
    if match:
        return float(match.group(1).replace("p", "."))
    return 7.5


def load_open_close(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_modified_at: pd.Timestamp | None = None
    if CACHE.exists() and not refresh:
        frame = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        source_modified_at = pd.Timestamp.fromtimestamp(
            CACHE.stat().st_mtime,
            tz="America/New_York",
        )
    else:
        end_date = (
            pd.Timestamp.now(tz="America/New_York").normalize()
            + pd.Timedelta(days=2)
        ).date().isoformat()
        raw = yf.download(
            list(TICKERS.values()),
            start="2011-12-01",
            end=end_date,
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            raise RuntimeError("Yahoo did not return multi-asset Open/Close data")
        symbol_to_asset = {symbol: asset for asset, symbol in TICKERS.items()}
        open_prices = raw["Open"].rename(columns=symbol_to_asset).reindex(
            columns=list(TICKERS)
        )
        close_prices = raw["Close"].rename(columns=symbol_to_asset).reindex(
            columns=list(TICKERS)
        )
        missing_vixy = (
            open_prices["VIX_HEDGE"].isna()
            & close_prices["VIX_HEDGE"].isna()
        )
        vixy_flat_proxy = close_prices["VIX_HEDGE"].ffill()
        open_prices.loc[missing_vixy, "VIX_HEDGE"] = vixy_flat_proxy.loc[
            missing_vixy
        ]
        close_prices.loc[missing_vixy, "VIX_HEDGE"] = vixy_flat_proxy.loc[
            missing_vixy
        ]
        frame = pd.concat(
            {
                "open": open_prices,
                "close": close_prices,
            },
            axis=1,
        ).dropna(how="any")
        frame.columns = [f"{field}_{asset}" for field, asset in frame.columns]
        downloaded_at = pd.Timestamp.now(tz="America/New_York")
        frame = completed_us_daily_prices(
            frame,
            now=downloaded_at,
            source_modified_at=downloaded_at,
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(CACHE, index_label="date")
        source_modified_at = downloaded_at
        metadata = {
            "market_data_source": "Yahoo Finance via yfinance",
            "market_data_adjustment": "auto_adjust=True",
            "ticker_map": TICKERS,
            "first_date": frame.index[0].date().isoformat(),
            "price_as_of": frame.index[-1].date().isoformat(),
            "row_count": len(frame),
            "cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
            "flat_proxy_bars": {
                "VIX_HEDGE": [
                    date.date().isoformat()
                    for date in open_prices.index[missing_vixy]
                ]
            },
        }
        CACHE.with_suffix(CACHE.suffix + ".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    frame = completed_us_daily_prices(
        frame,
        source_modified_at=source_modified_at,
    )
    opens = frame[[f"open_{asset}" for asset in TICKERS]].copy()
    closes = frame[[f"close_{asset}" for asset in TICKERS]].copy()
    opens.columns = list(TICKERS)
    closes.columns = list(TICKERS)
    return opens.astype(float), closes.astype(float)


def simulate_open_execution(
    strategy_directory: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: str | None = "2025-12-31",
    cost_bps: float = 7.5,
    financing_spread_bps: float = 100.0,
    short_borrow_spread_bps: float = 100.0,
) -> pd.DataFrame:
    source = Path("output") / strategy_directory
    weights = pd.read_csv(source / "weights.csv", index_col=0, parse_dates=True)
    daily = pd.read_csv(source / "daily_returns.csv", index_col=0, parse_dates=True)
    dates = (
        weights.index.intersection(daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    if end_date is not None:
        dates = dates[dates <= end_date]
    assets = list(weights.columns)
    current = np.zeros(len(assets), dtype=float)
    current[assets.index("CASH")] = 1.0
    rows = []
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
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )
        traded = float(daily.loc[date, "turnover"]) > 1e-14
        open_turnover = 0.0
        if traded:
            target = weights.loc[date, assets].to_numpy(dtype=float)
            open_turnover = 0.5 * float(np.abs(target - current).sum())
            current = target
        intraday_asset_returns = (
            closes.loc[date, assets].to_numpy(dtype=float)
            / opens.loc[date, assets].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        trading_cost = 2.0 * open_turnover * cost_bps / 10_000.0
        financing_cost = (
            max(-float(current[assets.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / 252.0
        )
        risky = np.ones(len(current), dtype=bool)
        risky[assets.index("CASH")] = False
        financing_cost += (
            float(np.maximum(-current[risky], 0.0).sum())
            * short_borrow_spread_bps
            / 10_000.0
            / 252.0
        )
        cost = trading_cost + financing_cost
        net_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0 - cost
        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth > 1e-12:
            current = current * (1.0 + intraday_asset_returns) / gross_intraday_growth
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "overnight_return_before_trade": overnight_return,
                "intraday_return_after_trade": intraday_return,
                "cost": cost,
                "open_turnover": open_turnover,
                "model_reported_cost": float(daily.loc[date, "cost"]),
                "traded_at_open": int(traded),
            }
        )
        previous_date = date
    frame = pd.DataFrame(rows).set_index("date")
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(args.refresh)
    metric_rows = []
    for label, directory in STRATEGIES.items():
        start_date = "2012-01-01" if label.endswith("_2012") else "2015-01-01"
        simulated = simulate_open_execution(
            directory,
            opens,
            closes,
            start_date=start_date,
            cost_bps=scenario_cost_bps(label),
        )
        simulated.to_csv(DESTINATION / f"{label}_daily.csv")
        reported = pd.read_csv(
            Path("output") / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        ).loc[simulated.index, "net_return"]
        for execution, returns in (
            ("reported_close_to_close", reported),
            ("open_execution_proxy", simulated["net_return"]),
        ):
            metric_rows.append(
                {
                    "strategy": label,
                    "execution": execution,
                    **performance_metrics(returns),
                }
            )
    benchmark_returns = closes[["SPX", "QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    for benchmark in benchmark_returns:
        for sample, start in (
            ("complete_2015_2025", "2015-01-01"),
            ("extended_2012_2025", "2012-01-01"),
        ):
            metric_rows.append(
                {
                    "strategy": f"benchmark_{benchmark}_{sample}",
                    "execution": "buy_and_hold",
                    **performance_metrics(
                        benchmark_returns.loc[start:"2025-12-31", benchmark]
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows).set_index(["strategy", "execution"])
    metrics.to_csv(DESTINATION / "metrics.csv")
    trough_rows = []
    for label in STRATEGIES:
        simulated = pd.read_csv(
            DESTINATION / f"{label}_daily.csv", index_col=0, parse_dates=True
        )
        trough_date = simulated["drawdown"].idxmin()
        trough_rows.append(
            {
                "strategy": label,
                "trough_date": trough_date,
                "max_drawdown": simulated.loc[trough_date, "drawdown"],
                "traded_at_trough_open": int(
                    simulated.loc[trough_date, "traded_at_open"]
                ),
                "trough_overnight_return": simulated.loc[
                    trough_date, "overnight_return_before_trade"
                ],
                "trough_intraday_return": simulated.loc[
                    trough_date, "intraday_return_after_trade"
                ],
            }
        )
    troughs = pd.DataFrame(trough_rows).set_index("strategy")
    troughs.to_csv(DESTINATION / "drawdown_troughs.csv")
    print(metrics[["cagr", "sharpe", "max_drawdown"]].round(6).to_string())
    print("\nOpen-execution drawdown troughs:")
    print(troughs.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
