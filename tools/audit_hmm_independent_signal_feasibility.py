from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT = Path("output/hmm_independent_signal_feasibility")
REFERENCE_DATE = pd.Timestamp("2026-07-30")
SEMICONDUCTOR_COLUMNS = [
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
]


def summarize_columns(path: Path, columns: Iterable[str]) -> dict[str, object]:
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path} has no date column")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path} dates must be unique and increasing")
    selected = list(columns)
    missing_columns = sorted(set(selected).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")
    valid = frame[selected].notna().all(axis=1)
    if not valid.any():
        raise ValueError(f"{path} has no complete observations for {selected}")
    valid_dates = dates.loc[valid]
    last_date = valid_dates.iloc[-1]
    lag = len(pd.bdate_range(last_date + pd.offsets.BDay(), REFERENCE_DATE))
    return {
        "source_file": str(path),
        "columns": selected,
        "complete_observations": int(valid.sum()),
        "first_complete_date": valid_dates.iloc[0].date().isoformat(),
        "last_complete_date": last_date.date().isoformat(),
        "business_day_lag_to_reference": lag,
        "missing_values_by_column": {
            column: int(frame[column].isna().sum()) for column in selected
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    recovery = Path("data/prices_recovery_quality.csv")
    stock_panel = Path("data/individual_stock_overlay_prices.csv")
    sector_panel = Path("data/sector_hmm_shadow_prices.csv")
    candidates = {
        "vix_term_structure": {
            **summarize_columns(recovery, ["VIX", "VIX3M"]),
            "causal_transform": "lagged VIX / VIX3M",
            "incremental_information": True,
            "feasibility": "already_used_in_parent_action_layer",
            "already_in_current_parent_action_layer": True,
            "limitations": [
                "cache trails the reference close",
                "levels are market-wide rather than semiconductor-specific",
                "adding the same signal to the HMM could double-count existing risk controls",
            ],
        },
        "credit_relative_strength": {
            **summarize_columns(sector_panel, ["HYG", "IEF"]),
            "causal_transform": "lagged HYG / IEF return and trend",
            "incremental_information": True,
            "feasibility": "research_ready",
            "limitations": [
                "HYG begins in 2007 and misses the earliest rows",
                "ETF liquidity and duration effects are mixed with credit risk",
            ],
        },
        "equal_weight_market_breadth_proxy": {
            **summarize_columns(recovery, ["BREADTH", "SPX"]),
            "causal_transform": "lagged RSP / SPX relative trend",
            "incremental_information": True,
            "feasibility": "macro_breadth_proxy_only",
            "limitations": [
                "not a semiconductor constituent breadth measure",
                "cache trails the reference close",
            ],
        },
        "semiconductor_constituent_dispersion": {
            **summarize_columns(stock_panel, SEMICONDUCTOR_COLUMNS),
            "causal_transform": "cross-sectional active-return dispersion and breadth",
            "incremental_information": True,
            "feasibility": "forward_shadow_only",
            "point_in_time_membership": False,
            "limitations": [
                "fixed present-day basket creates survivorship and membership look-ahead bias",
                "history starts in 2014 and contains few independent stress regimes",
                "ADR and U.S. listings mix different market hours",
            ],
        },
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_data_feasibility_diagnostic",
        "reference_completed_close": REFERENCE_DATE.date().isoformat(),
        "production_changed": False,
        "orders_generated": False,
        "candidates": candidates,
        "decision": {
            "ready_for_unbiased_historical_promotion_test": [],
            "preferred_next_shadow_inputs": [
                "credit_relative_strength",
            ],
            "existing_action_layer_inputs_not_new_hmm_information": [
                "vix_term_structure"
            ],
            "requires_point_in_time_data_before_backtest": [
                "semiconductor_constituent_dispersion"
            ],
            "do_not_relabel_as_semiconductor_breadth": [
                "equal_weight_market_breadth_proxy"
            ],
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
