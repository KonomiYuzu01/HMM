from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import load_strategy_inputs
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r38_state_dependent_rollout as rollout


PRICES = Path("data/prices_pre2008_crash_proxy.csv")
R9_DIRECTORY = "research_r39_pre2008_crash_proxy_r9"
OUTPUT = Path("output/pre2008_crash_replay")
R38_SHARE = 0.25
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
EVENTS = {
    "great_depression_continuation_1931_1932": ("1931-01-02", "1932-07-08"),
    "recession_crash_1937_1938": ("1937-03-10", "1938-03-31"),
    "kennedy_slide_1962": ("1961-12-12", "1962-06-26"),
    "bear_market_1968_1970": ("1968-11-29", "1970-05-26"),
    "oil_crisis_1973_1974": ("1973-01-11", "1974-10-03"),
    "black_monday_1987": ("1987-08-25", "1987-12-04"),
    "gulf_war_bear_1990": ("1990-07-16", "1990-10-11"),
    "ltcm_russia_1998": ("1998-07-20", "1998-10-08"),
    "dotcom_2000_2002": ("2000-03-10", "2002-10-09"),
}


def return_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    equity = pd.concat(
        [pd.Series([1.0]), (1.0 + clean).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity / equity.cummax() - 1.0
    return {
        "return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def price_metrics(prices: pd.Series) -> dict[str, float]:
    clean = prices.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("Price window requires at least two observations")
    equity = clean / clean.iloc[0]
    drawdown = equity / equity.cummax() - 1.0
    return {
        "return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def protection_label(strategy_mdd: float, growth_mdd: float) -> str:
    if growth_mdd >= 0.0:
        return "not_a_growth_drawdown"
    reduction = 1.0 - abs(strategy_mdd) / abs(growth_mdd)
    if strategy_mdd >= -0.20 and reduction >= 0.50:
        return "avoided_major_crash"
    if reduction >= 0.25:
        return "partially_protected"
    return "not_avoided"


def build_settings() -> tuple[dict[str, object], pd.DataFrame]:
    prices = pd.read_csv(PRICES, index_col="date", parse_dates=True)
    closes = prices[ASSETS].astype(float)
    opens = closes.shift(1)
    opens.iloc[0] = closes.iloc[0]
    weights, daily = load_strategy_inputs(R9_DIRECTORY)
    settings: dict[str, object] = {
        "directory": R9_DIRECTORY,
        "weights": weights,
        "daily": daily,
        "opens": opens,
        "closes": closes,
        "mode": "synthetic",
        "start": str(weights.index.min().date()),
        "periods": EVENTS,
    }
    return settings, prices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings, prices = build_settings()
    scenario = COST_SCENARIOS[0]
    _, r11_weights, r11_daily = simulate_fixed_r11(settings, scenario)
    r38_weights, r38_daily, r38_diagnostics = rollout.build_r38_targets(
        settings, scenario
    )
    index = (
        r11_weights.index.intersection(r11_daily.index)
        .intersection(r38_weights.index)
        .intersection(r38_daily.index)
        .intersection(r38_diagnostics.index)
    )
    share = pd.Series(R38_SHARE, index=index, dtype=float)
    strategy, account_diagnostics = rollout.simulate_blended_account(
        settings,
        scenario,
        r11_weights,
        r11_daily,
        r38_weights,
        r38_daily,
        share,
    )
    blended_weights = (
        (1.0 - R38_SHARE) * r11_weights.reindex(index)
        + R38_SHARE * r38_weights.reindex(index)
    )

    daily_proxy_returns = prices[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    growth_return = daily_proxy_returns.mean(axis=1).fillna(0.0)
    growth_index = 100.0 * (1.0 + growth_return).cumprod()
    rows: list[dict[str, object]] = []
    for event, (requested_start, requested_end) in EVENTS.items():
        event_dates = strategy.index[
            (strategy.index >= requested_start) & (strategy.index <= requested_end)
        ]
        if len(event_dates) < 2:
            raise ValueError(f"Insufficient strategy history for {event}")
        start = event_dates[0]
        end = event_dates[-1]
        strategy_returns = strategy.loc[
            (strategy.index > start) & (strategy.index <= end), "net_return"
        ]
        strategy_stats = return_metrics(strategy_returns)
        growth_stats = price_metrics(growth_index.loc[start:end])
        spx_stats = price_metrics(prices.loc[start:end, "SPX"])
        qqq_stats = price_metrics(prices.loc[start:end, "QQQ"])
        semis_stats = price_metrics(prices.loc[start:end, "SEMIS"])
        weight_window = blended_weights.loc[start:end]
        cash = weight_window["CASH"]
        growth_weight = weight_window["QQQ"] + weight_window["SEMIS"]
        de_risk = weight_window.index[
            cash.ge(0.50) | growth_weight.le(0.20)
        ]
        label = protection_label(
            strategy_stats["max_drawdown"], growth_stats["max_drawdown"]
        )
        rows.append(
            {
                "event": event,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "observed_start": start.date().isoformat(),
                "observed_end": end.date().isoformat(),
                "strategy_return": strategy_stats["return"],
                "strategy_max_drawdown": strategy_stats["max_drawdown"],
                "growth_proxy_return": growth_stats["return"],
                "growth_proxy_max_drawdown": growth_stats["max_drawdown"],
                "spx_proxy_return": spx_stats["return"],
                "spx_proxy_max_drawdown": spx_stats["max_drawdown"],
                "qqq_proxy_return": qqq_stats["return"],
                "qqq_proxy_max_drawdown": qqq_stats["max_drawdown"],
                "semis_proxy_return": semis_stats["return"],
                "semis_proxy_max_drawdown": semis_stats["max_drawdown"],
                "drawdown_reduction_vs_growth": (
                    1.0
                    - abs(strategy_stats["max_drawdown"])
                    / abs(growth_stats["max_drawdown"])
                ),
                "average_cash_target": float(cash.mean()),
                "maximum_cash_target": float(cash.max()),
                "minimum_growth_target": float(growth_weight.min()),
                "first_material_derisk_date": (
                    de_risk[0].date().isoformat() if len(de_risk) else None
                ),
                "shock_brake_days": int(
                    r38_diagnostics.reindex(weight_window.index)[
                        "pulse_veto_active"
                    ].fillna(False).astype(bool).sum()
                ),
                "volatility_block_days": int(
                    r38_diagnostics.reindex(weight_window.index)[
                        "volatility_acceleration_block"
                    ].fillna(False).astype(bool).sum()
                ),
                "protection_label": label,
            }
        )

    events = pd.DataFrame(rows)
    events.to_csv(OUTPUT / "event_metrics.csv", index=False)
    strategy.to_csv(OUTPUT / "strategy_daily.csv", index_label="date")
    blended_weights.to_csv(OUTPUT / "blended_target_weights.csv", index_label="date")
    r38_diagnostics.to_csv(OUTPUT / "r38_diagnostics.csv", index_label="date")
    account_diagnostics.to_csv(
        OUTPUT / "account_diagnostics.csv", index_label="date"
    )
    counts = events["protection_label"].value_counts().to_dict()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "counterfactual_pre2008_proxy_replay",
        "production_changed": False,
        "orders_generated": False,
        "strategy": "75% R11 plus 25% R38 with current rules",
        "events_tested": len(events),
        "protection_counts": counts,
        "unavailable_event": {
            "event": "wall_street_crash_1929_first_leg",
            "reason": (
                "The current 1008-session HMM has not completed its causal "
                "warm-up by October 1929. The 1931-1932 continuation is tested."
            ),
        },
        "proxy_disclosure": (
            "QQQ=Fama/French BusEq; SEMIS=Fama/French Chips; GOLD is a "
            "gold-mining-equity proxy after 1963 and flat before; no historical open gaps; "
            "VIX is realized-volatility proxy before 1990; VIX3M is neutral."
        ),
        "interpretation": (
            "A result is labeled avoided only when strategy drawdown stays at "
            "or above -20% and is at least 50% smaller than the equal-weight "
            "QQQ/SEMIS proxy drawdown."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
