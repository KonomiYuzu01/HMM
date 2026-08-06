from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_hierarchical_regime_portfolios import (
    ASSETS,
    PORTFOLIO_LIBRARY,
    build_causal_signals,
    build_factorized_daily_states,
    load_regime_schedule,
    risk_budget_preserving_target,
)
from tools.evaluate_open_execution import simulate_open_execution
from tools.evaluate_r10_gde_capital_efficiency import (
    load_adjusted_open_close,
    paired_block_bootstrap,
)


OUTPUT = Path("output/r10_crisis_decline_overlay")
SAMPLES = {
    "normal": {
        "directory": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
        ),
        "open_close": Path("data/adjusted_open_close_2011_present.csv"),
        "prices": Path("data/prices_vix_hedge.csv"),
        "start": "2015-01-01",
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "recent_2026": ("2026-01-01", "2026-12-31"),
            "complete_2015_2026": ("2015-01-01", "2026-12-31"),
        },
    },
    "proxy": {
        "directory": "experiment_r9_broad50_stage35_d10_20y_proxy",
        "open_close": Path("data/adjusted_open_close_20y_proxy.csv"),
        "prices": Path("data/prices_20y_proxy.csv"),
        "start": "2006-08-01",
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2026": ("2015-01-01", "2026-12-31"),
            "complete_2006_2026": ("2006-08-01", "2026-12-31"),
        },
    },
}


def crisis_tilt_target(anchor: np.ndarray, blend: float) -> np.ndarray:
    gold_defense = np.asarray(
        [
            PORTFOLIO_LIBRARY["gold_defense"][asset]
            for asset in ASSETS
        ],
        dtype=float,
    )
    return risk_budget_preserving_target(anchor, gold_defense, blend)


def simulate_crisis_overlay(
    strategy_directory: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    daily_state: pd.Series,
    *,
    start_date: str,
    blend: float,
    cost_bps: float,
    no_trade_turnover: float = 0.01,
    financing_spread_bps: float = 100.0,
) -> pd.DataFrame:
    source = Path("output") / strategy_directory
    weights = pd.read_csv(
        source / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    base_daily = pd.read_csv(
        source / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    dates = (
        weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    previous_crisis = False
    rows: list[dict[str, float | int | str]] = []
    for date in dates:
        if previous_date is None:
            overnight_return = 0.0
        else:
            overnight_asset_returns = (
                opens.loc[date, ASSETS].to_numpy(dtype=float)
                / closes.loc[previous_date, ASSETS].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )
        state = daily_state.get(date)
        crisis = bool(state == "crisis_decline")
        base_trade = float(base_daily.loc[date, "turnover"]) > 1e-14
        state_transition = crisis != previous_crisis
        turnover = 0.0
        trade_reason = "none"
        if base_trade or state_transition:
            anchor = weights.loc[date, ASSETS].to_numpy(dtype=float)
            desired = (
                crisis_tilt_target(anchor, blend)
                if crisis
                else anchor
            )
            proposed = 0.5 * float(np.abs(desired - current).sum())
            if proposed >= no_trade_turnover:
                current = desired
                turnover = proposed
                trade_reason = (
                    "base_and_state"
                    if base_trade and state_transition
                    else "base"
                    if base_trade
                    else "state"
                )
        intraday_asset_returns = (
            closes.loc[date, ASSETS].to_numpy(dtype=float)
            / opens.loc[date, ASSETS].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0
        financing_cost = (
            max(-float(current[ASSETS.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
        )
        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 0.0:
            raise RuntimeError("Non-positive intraday portfolio value")
        current = (
            current
            * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "crisis_decline": int(crisis),
                "state_transition": int(state_transition),
                "trade_reason": trade_reason,
            }
        )
        previous_crisis = crisis
        previous_date = date
    frame = pd.DataFrame(rows).set_index("date")
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    state_rows: list[dict[str, float | int | str]] = []
    for sample, settings in SAMPLES.items():
        opens, closes = load_adjusted_open_close(settings["open_close"])
        prices = pd.read_csv(
            settings["prices"],
            index_col=0,
            parse_dates=True,
        )
        schedule = load_regime_schedule(
            Path("output") / str(settings["directory"])
        )
        signals = build_causal_signals(prices)
        _, daily_state = build_factorized_daily_states(
            schedule,
            signals,
            prices.index,
        )
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for period, (start, end) in periods.items():
            selected_state = daily_state.loc[start:end]
            state_rows.append(
                {
                    "sample": sample,
                    "period": period,
                    "days": len(selected_state),
                    "crisis_decline_days": int(
                        selected_state.eq("crisis_decline").sum()
                    ),
                    "crisis_decline_share": float(
                        selected_state.eq("crisis_decline").mean()
                    ),
                }
            )
        for cost_bps in (7.5, 15.0, 30.0):
            baseline = simulate_open_execution(
                str(settings["directory"]),
                opens,
                closes,
                start_date=str(settings["start"]),
                end_date=None,
                cost_bps=cost_bps,
            )
            for blend in (0.25, 0.50, 0.75):
                candidate = simulate_crisis_overlay(
                    str(settings["directory"]),
                    opens,
                    closes,
                    daily_state,
                    start_date=str(settings["start"]),
                    blend=blend,
                    cost_bps=cost_bps,
                )
                common = baseline.index.intersection(candidate.index)
                for period, (start, end) in periods.items():
                    selected = common[
                        (common >= start) & (common <= end)
                    ]
                    base_metrics = performance_metrics(
                        baseline.loc[selected, "net_return"]
                    )
                    metrics = performance_metrics(
                        candidate.loc[selected, "net_return"]
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "cost_bps": cost_bps,
                            "blend": blend,
                            "period": period,
                            **metrics,
                            "cagr_delta": (
                                metrics["cagr"] - base_metrics["cagr"]
                            ),
                            "sharpe_delta": (
                                metrics["sharpe"] - base_metrics["sharpe"]
                            ),
                            "max_drawdown_delta": (
                                metrics["max_drawdown"]
                                - base_metrics["max_drawdown"]
                            ),
                            "annualized_one_way_turnover": float(
                                candidate.loc[
                                    selected,
                                    "turnover",
                                ].mean()
                                * 252.0
                            ),
                        }
                    )
                if cost_bps == 7.5:
                    label = int(round(blend * 100))
                    candidate.to_csv(
                        OUTPUT / f"{sample}_blend{label}_daily.csv",
                        index_label="date",
                    )
                    for block_days in (21, 63, 126):
                        bootstrap_rows.append(
                            {
                                "sample": sample,
                                "blend": blend,
                                **paired_block_bootstrap(
                                    baseline.loc[common, "net_return"],
                                    candidate.loc[common, "net_return"],
                                    block_days=block_days,
                                ),
                            }
                        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "paired_bootstrap.csv",
        index=False,
    )
    pd.DataFrame(state_rows).to_csv(
        OUTPUT / "state_coverage.csv",
        index=False,
    )
    print(
        metrics[
            [
                "sample",
                "cost_bps",
                "blend",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
