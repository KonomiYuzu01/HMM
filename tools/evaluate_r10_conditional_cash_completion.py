from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_open_execution import simulate_open_execution
from regime_strategy.report import performance_metrics


OUTPUT = Path("output/r10_conditional_cash_completion")
NORMAL_DIRECTORY = (
    "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
PROXY_DIRECTORY = "experiment_r9_broad50_stage35_d10_20y_proxy"
NORMAL_OPEN_CLOSE = Path("data/adjusted_open_close_2011_present.csv")
PROXY_OPEN_CLOSE = Path("data/adjusted_open_close_20y_proxy.csv")
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


@dataclass(frozen=True)
class CompletionRule:
    name: str
    maximum_account_share: float = 0.10
    cash_fraction: float = 0.50
    minimum_growth_exposure: float = 0.40
    slow_trend_days: int = 200
    fast_trend_days: int = 20
    volatility_days: int = 20
    maximum_annual_volatility: float = 0.30
    destination_asset: str = "QQQ"


PRIMARY_RULE = CompletionRule(name="primary_cash_half_cap10")
NEIGHBORHOOD = (
    CompletionRule(name="cap05", maximum_account_share=0.05),
    CompletionRule(name="cap075", maximum_account_share=0.075),
    PRIMARY_RULE,
    CompletionRule(name="cash_fraction25", cash_fraction=0.25),
    CompletionRule(name="cash_fraction75", cash_fraction=0.75),
    CompletionRule(name="growth_floor35", minimum_growth_exposure=0.35),
    CompletionRule(name="growth_floor45", minimum_growth_exposure=0.45),
    CompletionRule(name="volatility25", maximum_annual_volatility=0.25),
    CompletionRule(name="volatility35", maximum_annual_volatility=0.35),
)


def load_adjusted_open_close(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    opens = frame[[f"open_{asset}" for asset in ASSETS]].copy()
    closes = frame[[f"close_{asset}" for asset in ASSETS]].copy()
    opens.columns = ASSETS
    closes.columns = ASSETS
    return opens.astype(float), closes.astype(float)


def causal_completion_signals(
    closes: pd.DataFrame,
    rule: CompletionRule,
) -> pd.DataFrame:
    """Build signals for an open using prices available by the prior close."""
    prior_close = closes[["QQQ", "SEMIS"]].shift(1)
    slow_average = prior_close.rolling(
        rule.slow_trend_days,
        min_periods=rule.slow_trend_days,
    ).mean()
    slow_trend = (prior_close > slow_average).all(axis=1)
    fast_trend = (
        prior_close
        / prior_close.shift(rule.fast_trend_days)
        - 1.0
    ).gt(0.0).all(axis=1)
    growth_return = closes[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1)
    realized_volatility = (
        growth_return.shift(1)
        .rolling(
            rule.volatility_days,
            min_periods=rule.volatility_days,
        )
        .std(ddof=1)
        * np.sqrt(252.0)
    )
    result = pd.DataFrame(
        {
            "slow_trend": slow_trend,
            "fast_trend": fast_trend,
            "realized_volatility": realized_volatility,
        }
    )
    result["market_gate"] = (
        result["slow_trend"]
        & result["fast_trend"]
        & (
            result["realized_volatility"]
            <= rule.maximum_annual_volatility
        )
    )
    return result


def completion_target(
    base_target: pd.Series,
    market_gate: bool,
    rule: CompletionRule,
) -> tuple[pd.Series, float]:
    target = base_target.astype(float).copy()
    growth_exposure = float(target["QQQ"] + target["SEMIS"])
    available_cash = max(float(target["CASH"]), 0.0)
    if (
        not market_gate
        or growth_exposure < rule.minimum_growth_exposure
        or available_cash <= 0.0
    ):
        return target, 0.0
    completion_share = min(
        rule.maximum_account_share,
        rule.cash_fraction * available_cash,
    )
    target["CASH"] -= completion_share
    target[rule.destination_asset] += completion_share
    return target, completion_share


def simulate_completion(
    strategy_directory: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    rule: CompletionRule,
    *,
    start_date: str,
    end_date: str | None,
    cost_bps: float,
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
    if end_date is not None:
        dates = dates[dates <= end_date]
    signals = causal_completion_signals(closes, rule).reindex(dates)
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    overlay_active = False
    active_share = 0.0
    rows: list[dict[str, float | int]] = []
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

        base_trade = float(base_daily.loc[date, "turnover"]) > 1e-14
        market_gate = bool(signals.loc[date, "market_gate"])
        base_target = weights.loc[date, ASSETS]
        desired_target, desired_share = completion_target(
            base_target,
            market_gate,
            rule,
        )
        exit_now = overlay_active and desired_share <= 0.0
        trade_now = base_trade or exit_now
        open_turnover = 0.0
        if trade_now:
            target_values = desired_target.to_numpy(dtype=float)
            open_turnover = 0.5 * float(
                np.abs(target_values - current).sum()
            )
            current = target_values
            overlay_active = desired_share > 0.0
            active_share = desired_share

        intraday_asset_returns = (
            closes.loc[date, ASSETS].to_numpy(dtype=float)
            / opens.loc[date, ASSETS].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        trading_cost = 2.0 * open_turnover * cost_bps / 10_000.0
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
                "overnight_return_before_trade": overnight_return,
                "intraday_return_after_trade": intraday_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "open_turnover": open_turnover,
                "base_trade": int(base_trade),
                "candidate_trade": int(trade_now),
                "market_gate": int(market_gate),
                "overlay_active": int(overlay_active),
                "completion_share": active_share if overlay_active else 0.0,
            }
        )
        previous_date = date
    frame = pd.DataFrame(rows).set_index("date")
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def period_rows(
    sample: str,
    rule_name: str,
    cost_bps: float,
    baseline: pd.Series,
    candidate: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for period, (start, end) in periods.items():
        baseline_metrics = performance_metrics(baseline.loc[start:end])
        candidate_metrics = performance_metrics(candidate.loc[start:end])
        rows.append(
            {
                "sample": sample,
                "rule": rule_name,
                "cost_bps": cost_bps,
                "period": period,
                **{
                    f"baseline_{key}": value
                    for key, value in baseline_metrics.items()
                },
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                "cagr_delta": (
                    candidate_metrics["cagr"] - baseline_metrics["cagr"]
                ),
                "max_drawdown_delta": (
                    candidate_metrics["max_drawdown"]
                    - baseline_metrics["max_drawdown"]
                ),
            }
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = {
        "normal": {
            "directory": NORMAL_DIRECTORY,
            "path": NORMAL_OPEN_CLOSE,
            "start": "2015-01-01",
            "end": None,
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": ("2015-01-01", "2026-12-31"),
            },
        },
        "proxy": {
            "directory": PROXY_DIRECTORY,
            "path": PROXY_OPEN_CLOSE,
            "start": "2006-08-01",
            "end": None,
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": ("2006-08-01", "2026-12-31"),
            },
        },
    }
    metric_rows: list[dict[str, float | str]] = []
    diagnostic_rows: list[dict[str, float | str]] = []
    primary_paths: dict[str, pd.DataFrame] = {}
    for sample, specification in samples.items():
        opens, closes = load_adjusted_open_close(specification["path"])
        directory = str(specification["directory"])
        for cost_bps in (7.5, 15.0, 30.0):
            baseline = simulate_open_execution(
                directory,
                opens,
                closes,
                start_date=str(specification["start"]),
                end_date=specification["end"],
                cost_bps=cost_bps,
            )
            for rule in NEIGHBORHOOD:
                candidate = simulate_completion(
                    directory,
                    opens,
                    closes,
                    rule,
                    start_date=str(specification["start"]),
                    end_date=specification["end"],
                    cost_bps=cost_bps,
                )
                common = baseline.index.intersection(candidate.index)
                metric_rows.extend(
                    period_rows(
                        sample,
                        rule.name,
                        cost_bps,
                        baseline.loc[common, "net_return"],
                        candidate.loc[common, "net_return"],
                        specification["periods"],
                    )
                )
                diagnostic_rows.append(
                    {
                        "sample": sample,
                        "rule": rule.name,
                        "cost_bps": cost_bps,
                        "active_days": int(candidate["overlay_active"].sum()),
                        "active_share_mean": float(
                            candidate.loc[
                                candidate["overlay_active"].eq(1),
                                "completion_share",
                            ].mean()
                        ),
                        "candidate_trades": int(
                            candidate["candidate_trade"].sum()
                        ),
                        "annualized_turnover": float(
                            candidate["open_turnover"].mean() * 252.0
                        ),
                    }
                )
                if rule == PRIMARY_RULE and cost_bps == 7.5:
                    primary_paths[sample] = candidate
                    candidate.to_csv(
                        OUTPUT / f"{sample}_primary_daily.csv",
                        index_label="date",
                    )
                    baseline.to_csv(
                        OUTPUT / f"{sample}_baseline_daily.csv",
                        index_label="date",
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "diagnostics.csv",
        index=False,
    )
    primary = metrics[
        metrics["rule"].eq(PRIMARY_RULE.name)
        & metrics["cost_bps"].eq(7.5)
    ]
    print(primary[
        [
            "sample",
            "period",
            "baseline_cagr",
            "candidate_cagr",
            "cagr_delta",
            "baseline_max_drawdown",
            "candidate_max_drawdown",
        ]
    ].to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
