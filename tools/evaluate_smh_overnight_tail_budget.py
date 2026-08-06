from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import (
    ASSETS,
    STRATEGY,
    load_inputs,
    positive_capture,
    transfer_semis,
)
from evaluate_open_execution import simulate_open_execution
from evaluate_smh_shock_guard import five_day_compound
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_overnight_tail_budget")
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}


@dataclass(frozen=True)
class TailBudgetRule:
    name: str
    lookback_days: int | None = None
    tail_probability: float = 0.975
    estimator: str = "expected_shortfall"
    account_loss_budget: float = 1.0
    minimum_semis_cap: float = 0.0
    overflow_asset: str = "CASH"
    daily_reduce_only: bool = False
    no_trade_weight: float = 0.01


def variants() -> list[TailBudgetRule]:
    candidates = [TailBudgetRule("baseline")]
    for lookback_days in (126, 252, 504):
        for tail_probability in (0.95, 0.975, 0.99):
            tail_label = str(tail_probability).replace(".", "p")
            for estimator in ("quantile", "expected_shortfall"):
                for account_loss_budget in (0.015, 0.020, 0.025, 0.030):
                    budget_label = int(round(account_loss_budget * 10_000))
                    for minimum_semis_cap in (0.20, 0.30):
                        floor_label = int(round(minimum_semis_cap * 100))
                        for overflow_asset in ("CASH", "QQQ"):
                            for daily_reduce_only in (False, True):
                                frequency = (
                                    "daily_reduce"
                                    if daily_reduce_only
                                    else "scheduled"
                                )
                                candidates.append(
                                    TailBudgetRule(
                                        name=(
                                            f"{estimator}"
                                            f"_lb{lookback_days}"
                                            f"_q{tail_label}"
                                            f"_budget{budget_label}bp"
                                            f"_floor{floor_label}"
                                            f"_to{overflow_asset}"
                                            f"_{frequency}"
                                        ),
                                        lookback_days=lookback_days,
                                        tail_probability=tail_probability,
                                        estimator=estimator,
                                        account_loss_budget=(
                                            account_loss_budget
                                        ),
                                        minimum_semis_cap=minimum_semis_cap,
                                        overflow_asset=overflow_asset,
                                        daily_reduce_only=daily_reduce_only,
                                    )
                                )
    return candidates


def rolling_tail_estimate(
    overnight_losses: pd.Series,
    lookback_days: int,
    tail_probability: float,
    estimator: str,
) -> pd.Series:
    known_losses = overnight_losses.shift(1)
    rolling = known_losses.rolling(
        lookback_days,
        min_periods=lookback_days,
    )
    if estimator == "quantile":
        return rolling.quantile(tail_probability)
    if estimator != "expected_shortfall":
        raise ValueError(f"Unknown tail estimator: {estimator}")

    def expected_shortfall(values: np.ndarray) -> float:
        threshold = float(np.quantile(values, tail_probability))
        tail = values[values >= threshold]
        return float(tail.mean())

    return rolling.apply(expected_shortfall, raw=True)


def semis_cap_from_tail_budget(
    account_loss_budget: float,
    tail_loss: float,
    minimum_semis_cap: float,
) -> float:
    if not np.isfinite(tail_loss) or tail_loss <= 1e-12:
        return 1.0
    return float(
        np.clip(
            account_loss_budget / tail_loss,
            minimum_semis_cap,
            1.0,
        )
    )


def simulate(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    rule: TailBudgetRule,
    cost_bps: float = 7.5,
    precomputed_tail_estimates: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = (
        base_weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= "2015-01-01"]
    base_weight_values = base_weights.loc[
        dates,
        ASSETS,
    ].to_numpy(dtype=float)
    base_trade_values = (
        base_daily.loc[dates, "turnover"].to_numpy(dtype=float) > 1e-14
    )
    open_frame = opens.loc[:, ASSETS]
    close_frame = closes.loc[:, ASSETS]
    open_values = open_frame.loc[dates].to_numpy(dtype=float)
    close_values = close_frame.loc[dates].to_numpy(dtype=float)
    overnight_asset_return_values = np.zeros_like(open_values)
    overnight_asset_return_values[1:] = (
        open_values[1:] / close_values[:-1] - 1.0
    )
    intraday_asset_return_values = close_values / open_values - 1.0
    if precomputed_tail_estimates is not None:
        tail_estimates = precomputed_tail_estimates
    else:
        overnight_losses = (
            1.0
            - open_frame["SEMIS"] / close_frame["SEMIS"].shift(1)
        )
        overnight_losses = overnight_losses.clip(lower=0.0)
        tail_estimates = (
            pd.Series(np.nan, index=open_frame.index)
            if rule.lookback_days is None
            else rolling_tail_estimate(
                overnight_losses,
                rule.lookback_days,
                rule.tail_probability,
                rule.estimator,
            )
        )
    tail_estimate_values = tail_estimates.reindex(dates).to_numpy(dtype=float)

    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    rows: list[dict[str, float | int | bool | str]] = []
    weight_rows: list[np.ndarray] = []

    for position, date in enumerate(dates):
        overnight_asset_returns = overnight_asset_return_values[position]
        overnight_return = 0.0
        if previous_date is not None:
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )

        base_trade = bool(base_trade_values[position])
        desired = (
            base_weight_values[position].copy()
            if base_trade
            else None
        )
        tail_loss = float(tail_estimate_values[position])
        semis_cap = semis_cap_from_tail_budget(
            rule.account_loss_budget,
            tail_loss,
            rule.minimum_semis_cap,
        )
        reference = desired if desired is not None else current
        reference_semis = float(reference[ASSETS.index("SEMIS")])
        cap_active = bool(
            rule.lookback_days is not None
            and reference_semis > semis_cap + rule.no_trade_weight
        )
        risk_trade = bool(
            cap_active and (base_trade or rule.daily_reduce_only)
        )
        trade_reason = "base" if base_trade else "none"
        if risk_trade:
            desired = transfer_semis(
                reference,
                1.0,
                rule.overflow_asset,
                semis_cap,
            )
            trade_reason = "tail_budget"

        proposed_turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current).sum())
        )
        executed = bool(
            desired is not None
            and (
                base_trade
                or proposed_turnover >= rule.no_trade_weight
            )
        )
        turnover = proposed_turnover if executed else 0.0
        if executed and desired is not None:
            current = desired
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0

        intraday_asset_returns = intraday_asset_return_values[position]
        intraday_return = float(current @ intraday_asset_returns)
        financing_cost = (
            max(-float(current[ASSETS.index("CASH")]), 0.0)
            * 100.0
            / 10_000.0
            / 252.0
        )
        risky = np.ones(len(current), dtype=bool)
        risky[ASSETS.index("CASH")] = False
        financing_cost += (
            float(np.maximum(-current[risky], 0.0).sum())
            * 100.0
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
        )
        rows.append(
            {
                "net_return": net_return,
                "overnight_return": overnight_return,
                "intraday_return": intraday_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "turnover": turnover,
                "tail_loss_estimate": tail_loss,
                "semis_cap": semis_cap,
                "cap_active": cap_active,
                "risk_trade": risk_trade,
                "base_trade": base_trade,
                "trade_reason": trade_reason,
            }
        )
        weight_rows.append(current.copy())

        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 1e-12:
            raise ValueError("Portfolio lost all capital intraday")
        current = (
            current
            * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        previous_date = date

    return (
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
    )


def metric_rows(
    rule: TailBudgetRule,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    baseline: pd.Series,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for period, (start, end) in PERIODS.items():
        selected = daily.loc[start:end]
        reference = baseline.loc[start:end]
        values = performance_metrics(selected["net_return"])
        rows.append(
            {
                "variant": rule.name,
                "period": period,
                **values,
                "worst_day": float(selected["net_return"].min()),
                "worst_five_days": float(
                    five_day_compound(selected["net_return"]).min()
                ),
                "positive_capture": positive_capture(
                    selected["net_return"],
                    reference,
                ),
                "maximum_semis_weight": float(
                    weights.loc[selected.index, "SEMIS"].max()
                ),
                "average_semis_weight": float(
                    weights.loc[selected.index, "SEMIS"].mean()
                ),
                "cap_active_days": int(selected["cap_active"].sum()),
                "risk_trade_days": int(selected["risk_trade"].sum()),
                "annualized_turnover": float(
                    selected["turnover"].mean() * 252.0
                ),
                "annualized_trading_cost": float(
                    selected["trading_cost"].mean() * 252.0
                ),
            }
        )
    return rows


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["variant", "period"])
    for column in (
        "cagr",
        "max_drawdown",
        "worst_day",
        "worst_five_days",
        "positive_capture",
    ):
        baseline = indexed.xs("baseline", level="variant")[column]
        indexed[f"{column}_delta_vs_baseline"] = [
            float(row[column] - baseline.loc[period])
            for (_, period), row in indexed.iterrows()
        ]
    return indexed.reset_index()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    base_weights, base_daily, opens, closes = load_inputs()
    all_rows: list[dict[str, float | int | str]] = []
    baseline_returns: pd.Series | None = None
    reconstruction_rows: list[dict[str, float]] = []
    overnight_losses = (
        1.0 - opens["SEMIS"] / closes["SEMIS"].shift(1)
    ).clip(lower=0.0)
    tail_cache: dict[tuple[int, float, str], pd.Series] = {}

    for rule in variants():
        tail_estimates: pd.Series | None = None
        if rule.lookback_days is not None:
            cache_key = (
                rule.lookback_days,
                rule.tail_probability,
                rule.estimator,
            )
            if cache_key not in tail_cache:
                tail_cache[cache_key] = rolling_tail_estimate(
                    overnight_losses,
                    rule.lookback_days,
                    rule.tail_probability,
                    rule.estimator,
                )
            tail_estimates = tail_cache[cache_key]
        daily, weights = simulate(
            base_weights,
            base_daily,
            opens,
            closes,
            rule,
            precomputed_tail_estimates=tail_estimates,
        )
        if rule.name == "baseline":
            baseline_returns = daily["net_return"].copy()
            reference = simulate_open_execution(
                STRATEGY,
                opens,
                closes,
                start_date="2015-01-01",
                end_date=None,
                cost_bps=7.5,
            )
            shared = daily.index.intersection(reference.index)
            reconstruction_rows.append(
                {
                    "baseline_rows": float(len(daily)),
                    "source_rows": float(len(reference)),
                    "shared_rows": float(len(shared)),
                    "maximum_daily_return_error": float(
                        (
                            daily.loc[shared, "net_return"]
                            - reference.loc[shared, "net_return"]
                        )
                        .abs()
                        .max()
                    ),
                }
            )
        if baseline_returns is None:
            raise RuntimeError("Baseline must be evaluated first")
        all_rows.extend(
            metric_rows(
                rule,
                daily,
                weights,
                baseline_returns,
            )
        )

    metrics = add_deltas(pd.DataFrame(all_rows))
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(
        DESTINATION / "reconstruction_check.csv",
        index=False,
    )

    pivot = metrics.pivot(
        index="variant",
        columns="period",
        values="cagr_delta_vs_baseline",
    )
    full = metrics.loc[
        metrics["period"] == "complete_2015_present"
    ].set_index("variant")
    summary = pivot.join(
        full[
            [
                "positive_capture",
                "max_drawdown_delta_vs_baseline",
                "worst_day_delta_vs_baseline",
                "worst_five_days_delta_vs_baseline",
                "risk_trade_days",
            ]
        ]
    )
    periods = list(PERIODS)
    all_period_positive = summary.loc[
        (summary[periods] > 0.0).all(axis=1)
    ].sort_values(
        [
            "complete_2015_present",
            "positive_capture",
        ],
        ascending=False,
    )
    print("All-period positive candidates:")
    print(all_period_positive.head(40).round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
