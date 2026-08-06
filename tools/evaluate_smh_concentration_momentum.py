from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_open_execution import (
    load_open_close,
    simulate_open_execution,
)
from evaluate_smh_dynamic_guard import (
    ASSETS,
    STRATEGY,
    positive_capture,
    restore_pair_mix,
    transfer_semis,
)
from evaluate_smh_shock_guard import five_day_compound
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_concentration_momentum")
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}


@dataclass(frozen=True)
class ConcentrationMomentumGuard:
    name: str
    horizon_days: int | None = None
    entry_score: float = 0.0
    exit_score: float = 0.0
    minimum_semis_weight: float = 0.0
    semis_cap: float | None = None
    daily_check: bool = True
    emergency_slippage_bps: float = 10.0


def variants() -> list[ConcentrationMomentumGuard]:
    candidates = [ConcentrationMomentumGuard("baseline")]
    for horizon_days in (10, 20, 40, 60, 126):
        for entry_score in (-1.0, -0.5, 0.0):
            entry_label = str(abs(entry_score)).replace(".", "p")
            for minimum_weight in (0.50, 0.60):
                minimum_label = int(round(minimum_weight * 100))
                for semis_cap in (0.30, 0.40, 0.50):
                    cap_label = int(round(semis_cap * 100))
                    for daily_check in (False, True):
                        frequency = "daily" if daily_check else "scheduled"
                        candidates.append(
                            ConcentrationMomentumGuard(
                                name=(
                                    f"relz{horizon_days}"
                                    f"_entryminus{entry_label}"
                                    f"_min{minimum_label}"
                                    f"_cap{cap_label}"
                                    f"_{frequency}"
                                ),
                                horizon_days=horizon_days,
                                entry_score=entry_score,
                                exit_score=0.0,
                                minimum_semis_weight=minimum_weight,
                                semis_cap=semis_cap,
                                daily_check=daily_check,
                            )
                        )
    return candidates


def relative_momentum_score(
    closes: pd.DataFrame,
    horizon_days: int,
) -> pd.Series:
    relative_log_return = (
        np.log(closes["SEMIS"]).diff()
        - np.log(closes["QQQ"]).diff()
    )
    cumulative = relative_log_return.rolling(
        horizon_days,
        min_periods=horizon_days,
    ).sum()
    volatility = relative_log_return.rolling(
        max(horizon_days, 20),
        min_periods=max(horizon_days, 20),
    ).std(ddof=1)
    score = cumulative / (
        volatility.clip(lower=1e-6) * np.sqrt(horizon_days)
    )
    return score.shift(1)


def simulate(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    guard: ConcentrationMomentumGuard,
    cost_bps: float = 7.5,
    no_trade_turnover: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = (
        base_weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= "2015-01-01"]
    base_weights = base_weights.loc[dates, ASSETS]
    base_daily = base_daily.loc[dates]
    opens = opens.loc[dates, ASSETS]
    closes = closes.loc[dates, ASSETS]
    scores = (
        pd.Series(np.nan, index=dates)
        if guard.horizon_days is None
        else relative_momentum_score(closes, guard.horizon_days).reindex(dates)
    )

    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    active = False
    previous_date: pd.Timestamp | None = None
    rows: list[dict[str, float | int | bool | str]] = []
    weight_rows: list[np.ndarray] = []

    for date in dates:
        overnight_return = 0.0
        if previous_date is not None:
            overnight_asset_returns = (
                opens.loc[date].to_numpy(dtype=float)
                / closes.loc[previous_date].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )

        base_trade = bool(base_daily.loc[date, "turnover"] > 1e-14)
        desired = (
            base_weights.loc[date].to_numpy(dtype=float)
            if base_trade
            else None
        )
        score = float(scores.loc[date])
        evaluate = bool(guard.daily_check or base_trade)
        reference = desired if desired is not None else current
        entered = False
        exited = False
        state_changed = False
        trade_reason = "base" if base_trade else "none"

        if (
            active
            and evaluate
            and np.isfinite(score)
            and score >= guard.exit_score
        ):
            if desired is None:
                desired = restore_pair_mix(
                    current,
                    base_weights.loc[date].to_numpy(dtype=float),
                    "QQQ",
                )
            active = False
            exited = True
            state_changed = True
            trade_reason = "relative_recovery"
        elif (
            not active
            and evaluate
            and guard.semis_cap is not None
            and np.isfinite(score)
            and score <= guard.entry_score
            and float(reference[ASSETS.index("SEMIS")])
            >= guard.minimum_semis_weight
        ):
            desired = transfer_semis(
                reference,
                1.0,
                "QQQ",
                guard.semis_cap,
            )
            active = True
            entered = True
            state_changed = True
            trade_reason = "relative_weakness"
        elif active and guard.semis_cap is not None:
            if base_trade or guard.daily_check:
                desired = transfer_semis(
                    reference,
                    1.0,
                    "QQQ",
                    guard.semis_cap,
                )
                trade_reason = "maintain_cap"

        proposed_turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current).sum())
        )
        executed = bool(
            base_trade or proposed_turnover >= no_trade_turnover
        )
        turnover = proposed_turnover if executed else 0.0
        if executed and desired is not None:
            current = desired
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0
        slippage_cost = (
            2.0
            * turnover
            * guard.emergency_slippage_bps
            / 10_000.0
            if state_changed and executed
            else 0.0
        )

        intraday_asset_returns = (
            closes.loc[date].to_numpy(dtype=float)
            / opens.loc[date].to_numpy(dtype=float)
            - 1.0
        )
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
            - slippage_cost
            - financing_cost
        )
        rows.append(
            {
                "net_return": net_return,
                "overnight_return": overnight_return,
                "intraday_return": intraday_return,
                "trading_cost": trading_cost,
                "slippage_cost": slippage_cost,
                "financing_cost": financing_cost,
                "turnover": turnover,
                "relative_momentum_score": score,
                "base_trade": base_trade,
                "entered": entered,
                "exited": exited,
                "active": active,
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

    result = pd.DataFrame(rows, index=dates)
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    result["drawdown"] = (
        result["equity"] / result["equity"].cummax() - 1.0
    )
    return result, pd.DataFrame(
        weight_rows,
        index=dates,
        columns=ASSETS,
    )


def metric_rows(
    guard: ConcentrationMomentumGuard,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    baseline: pd.Series,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for period, (start, end) in PERIODS.items():
        selected = daily.loc[start:end]
        reference = baseline.reindex(selected.index)
        rows.append(
            {
                "variant": guard.name,
                "period": period,
                "horizon_days": guard.horizon_days,
                "entry_score": guard.entry_score,
                "minimum_semis_weight": guard.minimum_semis_weight,
                "semis_cap": guard.semis_cap,
                "daily_check": guard.daily_check,
                **performance_metrics(selected["net_return"]),
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
                "active_days": int(selected["active"].sum()),
                "entry_count": int(selected["entered"].sum()),
                "annualized_turnover": float(
                    selected["turnover"].mean() * 252.0
                ),
                "annualized_cost": float(
                    (
                        selected["trading_cost"]
                        + selected["slippage_cost"]
                    ).mean()
                    * 252.0
                ),
            }
        )
    return rows


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["variant", "period"])
    baseline = indexed.xs("baseline", level="variant")
    for column in (
        "cagr",
        "max_drawdown",
        "worst_day",
        "worst_five_days",
        "positive_capture",
    ):
        indexed[f"{column}_delta_vs_baseline"] = [
            float(row[column] - baseline.loc[period, column])
            for (_, period), row in indexed.iterrows()
        ]
    return indexed.reset_index()


def event_rows(
    guard: ConcentrationMomentumGuard,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    closes: pd.DataFrame,
) -> list[dict[str, float | str | bool]]:
    semis_returns = closes["SEMIS"].pct_change(fill_method=None).reindex(
        daily.index
    )
    event_dates = semis_returns.index[semis_returns <= -0.07]
    rows: list[dict[str, float | str | bool]] = []
    for date in event_dates:
        previous_position = max(int(daily.index.get_loc(date)) - 1, 0)
        previous_date = daily.index[previous_position]
        rows.append(
            {
                "variant": guard.name,
                "date": date,
                "semis_return": float(semis_returns.loc[date]),
                "strategy_return": float(daily.loc[date, "net_return"]),
                "previous_semis_weight": float(
                    weights.loc[previous_date, "SEMIS"]
                ),
                "event_semis_weight": float(weights.loc[date, "SEMIS"]),
                "guard_active_before_event": bool(
                    daily.loc[previous_date, "active"]
                ),
                "guard_active_on_event": bool(daily.loc[date, "active"]),
            }
        )
    return rows


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    root = Path("output") / STRATEGY
    base_weights = pd.read_csv(
        root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    base_daily = pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    opens, closes = load_open_close(refresh=False)
    reference = simulate_open_execution(
        STRATEGY,
        opens,
        closes,
        start_date="2015-01-01",
        end_date=None,
    )

    all_metrics: list[dict[str, float | int | str | bool]] = []
    all_events: list[dict[str, float | str | bool]] = []
    baseline_returns: pd.Series | None = None
    reconstruction_rows: list[dict[str, float]] = []
    for guard in variants():
        daily, weights = simulate(
            base_weights,
            base_daily,
            opens,
            closes,
            guard,
        )
        if guard.name == "baseline":
            baseline_returns = daily["net_return"].copy()
            common = daily.index.intersection(reference.index)
            reconstruction_rows.append(
                {
                    "maximum_absolute_return_difference": float(
                        (
                            daily.loc[common, "net_return"]
                            - reference.loc[common, "net_return"]
                        )
                        .abs()
                        .max()
                    )
                }
            )
        if baseline_returns is None:
            raise RuntimeError("Baseline must be evaluated first")
        all_metrics.extend(
            metric_rows(
                guard,
                daily,
                weights,
                baseline_returns,
            )
        )
        all_events.extend(
            event_rows(
                guard,
                daily,
                weights,
                closes,
            )
        )

    metrics = add_deltas(pd.DataFrame(all_metrics))
    events = pd.DataFrame(all_events)
    reconstruction = pd.DataFrame(reconstruction_rows)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    events.to_csv(DESTINATION / "events.csv", index=False)
    reconstruction.to_csv(
        DESTINATION / "reconstruction_check.csv",
        index=False,
    )

    pivot = metrics.pivot(
        index="variant",
        columns="period",
        values="cagr_delta_vs_baseline",
    )
    all_periods_positive = (pivot[list(PERIODS)] > 0.0).all(axis=1)
    eligible = metrics.loc[
        (metrics["period"] == "complete_2015_present")
        & metrics["variant"].isin(
            all_periods_positive.index[all_periods_positive]
        )
    ].sort_values(
        [
            "cagr_delta_vs_baseline",
            "positive_capture_delta_vs_baseline",
        ],
        ascending=False,
    )
    eligible.to_csv(DESTINATION / "all_periods_positive.csv", index=False)

    print("Reconstruction:")
    print(reconstruction.round(12).to_string(index=False))
    print(
        f"\nAll-period positive candidates: {int(all_periods_positive.sum())}"
        f" / {len(all_periods_positive) - 1}"
    )
    print("\nTop all-period positive candidates:")
    print(
        eligible[
            [
                "variant",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
                "worst_day_delta_vs_baseline",
                "positive_capture",
                "entry_count",
                "annualized_turnover",
                "annualized_cost",
            ]
        ]
        .head(30)
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
