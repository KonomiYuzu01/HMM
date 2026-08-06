from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "smh_shock_guard"
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
SAMPLES = {
    "normal": {
        "strategy": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
        ),
        "prices": "data/prices_vix_hedge.csv",
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "complete_2015_present": ("2015-01-01", None),
        },
    },
    "proxy": {
        "strategy": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
            "_20y_proxy"
        ),
        "prices": "data/prices_20y_proxy.csv",
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_present": ("2015-01-01", None),
            "complete_proxy": ("2006-08-01", None),
        },
    },
}


@dataclass(frozen=True)
class Guard:
    name: str
    scheduled_cap: float | None = None
    growth_share_cap: float | None = None
    daily_cap: bool = False
    shock_cap: float | None = None
    shock_raw_return: float = -0.05
    shock_z: float = 3.0
    shock_hold_days: int = 5
    overflow_asset: str = "QQQ"


def variants() -> list[Guard]:
    candidates = [Guard("baseline")]
    for cap in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        label = int(round(cap * 100))
        candidates.extend(
            [
                Guard(f"scheduled_cap{label}", scheduled_cap=cap),
                Guard(
                    f"daily_cap{label}",
                    scheduled_cap=cap,
                    daily_cap=True,
                ),
            ]
        )
    for cap in (0.50, 0.55, 0.60, 0.65, 0.70):
        cap_label = int(round(cap * 100))
        for overflow_asset in ("QQQ", "BOND", "GOLD", "CASH"):
            candidates.append(
                Guard(
                    name=(
                        f"diversified_cap{cap_label}_to{overflow_asset}"
                    ),
                    scheduled_cap=cap,
                    overflow_asset=overflow_asset,
                )
            )
        candidates.append(
            Guard(
                name=f"daily_diversified_cap{cap_label}_toQQQ",
                scheduled_cap=cap,
                daily_cap=True,
                overflow_asset="QQQ",
            )
        )
    for growth_share_cap in (0.50, 0.60, 0.70, 0.80, 0.90):
        label = int(round(growth_share_cap * 100))
        candidates.extend(
            [
                Guard(
                    f"scheduled_growth_share{label}",
                    growth_share_cap=growth_share_cap,
                ),
                Guard(
                    f"daily_growth_share{label}",
                    growth_share_cap=growth_share_cap,
                    daily_cap=True,
                ),
            ]
        )
    for shock_cap in (0.10, 0.15, 0.20):
        label = int(round(shock_cap * 100))
        for hold_days in (3, 5, 10):
            candidates.append(
                Guard(
                    f"shock_cap{label}_hold{hold_days}",
                    shock_cap=shock_cap,
                    shock_hold_days=hold_days,
                )
            )
    for scheduled_cap in (0.25, 0.30, 0.35, 0.40):
        scheduled_label = int(round(scheduled_cap * 100))
        for shock_cap in (0.10, 0.15, 0.20):
            shock_label = int(round(shock_cap * 100))
            candidates.append(
                Guard(
                    (
                        f"scheduled{scheduled_label}_shock{shock_label}"
                        "_hold5"
                    ),
                    scheduled_cap=scheduled_cap,
                    shock_cap=shock_cap,
                    shock_hold_days=5,
                )
            )
    return candidates


def cap_asset(
    weights: np.ndarray,
    overflow_asset: str,
    absolute_cap: float | None = None,
    growth_share_cap: float | None = None,
) -> np.ndarray:
    adjusted = weights.copy()
    semis_index = ASSETS.index("SEMIS")
    qqq_index = ASSETS.index("QQQ")
    overflow_index = ASSETS.index(overflow_asset)
    cap = np.inf if absolute_cap is None else absolute_cap
    if growth_share_cap is not None:
        growth_total = float(adjusted[semis_index] + adjusted[qqq_index])
        cap = min(cap, growth_share_cap * growth_total)
    excess = max(float(adjusted[semis_index]) - cap, 0.0)
    adjusted[semis_index] -= excess
    adjusted[overflow_index] += excess
    return adjusted


def restore_asset_pair(
    weights: np.ndarray,
    baseline_weights: np.ndarray,
    overflow_asset: str,
) -> np.ndarray:
    adjusted = weights.copy()
    semis_index = ASSETS.index("SEMIS")
    overflow_index = ASSETS.index(overflow_asset)
    pair_total = float(adjusted[semis_index] + adjusted[overflow_index])
    baseline_pair_total = float(
        baseline_weights[semis_index] + baseline_weights[overflow_index]
    )
    if pair_total <= 1e-12 or baseline_pair_total <= 1e-12:
        return adjusted
    semis_share = float(
        baseline_weights[semis_index] / baseline_pair_total
    )
    adjusted[semis_index] = pair_total * semis_share
    adjusted[overflow_index] = pair_total * (1.0 - semis_share)
    return adjusted


def shock_signals(
    semis_returns: pd.Series,
    raw_return: float,
    z_score: float,
) -> pd.Series:
    previous_return = semis_returns.shift(1)
    pre_shock_volatility = semis_returns.shift(2).rolling(
        20,
        min_periods=20,
    ).std(ddof=1)
    return (
        (previous_return <= raw_return)
        & (previous_return <= -z_score * pre_shock_volatility)
    ).fillna(False)


def simulate(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    guard: Guard,
    cost_bps: float = 7.5,
    no_trade_turnover: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = base_weights.index
    base_weights = base_weights.loc[dates, ASSETS]
    base_daily = base_daily.loc[dates]
    asset_returns = asset_returns.reindex(dates).loc[:, ASSETS].fillna(0.0)
    signals = shock_signals(
        asset_returns["SEMIS"],
        guard.shock_raw_return,
        guard.shock_z,
    )

    current_weights = np.zeros(len(ASSETS), dtype=float)
    current_weights[ASSETS.index("CASH")] = 1.0
    equity = 1.0
    peak = 1.0
    remaining_guard_days = 0
    rows: list[dict[str, float | int | bool]] = []
    weight_rows: list[np.ndarray] = []
    previous_shock_active = False

    for position, date in enumerate(dates):
        signal = bool(signals.loc[date])
        if signal and guard.shock_cap is not None:
            remaining_guard_days = guard.shock_hold_days
        shock_active = remaining_guard_days > 0
        effective_cap = guard.scheduled_cap
        if shock_active and guard.shock_cap is not None:
            effective_cap = (
                guard.shock_cap
                if effective_cap is None
                else min(effective_cap, guard.shock_cap)
            )

        base_trade = bool(base_daily.loc[date, "turnover"] > 0.0)
        desired: np.ndarray | None = None
        if base_trade:
            desired = base_weights.loc[date].to_numpy(dtype=float)
            if effective_cap is not None or guard.growth_share_cap is not None:
                desired = cap_asset(
                    desired,
                    guard.overflow_asset,
                    effective_cap,
                    guard.growth_share_cap,
                )
        elif (
            previous_shock_active
            and not shock_active
            and guard.shock_cap is not None
        ):
            desired = restore_asset_pair(
                current_weights,
                base_weights.loc[date].to_numpy(dtype=float),
                guard.overflow_asset,
            )
        elif (
            (effective_cap is not None or guard.growth_share_cap is not None)
            and (
                shock_active
                or guard.daily_cap
                or position == 0
            )
        ):
            capped = cap_asset(
                current_weights,
                guard.overflow_asset,
                effective_cap,
                guard.growth_share_cap,
            )
            if not np.allclose(capped, current_weights):
                desired = capped

        proposed_turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current_weights).sum())
        )
        executed = proposed_turnover >= no_trade_turnover
        turnover = proposed_turnover if executed else 0.0
        trading_cost = (
            2.0 * turnover * cost_bps / 10_000.0
            if executed
            else 0.0
        )
        if executed and desired is not None:
            current_weights = desired

        day_returns = asset_returns.loc[date].to_numpy(dtype=float)
        gross_return = float(current_weights @ day_returns)
        financing_cost = float(base_daily.loc[date, "financing_cost"])
        net_return = gross_return - financing_cost - trading_cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        weight_rows.append(current_weights.copy())
        rows.append(
            {
                "net_return": net_return,
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "turnover": turnover,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "base_trade": base_trade,
                "shock_signal": signal,
                "shock_active": shock_active,
            }
        )

        denominator = 1.0 + gross_return
        if denominator <= 0.0:
            raise ValueError("Candidate portfolio lost all capital in one day")
        current_weights = (
            current_weights * (1.0 + day_returns) / denominator
        )
        if shock_active:
            remaining_guard_days -= 1
        previous_shock_active = shock_active

    return (
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
    )


def five_day_compound(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).rolling(5).apply(np.prod, raw=True) - 1.0


def load_sample(
    sample: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = SAMPLES[sample]
    root = OUTPUT / str(settings["strategy"])
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
    prices = pd.read_csv(
        str(settings["prices"]),
        index_col=0,
        parse_dates=True,
    )
    returns = prices.loc[:, ASSETS].pct_change(fill_method=None)
    return base_weights, base_daily, returns


def validate_reconstruction(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    reconstructed: pd.DataFrame,
    reconstructed_weights: pd.DataFrame,
) -> dict[str, float]:
    return_difference = (
        reconstructed["net_return"] - base_daily["net_return"]
    ).abs()
    weight_difference = (
        reconstructed_weights - base_weights.loc[:, ASSETS]
    ).abs()
    return {
        "maximum_absolute_return_difference": float(
            return_difference.max()
        ),
        "maximum_absolute_weight_difference": float(
            weight_difference.to_numpy().max()
        ),
    }


def metric_rows(
    sample: str,
    guard: Guard,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
) -> list[dict[str, object]]:
    settings = SAMPLES[sample]
    rows: list[dict[str, object]] = []
    for period, boundaries in settings["periods"].items():
        start, end = boundaries
        selected = daily.loc[start:end]
        selected_weights = weights.loc[selected.index]
        values = performance_metrics(selected["net_return"])
        rows.append(
            {
                "sample": sample,
                "variant": guard.name,
                "period": period,
                **values,
                "worst_day": float(selected["net_return"].min()),
                "worst_five_days": float(
                    five_day_compound(selected["net_return"]).min()
                ),
                "maximum_semis_weight": float(
                    selected_weights["SEMIS"].max()
                ),
                "smh_15pct_direct_loss_budget": float(
                    -0.15 * selected_weights["SEMIS"].max()
                ),
                "annualized_turnover": float(
                    selected["turnover"].mean() * 252.0
                ),
                "annualized_trading_cost": float(
                    selected["trading_cost"].mean() * 252.0
                ),
                "shock_signal_days": int(selected["shock_signal"].sum()),
                "shock_active_days": int(selected["shock_active"].sum()),
            }
        )
    return rows


def event_rows(
    sample: str,
    guard: Guard,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> list[dict[str, object]]:
    aligned_returns = asset_returns.reindex(daily.index)
    semis_five_day = five_day_compound(aligned_returns["SEMIS"])
    daily_events = aligned_returns["SEMIS"] <= -0.07
    weekly_events = semis_five_day <= -0.15
    rows: list[dict[str, object]] = []
    for event_type, event_mask in (
        ("daily_le_minus_7pct", daily_events),
        ("five_day_le_minus_15pct", weekly_events),
    ):
        for date in event_mask.index[event_mask.fillna(False)]:
            rows.append(
                {
                    "sample": sample,
                    "variant": guard.name,
                    "event_type": event_type,
                    "date": date,
                    "semis_return": float(
                        aligned_returns.loc[date, "SEMIS"]
                        if event_type == "daily_le_minus_7pct"
                        else semis_five_day.loc[date]
                    ),
                    "strategy_return": float(
                        daily.loc[date, "net_return"]
                        if event_type == "daily_le_minus_7pct"
                        else five_day_compound(
                            daily["net_return"]
                        ).loc[date]
                    ),
                    "semis_weight": float(weights.loc[date, "SEMIS"]),
                    "qqq_weight": float(weights.loc[date, "QQQ"]),
                    "shock_active": bool(
                        daily.loc[date, "shock_active"]
                    ),
                }
            )
    return rows


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["sample", "variant", "period"])
    for column in ("cagr", "max_drawdown", "worst_day", "worst_five_days"):
        baseline = indexed.xs("baseline", level="variant")[column]
        indexed[f"{column}_delta_vs_baseline"] = [
            float(row[column] - baseline.loc[(sample, period)])
            for (sample, _, period), row in indexed.iterrows()
        ]
    return indexed.reset_index()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        base_weights, base_daily, asset_returns = load_sample(sample)
        for guard in variants():
            daily, weights = simulate(
                base_weights,
                base_daily,
                asset_returns,
                guard,
            )
            if guard.name == "baseline":
                reconstruction_rows.append(
                    {
                        "sample": sample,
                        **validate_reconstruction(
                            base_weights,
                            base_daily,
                            daily,
                            weights,
                        ),
                    }
                )
            all_metrics.extend(metric_rows(sample, guard, daily, weights))
            all_events.extend(
                event_rows(
                    sample,
                    guard,
                    daily,
                    weights,
                    asset_returns,
                )
            )

    reconstruction = pd.DataFrame(reconstruction_rows).set_index("sample")
    metrics = add_deltas(pd.DataFrame(all_metrics))
    events = pd.DataFrame(all_events)
    reconstruction.to_csv(DESTINATION / "reconstruction_check.csv")
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    events.to_csv(DESTINATION / "events.csv", index=False)

    complete = metrics.loc[
        metrics["period"].isin(
            ["complete_2015_present", "complete_proxy"]
        )
    ]
    normal = complete.loc[
        complete["period"] == "complete_2015_present"
    ].set_index("variant")
    proxy = complete.loc[
        complete["period"] == "complete_proxy"
    ].set_index("variant")
    summary = normal[
        [
            "cagr",
            "cagr_delta_vs_baseline",
            "max_drawdown",
            "max_drawdown_delta_vs_baseline",
            "worst_day",
            "worst_five_days",
            "maximum_semis_weight",
            "annualized_turnover",
        ]
    ].join(
        proxy[
            [
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
            ]
        ].rename(
            columns={
                "cagr_delta_vs_baseline": "proxy_cagr_delta",
                "max_drawdown_delta_vs_baseline": "proxy_mdd_delta",
            }
        )
    )
    print("Reconstruction:")
    print(reconstruction.to_string())
    print("\nComplete-period summary:")
    print(
        summary.sort_values(
            ["max_drawdown_delta_vs_baseline", "cagr_delta_vs_baseline"],
            ascending=False,
        )
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
