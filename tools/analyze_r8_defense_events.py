from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN = Path(
    "output/"
    "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
    "inverse_momentum_netted_ensemble_20y_proxy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain how the R8 defensive layers behaved during growth "
            "benchmark drawdowns."
        )
    )
    parser.add_argument("--prices", default="data/prices_20y_proxy.csv")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/r8_defense_event_analysis"),
    )
    parser.add_argument("--threshold", type=float, default=-0.20)
    return parser.parse_args()


def drawdown_episodes(
    returns: pd.Series,
    threshold: float,
) -> list[dict[str, object]]:
    clean = returns.dropna().astype(float)
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    episodes: list[dict[str, object]] = []
    in_episode = False
    peak_date: pd.Timestamp | None = None
    trough_date: pd.Timestamp | None = None
    trough_drawdown = 0.0

    for date, value in drawdown.items():
        if not in_episode and value < 0.0:
            in_episode = True
            prior = equity.loc[:date].iloc[:-1]
            peak_date = prior.idxmax() if not prior.empty else date
            trough_date = date
            trough_drawdown = float(value)
        elif in_episode and value < trough_drawdown:
            trough_date = date
            trough_drawdown = float(value)

        if in_episode and value >= -1.0e-12:
            if trough_drawdown <= threshold:
                episodes.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": date,
                        "growth_max_drawdown": trough_drawdown,
                    }
                )
            in_episode = False
            peak_date = None
            trough_date = None
            trough_drawdown = 0.0

    if in_episode and trough_drawdown <= threshold:
        episodes.append(
            {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "growth_max_drawdown": trough_drawdown,
            }
        )
    return episodes


def compounded_return(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    sample = returns.loc[(returns.index > start) & (returns.index <= end)]
    return float((1.0 + sample).prod() - 1.0)


def path_max_drawdown(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    sample = returns.loc[(returns.index > start) & (returns.index <= end)]
    wealth = pd.concat(
        [pd.Series([1.0], index=[start]), (1.0 + sample).cumprod()]
    )
    return float((wealth / wealth.cummax() - 1.0).min())


def first_date(
    values: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    predicate: pd.Series,
) -> pd.Timestamp | pd.NaT:
    matches = values.index[
        (values.index >= start)
        & (values.index <= end)
        & predicate.reindex(values.index, fill_value=False)
    ]
    return matches[0] if len(matches) else pd.NaT


def sessions_between(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp | pd.NaT,
) -> int | pd.NA:
    if pd.isna(end):
        return pd.NA
    return int(index.get_loc(end) - index.get_loc(start))


def turning_state(fast: pd.Series, slow: pd.Series) -> pd.Series:
    result = pd.Series("insufficient", index=fast.index, dtype=object)
    valid = fast.notna() & slow.notna()
    result.loc[valid & fast.ge(0.0) & slow.ge(0.0)] = "bull"
    result.loc[valid & fast.lt(0.0) & slow.ge(0.0)] = "correction"
    result.loc[valid & fast.lt(0.0) & slow.lt(0.0)] = "bear"
    result.loc[valid & fast.ge(0.0) & slow.lt(0.0)] = "rebound"
    return result


def event_name(peak: pd.Timestamp) -> str:
    names = {
        2007: "全球金融危机",
        2011: "欧债与美国评级冲击",
        2018: "2018年第四季度下跌",
        2020: "新冠疫情冲击",
        2021: "2022年成长股熊市",
        2025: "2025年关税冲击",
    }
    return names.get(peak.year, peak.date().isoformat())


def load_risk_on_votes(
    run_dir: Path,
    index: pd.DatetimeIndex,
) -> pd.Series:
    members: list[pd.Series] = []
    for seed in (7, 42, 123):
        path = run_dir / "members" / f"seed_{seed}" / "regimes.csv"
        regimes = pd.read_csv(path, index_col=0, parse_dates=True)
        members.append(
            regimes["paper_risk_on_candidate"]
            .astype(float)
            .reindex(index)
            .ffill()
            .fillna(0.0)
        )
    return pd.concat(members, axis=1).sum(axis=1).rename("risk_on_votes")


def build_diagnostics(
    prices: pd.DataFrame,
    analysis_index: pd.DatetimeIndex,
    risk_on_votes: pd.Series,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    growth = returns[["QQQ", "SEMIS"]].mean(axis=1)
    cash = returns["CASH"]
    growth_log = np.log1p(growth.clip(lower=-0.999999))
    cash_log = np.log1p(cash.clip(lower=-0.999999))
    qqq_log = np.log1p(returns["QQQ"].clip(lower=-0.999999))

    diagnostics = pd.DataFrame(index=prices.index)
    diagnostics["qqq_volatility_20d"] = (
        returns["QQQ"].rolling(20).std(ddof=1) * np.sqrt(252.0)
    )
    diagnostics["qqq_return_200d"] = np.expm1(qqq_log.rolling(200).sum())
    diagnostics["growth_excess_return_20d"] = np.expm1(
        (growth_log - cash_log).rolling(20).sum()
    )
    diagnostics["growth_excess_return_63d"] = np.expm1(
        (growth_log - cash_log).rolling(63).sum()
    )
    diagnostics["growth_excess_return_252d"] = np.expm1(
        (growth_log - cash_log).rolling(252).sum()
    )
    diagnostics["zero_entry_state"] = turning_state(
        diagnostics["growth_excess_return_20d"],
        diagnostics["growth_excess_return_252d"],
    )
    diagnostics["established_position_state"] = turning_state(
        diagnostics["growth_excess_return_63d"],
        diagnostics["growth_excess_return_252d"],
    )
    growth_volatility = growth.rolling(20).std(ddof=1) * np.sqrt(252.0)
    diagnostics["growth_volatility_20d"] = growth_volatility
    diagnostics["growth_volatility_high_threshold"] = (
        growth_volatility.rolling(756, min_periods=504).quantile(0.90)
    )
    diagnostics["growth_volatility_state"] = "medium"
    diagnostics.loc[
        growth_volatility
        <= growth_volatility.rolling(756, min_periods=504).quantile(0.45),
        "growth_volatility_state",
    ] = "low"
    diagnostics.loc[
        growth_volatility
        >= diagnostics["growth_volatility_high_threshold"],
        "growth_volatility_state",
    ] = "high"
    diagnostics.loc[
        diagnostics["growth_volatility_high_threshold"].isna(),
        "growth_volatility_state",
    ] = "insufficient"
    total_variation = growth.pow(2).rolling(20).sum()
    downside_variation = growth.clip(upper=0.0).pow(2).rolling(20).sum()
    diagnostics["downside_variation_share_20d"] = (
        downside_variation / total_variation
    )
    signal_columns = list(diagnostics.columns)
    diagnostics[signal_columns] = diagnostics[signal_columns].shift(1)
    diagnostics["risk_on_votes"] = risk_on_votes.reindex(prices.index).ffill()
    return diagnostics.reindex(analysis_index)


def add_timeline_row(
    rows: list[dict[str, object]],
    event: str,
    stage: str,
    date: pd.Timestamp | pd.NaT,
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    if pd.isna(date):
        return
    date = pd.Timestamp(date)
    row = diagnostics.loc[date]
    rows.append(
        {
            "event": event,
            "stage": stage,
            "date": date,
            "growth_weight": float(
                weights.loc[date, ["QQQ", "SEMIS"]].sum()
            ),
            "QQQ_weight": float(weights.loc[date, "QQQ"]),
            "SEMIS_weight": float(weights.loc[date, "SEMIS"]),
            "GOLD_weight": float(weights.loc[date, "GOLD"]),
            "BOND_weight": float(weights.loc[date, "BOND"]),
            "CASH_weight": float(weights.loc[date, "CASH"]),
            "risk_on_votes": int(row["risk_on_votes"]),
            "qqq_volatility_20d": float(row["qqq_volatility_20d"]),
            "qqq_return_200d": float(row["qqq_return_200d"]),
            "zero_entry_state": row["zero_entry_state"],
            "established_position_state": row[
                "established_position_state"
            ],
            "growth_volatility_state": row["growth_volatility_state"],
            "growth_excess_return_20d": float(
                row["growth_excess_return_20d"]
            ),
            "growth_excess_return_252d": float(
                row["growth_excess_return_252d"]
            ),
            "downside_variation_share_20d": float(
                row["downside_variation_share_20d"]
            ),
        }
    )


def main() -> None:
    args = parse_args()
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    daily = pd.read_csv(
        args.run_dir / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    weights = pd.read_csv(
        args.run_dir / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    common_index = daily.index.intersection(weights.index)
    daily = daily.reindex(common_index)
    weights = weights.reindex(common_index)
    price_returns = prices.pct_change(fill_method=None)
    growth_returns = (
        price_returns[["QQQ", "SEMIS"]].mean(axis=1).reindex(common_index)
    )
    qqq_returns = price_returns["QQQ"].reindex(common_index)
    semis_returns = price_returns["SEMIS"].reindex(common_index)
    risk_on_votes = load_risk_on_votes(args.run_dir, common_index)
    diagnostics = build_diagnostics(prices, common_index, risk_on_votes)
    growth_weight = weights[["QQQ", "SEMIS"]].sum(axis=1)
    growth_equity = (1.0 + growth_returns.fillna(0.0)).cumprod()
    growth_drawdown = growth_equity / growth_equity.cummax() - 1.0
    episodes = drawdown_episodes(growth_returns, args.threshold)

    event_rows: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    last_available = common_index[-1]
    for episode in episodes:
        peak = pd.Timestamp(episode["peak_date"])
        trough = pd.Timestamp(episode["trough_date"])
        recovery_value = episode["recovery_date"]
        recovery = (
            pd.Timestamp(recovery_value)
            if not pd.isna(recovery_value)
            else last_available
        )
        event = event_name(peak)
        event_weights = growth_weight.loc[peak:recovery]
        first_below_45 = first_date(
            growth_weight,
            peak,
            trough,
            growth_weight.le(0.46),
        )
        first_at_20 = first_date(
            growth_weight,
            peak,
            trough,
            growth_weight.le(0.205),
        )
        first_zero = first_date(
            growth_weight,
            peak,
            trough,
            growth_weight.le(0.001),
        )
        first_zero_episode = first_date(
            growth_weight,
            peak,
            recovery,
            growth_weight.le(0.001),
        )
        zero_dates = event_weights.index[event_weights.le(0.001)]
        last_zero_episode = (
            zero_dates[-1] if len(zero_dates) else pd.NaT
        )
        durable_reentry_start = trough
        if not pd.isna(last_zero_episode) and last_zero_episode >= trough:
            last_zero_position = common_index.get_loc(last_zero_episode)
            if last_zero_position + 1 < len(common_index):
                durable_reentry_start = common_index[last_zero_position + 1]
        first_reentry_20 = first_date(
            growth_weight,
            trough,
            recovery,
            growth_weight.ge(0.195),
        )
        first_reentry_45 = first_date(
            growth_weight,
            trough,
            recovery,
            growth_weight.ge(0.44),
        )
        first_reentry_75 = first_date(
            growth_weight,
            trough,
            recovery,
            growth_weight.ge(0.74),
        )
        durable_reentry_20 = first_date(
            growth_weight,
            durable_reentry_start,
            recovery,
            growth_weight.ge(0.195),
        )
        durable_reentry_45 = first_date(
            growth_weight,
            durable_reentry_start,
            recovery,
            growth_weight.ge(0.44),
        )
        durable_reentry_75 = first_date(
            growth_weight,
            durable_reentry_start,
            recovery,
            growth_weight.ge(0.74),
        )
        strategy_peak_to_trough = compounded_return(
            daily["net_return"], peak, trough
        )
        growth_peak_to_trough = compounded_return(
            growth_returns, peak, trough
        )
        growth_rebound = compounded_return(growth_returns, trough, recovery)
        strategy_rebound = compounded_return(
            daily["net_return"], trough, recovery
        )
        event_rows.append(
            {
                "event": event,
                "peak_date": peak,
                "trough_date": trough,
                "recovery_date": (
                    recovery if not pd.isna(recovery_value) else pd.NaT
                ),
                "growth_max_drawdown": episode["growth_max_drawdown"],
                "QQQ_peak_to_trough_return": compounded_return(
                    qqq_returns, peak, trough
                ),
                "SEMIS_peak_to_trough_return": compounded_return(
                    semis_returns, peak, trough
                ),
                "strategy_peak_to_trough_return": strategy_peak_to_trough,
                "strategy_window_max_drawdown": path_max_drawdown(
                    daily["net_return"], peak, trough
                ),
                "loss_avoided_at_growth_trough": (
                    strategy_peak_to_trough - growth_peak_to_trough
                ),
                "growth_weight_at_peak": float(growth_weight.loc[peak]),
                "average_growth_weight_peak_to_trough": float(
                    growth_weight.loc[peak:trough].mean()
                ),
                "minimum_growth_weight_peak_to_trough": float(
                    growth_weight.loc[peak:trough].min()
                ),
                "minimum_growth_weight_peak_to_recovery": float(
                    event_weights.min()
                ),
                "growth_weight_at_trough": float(growth_weight.loc[trough]),
                "first_below_45_date": first_below_45,
                "first_below_45_lag_sessions_from_peak": sessions_between(
                    common_index, peak, first_below_45
                ),
                "growth_drawdown_at_first_below_45": (
                    float(growth_drawdown.loc[first_below_45])
                    if not pd.isna(first_below_45)
                    else np.nan
                ),
                "first_20_or_lower_date": first_at_20,
                "first_zero_before_trough_date": first_zero,
                "first_zero_lag_sessions_from_peak": sessions_between(
                    common_index, peak, first_zero_episode
                ),
                "growth_drawdown_at_first_zero": (
                    float(growth_drawdown.loc[first_zero_episode])
                    if not pd.isna(first_zero_episode)
                    else np.nan
                ),
                "first_zero_date": first_zero_episode,
                "last_zero_date": last_zero_episode,
                "zero_weight_sessions_peak_to_recovery": int(
                    event_weights.le(0.001).sum()
                ),
                "growth_rebound_to_recovery": growth_rebound,
                "strategy_rebound_to_growth_recovery": strategy_rebound,
                "rebound_capture_ratio": (
                    strategy_rebound / growth_rebound
                    if growth_rebound > 0.0
                    else np.nan
                ),
                "first_reentry_20_date": first_reentry_20,
                "reentry_20_lag_sessions_from_trough": sessions_between(
                    common_index, trough, first_reentry_20
                ),
                "first_reentry_45_date": first_reentry_45,
                "reentry_45_lag_sessions_from_trough": sessions_between(
                    common_index, trough, first_reentry_45
                ),
                "first_reentry_75_date": first_reentry_75,
                "reentry_75_lag_sessions_from_trough": sessions_between(
                    common_index, trough, first_reentry_75
                ),
                "durable_reentry_20_date": durable_reentry_20,
                "durable_reentry_20_lag_sessions_from_trough": sessions_between(
                    common_index, trough, durable_reentry_20
                ),
                "durable_reentry_45_date": durable_reentry_45,
                "durable_reentry_45_lag_sessions_from_trough": sessions_between(
                    common_index, trough, durable_reentry_45
                ),
                "durable_reentry_75_date": durable_reentry_75,
                "durable_reentry_75_lag_sessions_from_trough": sessions_between(
                    common_index, trough, durable_reentry_75
                ),
            }
        )

        stages = [
            ("成长资产峰值", peak),
            ("成长仓首次降至45%附近或以下", first_below_45),
            ("成长仓首次降至20%附近或以下", first_at_20),
            ("成长仓首次降至0", first_zero_episode),
            ("成长资产谷底", trough),
            ("谷底后恢复至少20%", first_reentry_20),
            ("谷底后恢复至少45%", first_reentry_45),
            ("谷底后恢复至少75%", first_reentry_75),
            ("事件内最后一次归零", last_zero_episode),
            ("最后归零后稳定恢复至少20%", durable_reentry_20),
            ("最后归零后稳定恢复至少45%", durable_reentry_45),
            ("最后归零后稳定恢复至少75%", durable_reentry_75),
            ("成长资产恢复前高", recovery),
        ]
        seen: set[tuple[str, pd.Timestamp]] = set()
        for stage, date in stages:
            if pd.isna(date):
                continue
            key = (stage, pd.Timestamp(date))
            if key in seen:
                continue
            seen.add(key)
            add_timeline_row(
                timeline_rows,
                event,
                stage,
                pd.Timestamp(date),
                weights,
                diagnostics,
            )

    summary = pd.DataFrame(event_rows)
    timeline = pd.DataFrame(timeline_rows).sort_values(
        ["event", "date"],
        kind="stable",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "event_summary.csv", index=False)
    timeline.to_csv(args.output_dir / "event_timelines.csv", index=False)
    diagnostics.to_csv(args.output_dir / "daily_defense_diagnostics.csv")

    display = summary[
        [
            "event",
            "peak_date",
            "trough_date",
            "growth_max_drawdown",
            "strategy_peak_to_trough_return",
            "loss_avoided_at_growth_trough",
            "average_growth_weight_peak_to_trough",
            "minimum_growth_weight_peak_to_trough",
            "first_zero_date",
            "zero_weight_sessions_peak_to_recovery",
            "reentry_45_lag_sessions_from_trough",
            "rebound_capture_ratio",
        ]
    ].copy()
    print(display.to_string(index=False))
    print(f"\nArtifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
