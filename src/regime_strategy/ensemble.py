from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .report import performance_metrics
from .schedule import anchored_business_day_position


@dataclass
class ModelEnsembleResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    sleeve_weights: pd.DataFrame
    member_returns: pd.DataFrame
    benchmarks: pd.DataFrame
    asset_returns: pd.DataFrame


def combine_model_sleeves(
    members: dict[str, BacktestResult],
    rebalance_every_days: int,
    no_trade_turnover: float,
    cost_bps_per_dollar_traded: float,
    calendar_anchor: pd.Index | None = None,
    net_member_trades: bool = False,
    disagreement_risk_control: dict[str, object] | None = None,
) -> ModelEnsembleResult:
    if len(members) < 2:
        raise ValueError("A model ensemble requires at least two members")
    if rebalance_every_days < 1:
        raise ValueError("Ensemble rebalance interval must be positive")
    names = list(members)
    dates = members[names[0]].daily.index
    assets = list(members[names[0]].weights.columns)
    for result in members.values():
        dates = dates.intersection(result.daily.index)
        if list(result.weights.columns) != assets:
            raise ValueError("All ensemble members must use the same assets")
    if dates.empty:
        raise ValueError("The ensemble members have no overlapping returns")
    if net_member_trades:
        return _combine_netted_model_targets(
            members,
            names,
            dates,
            assets,
            rebalance_every_days,
            no_trade_turnover,
            cost_bps_per_dollar_traded,
            calendar_anchor,
            disagreement_risk_control,
        )
    if isinstance(disagreement_risk_control, dict) and bool(
        disagreement_risk_control.get("enabled", False)
    ):
        raise ValueError("Model-disagreement risk control requires netted execution")

    target_shares = np.full(len(names), 1.0 / len(names))
    sleeve_values = target_shares.copy()
    peak = 1.0
    daily_rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    sleeve_rows: list[np.ndarray] = []
    member_return_rows: list[np.ndarray] = []

    for position, date in enumerate(dates):
        starting_equity = float(sleeve_values.sum())
        starting_shares = sleeve_values / starting_equity
        outer_cost = 0.0
        outer_turnover = 0.0
        schedule_position = (
            anchored_business_day_position(date)
            if calendar_anchor is not None
            else position
        )
        if position > 0 and schedule_position % rebalance_every_days == 0:
            proposed_turnover = 0.5 * float(
                np.abs(starting_shares - target_shares).sum()
            )
            if proposed_turnover >= no_trade_turnover:
                traded_notional = 2.0 * proposed_turnover
                outer_cost = traded_notional * cost_bps_per_dollar_traded / 10_000.0
                investable_equity = starting_equity * (1.0 - outer_cost)
                sleeve_values = investable_equity * target_shares
                starting_shares = target_shares.copy()
                outer_turnover = proposed_turnover

        combined_weights = sum(
            starting_shares[index]
            * members[name].weights.loc[date].to_numpy(dtype=float)
            for index, name in enumerate(names)
        )
        member_day_returns = np.asarray(
            [float(members[name].daily.loc[date, "net_return"]) for name in names]
        )
        weighted_internal_cost = float(
            sum(
                starting_shares[index]
                * float(members[name].daily.loc[date, "cost"])
                for index, name in enumerate(names)
            )
        )
        weighted_internal_turnover = float(
            sum(
                starting_shares[index]
                * float(members[name].daily.loc[date, "turnover"])
                for index, name in enumerate(names)
            )
        )
        sleeve_values *= 1.0 + member_day_returns
        ending_equity = float(sleeve_values.sum())
        net_return = ending_equity / starting_equity - 1.0
        peak = max(peak, ending_equity)

        daily_rows.append(
            {
                "net_return": net_return,
                "cost": outer_cost + weighted_internal_cost,
                "outer_cost": outer_cost,
                "weighted_internal_cost": weighted_internal_cost,
                "turnover": outer_turnover + weighted_internal_turnover,
                "outer_turnover": outer_turnover,
                "weighted_internal_turnover": weighted_internal_turnover,
                "equity": ending_equity,
                "drawdown": ending_equity / peak - 1.0,
            }
        )
        weight_rows.append(np.asarray(combined_weights, dtype=float))
        sleeve_rows.append(starting_shares)
        member_return_rows.append(member_day_returns)

    first = members[names[0]]
    return ModelEnsembleResult(
        daily=pd.DataFrame(daily_rows, index=dates),
        weights=pd.DataFrame(weight_rows, index=dates, columns=assets),
        sleeve_weights=pd.DataFrame(sleeve_rows, index=dates, columns=names),
        member_returns=pd.DataFrame(member_return_rows, index=dates, columns=names),
        benchmarks=first.benchmarks.reindex(dates),
        asset_returns=first.asset_returns.reindex(dates),
    )


def _combine_netted_model_targets(
    members: dict[str, BacktestResult],
    names: list[str],
    dates: pd.Index,
    assets: list[str],
    rebalance_every_days: int,
    no_trade_turnover: float,
    cost_bps_per_dollar_traded: float,
    calendar_anchor: pd.Index | None,
    disagreement_risk_control: dict[str, object] | None,
) -> ModelEnsembleResult:
    """Execute the capital-weighted aggregate target as one brokerage account."""
    first = members[names[0]]
    if "CASH" not in assets:
        raise ValueError("Netted ensemble execution requires a CASH asset")
    target_shares = np.full(len(names), 1.0 / len(names))
    sleeve_values = target_shares.copy()
    equity = 1.0
    peak = 1.0
    current_weights = np.zeros(len(assets), dtype=float)
    current_weights[assets.index("CASH")] = 1.0
    daily_rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    sleeve_rows: list[np.ndarray] = []
    member_return_rows: list[np.ndarray] = []

    for position, date in enumerate(dates):
        starting_shares = sleeve_values / sleeve_values.sum()
        schedule_position = (
            anchored_business_day_position(date)
            if calendar_anchor is not None
            else position
        )
        if position > 0 and schedule_position % rebalance_every_days == 0:
            starting_shares = target_shares.copy()
            sleeve_values = equity * target_shares

        aggregate_target = sum(
            starting_shares[index]
            * members[name].weights.loc[date].to_numpy(dtype=float)
            for index, name in enumerate(names)
        )
        member_targets = np.asarray(
            [members[name].weights.loc[date].to_numpy(dtype=float) for name in names]
        )
        (
            aggregate_target,
            disagreement_multiplier,
            disagreement_reallocated,
            model_growth_disagreement,
        ) = apply_model_disagreement_risk_control(
            np.asarray(aggregate_target, dtype=float),
            member_targets,
            starting_shares,
            assets,
            disagreement_risk_control,
        )
        proposed_turnover = 0.5 * float(
            np.abs(aggregate_target - current_weights).sum()
        )
        executed = proposed_turnover >= no_trade_turnover
        if executed:
            traded_notional = 2.0 * proposed_turnover
            trading_cost = (
                traded_notional * cost_bps_per_dollar_traded / 10_000.0
            )
            current_weights = aggregate_target
            turnover = proposed_turnover
        else:
            trading_cost = 0.0
            turnover = 0.0

        member_day_returns = np.asarray(
            [float(members[name].daily.loc[date, "net_return"]) for name in names]
        )
        member_virtual_returns = np.asarray(
            [
                float(members[name].daily.loc[date, "gross_return"])
                - float(members[name].daily.loc[date, "financing_cost"])
                for name in names
            ]
        )
        weighted_internal_cost = float(
            sum(
                starting_shares[index]
                * float(members[name].daily.loc[date, "cost"])
                for index, name in enumerate(names)
            )
        )
        asset_day_returns = first.asset_returns.loc[
            date, assets
        ].to_numpy(dtype=float)
        gross_return = float(current_weights @ asset_day_returns)
        financing_cost = float(
            sum(
                starting_shares[index]
                * float(members[name].daily.loc[date, "financing_cost"])
                for index, name in enumerate(names)
            )
        )
        net_return = gross_return - financing_cost - trading_cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)

        sleeve_values *= 1.0 + member_virtual_returns
        sleeve_values *= equity / sleeve_values.sum()
        weight_rows.append(current_weights.copy())
        sleeve_rows.append(starting_shares)
        member_return_rows.append(member_day_returns)
        daily_rows.append(
            {
                "net_return": net_return,
                "gross_return": gross_return,
                "cost": trading_cost + financing_cost,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "outer_cost": 0.0,
                "weighted_internal_cost": weighted_internal_cost,
                "turnover": turnover,
                "outer_turnover": 0.0,
                "weighted_internal_turnover": float(
                    sum(
                        starting_shares[index]
                        * float(members[name].daily.loc[date, "turnover"])
                        for index, name in enumerate(names)
                    )
                ),
                "model_disagreement_multiplier": disagreement_multiplier,
                "model_growth_disagreement": model_growth_disagreement,
                "model_disagreement_reallocated": disagreement_reallocated,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )

        denominator = 1.0 + gross_return
        if denominator <= 0.0:
            raise ValueError("Aggregate portfolio lost all capital in one session")
        current_weights = (
            current_weights * (1.0 + asset_day_returns) / denominator
        )

    return ModelEnsembleResult(
        daily=pd.DataFrame(daily_rows, index=dates),
        weights=pd.DataFrame(weight_rows, index=dates, columns=assets),
        sleeve_weights=pd.DataFrame(sleeve_rows, index=dates, columns=names),
        member_returns=pd.DataFrame(
            member_return_rows,
            index=dates,
            columns=names,
        ),
        benchmarks=first.benchmarks.reindex(dates),
        asset_returns=first.asset_returns.reindex(dates),
    )


def apply_model_disagreement_risk_control(
    aggregate_target: np.ndarray,
    member_targets: np.ndarray,
    sleeve_shares: np.ndarray,
    assets: list[str],
    config: dict[str, object] | None,
) -> tuple[np.ndarray, float, float, float]:
    """Haircut growth by its cross-model reliability and move the rest to cash."""
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return aggregate_target.copy(), 1.0, 0.0, 0.0
    growth_assets = [str(asset) for asset in config["assets"]]  # type: ignore[index]
    destination = str(config.get("destination", "CASH"))
    missing = [asset for asset in [*growth_assets, destination] if asset not in assets]
    if missing:
        raise ValueError(f"Disagreement risk control is missing assets: {missing}")
    probabilities = np.maximum(np.asarray(sleeve_shares, dtype=float), 0.0)
    probabilities /= probabilities.sum()
    growth_indices = [assets.index(asset) for asset in growth_assets]
    exposures = member_targets[:, growth_indices].sum(axis=1)
    mean_exposure = float(probabilities @ exposures)
    disagreement = float(
        probabilities @ np.square(exposures - mean_exposure)
    )
    if mean_exposure <= 1e-12:
        return aggregate_target.copy(), 1.0, 0.0, disagreement
    multiplier = float(
        np.square(mean_exposure) / (np.square(mean_exposure) + disagreement)
    )
    result = aggregate_target.copy()
    growth_before = float(result[growth_indices].sum())
    result[growth_indices] *= multiplier
    reallocated = growth_before - float(result[growth_indices].sum())
    result[assets.index(destination)] += reallocated
    return result, multiplier, reallocated, disagreement


def write_ensemble_report(
    result: ModelEnsembleResult,
    output_dir: str | Path,
    annualization: int = 252,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    strategies: dict[str, pd.Series] = {"ENSEMBLE": result.daily["net_return"]}
    strategies.update(
        {f"MEMBER_{name}": result.member_returns[name] for name in result.member_returns}
    )
    strategies.update(
        {name: result.benchmarks[name] for name in result.benchmarks.columns}
    )
    metrics = pd.DataFrame(
        {
            name: performance_metrics(series, annualization)
            for name, series in strategies.items()
        }
    ).T
    cash_returns = result.asset_returns["CASH"]
    for name, series in strategies.items():
        excess = series.reindex(cash_returns.index) - cash_returns
        metrics.loc[name, "cash_excess_sharpe"] = float(
            excess.mean() / excess.std(ddof=1) * np.sqrt(annualization)
        )
    metrics.loc["ENSEMBLE", "average_daily_turnover"] = float(
        result.daily["turnover"].mean()
    )
    metrics.loc["ENSEMBLE", "annualized_cost_drag"] = float(
        result.daily["cost"].mean() * annualization
    )

    result.daily.to_csv(destination / "daily_returns.csv")
    result.weights.to_csv(destination / "weights.csv")
    result.sleeve_weights.to_csv(destination / "sleeve_weights.csv")
    result.member_returns.to_csv(destination / "member_returns.csv")
    metrics.to_csv(destination / "metrics.csv")
    (destination / "metrics.json").write_text(
        json.dumps(metrics.round(8).to_dict(orient="index"), indent=2, allow_nan=True),
        encoding="utf-8",
    )

    annual_rows: list[dict[str, float | int]] = []
    for year, group in result.daily["net_return"].groupby(result.daily.index.year):
        annual_rows.append(
            {"year": int(year), **performance_metrics(group, annualization)}
        )
    pd.DataFrame(annual_rows).set_index("year").to_csv(
        destination / "annual_metrics.csv"
    )

    periods = {
        "development_2015_2021": ("2015-01-01", "2021-12-31"),
        "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
        "complete_2015_2025": ("2015-01-01", "2025-12-31"),
        "holdout_2022_present": ("2022-01-01", None),
    }
    period_rows: list[dict[str, float | str]] = []
    for period, (start, end) in periods.items():
        for name, series in strategies.items():
            sample = series.loc[start:end]
            if sample.empty:
                continue
            period_rows.append(
                {
                    "period": period,
                    "strategy": name,
                    **performance_metrics(sample, annualization),
                }
            )
    if period_rows:
        period_metrics = pd.DataFrame(period_rows).set_index(
            ["period", "strategy"]
        )
    else:
        period_metrics = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["period", "strategy"]
            )
        )
    period_metrics.to_csv(destination / "fixed_period_metrics.csv")
    return metrics
