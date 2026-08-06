from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .features import simple_returns
from .portfolio import (
    apply_drawdown_overlay,
    financing_spread_cost,
    transaction_cost,
)
from .report import performance_metrics
from .schedule import anchored_business_day_position


@dataclass
class TrendSleeveResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame


@dataclass
class HybridResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    sleeve_weights: pd.DataFrame
    trend_signals: pd.DataFrame


def trend_target_weights(
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    momentum_days: int,
    fast_volatility_days: int,
    slow_volatility_days: int,
    target_volatility: float,
    max_gross_leverage: float,
) -> tuple[np.ndarray, bool, float, float]:
    required = max(momentum_days, slow_volatility_days)
    if len(historical_returns) < required:
        raise ValueError("Insufficient history for the trend sleeve")
    growth_history = historical_returns[growth_assets]
    momentum = (
        (1.0 + growth_history.iloc[-momentum_days:]).prod(axis=0) - 1.0
    )
    trend_on = float(momentum.mean()) > 0.0
    target = np.zeros(len(assets), dtype=float)
    if not trend_on:
        target[cash_index] = 1.0
        return target, False, 0.0, 0.0

    unit_weights = np.full(len(growth_assets), 1.0 / len(growth_assets))
    fast_covariance = (
        growth_history.iloc[-fast_volatility_days:].cov().to_numpy() * 252
    )
    slow_covariance = (
        growth_history.iloc[-slow_volatility_days:].cov().to_numpy() * 252
    )
    fast_volatility = float(
        np.sqrt(max(unit_weights @ fast_covariance @ unit_weights, 0.0))
    )
    slow_volatility = float(
        np.sqrt(max(unit_weights @ slow_covariance @ unit_weights, 0.0))
    )
    sizing_volatility = max(fast_volatility, slow_volatility, 1e-12)
    gross_exposure = min(
        max_gross_leverage,
        target_volatility / sizing_volatility,
    )
    for asset, unit_weight in zip(growth_assets, unit_weights, strict=True):
        target[assets.index(asset)] = gross_exposure * unit_weight
    target[cash_index] = 1.0 - gross_exposure
    return target, True, gross_exposure, sizing_volatility


class TrendSleeveBacktester:
    def __init__(self, config: dict[str, object]):
        self.config = config

    def run(self, prices: pd.DataFrame) -> TrendSleeveResult:
        assets = [str(asset) for asset in self.config["assets"]]  # type: ignore[union-attr]
        growth_assets = [
            str(asset) for asset in self.config["growth_assets"]  # type: ignore[union-attr]
        ]
        cash_index = assets.index(str(self.config["cash_asset"]))
        returns = simple_returns(prices[assets]).dropna(how="any")
        dates = returns.index[returns.index >= pd.Timestamp(self.config["oos_start"])]
        weights = np.zeros(len(assets), dtype=float)
        weights[cash_index] = 1.0
        equity = 1.0
        peak = 1.0
        daily_rows: list[dict[str, float]] = []
        weight_rows: list[np.ndarray] = []
        signal_rows: list[dict[str, float | int | pd.Timestamp]] = []
        rebalance_days = int(self.config["rebalance_every_days"])
        annualization = int(self.config.get("annualization", 252))

        for position, date in enumerate(dates):
            trading_cost = 0.0
            turnover = 0.0
            drawdown_before = equity / peak - 1.0
            schedule_position = (
                anchored_business_day_position(date)
                if bool(self.config.get("calendar_anchored_schedule", False))
                else position
            )
            if schedule_position % rebalance_days == 0:
                history = returns.loc[returns.index < date]
                target, trend_on, gross_exposure, sizing_volatility = (
                    trend_target_weights(
                        history,
                        assets,
                        growth_assets,
                        cash_index,
                        int(self.config["momentum_days"]),
                        int(self.config["fast_volatility_days"]),
                        int(self.config["slow_volatility_days"]),
                        float(self.config["target_volatility"]),
                        float(self.config["max_gross_leverage"]),
                    )
                )
                target, drawdown_multiplier = apply_drawdown_overlay(
                    target,
                    cash_index,
                    drawdown_before,
                    self.config["drawdown_overlay"],  # type: ignore[arg-type]
                )
                proposed_turnover = 0.5 * float(np.abs(target - weights).sum())
                if proposed_turnover >= float(self.config["no_trade_turnover"]):
                    trading_cost, turnover = transaction_cost(
                        weights,
                        target,
                        float(self.config["cost_bps_per_dollar_traded"]),
                    )
                    weights = target
                signal_rows.append(
                    {
                        "date": date,
                        "trend_on": int(trend_on),
                        "gross_exposure": gross_exposure,
                        "sizing_volatility": sizing_volatility,
                        "drawdown_multiplier": drawdown_multiplier,
                    }
                )

            day_returns = returns.loc[date].to_numpy(dtype=float)
            gross_return = float(weights @ day_returns)
            financing_cost = financing_spread_cost(
                weights,
                cash_index,
                float(self.config.get("financing_spread_bps", 0.0)),
                annualization,
            )
            total_cost = trading_cost + financing_cost
            net_return = gross_return - total_cost
            equity *= 1.0 + net_return
            peak = max(peak, equity)
            daily_rows.append(
                {
                    "gross_return": gross_return,
                    "cost": total_cost,
                    "trading_cost": trading_cost,
                    "financing_cost": financing_cost,
                    "net_return": net_return,
                    "turnover": turnover,
                    "equity": equity,
                    "drawdown": equity / peak - 1.0,
                }
            )
            weight_rows.append(weights.copy())

            gross_growth = 1.0 + gross_return
            if gross_growth > 1e-12:
                weights = weights * (1.0 + day_returns) / gross_growth
                risky = np.ones(len(weights), dtype=bool)
                risky[cash_index] = False
                weights[risky] = np.maximum(weights[risky], 0.0)
                weights[cash_index] = 1.0 - weights[risky].sum()
                weights /= weights.sum()

        return TrendSleeveResult(
            pd.DataFrame(daily_rows, index=dates),
            pd.DataFrame(weight_rows, index=dates, columns=assets),
            pd.DataFrame(signal_rows).set_index("date"),
        )


def combine_sleeves(
    hmm: BacktestResult,
    trend: TrendSleeveResult,
    hmm_share: float,
    rebalance_every_days: int,
    no_trade_turnover: float,
    cost_bps_per_dollar_traded: float,
    calendar_anchor: pd.Index | None = None,
) -> HybridResult:
    if not 0.0 < hmm_share < 1.0:
        raise ValueError("HMM sleeve share must be strictly between zero and one")
    dates = hmm.daily.index.intersection(trend.daily.index)
    if dates.empty:
        raise ValueError("The HMM and trend sleeves have no overlapping returns")
    assets = list(hmm.weights.columns)
    if assets != list(trend.weights.columns):
        raise ValueError("The HMM and trend sleeve assets must match")

    hmm_value = hmm_share
    trend_value = 1.0 - hmm_share
    peak = 1.0
    daily_rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    sleeve_rows: list[np.ndarray] = []
    for position, date in enumerate(dates):
        starting_equity = hmm_value + trend_value
        current_hmm_share = hmm_value / starting_equity
        outer_cost = 0.0
        outer_turnover = 0.0
        schedule_position = (
            anchored_business_day_position(date)
            if calendar_anchor is not None
            else position
        )
        if position > 0 and schedule_position % rebalance_every_days == 0:
            proposed_turnover = abs(current_hmm_share - hmm_share)
            if proposed_turnover >= no_trade_turnover:
                traded_notional = 2.0 * proposed_turnover
                outer_cost = (
                    traded_notional * cost_bps_per_dollar_traded / 10_000.0
                )
                investable_equity = starting_equity * (1.0 - outer_cost)
                hmm_value = investable_equity * hmm_share
                trend_value = investable_equity * (1.0 - hmm_share)
                starting_hmm_share = hmm_share
                outer_turnover = proposed_turnover
            else:
                starting_hmm_share = current_hmm_share
        else:
            starting_hmm_share = current_hmm_share
        starting_trend_share = 1.0 - starting_hmm_share
        combined_weights = (
            starting_hmm_share * hmm.weights.loc[date].to_numpy(dtype=float)
            + starting_trend_share * trend.weights.loc[date].to_numpy(dtype=float)
        )
        weight_rows.append(combined_weights)
        sleeve_rows.append(np.array([starting_hmm_share, starting_trend_share]))

        hmm_return = float(hmm.daily.loc[date, "net_return"])
        trend_return = float(trend.daily.loc[date, "net_return"])
        hmm_value *= 1.0 + hmm_return
        trend_value *= 1.0 + trend_return
        ending_equity = hmm_value + trend_value
        net_return = ending_equity / starting_equity - 1.0
        peak = max(peak, ending_equity)
        weighted_internal_cost = float(
            starting_hmm_share * hmm.daily.loc[date, "cost"]
            + starting_trend_share * trend.daily.loc[date, "cost"]
        )
        daily_rows.append(
            {
                "hmm_return": hmm_return,
                "trend_return": trend_return,
                "outer_cost": outer_cost,
                "weighted_internal_cost": weighted_internal_cost,
                "cost": outer_cost + weighted_internal_cost,
                "outer_turnover": outer_turnover,
                "net_return": net_return,
                "equity": ending_equity,
                "drawdown": ending_equity / peak - 1.0,
            }
        )

    return HybridResult(
        pd.DataFrame(daily_rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=assets),
        pd.DataFrame(sleeve_rows, index=dates, columns=["HMM", "TREND"]),
        trend.signals,
    )


def write_hybrid_report(
    result: HybridResult,
    output_dir: str | Path,
    annualization: int = 252,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        {
            "HYBRID": performance_metrics(result.daily["net_return"], annualization),
            "HMM_SLEEVE": performance_metrics(result.daily["hmm_return"], annualization),
            "TREND_SLEEVE": performance_metrics(
                result.daily["trend_return"], annualization
            ),
        }
    ).T
    metrics.loc["HYBRID", "annualized_cost_drag"] = (
        result.daily["cost"].mean() * annualization
    )
    metrics.loc["HYBRID", "average_outer_turnover"] = result.daily[
        "outer_turnover"
    ].mean()
    result.daily.to_csv(destination / "daily_returns.csv")
    result.weights.to_csv(destination / "weights.csv")
    result.sleeve_weights.to_csv(destination / "sleeve_weights.csv")
    result.trend_signals.to_csv(destination / "trend_signals.csv")
    metrics.to_csv(destination / "metrics.csv")
    (destination / "metrics.json").write_text(
        json.dumps(metrics.round(8).to_dict(orient="index"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    annual = result.daily["net_return"].groupby(result.daily.index.year).apply(
        lambda values: pd.Series(performance_metrics(values, annualization))
    )
    annual.to_csv(destination / "annual_metrics.csv")
    return metrics
