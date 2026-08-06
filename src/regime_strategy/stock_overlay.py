from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class BasketRiskEstimate:
    risk_multiple: float
    long_volatility_ratio: float
    short_volatility_ratio: float
    tail_loss_ratio: float
    beta: float
    observations: int


@dataclass(frozen=True)
class ActiveRiskEstimate:
    long_tracking_error: float
    short_tracking_error: float
    tail_loss: float
    basket_beta: float
    observations: int


@dataclass(frozen=True)
class ReplacementAllocation:
    stock_weights: pd.Series
    risk_budget: float
    actual_stock_weight: float
    cash_reserve_weight: float
    risk: BasketRiskEstimate


def _normalized_mix(
    tickers: Iterable[str],
    desired_mix: pd.Series | None,
) -> pd.Series:
    names = list(tickers)
    if not names:
        raise ValueError("At least one stock is required")
    if len(set(names)) != len(names):
        raise ValueError("Stock names must be unique")
    if desired_mix is None:
        return pd.Series(1.0 / len(names), index=names, dtype=float)
    mix = desired_mix.reindex(names).astype(float)
    if mix.isna().any() or (mix < 0.0).any() or float(mix.sum()) <= 0.0:
        raise ValueError("Desired stock mix must be finite and non-negative")
    return mix / float(mix.sum())


def _shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    if len(returns) < 20:
        raise ValueError("At least 20 return observations are required")
    return LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_


def _expected_shortfall_loss(returns: pd.Series, probability: float = 0.05) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Expected shortfall requires return observations")
    cutoff = float(clean.quantile(probability))
    tail = clean.loc[clean <= cutoff]
    return max(0.0, -float(tail.mean()))


def estimate_basket_risk(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    desired_mix: pd.Series | None = None,
    *,
    long_window: int = 252,
    short_window: int = 60,
) -> BasketRiskEstimate:
    """Estimate conservative risk of one dollar in a stock basket.

    The estimate uses only the supplied history. It compares the basket with its
    replacement ETF under long-window covariance, short-window covariance, and
    five-percent expected-shortfall loss, then keeps the largest ratio.
    """

    mix = _normalized_mix(stock_returns.columns, desired_mix)
    aligned = pd.concat(
        [stock_returns[mix.index], benchmark_returns.rename("__benchmark__")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < long_window:
        raise ValueError(
            f"At least {long_window} common observations are required"
        )
    long_sample = aligned.iloc[-long_window:]
    short_sample = aligned.iloc[-short_window:]
    long_benchmark_volatility = float(
        long_sample["__benchmark__"].std(ddof=1)
    )
    short_benchmark_volatility = float(
        short_sample["__benchmark__"].std(ddof=1)
    )
    if long_benchmark_volatility <= 0.0 or short_benchmark_volatility <= 0.0:
        raise ValueError("Benchmark volatility must be positive")

    long_covariance = _shrunk_covariance(long_sample[mix.index])
    short_covariance = _shrunk_covariance(short_sample[mix.index])
    vector = mix.to_numpy(dtype=float)
    long_basket_volatility = float(
        np.sqrt(vector @ long_covariance @ vector)
    )
    short_basket_volatility = float(
        np.sqrt(vector @ short_covariance @ vector)
    )
    long_ratio = long_basket_volatility / long_benchmark_volatility
    short_ratio = short_basket_volatility / short_benchmark_volatility

    basket_returns = long_sample[mix.index].mul(mix, axis=1).sum(axis=1)
    benchmark_tail_loss = _expected_shortfall_loss(
        long_sample["__benchmark__"]
    )
    basket_tail_loss = _expected_shortfall_loss(basket_returns)
    tail_ratio = (
        basket_tail_loss / benchmark_tail_loss
        if benchmark_tail_loss > 0.0
        else 1.0
    )
    benchmark_variance = float(
        long_sample["__benchmark__"].var(ddof=1)
    )
    basket_covariance = float(
        basket_returns.cov(long_sample["__benchmark__"])
    )
    beta = (
        basket_covariance / benchmark_variance
        if benchmark_variance > 0.0
        else 0.0
    )
    risk_multiple = max(1.0, long_ratio, short_ratio, tail_ratio)
    return BasketRiskEstimate(
        risk_multiple=risk_multiple,
        long_volatility_ratio=long_ratio,
        short_volatility_ratio=short_ratio,
        tail_loss_ratio=tail_ratio,
        beta=beta,
        observations=len(long_sample),
    )


def estimate_active_risk(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    desired_mix: pd.Series | None = None,
    *,
    long_window: int = 252,
    short_window: int = 60,
    annualization: int = 252,
) -> ActiveRiskEstimate:
    """Estimate the risk added by replacing an ETF with a stock basket."""

    mix = _normalized_mix(stock_returns.columns, desired_mix)
    aligned = pd.concat(
        [stock_returns[mix.index], benchmark_returns.rename("__benchmark__")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < long_window:
        raise ValueError(
            f"At least {long_window} common observations are required"
        )
    long_sample = aligned.iloc[-long_window:]
    short_sample = aligned.iloc[-short_window:]
    long_basket = long_sample[mix.index].mul(mix, axis=1).sum(axis=1)
    short_basket = short_sample[mix.index].mul(mix, axis=1).sum(axis=1)
    long_active = long_basket - long_sample["__benchmark__"]
    short_active = short_basket - short_sample["__benchmark__"]
    benchmark_variance = float(
        long_sample["__benchmark__"].var(ddof=1)
    )
    basket_covariance = float(
        long_basket.cov(long_sample["__benchmark__"])
    )
    return ActiveRiskEstimate(
        long_tracking_error=float(
            long_active.std(ddof=1) * np.sqrt(annualization)
        ),
        short_tracking_error=float(
            short_active.std(ddof=1) * np.sqrt(annualization)
        ),
        tail_loss=_expected_shortfall_loss(long_active),
        basket_beta=(
            basket_covariance / benchmark_variance
            if benchmark_variance > 0.0
            else 0.0
        ),
        observations=len(long_sample),
    )


def active_relative_strength_confirmed(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    desired_mix: pd.Series | None = None,
    *,
    windows: Sequence[int] = (126,),
    minimum_active_log_return: float = 0.0,
) -> bool:
    """Confirm that a stock basket has outperformed its ETF over every window.

    Only the supplied return history is used. The signal is intended for a
    decision made after the latest close and applied on the next session.
    """

    clean_windows = tuple(int(window) for window in windows)
    if not clean_windows or any(window <= 0 for window in clean_windows):
        raise ValueError("Relative-strength windows must be positive")
    mix = _normalized_mix(stock_returns.columns, desired_mix)
    aligned = pd.concat(
        [stock_returns[mix.index], benchmark_returns.rename("__benchmark__")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < max(clean_windows):
        return False
    basket_returns = aligned[mix.index].mul(mix, axis=1).sum(axis=1)
    active_log_returns = np.log1p(basket_returns) - np.log1p(
        aligned["__benchmark__"]
    )
    return all(
        float(active_log_returns.iloc[-window:].sum())
        > minimum_active_log_return
        for window in clean_windows
    )


def allocate_joint_replacement(
    risk_budget: float,
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    desired_mix: pd.Series | None = None,
    *,
    maximum_name_weight: float = 0.075,
    maximum_basket_weight: float = 0.25,
) -> ReplacementAllocation:
    """Size a stock basket inside an ETF-denominated risk budget."""

    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("Risk budget must be between zero and one")
    if not 0.0 < maximum_name_weight <= 1.0:
        raise ValueError("Maximum name weight must be positive")
    if not 0.0 < maximum_basket_weight <= 1.0:
        raise ValueError("Maximum basket weight must be positive")

    mix = _normalized_mix(stock_returns.columns, desired_mix)
    risk = estimate_basket_risk(
        stock_returns,
        benchmark_returns,
        mix,
    )
    actual_stock_weight = min(
        risk_budget / risk.risk_multiple,
        maximum_basket_weight,
    )
    if float(mix.max()) > 0.0:
        actual_stock_weight = min(
            actual_stock_weight,
            maximum_name_weight / float(mix.max()),
        )
    weights = mix * actual_stock_weight
    return ReplacementAllocation(
        stock_weights=weights,
        risk_budget=risk_budget,
        actual_stock_weight=float(weights.sum()),
        cash_reserve_weight=max(0.0, risk_budget - float(weights.sum())),
        risk=risk,
    )


def allocate_standalone_replacement(
    risk_budget: float,
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    desired_mix: pd.Series | None = None,
    *,
    maximum_name_weight: float = 0.075,
    maximum_basket_weight: float = 0.25,
) -> ReplacementAllocation:
    """Size every stock as if its idiosyncratic risk could not diversify."""

    mix = _normalized_mix(stock_returns.columns, desired_mix)
    individual_weights: dict[str, float] = {}
    individual_risks: list[BasketRiskEstimate] = []
    for ticker, share in mix.items():
        risk = estimate_basket_risk(
            stock_returns[[ticker]],
            benchmark_returns,
        )
        individual_risks.append(risk)
        individual_weights[ticker] = min(
            risk_budget * float(share) / risk.risk_multiple,
            maximum_name_weight,
        )
    weights = pd.Series(individual_weights, dtype=float)
    if float(weights.sum()) > maximum_basket_weight:
        weights *= maximum_basket_weight / float(weights.sum())
    conservative_risk = BasketRiskEstimate(
        risk_multiple=max(risk.risk_multiple for risk in individual_risks),
        long_volatility_ratio=max(
            risk.long_volatility_ratio for risk in individual_risks
        ),
        short_volatility_ratio=max(
            risk.short_volatility_ratio for risk in individual_risks
        ),
        tail_loss_ratio=max(
            risk.tail_loss_ratio for risk in individual_risks
        ),
        beta=max(risk.beta for risk in individual_risks),
        observations=min(risk.observations for risk in individual_risks),
    )
    return ReplacementAllocation(
        stock_weights=weights,
        risk_budget=risk_budget,
        actual_stock_weight=float(weights.sum()),
        cash_reserve_weight=max(0.0, risk_budget - float(weights.sum())),
        risk=conservative_risk,
    )


def regime_replacement_share(
    state: str,
    growth_exposure: float,
    *,
    confirmed_growth_threshold: float = 0.35,
    replacement_shares: Mapping[str, float] | None = None,
) -> float:
    """Return the ETF risk-budget share available to managed individual stocks."""

    if growth_exposure <= confirmed_growth_threshold + 1.0e-12:
        return 0.0
    shares = replacement_shares or {
        "quiet_bull": 0.50,
        "normal_bull": 0.50,
        "fragile_bull": 0.25,
        "rebound": 0.25,
    }
    share = float(shares.get(state, 0.0))
    if not 0.0 <= share <= 1.0:
        raise ValueError("Replacement shares must be between zero and one")
    return share
