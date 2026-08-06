from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def derived_signal_series(
    prices: pd.DataFrame,
    specification: dict[str, object],
    annualization: int = 252,
) -> pd.Series:
    """Build an unlagged derived signal from information available at each close."""
    kind = str(specification["kind"])
    name = str(specification["name"])
    if kind != "variance_risk_premium":
        raise ValueError(f"Unsupported derived signal kind: {kind}")
    implied_signal = str(specification["implied_signal"])
    realized_asset = str(specification["realized_asset"])
    realized_days = int(specification["realized_days"])
    if realized_days <= 1:
        raise ValueError("VRP realized-variance window must exceed one day")
    missing = [column for column in (implied_signal, realized_asset) if column not in prices]
    if missing:
        raise ValueError(f"Derived signal is missing price columns: {missing}")
    realized_returns = np.log(prices[realized_asset]).diff()
    realized_variance = realized_returns.pow(2).rolling(realized_days).mean() * annualization
    implied_variance = np.square(prices[implied_signal].astype(float) / 100.0)
    return (implied_variance - realized_variance).rename(f"derived__{name}")


def build_causal_features(
    prices: pd.DataFrame,
    regime_assets: list[str],
    volatility_days: int = 60,
    momentum_days: int = 20,
    derived_signals: list[dict[str, object]] | None = None,
    annualization: int = 252,
    components: list[str] | None = None,
) -> pd.DataFrame:
    """Build x_t exclusively from observations available at the close of t-1."""
    returns = log_returns(prices[regime_assets])
    lagged_return = returns.shift(1)
    lagged_volatility = returns.rolling(volatility_days).std(ddof=1).shift(1)
    lagged_downside_volatility = (
        returns.clip(upper=0.0).pow(2).rolling(volatility_days).mean().pow(0.5).shift(1)
    )
    fast_volatility = returns.rolling(momentum_days).std(ddof=1)
    slow_volatility = returns.rolling(volatility_days).std(ddof=1)
    lagged_volatility_acceleration = (
        fast_volatility.div(slow_volatility.where(slow_volatility > 1e-12))
        .sub(1.0)
        .shift(1)
    )
    lagged_momentum = returns.rolling(momentum_days).mean().shift(1)

    selected_components = components or ["return", "volatility", "momentum"]
    unknown = sorted(
        set(selected_components).difference(
            {
                "return",
                "volatility",
                "downside_volatility",
                "volatility_acceleration",
                "momentum",
            }
        )
    )
    if unknown:
        raise ValueError(f"Unsupported feature components: {unknown}")
    candidates = {
        "return": ("ret1", lagged_return),
        "volatility": (f"vol{volatility_days}", lagged_volatility),
        "downside_volatility": (
            f"downvol{volatility_days}",
            lagged_downside_volatility,
        ),
        "volatility_acceleration": (
            f"volaccel{momentum_days}_{volatility_days}",
            lagged_volatility_acceleration,
        ),
        "momentum": (f"mean{momentum_days}", lagged_momentum),
    }
    parts: list[pd.DataFrame] = []
    for component in selected_components:
        name, frame = candidates[component]
        renamed = frame.copy()
        renamed.columns = [f"{name}__{column}" for column in frame.columns]
        parts.append(renamed)
    for specification in derived_signals or []:
        signal = derived_signal_series(prices, specification, annualization).shift(1)
        parts.append(signal.to_frame())
    if not parts:
        raise ValueError("At least one base or derived feature is required")
    return pd.concat(parts, axis=1).dropna(how="any")


def build_next_feature(
    prices: pd.DataFrame,
    regime_assets: list[str],
    volatility_days: int = 60,
    momentum_days: int = 20,
    derived_signals: list[dict[str, object]] | None = None,
    annualization: int = 252,
    components: list[str] | None = None,
) -> pd.Series:
    """Feature vector for the next session, using the latest completed close."""
    returns = log_returns(prices[regime_assets])
    if len(returns.dropna()) < max(volatility_days, momentum_days):
        raise ValueError("Insufficient history for the configured feature windows")
    selected_components = components or ["return", "volatility", "momentum"]
    unknown = sorted(
        set(selected_components).difference(
            {
                "return",
                "volatility",
                "downside_volatility",
                "volatility_acceleration",
                "momentum",
            }
        )
    )
    if unknown:
        raise ValueError(f"Unsupported feature components: {unknown}")
    candidates = {
        "return": (
            returns.iloc[-1].to_numpy(),
            [f"ret1__{asset}" for asset in regime_assets],
        ),
        "volatility": (
            returns.iloc[-volatility_days:].std(ddof=1).to_numpy(),
            [f"vol{volatility_days}__{asset}" for asset in regime_assets],
        ),
        "downside_volatility": (
            returns.iloc[-volatility_days:]
            .clip(upper=0.0)
            .pow(2)
            .mean()
            .pow(0.5)
            .to_numpy(),
            [f"downvol{volatility_days}__{asset}" for asset in regime_assets],
        ),
        "volatility_acceleration": (
            returns.iloc[-momentum_days:]
            .std(ddof=1)
            .div(returns.iloc[-volatility_days:].std(ddof=1).where(lambda x: x > 1e-12))
            .sub(1.0)
            .to_numpy(),
            [
                f"volaccel{momentum_days}_{volatility_days}__{asset}"
                for asset in regime_assets
            ],
        ),
        "momentum": (
            returns.iloc[-momentum_days:].mean().to_numpy(),
            [f"mean{momentum_days}__{asset}" for asset in regime_assets],
        ),
    }
    value_parts = [candidates[component][0] for component in selected_components]
    names = [
        name
        for component in selected_components
        for name in candidates[component][1]
    ]
    for specification in derived_signals or []:
        signal = derived_signal_series(prices, specification, annualization)
        if signal.empty or pd.isna(signal.iloc[-1]):
            raise ValueError("Insufficient history for a configured derived signal")
        value_parts.append(np.asarray([float(signal.iloc[-1])]))
        names.append(str(signal.name))
    if not value_parts:
        raise ValueError("At least one base or derived feature is required")
    values = np.concatenate(value_parts)
    return pd.Series(values, index=names, name=prices.index[-1] + pd.offsets.BDay(1))
