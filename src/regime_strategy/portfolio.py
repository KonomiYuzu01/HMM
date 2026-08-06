from __future__ import annotations

from statistics import NormalDist

import cvxpy as cp
import numpy as np
import pandas as pd


def project_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def transaction_cost(
    previous_weights: np.ndarray,
    target_weights: np.ndarray,
    bps_per_dollar_traded: float,
) -> tuple[float, float]:
    traded_notional = float(np.abs(target_weights - previous_weights).sum())
    cost = traded_notional * bps_per_dollar_traded / 10_000.0
    return cost, 0.5 * traded_notional


def multi_horizon_trend_forecast(
    historical_returns: pd.DataFrame,
    horizons: list[int],
    horizon_weights: list[float],
    annual_forecast_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map volatility-normalized multi-horizon momentum to a bounded annual forecast."""
    if len(horizons) != len(horizon_weights) or not horizons:
        raise ValueError("Trend horizons and weights must have the same non-zero length")
    if len(historical_returns) < max(horizons):
        raise ValueError("Insufficient return history for trend forecast")
    weights = np.asarray(horizon_weights, dtype=float)
    weights = weights / weights.sum()
    daily_volatility = historical_returns.iloc[-max(horizons) :].std(ddof=1).to_numpy()
    daily_volatility = np.maximum(daily_volatility, 1e-6)
    score = np.zeros(historical_returns.shape[1], dtype=float)
    for horizon, weight in zip(horizons, weights, strict=True):
        window = historical_returns.iloc[-horizon:].clip(lower=-0.999999)
        log_return = np.log1p(window).sum(axis=0).to_numpy()
        standardized = log_return / (daily_volatility * np.sqrt(horizon))
        score += weight * np.tanh(standardized)
    return annual_forecast_scale * score, score


def time_series_momentum_overlay_weights(
    historical_returns: pd.DataFrame,
    assets: list[str],
    signal_assets: list[str],
    cash_index: int,
    horizons: list[int],
    volatility_lookback_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a self-financing, equal-risk, long-short trend overlay.

    The construction follows Hurst, Ooi, and Pedersen: equally combine the
    signs of 1-, 3-, and 12-month excess returns, equalize standalone risk,
    and scale the combined sleeve to a fixed ex-ante volatility using a
    three-year equally weighted covariance estimate.
    """
    if not signal_assets:
        raise ValueError("Time-series momentum requires at least one asset")
    if any(horizon <= 1 for horizon in horizons) or not horizons:
        raise ValueError("Trend horizons must contain values greater than one day")
    if volatility_lookback_days <= 1 or target_volatility <= 0.0:
        raise ValueError("Time-series momentum risk settings are invalid")
    if assets[cash_index] in signal_assets:
        raise ValueError("The cash asset cannot be a trend market")
    unknown = sorted(
        set(signal_assets + [assets[cash_index]]).difference(historical_returns)
    )
    if unknown:
        raise ValueError(f"Unknown time-series momentum assets: {unknown}")

    overlay = np.zeros(len(assets), dtype=float)
    signals = np.full(len(signal_assets), np.nan)
    minimum_history = max(max(horizons), volatility_lookback_days)
    clean = historical_returns[signal_assets + [assets[cash_index]]].dropna(how="any")
    if len(clean) < minimum_history:
        return overlay, signals, float("nan")

    cash_log_return = np.log1p(
        clean[assets[cash_index]].clip(lower=-0.999999)
    )
    excess_log_returns = pd.DataFrame(
        {
            asset: np.log1p(clean[asset].clip(lower=-0.999999)) - cash_log_return
            for asset in signal_assets
        },
        index=clean.index,
    )
    signals = np.mean(
        np.vstack(
            [
                np.sign(
                    excess_log_returns.iloc[-horizon:].sum(axis=0).to_numpy(
                        dtype=float
                    )
                )
                for horizon in horizons
            ]
        ),
        axis=0,
    )
    risk_window = excess_log_returns.iloc[-volatility_lookback_days:]
    annual_covariance = project_psd(
        risk_window.cov().to_numpy(dtype=float) * annualization
    )
    annual_volatility = np.sqrt(
        np.maximum(np.diag(annual_covariance), 1e-12)
    )
    equal_risk_positions = signals / annual_volatility
    unit_volatility = float(
        np.sqrt(
            max(
                equal_risk_positions
                @ annual_covariance
                @ equal_risk_positions,
                0.0,
            )
        )
    )
    if unit_volatility <= 1e-12:
        return overlay, signals, 0.0
    positions = equal_risk_positions * target_volatility / unit_volatility
    for asset, position in zip(signal_assets, positions, strict=True):
        overlay[assets.index(asset)] = float(position)
    overlay[cash_index] = -float(overlay.sum())
    realized_target = float(
        np.sqrt(max(positions @ annual_covariance @ positions, 0.0))
    )
    return overlay, signals, realized_target


def apply_self_financing_overlay(
    base_weights: np.ndarray,
    overlay_weights: np.ndarray,
    cash_index: int,
    overlay_share: float,
    maximum_gross: float,
) -> tuple[np.ndarray, float, float]:
    """Add as much of a zero-cost overlay as fits inside a gross-risk ceiling."""
    if not 0.0 <= overlay_share <= 1.0:
        raise ValueError("Overlay share must be between zero and one")
    if maximum_gross <= 0.0:
        raise ValueError("Maximum gross exposure must be positive")
    if not np.isclose(base_weights.sum(), 1.0):
        raise ValueError("Base weights must sum to one")
    if not np.isclose(overlay_weights.sum(), 0.0):
        raise ValueError("Overlay weights must be self-financing")

    risky = np.ones(len(base_weights), dtype=bool)
    risky[cash_index] = False

    def combined(scale: float) -> np.ndarray:
        result = base_weights + scale * overlay_weights
        result[cash_index] = 1.0 - float(result[risky].sum())
        return result

    def gross(scale: float) -> float:
        return float(np.abs(combined(scale)[risky]).sum())

    applied_share = overlay_share
    if gross(applied_share) > maximum_gross + 1e-12:
        if gross(0.0) > maximum_gross + 1e-12:
            applied_share = 0.0
        else:
            lower = 0.0
            upper = overlay_share
            for _ in range(60):
                midpoint = (lower + upper) / 2.0
                if gross(midpoint) <= maximum_gross:
                    lower = midpoint
                else:
                    upper = midpoint
            applied_share = lower
    result = combined(applied_share)
    return result, float(applied_share), gross(applied_share)


def self_financing_overlay_share_for_mode(
    configured_share: float,
    activation_mode: str,
    risk_on: bool,
) -> float:
    """Activate a self-financing sleeve permanently or only after risk-off."""
    if not 0.0 <= configured_share <= 1.0:
        raise ValueError("Overlay share must be between zero and one")
    if activation_mode == "always":
        return configured_share
    if activation_mode == "risk_off":
        return 0.0 if risk_on else configured_share
    raise ValueError(f"Unsupported overlay activation mode: {activation_mode}")


def joint_excess_trend_leverage_gate(
    historical_returns: pd.DataFrame,
    signal_assets: list[str],
    cash_asset: str,
    lookback_days: int | list[int] = 252,
) -> tuple[bool, np.ndarray]:
    """Permit incremental leverage only when every sleeve/horizon trend is positive.

    Moskowitz, Ooi, and Pedersen form time-series momentum from an
    instrument's own past 12-month excess return.  This gate uses the same
    causal sign rule only for incremental leverage.  Multiple horizons support
    the fast/slow trend intersection in Goulding, Harvey, and Mazzoleni without
    liquidating or altering the unlevered strategic allocation.
    """
    lookbacks = (
        [int(lookback_days)]
        if isinstance(lookback_days, (int, np.integer))
        else [int(days) for days in lookback_days]
    )
    if not lookbacks or any(days <= 1 for days in lookbacks):
        raise ValueError("Leverage trend lookback must exceed one day")
    if not signal_assets:
        raise ValueError("Leverage trend gate requires at least one signal asset")
    unknown = sorted(set(signal_assets + [cash_asset]).difference(historical_returns))
    if unknown:
        raise ValueError(f"Unknown leverage trend assets: {unknown}")
    clean = historical_returns[signal_assets + [cash_asset]].dropna(how="any")
    if len(clean) < max(lookbacks):
        return False, np.full(len(signal_assets) * len(lookbacks), np.nan)
    scores: list[float] = []
    for days in lookbacks:
        window = clean.iloc[-days:].clip(lower=-0.999999)
        cash_log_return = np.log1p(window[cash_asset]).to_numpy(dtype=float)
        scores.extend(
            float(
                (
                    np.log1p(window[asset]).to_numpy(dtype=float)
                    - cash_log_return
                ).sum()
            )
            for asset in signal_assets
        )
    excess_log_returns = np.asarray(scores, dtype=float)
    return bool(np.all(excess_log_returns > 0.0)), excess_log_returns


def realized_volatility_stress(
    asset_returns: pd.Series,
    volatility_days: int,
    volatility_threshold: float,
    annualization: int = 252,
    volatility_estimator: str = "standard",
) -> bool:
    """Identify the acute-volatility component of the growth stress guard."""
    if len(asset_returns) < volatility_days:
        return False
    if volatility_estimator == "standard":
        volatility = float(
            asset_returns.iloc[-volatility_days:].std(ddof=1) * np.sqrt(annualization)
        )
    elif volatility_estimator == "bipower":
        volatility = annualized_bipower_volatility(
            asset_returns,
            volatility_days,
            annualization,
        )
    else:
        raise ValueError(f"Unsupported stress volatility estimator: {volatility_estimator}")
    return bool(volatility > volatility_threshold)


def apply_growth_stress_guard(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    momentum_days: int,
    volatility_days: int,
    volatility_threshold: float,
    growth_multiplier: float,
    signal_asset: str = "SPX",
    annualization: int = 252,
    volatility_estimator: str = "standard",
) -> tuple[np.ndarray, bool]:
    """Cut cyclical sleeves when equity momentum is negative or short vol is extreme."""
    if len(historical_returns) < max(momentum_days, volatility_days):
        return weights, False
    signal_returns = historical_returns[signal_asset]
    momentum = float(
        np.log1p(signal_returns.iloc[-momentum_days:].clip(lower=-0.999999)).sum()
    )
    volatility_stressed = realized_volatility_stress(
        signal_returns,
        volatility_days,
        volatility_threshold,
        annualization,
        volatility_estimator,
    )
    stressed = momentum < 0.0 or volatility_stressed
    if not stressed:
        return weights, False
    adjusted = weights.copy()
    released = 0.0
    for asset in growth_assets:
        index = assets.index(asset)
        reduction = adjusted[index] * (1.0 - growth_multiplier)
        adjusted[index] *= growth_multiplier
        released += float(reduction)
    adjusted[cash_index] += released
    return adjusted / adjusted.sum(), True


def implied_volatility_term_structure(
    spot_volatility: float,
    three_month_volatility: float,
) -> tuple[float, bool]:
    """Return the VIX/VIX3M ratio and the economically defined backwardation state."""
    levels = np.asarray([spot_volatility, three_month_volatility], dtype=float)
    if not np.isfinite(levels).all() or np.any(levels <= 0.0):
        raise ValueError("Implied-volatility levels must be finite and positive")
    ratio = float(spot_volatility / three_month_volatility)
    return ratio, bool(ratio > 1.0)


def incremental_leverage_term_structure_gate(
    backwardation: bool,
    enabled: bool,
) -> bool:
    """Veto only incremental leverage during an inverted volatility curve."""
    return not enabled or not backwardation


def incremental_growth_floor_term_structure_cap(
    requested_share: float,
    backwardation_cap: float,
    backwardation: bool,
) -> float:
    """Remove only the incremental growth floor during VIX backwardation."""
    if not 0.0 <= requested_share <= 1.0:
        raise ValueError("Growth floor share must be between zero and one")
    if not 0.0 <= backwardation_cap <= requested_share:
        raise ValueError("Backwardation floor cap must not exceed the requested share")
    return backwardation_cap if backwardation else requested_share


def apply_conditional_cash_funded_hedge(
    weights: np.ndarray,
    hedge_index: int,
    cash_index: int,
    account_share: float,
    active: bool,
) -> np.ndarray:
    """Set a small hedge sleeve and fund it through the cash weight."""
    if weights.ndim != 1:
        raise ValueError("Conditional hedge weights must be one-dimensional")
    if hedge_index == cash_index:
        raise ValueError("Conditional hedge asset cannot be the cash asset")
    if not 0 <= hedge_index < len(weights) or not 0 <= cash_index < len(weights):
        raise ValueError("Conditional hedge asset index is out of bounds")
    if not 0.0 <= account_share <= 1.0:
        raise ValueError("Conditional hedge share must be between zero and one")
    target = weights.copy()
    desired_share = account_share if active else 0.0
    change = desired_share - target[hedge_index]
    target[hedge_index] = desired_share
    target[cash_index] -= change
    return target


def annualized_bipower_volatility(
    asset_returns: pd.Series,
    window: int,
    annualization: int = 252,
) -> float:
    """Estimate persistent volatility from adjacent daily absolute log returns."""
    if window <= 1:
        raise ValueError("Bipower window must exceed one day")
    clean = asset_returns.dropna().clip(lower=-0.999999)
    if len(clean) < window + 1:
        return float("nan")
    absolute_log_returns = np.log1p(clean).abs()
    products = absolute_log_returns * absolute_log_returns.shift(1)
    bipower_variance = (
        np.pi / 2.0 * float(products.iloc[-window:].mean()) * annualization
    )
    return float(np.sqrt(max(bipower_variance, 0.0)))


def annualized_downside_semivolatility(
    asset_returns: pd.Series,
    window: int,
    annualization: int = 252,
) -> float:
    """Estimate bad volatility from negative returns on a total-volatility scale.

    Under a continuous symmetric return process, positive and negative realized
    semivariances each converge to half of integrated variance. Multiplying the
    downside semivariance by two therefore makes this estimator comparable to
    ordinary total volatility without fitting a scale parameter.
    """
    if window <= 1:
        raise ValueError("Downside-semivolatility window must exceed one day")
    clean = asset_returns.dropna().iloc[-window:].to_numpy(dtype=float)
    if len(clean) < window:
        return float("nan")
    downside_variance = float(np.mean(np.minimum(clean, 0.0) ** 2))
    return float(np.sqrt(max(2.0 * downside_variance * annualization, 0.0)))


def annualized_semiskew_effective_volatility(
    asset_returns: pd.Series,
    window: int,
    annualization: int = 252,
) -> float:
    """Combine downside volatility and skewness on a volatility scale.

    Batista and Fernandes scale exposure in proportion to upside semivolatility
    divided by downside semivariance. Expressed as a risk denominator and with
    both semivolatilities normalized to total-volatility scale, the corresponding
    effective volatility is downside_volatility squared divided by upside
    volatility. It reduces to ordinary volatility under sign symmetry.
    """
    if window <= 1:
        raise ValueError("Semiskew-volatility window must exceed one day")
    clean = asset_returns.dropna().iloc[-window:].to_numpy(dtype=float)
    if len(clean) < window:
        return float("nan")
    downside_variance = float(np.mean(np.minimum(clean, 0.0) ** 2))
    upside_variance = float(np.mean(np.maximum(clean, 0.0) ** 2))
    downside_volatility = float(
        np.sqrt(max(2.0 * downside_variance * annualization, 0.0))
    )
    upside_volatility = float(
        np.sqrt(max(2.0 * upside_variance * annualization, 0.0))
    )
    return float(
        downside_volatility**2 / max(upside_volatility, 1e-6)
    )


def high_volatility_core_weights(
    covariance: np.ndarray,
    base_weights: np.ndarray,
    signal_returns: pd.Series,
    volatility_days: int,
    lookback_days: int,
    high_quantile: float,
    annualization: int = 252,
) -> tuple[np.ndarray, bool, float, float]:
    """Use inverse volatility only when the signal asset enters its causal high-vol tail."""
    if volatility_days <= 1 or lookback_days <= volatility_days:
        raise ValueError("Conditional core volatility windows are invalid")
    if not 0.0 < high_quantile < 1.0:
        raise ValueError("High-volatility quantile must be inside (0, 1)")
    base = np.asarray(base_weights, dtype=float)
    if np.any(base < 0.0) or base.sum() <= 0.0:
        raise ValueError("Base growth weights must be non-negative and non-empty")
    base = base / base.sum()
    realized = (
        signal_returns.dropna().rolling(volatility_days).std(ddof=1)
        * np.sqrt(annualization)
    ).dropna()
    history = realized.iloc[-lookback_days:]
    if history.empty:
        return base, False, float("nan"), float("nan")
    current = float(history.iloc[-1])
    threshold = float(history.quantile(high_quantile))
    high_state = bool(current >= threshold)
    weights = inverse_volatility_weights(covariance) if high_state else base
    return weights, high_state, current, threshold


def realized_stress_covariance(
    historical_returns: pd.DataFrame,
    fast_days: int,
    slow_days: int,
    annualization: int = 252,
    volatility_estimator: str = "standard",
    correlation_estimator: str = "slow",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a causal covariance with fast-out, slow-recovery asset volatilities."""
    if fast_days <= 1 or slow_days < fast_days:
        raise ValueError("Realized-risk windows must satisfy 1 < fast <= slow")
    clean = historical_returns.dropna(how="any")
    minimum_history = slow_days + 1 if volatility_estimator == "jump_aware" else slow_days
    if len(clean) < minimum_history:
        raise ValueError("Insufficient history for realized-risk covariance")
    if volatility_estimator in {"standard", "jump_aware"}:
        fast_volatility = (
            clean.iloc[-fast_days:].std(ddof=1).to_numpy(dtype=float)
            * np.sqrt(annualization)
        )
    elif volatility_estimator in {"downside", "semiskew"}:
        estimator = (
            annualized_downside_semivolatility
            if volatility_estimator == "downside"
            else annualized_semiskew_effective_volatility
        )
        fast_volatility = np.asarray(
            [
                estimator(clean[asset], fast_days, annualization)
                for asset in clean
            ],
            dtype=float,
        )
    else:
        raise ValueError(
            f"Unsupported realized-risk volatility estimator: {volatility_estimator}"
        )
    if volatility_estimator == "standard":
        slow_volatility = (
            clean.iloc[-slow_days:].std(ddof=1).to_numpy(dtype=float)
            * np.sqrt(annualization)
        )
    elif volatility_estimator == "jump_aware":
        slow_volatility = np.asarray(
            [
                annualized_bipower_volatility(
                    clean[asset], slow_days, annualization
                )
                for asset in clean
            ],
            dtype=float,
        )
    elif volatility_estimator in {"downside", "semiskew"}:
        estimator = (
            annualized_downside_semivolatility
            if volatility_estimator == "downside"
            else annualized_semiskew_effective_volatility
        )
        slow_volatility = np.asarray(
            [
                estimator(clean[asset], slow_days, annualization)
                for asset in clean
            ],
            dtype=float,
        )
    else:
        raise ValueError(
            f"Unsupported realized-risk volatility estimator: {volatility_estimator}"
        )
    effective_volatility = np.maximum(
        np.maximum(fast_volatility, slow_volatility),
        1e-6,
    )
    slow_correlation = clean.iloc[-slow_days:].corr().to_numpy(dtype=float)
    if correlation_estimator == "slow":
        correlation = slow_correlation
    elif correlation_estimator == "stress_max":
        fast_correlation = clean.iloc[-fast_days:].corr().to_numpy(dtype=float)
        correlation = np.maximum(fast_correlation, slow_correlation)
    elif correlation_estimator == "downside_stress_max":
        fast_window = clean.iloc[-fast_days:]
        slow_window = clean.iloc[-slow_days:]
        fast_correlation = fast_window.corr().to_numpy(dtype=float)
        fast_downside_correlation = pd.DataFrame(
            np.minimum(fast_window.to_numpy(dtype=float), 0.0)
        ).corr().to_numpy(dtype=float)
        slow_downside_correlation = pd.DataFrame(
            np.minimum(slow_window.to_numpy(dtype=float), 0.0)
        ).corr().to_numpy(dtype=float)
        correlation = np.maximum.reduce(
            [
                fast_correlation,
                slow_correlation,
                fast_downside_correlation,
                slow_downside_correlation,
            ]
        )
    else:
        raise ValueError(
            f"Unsupported realized-risk correlation estimator: {correlation_estimator}"
        )
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    covariance = project_psd(
        np.outer(effective_volatility, effective_volatility) * correlation
    )
    return covariance, effective_volatility


def capped_inverse_volatility_weights(
    volatility: np.ndarray,
    maximum_weights: np.ndarray,
) -> np.ndarray:
    """Allocate inverse volatility subject to fixed sleeve-share ceilings."""
    volatility = np.asarray(volatility, dtype=float)
    caps = np.asarray(maximum_weights, dtype=float)
    if volatility.ndim != 1 or caps.shape != volatility.shape:
        raise ValueError("Volatility and maximum weights must be aligned vectors")
    if np.any(volatility <= 0.0) or np.any(caps <= 0.0) or np.any(caps > 1.0):
        raise ValueError("Volatility and sleeve caps must be positive")
    if caps.sum() < 1.0 - 1e-12:
        raise ValueError("Sleeve caps cannot fund the full growth budget")

    inverse = 1.0 / volatility
    weights = np.zeros_like(inverse)
    active = np.ones(len(inverse), dtype=bool)
    remaining = 1.0
    while active.any():
        proposal = remaining * inverse[active] / inverse[active].sum()
        active_indices = np.flatnonzero(active)
        violated = proposal > caps[active] + 1e-12
        if not violated.any():
            weights[active_indices] = proposal
            break
        for index in active_indices[violated]:
            weights[index] = caps[index]
            remaining -= caps[index]
            active[index] = False
        if remaining < -1e-12:
            raise ValueError("Sleeve caps are internally inconsistent")
    return weights / weights.sum()


def apply_asset_aware_growth_risk_budget(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    maximum_core_weights: dict[str, float] | None = None,
    annualization: int = 252,
    activate_only_when_breached: bool = False,
    cap_total_volatility: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Reallocate growth risk by asset and apply a non-relevering realized-vol cap.

    When ``activate_only_when_breached`` is true, preserve the proposed growth mix
    unless its account-level realized volatility exceeds the risk budget.  This
    keeps the overlay focused on tail-risk control instead of permanently
    replacing the strategic allocation.
    """
    if target_volatility <= 0.0:
        raise ValueError("Target volatility must be positive")
    unknown = sorted(set(growth_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown growth-risk assets: {unknown}")
    growth_indices = [assets.index(asset) for asset in growth_assets]
    growth_total = float(weights[growth_indices].sum())
    if growth_total <= 1e-12:
        empty = np.zeros(len(growth_assets), dtype=float)
        return weights, empty, empty, 0.0, 0.0

    covariance, effective_volatility = realized_stress_covariance(
        historical_returns[growth_assets],
        fast_days,
        slow_days,
        annualization,
    )
    original_growth_weights = weights[growth_indices]
    original_volatility = float(
        np.sqrt(
            max(
                original_growth_weights
                @ covariance
                @ original_growth_weights,
                0.0,
            )
        )
    )
    if activate_only_when_breached and original_volatility <= target_volatility:
        original_core = original_growth_weights / growth_total
        original_unit_volatility = original_volatility / growth_total
        return (
            weights,
            original_core,
            effective_volatility,
            original_unit_volatility,
            growth_total,
        )
    configured_caps = maximum_core_weights or {}
    unknown_caps = sorted(set(configured_caps).difference(growth_assets))
    if unknown_caps:
        raise ValueError(f"Unknown growth-risk sleeve caps: {unknown_caps}")
    caps = np.asarray(
        [float(configured_caps.get(asset, 1.0)) for asset in growth_assets],
        dtype=float,
    )
    core_weights = capped_inverse_volatility_weights(effective_volatility, caps)
    unit_volatility = float(
        np.sqrt(max(core_weights @ covariance @ core_weights, 0.0))
    )
    volatility_cap = growth_total
    if cap_total_volatility and unit_volatility > 1e-12:
        volatility_cap = min(growth_total, target_volatility / unit_volatility)
    adjusted = weights.copy()
    adjusted[growth_indices] = core_weights * volatility_cap
    adjusted[cash_index] += growth_total - volatility_cap
    return (
        adjusted / adjusted.sum(),
        core_weights,
        effective_volatility,
        unit_volatility,
        float(volatility_cap),
    )


def apply_daily_growth_risk_reduction(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    annualization: int = 252,
    volatility_estimator: str = "standard",
    correlation_estimator: str = "slow",
) -> tuple[np.ndarray, float, float]:
    """Only reduce an existing growth mix when its realized risk breaches the cap."""
    growth_indices = [assets.index(asset) for asset in growth_assets]
    covariance, _ = realized_stress_covariance(
        historical_returns[growth_assets],
        fast_days,
        slow_days,
        annualization,
        volatility_estimator,
        correlation_estimator,
    )
    growth_weights = weights[growth_indices]
    realized_volatility = float(
        np.sqrt(max(growth_weights @ covariance @ growth_weights, 0.0))
    )
    if realized_volatility <= target_volatility or realized_volatility <= 1e-12:
        return weights, realized_volatility, 1.0
    multiplier = target_volatility / realized_volatility
    adjusted = weights.copy()
    released = float(growth_weights.sum() * (1.0 - multiplier))
    adjusted[growth_indices] *= multiplier
    adjusted[cash_index] += released
    return adjusted / adjusted.sum(), realized_volatility, float(multiplier)


def apply_trend_cvar_risk_reduction(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    trend_lookback_days: int,
    fhs_lookback_days: int,
    tail_probability: float,
    target_volatility: float,
    ewma_decay: float = 0.94,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float, bool]:
    """Reduce growth exposure by conditional CVaR only in a downtrend.

    The state switch follows Rickenberg's 200-day TSMOM rule. Conditional
    CVaR uses EWMA-filtered historical simulation so old tail observations
    are rescaled to the current volatility environment. The overlay never
    restores exposure removed by another risk control.
    """
    if trend_lookback_days <= 1:
        raise ValueError("CVaR trend lookback must exceed one day")
    if fhs_lookback_days <= 1:
        raise ValueError("CVaR FHS lookback must exceed one day")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("CVaR tail probability must lie between zero and 0.5")
    if target_volatility <= 0.0:
        raise ValueError("CVaR target volatility must be positive")
    if not 0.0 < ewma_decay < 1.0:
        raise ValueError("CVaR EWMA decay must lie between zero and one")
    unknown = sorted(set(growth_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown CVaR growth assets: {unknown}")

    growth_indices = [assets.index(asset) for asset in growth_assets]
    growth_weights = weights[growth_indices]
    growth_total = float(growth_weights.sum())
    if growth_total <= 1e-12:
        return weights, float("nan"), 1.0, False

    clean = historical_returns[growth_assets].dropna(how="any")
    required_history = max(trend_lookback_days, fhs_lookback_days + 1)
    if len(clean) < required_history:
        return weights, float("nan"), 1.0, False

    unit_weights = growth_weights / growth_total
    unit_returns = clean.to_numpy(dtype=float) @ unit_weights
    trend_window = np.clip(unit_returns[-trend_lookback_days:], -0.999999, None)
    bear_regime = bool(float(np.log1p(trend_window).sum()) <= 0.0)
    if not bear_regime:
        return weights, float("nan"), 1.0, False

    variance = np.empty(len(unit_returns), dtype=float)
    variance[0] = max(float(unit_returns[0] ** 2), 1e-8)
    innovation_weight = 1.0 - ewma_decay
    for index in range(1, len(unit_returns)):
        variance[index] = (
            innovation_weight * float(unit_returns[index - 1] ** 2)
            + ewma_decay * variance[index - 1]
        )
    standardized_losses = -unit_returns / np.sqrt(np.maximum(variance, 1e-12))
    filtered_losses = standardized_losses[-fhs_lookback_days:]
    tail_count = max(1, int(np.ceil(tail_probability * fhs_lookback_days)))
    standardized_cvar = float(
        np.mean(np.partition(filtered_losses, -tail_count)[-tail_count:])
    )
    next_variance = (
        innovation_weight * float(unit_returns[-1] ** 2)
        + ewma_decay * variance[-1]
    )
    unit_cvar = max(float(np.sqrt(next_variance) * standardized_cvar), 0.0)
    if unit_cvar <= 1e-12:
        return weights, unit_cvar, 1.0, True

    normal_quantile = NormalDist().inv_cdf(tail_probability)
    normal_tail_scale = float(
        np.exp(-0.5 * normal_quantile**2)
        / (np.sqrt(2.0 * np.pi) * tail_probability)
    )
    target_daily_cvar = (
        target_volatility / np.sqrt(annualization) * normal_tail_scale
    )
    target_growth_total = min(growth_total, target_daily_cvar / unit_cvar)
    multiplier = target_growth_total / growth_total
    if multiplier >= 1.0:
        return weights, unit_cvar, 1.0, True

    adjusted = weights.copy()
    released = growth_total * (1.0 - multiplier)
    adjusted[growth_indices] *= multiplier
    adjusted[cash_index] += released
    return adjusted / adjusted.sum(), unit_cvar, float(multiplier), True


def apply_daily_controlled_asset_risk_reduction(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    anchor_asset: str,
    controlled_asset: str,
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float]:
    """Meet a pair risk cap by reducing the higher-beta sleeve first.

    Fast and slow correlations are combined conservatively.  The anchor sleeve is
    preserved whenever reducing the controlled sleeve alone can meet the cap.  If
    the anchor sleeve independently breaches the budget, it is reduced only after
    the controlled sleeve reaches zero.  The overlay never restores exposure.
    """
    if target_volatility <= 0.0:
        raise ValueError("Target volatility must be positive")
    if anchor_asset == controlled_asset:
        raise ValueError("Anchor and controlled assets must differ")
    unknown = sorted({anchor_asset, controlled_asset}.difference(assets))
    if unknown:
        raise ValueError(f"Unknown controlled-risk assets: {unknown}")
    covariance, _ = realized_stress_covariance(
        historical_returns[[anchor_asset, controlled_asset]],
        fast_days,
        slow_days,
        annualization,
        correlation_estimator="stress_max",
    )
    anchor_index = assets.index(anchor_asset)
    controlled_index = assets.index(controlled_asset)
    pair = weights[[anchor_index, controlled_index]]
    realized_volatility = float(np.sqrt(max(pair @ covariance @ pair, 0.0)))
    if realized_volatility <= target_volatility or realized_volatility <= 1e-12:
        return weights, realized_volatility, 1.0

    anchor_weight = float(weights[anchor_index])
    controlled_weight = float(weights[controlled_index])
    anchor_volatility = float(np.sqrt(max(covariance[0, 0], 0.0)))
    adjusted_anchor = anchor_weight
    adjusted_controlled = controlled_weight
    anchor_only_risk = anchor_weight * anchor_volatility
    if anchor_only_risk > target_volatility:
        adjusted_controlled = 0.0
        adjusted_anchor = (
            target_volatility / anchor_volatility
            if anchor_volatility > 1e-12
            else 0.0
        )
    else:
        lower = 0.0
        upper = controlled_weight
        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            trial = np.asarray([anchor_weight, midpoint])
            risk = float(np.sqrt(max(trial @ covariance @ trial, 0.0)))
            if risk <= target_volatility:
                lower = midpoint
            else:
                upper = midpoint
        adjusted_controlled = lower

    adjusted = weights.copy()
    released = (
        anchor_weight
        + controlled_weight
        - adjusted_anchor
        - adjusted_controlled
    )
    adjusted[anchor_index] = adjusted_anchor
    adjusted[controlled_index] = adjusted_controlled
    adjusted[cash_index] += released
    multiplier = (
        adjusted_controlled / controlled_weight
        if controlled_weight > 1e-12
        else 1.0
    )
    return adjusted / adjusted.sum(), realized_volatility, float(multiplier)


def apply_daily_growth_risk_target(
    weights: np.ndarray,
    desired_weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float]:
    """Move toward the last scheduled mix, capped by today's stress volatility.

    This is the symmetric counterpart to the reduction-only rule. It can restore
    growth exposure after volatility falls, but never exceeds the most recent
    scheduled allocation or the cash currently available alongside that sleeve.
    """
    if target_volatility <= 0.0:
        raise ValueError("Target volatility must be positive")
    if weights.shape != desired_weights.shape:
        raise ValueError("Current and desired weights must have the same shape")
    unknown = sorted(set(growth_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown growth-risk assets: {unknown}")
    growth_indices = [assets.index(asset) for asset in growth_assets]
    covariance, _ = realized_stress_covariance(
        historical_returns[growth_assets],
        fast_days,
        slow_days,
        annualization,
    )
    desired_growth = np.maximum(desired_weights[growth_indices], 0.0)
    desired_volatility = float(
        np.sqrt(max(desired_growth @ covariance @ desired_growth, 0.0))
    )
    multiplier = (
        min(1.0, target_volatility / desired_volatility)
        if desired_volatility > 1e-12
        else 1.0
    )
    target_growth = desired_growth * multiplier
    available = max(float(weights[growth_indices].sum() + weights[cash_index]), 0.0)
    if target_growth.sum() > available and target_growth.sum() > 1e-12:
        target_growth *= available / target_growth.sum()
    adjusted = weights.copy()
    growth_change = float(target_growth.sum() - weights[growth_indices].sum())
    adjusted[growth_indices] = target_growth
    adjusted[cash_index] -= growth_change
    return adjusted / adjusted.sum(), desired_volatility, float(multiplier)


def apply_daily_growth_risk_reallocation_target(
    weights: np.ndarray,
    desired_weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    maximum_core_weights: dict[str, float] | None = None,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float]:
    """Cap daily growth risk while reallocating away from the volatile sleeve."""
    if target_volatility <= 0.0:
        raise ValueError("Target volatility must be positive")
    if weights.shape != desired_weights.shape:
        raise ValueError("Current and desired weights must have the same shape")
    unknown = sorted(set(growth_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown growth-risk assets: {unknown}")
    growth_indices = [assets.index(asset) for asset in growth_assets]
    covariance, effective_volatility = realized_stress_covariance(
        historical_returns[growth_assets],
        fast_days,
        slow_days,
        annualization,
    )
    desired_growth = np.maximum(desired_weights[growth_indices], 0.0)
    desired_total = float(desired_growth.sum())
    desired_volatility = float(
        np.sqrt(max(desired_growth @ covariance @ desired_growth, 0.0))
    )
    target_growth = desired_growth.copy()
    if desired_volatility > target_volatility and desired_total > 1e-12:
        configured_caps = maximum_core_weights or {}
        unknown_caps = sorted(set(configured_caps).difference(growth_assets))
        if unknown_caps:
            raise ValueError(f"Unknown growth-risk sleeve caps: {unknown_caps}")
        caps = np.asarray(
            [float(configured_caps.get(asset, 1.0)) for asset in growth_assets]
        )
        core = capped_inverse_volatility_weights(effective_volatility, caps)
        unit_volatility = float(
            np.sqrt(max(core @ covariance @ core, 0.0))
        )
        capped_total = min(
            desired_total,
            target_volatility / unit_volatility
            if unit_volatility > 1e-12
            else desired_total,
        )
        target_growth = core * capped_total
    available = max(float(weights[growth_indices].sum() + weights[cash_index]), 0.0)
    if target_growth.sum() > available and target_growth.sum() > 1e-12:
        target_growth *= available / target_growth.sum()
    adjusted = weights.copy()
    growth_change = float(target_growth.sum() - weights[growth_indices].sum())
    adjusted[growth_indices] = target_growth
    adjusted[cash_index] -= growth_change
    multiplier = (
        float(target_growth.sum() / desired_total) if desired_total > 1e-12 else 1.0
    )
    return adjusted / adjusted.sum(), desired_volatility, multiplier


def apply_account_stress_volatility_cap(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    risk_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float]:
    """Scale a complete risky allocation to a causal realized-volatility budget."""
    if target_volatility <= 0.0:
        raise ValueError("Target volatility must be positive")
    unknown = sorted(set(risk_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown account-risk assets: {unknown}")
    risk_indices = [assets.index(asset) for asset in risk_assets]
    covariance, _ = realized_stress_covariance(
        historical_returns[risk_assets],
        fast_days,
        slow_days,
        annualization,
    )
    risk_weights = weights[risk_indices]
    realized_volatility = float(
        np.sqrt(max(risk_weights @ covariance @ risk_weights, 0.0))
    )
    if realized_volatility <= target_volatility or realized_volatility <= 1e-12:
        return weights, realized_volatility, 1.0
    multiplier = target_volatility / realized_volatility
    adjusted = weights.copy()
    released = float(risk_weights.sum() * (1.0 - multiplier))
    adjusted[risk_indices] *= multiplier
    adjusted[cash_index] += released
    return adjusted / adjusted.sum(), realized_volatility, float(multiplier)


def apply_relative_asset_risk_cap(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    anchor_asset: str,
    controlled_asset: str,
    cash_index: int,
    fast_days: int,
    slow_days: int,
    annualization: int = 252,
    activation_volatility_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cap one sleeve's standalone risk at the anchor sleeve's risk.

    The anchor allocation is never increased or reduced. Any excess allocation
    from the controlled sleeve moves to cash, so the overlay cannot re-lever the
    signal that triggered entry.
    """
    if anchor_asset == controlled_asset:
        raise ValueError("Anchor and controlled assets must differ")
    unknown = sorted({anchor_asset, controlled_asset}.difference(assets))
    if unknown:
        raise ValueError(f"Unknown relative-risk assets: {unknown}")
    covariance, effective_volatility = realized_stress_covariance(
        historical_returns[[anchor_asset, controlled_asset]],
        fast_days,
        slow_days,
        annualization,
    )
    anchor_index = assets.index(anchor_asset)
    controlled_index = assets.index(controlled_asset)
    pair_weights = weights[[anchor_index, controlled_index]]
    pair_volatility = float(
        np.sqrt(max(pair_weights @ covariance @ pair_weights, 0.0))
    )
    if (
        activation_volatility_cap is not None
        and pair_volatility <= activation_volatility_cap
    ):
        return weights, effective_volatility, 1.0
    anchor_risk = float(weights[anchor_index] * effective_volatility[0])
    controlled_risk = float(weights[controlled_index] * effective_volatility[1])
    if controlled_risk <= anchor_risk or effective_volatility[1] <= 1e-12:
        return weights, effective_volatility, 1.0
    controlled_cap = anchor_risk / effective_volatility[1]
    multiplier = controlled_cap / weights[controlled_index]
    adjusted = weights.copy()
    reduction = adjusted[controlled_index] - controlled_cap
    adjusted[controlled_index] = controlled_cap
    adjusted[cash_index] += reduction
    return adjusted / adjusted.sum(), effective_volatility, float(multiplier)


def zero_entry_allocation_method(relative_return: float) -> str:
    """Choose cash protection only when the controlled asset lagged the anchor."""
    if np.isfinite(relative_return) and relative_return < 0.0:
        return "relative_asset_cap"
    return "total_volatility"


def relative_momentum_pair_weight(
    anchor_returns: pd.Series,
    tilt_returns: pd.Series,
    horizons: list[int],
    horizon_weights: list[float],
    skip_days: int,
    base_tilt_weight: float,
    max_tilt: float,
) -> tuple[float, float]:
    """Map causal multi-horizon relative momentum to a bounded pair allocation."""
    if len(horizons) != len(horizon_weights) or not horizons:
        raise ValueError("Relative-momentum horizons and weights must align")
    if min(horizons) <= 1 or skip_days < 0:
        raise ValueError("Relative-momentum lookback settings are invalid")
    if not 0.0 <= base_tilt_weight <= 1.0 or max_tilt < 0.0:
        raise ValueError("Relative-momentum allocation bounds are invalid")
    if base_tilt_weight - max_tilt < 0.0 or base_tilt_weight + max_tilt > 1.0:
        raise ValueError("Relative-momentum tilt exceeds the pair budget")

    aligned = pd.concat([anchor_returns, tilt_returns], axis=1).dropna()
    required = max(horizons) + skip_days
    if len(aligned) < required:
        return base_tilt_weight, float("nan")
    relative_log_returns = (
        np.log1p(aligned.iloc[:, 1].clip(lower=-0.999999))
        - np.log1p(aligned.iloc[:, 0].clip(lower=-0.999999))
    )
    signal_history = (
        relative_log_returns.iloc[-required:-skip_days]
        if skip_days
        else relative_log_returns.iloc[-required:]
    )
    daily_relative_volatility = max(float(signal_history.std(ddof=1)), 1e-6)
    weights = np.asarray(horizon_weights, dtype=float)
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Relative-momentum horizon weights must be non-negative")
    weights /= weights.sum()
    score = 0.0
    for horizon, horizon_weight in zip(horizons, weights, strict=True):
        cumulative_relative_return = float(signal_history.iloc[-horizon:].sum())
        standardized = cumulative_relative_return / (
            daily_relative_volatility * np.sqrt(horizon)
        )
        score += float(horizon_weight * np.tanh(standardized))
    tilt_weight = float(
        np.clip(
            base_tilt_weight + max_tilt * score,
            base_tilt_weight - max_tilt,
            base_tilt_weight + max_tilt,
        )
    )
    return tilt_weight, float(score)


def volatility_managed_relative_momentum_weight(
    raw_tilt_weight: float,
    base_tilt_weight: float,
    anchor_returns: pd.Series,
    tilt_returns: pd.Series,
    lookback_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[float, float, float]:
    """Cap active pair risk using the momentum spread's realized volatility.

    Barroso and Santa-Clara estimate momentum variance from six months of
    squared daily returns and scale toward a 12% annual target. Here the same
    causal estimator only reduces the active tilt; it never amplifies a signal.
    """
    if not 0.0 <= raw_tilt_weight <= 1.0:
        raise ValueError("Raw relative-momentum weight must be between zero and one")
    if not 0.0 <= base_tilt_weight <= 1.0:
        raise ValueError("Base relative-momentum weight must be between zero and one")
    if lookback_days <= 1:
        raise ValueError("Relative-momentum volatility lookback must exceed one day")
    if target_volatility <= 0.0:
        raise ValueError("Relative-momentum volatility target must be positive")

    aligned = pd.concat([anchor_returns, tilt_returns], axis=1).dropna()
    if len(aligned) < lookback_days:
        return raw_tilt_weight, float("nan"), 1.0
    spread_returns = (
        aligned.iloc[-lookback_days:, 1].to_numpy(dtype=float)
        - aligned.iloc[-lookback_days:, 0].to_numpy(dtype=float)
    )
    spread_volatility = float(
        np.sqrt(annualization * np.mean(np.square(spread_returns)))
    )
    active_tilt = raw_tilt_weight - base_tilt_weight
    if abs(active_tilt) <= 1e-12 or spread_volatility <= 1e-12:
        return raw_tilt_weight, spread_volatility, 1.0

    active_risk_cap = target_volatility / spread_volatility
    adjusted_active_tilt = float(
        np.sign(active_tilt) * min(abs(active_tilt), active_risk_cap)
    )
    multiplier = abs(adjusted_active_tilt / active_tilt)
    return (
        float(base_tilt_weight + adjusted_active_tilt),
        spread_volatility,
        multiplier,
    )


def relative_momentum_spread_overlay_weights(
    anchor_returns: pd.Series,
    tilt_returns: pd.Series,
    assets: list[str],
    anchor_asset: str,
    tilt_asset: str,
    cash_index: int,
    momentum_score: float,
    lookback_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> tuple[np.ndarray, float, float]:
    """Express a relative-momentum score as a volatility-managed pair spread."""
    if anchor_asset == tilt_asset:
        raise ValueError("Relative-momentum spread assets must be distinct")
    if lookback_days <= 1 or target_volatility <= 0.0:
        raise ValueError("Relative-momentum spread risk settings are invalid")
    if np.isfinite(momentum_score) and not -1.0 <= momentum_score <= 1.0:
        raise ValueError("Relative-momentum score must be between -1 and one")
    overlay = np.zeros(len(assets), dtype=float)
    if not np.isfinite(momentum_score):
        return overlay, float("nan"), 0.0
    aligned = pd.concat([anchor_returns, tilt_returns], axis=1).dropna()
    if len(aligned) < lookback_days:
        return overlay, float("nan"), 0.0
    spread_returns = (
        aligned.iloc[-lookback_days:, 1]
        - aligned.iloc[-lookback_days:, 0]
    ).to_numpy(dtype=float)
    spread_volatility = float(
        np.sqrt(annualization * np.mean(np.square(spread_returns)))
    )
    if spread_volatility <= 1e-12:
        return overlay, spread_volatility, 0.0
    tilt_position = float(
        momentum_score * target_volatility / spread_volatility
    )
    overlay[assets.index(tilt_asset)] = tilt_position
    overlay[assets.index(anchor_asset)] = -tilt_position
    overlay[cash_index] = -float(overlay.sum())
    active_risk = abs(tilt_position) * spread_volatility
    return overlay, spread_volatility, float(active_risk)


def residual_momentum_beta_pair_weight(
    anchor_returns: pd.Series,
    controlled_returns: pd.Series,
    beta_months: int,
    formation_months: int,
    skip_months: int,
    base_controlled_weight: float,
) -> tuple[float, float, float]:
    """Shrink a high-beta sleeve only after negative residual momentum.

    The factor loading is estimated from completed monthly returns over a rolling
    window.  Residual momentum uses the same causal regression and excludes the
    most recent completed month.  A negative signal caps the controlled sleeve at
    equal systematic-risk contribution with the anchor; a non-negative signal
    preserves the strategic allocation.
    """
    if beta_months < formation_months or formation_months <= skip_months:
        raise ValueError("Residual-momentum windows are invalid")
    if skip_months < 0:
        raise ValueError("Residual-momentum skip must be non-negative")
    if not 0.0 <= base_controlled_weight <= 1.0:
        raise ValueError("Base controlled weight must be between zero and one")

    aligned = pd.concat([anchor_returns, controlled_returns], axis=1).dropna()
    if aligned.empty or not isinstance(aligned.index, pd.DatetimeIndex):
        return base_controlled_weight, float("nan"), float("nan")
    current_period = aligned.index[-1].to_period("M")
    completed = aligned.loc[aligned.index.to_period("M") < current_period]
    if completed.empty:
        return base_controlled_weight, float("nan"), float("nan")
    monthly = (1.0 + completed).resample("ME").prod() - 1.0
    monthly = monthly.dropna(how="any")
    if len(monthly) < beta_months:
        return base_controlled_weight, float("nan"), float("nan")

    regression = monthly.iloc[-beta_months:]
    anchor = regression.iloc[:, 0].to_numpy(dtype=float)
    controlled = regression.iloc[:, 1].to_numpy(dtype=float)
    anchor_centered = anchor - anchor.mean()
    denominator = float(anchor_centered @ anchor_centered)
    if denominator <= 1e-12:
        return base_controlled_weight, float("nan"), float("nan")
    beta = float(anchor_centered @ (controlled - controlled.mean()) / denominator)
    alpha = float(controlled.mean() - beta * anchor.mean())
    residual = controlled - alpha - beta * anchor

    signal_residual = residual[-formation_months:]
    if skip_months:
        signal_residual = signal_residual[:-skip_months]
    if len(signal_residual) < 2:
        return base_controlled_weight, float("nan"), beta
    residual_volatility = float(np.std(signal_residual, ddof=1))
    if residual_volatility <= 1e-12:
        score = 0.0
    else:
        score = float(
            signal_residual.sum()
            / (residual_volatility * np.sqrt(len(signal_residual)))
        )

    controlled_weight = base_controlled_weight
    if score < 0.0 and beta > 0.0:
        beta_balanced_weight = 1.0 / (1.0 + beta)
        controlled_weight = min(base_controlled_weight, beta_balanced_weight)
    return float(controlled_weight), score, beta


def reallocate_pair_weights(
    weights: np.ndarray,
    assets: list[str],
    anchor_asset: str,
    tilt_asset: str,
    tilt_weight: float,
) -> tuple[np.ndarray, float]:
    """Redistribute a pair without changing its total account exposure."""
    if anchor_asset == tilt_asset:
        raise ValueError("Relative-momentum pair assets must be distinct")
    if not 0.0 <= tilt_weight <= 1.0:
        raise ValueError("Pair tilt weight must be between zero and one")
    anchor_index = assets.index(anchor_asset)
    tilt_index = assets.index(tilt_asset)
    adjusted = weights.copy()
    pair_total = float(adjusted[anchor_index] + adjusted[tilt_index])
    adjusted[anchor_index] = pair_total * (1.0 - tilt_weight)
    adjusted[tilt_index] = pair_total * tilt_weight
    return adjusted, pair_total


def cap_pair_asset_weight(
    weights: np.ndarray,
    assets: list[str],
    anchor_asset: str,
    controlled_asset: str,
    maximum_account_weight: float,
) -> tuple[np.ndarray, float]:
    """Move a pair sleeve's account-level concentration excess to its anchor."""
    if anchor_asset == controlled_asset:
        raise ValueError("Pair concentration assets must be distinct")
    if not 0.0 <= maximum_account_weight <= 1.0:
        raise ValueError("Maximum account weight must be between zero and one")
    anchor_index = assets.index(anchor_asset)
    controlled_index = assets.index(controlled_asset)
    adjusted = weights.copy()
    excess = max(
        float(adjusted[controlled_index]) - maximum_account_weight,
        0.0,
    )
    adjusted[controlled_index] -= excess
    adjusted[anchor_index] += excess
    return adjusted, float(excess)


def cap_pair_standalone_risk_share(
    weights: np.ndarray,
    assets: list[str],
    anchor_asset: str,
    controlled_asset: str,
    anchor_volatility: float,
    controlled_volatility: float,
    maximum_controlled_risk_share: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """Cap one pair sleeve's standalone-volatility risk share.

    The proxy is ``weight * effective volatility``. It intentionally ignores
    covariance because the two growth sleeves are highly correlated and their
    marginal covariance estimates are unstable during abrupt regime changes.
    Any excess controlled weight moves to the anchor, preserving the pair's
    account exposure.
    """
    if anchor_asset == controlled_asset:
        raise ValueError("Pair risk-share assets must be distinct")
    unknown = sorted({anchor_asset, controlled_asset}.difference(assets))
    if unknown:
        raise ValueError(f"Unknown pair risk-share assets: {unknown}")
    if anchor_volatility <= 0.0 or controlled_volatility <= 0.0:
        raise ValueError("Pair effective volatilities must be positive")
    if not 0.0 < maximum_controlled_risk_share < 1.0:
        raise ValueError("Maximum controlled risk share must be inside (0, 1)")

    anchor_index = assets.index(anchor_asset)
    controlled_index = assets.index(controlled_asset)
    adjusted = weights.copy()
    anchor_weight = float(adjusted[anchor_index])
    controlled_weight = float(adjusted[controlled_index])
    if anchor_weight < 0.0 or controlled_weight < 0.0:
        raise ValueError("Pair risk-share weights must be non-negative")

    pair_total = anchor_weight + controlled_weight
    risk_total = (
        anchor_weight * anchor_volatility
        + controlled_weight * controlled_volatility
    )
    before_share = (
        controlled_weight * controlled_volatility / risk_total
        if risk_total > 1e-12
        else 0.0
    )
    denominator = (
        (1.0 - maximum_controlled_risk_share) * controlled_volatility
        + maximum_controlled_risk_share * anchor_volatility
    )
    maximum_controlled_weight = (
        maximum_controlled_risk_share
        * pair_total
        * anchor_volatility
        / denominator
    )
    reallocated = max(controlled_weight - maximum_controlled_weight, 0.0)
    adjusted[controlled_index] -= reallocated
    adjusted[anchor_index] += reallocated

    after_risk_total = (
        float(adjusted[anchor_index]) * anchor_volatility
        + float(adjusted[controlled_index]) * controlled_volatility
    )
    after_share = (
        float(adjusted[controlled_index])
        * controlled_volatility
        / after_risk_total
        if after_risk_total > 1e-12
        else 0.0
    )
    return (
        adjusted,
        float(reallocated),
        float(before_share),
        float(after_share),
        float(maximum_controlled_weight),
    )


def cap_asset_standalone_risk_contribution(
    weights: np.ndarray,
    asset_index: int,
    cash_index: int,
    effective_volatility: float,
    maximum_standalone_contribution: float,
) -> tuple[np.ndarray, float, float, float]:
    """Cap an asset's account-level ``weight * volatility`` contribution."""
    if asset_index == cash_index:
        raise ValueError("The capped asset cannot be cash")
    if effective_volatility <= 0.0:
        raise ValueError("Effective volatility must be positive")
    if maximum_standalone_contribution < 0.0:
        raise ValueError("Maximum standalone contribution cannot be negative")

    adjusted = weights.copy()
    asset_weight = float(adjusted[asset_index])
    if asset_weight < 0.0:
        raise ValueError("The capped asset weight must be non-negative")
    before_contribution = asset_weight * effective_volatility
    maximum_weight = maximum_standalone_contribution / effective_volatility
    released = max(asset_weight - maximum_weight, 0.0)
    adjusted[asset_index] -= released
    adjusted[cash_index] += released
    after_contribution = float(adjusted[asset_index]) * effective_volatility
    return (
        adjusted,
        float(released),
        float(before_contribution),
        float(after_contribution),
    )


def idiosyncratic_downside_survival_multiplier(
    anchor_returns: pd.Series,
    controlled_returns: pd.Series,
    beta_days: int,
    shock_horizons: list[int],
    soft_z_score: float,
    hard_z_score: float,
    minimum_multiplier: float = 0.0,
) -> tuple[float, float, float]:
    """Measure a recent controlled-asset shock unexplained by its anchor.

    Alpha, beta, and residual volatility are estimated before the longest
    shock window. The most negative standardized residual return across the
    fixed horizons drives a continuous survival multiplier.
    """
    if beta_days <= 2:
        raise ValueError("Idiosyncratic beta window must exceed two days")
    if not shock_horizons or min(shock_horizons) <= 0:
        raise ValueError("Shock horizons must contain positive days")
    if hard_z_score >= soft_z_score:
        raise ValueError("Hard z-score must be below the soft z-score")
    if not 0.0 <= minimum_multiplier <= 1.0:
        raise ValueError("Minimum survival multiplier must be between zero and one")

    pair = pd.concat(
        [
            anchor_returns.rename("anchor"),
            controlled_returns.rename("controlled"),
        ],
        axis=1,
    ).dropna(how="any")
    maximum_horizon = max(shock_horizons)
    if len(pair) < beta_days + maximum_horizon:
        raise ValueError("Insufficient history for idiosyncratic shock estimate")

    reference = pair.iloc[-(beta_days + maximum_horizon) : -maximum_horizon]
    recent = pair.iloc[-maximum_horizon:]
    anchor_variance = float(reference["anchor"].var(ddof=1))
    if anchor_variance <= 1e-12:
        raise ValueError("Anchor variance is too small for a stable beta")
    beta = float(
        reference[["anchor", "controlled"]].cov().iloc[0, 1]
        / anchor_variance
    )
    alpha = float(
        reference["controlled"].mean() - beta * reference["anchor"].mean()
    )
    reference_residual = (
        reference["controlled"] - alpha - beta * reference["anchor"]
    )
    residual_volatility = float(reference_residual.std(ddof=1))
    if residual_volatility <= 1e-12:
        raise ValueError("Residual volatility is too small for a stable z-score")
    recent_residual = recent["controlled"] - alpha - beta * recent["anchor"]
    scores = [
        float(
            recent_residual.iloc[-horizon:].sum()
            / (residual_volatility * np.sqrt(horizon))
        )
        for horizon in shock_horizons
    ]
    downside_score = min(scores)
    if downside_score >= soft_z_score:
        multiplier = 1.0
    elif downside_score <= hard_z_score:
        multiplier = minimum_multiplier
    else:
        interpolation = (
            (downside_score - hard_z_score)
            / (soft_z_score - hard_z_score)
        )
        multiplier = minimum_multiplier + interpolation * (
            1.0 - minimum_multiplier
        )
    return float(multiplier), float(downside_score), beta


def apply_pair_survival_multiplier(
    weights: np.ndarray,
    assets: list[str],
    anchor_asset: str,
    controlled_asset: str,
    multiplier: float,
) -> tuple[np.ndarray, float]:
    """Cap the controlled sleeve's pair share at the survival multiplier.

    Interpreting the multiplier as a share cap makes the rule idempotent:
    repeated daily evaluation of the same signal cannot compound reductions.
    """
    if anchor_asset == controlled_asset:
        raise ValueError("Pair survival assets must be distinct")
    unknown = sorted({anchor_asset, controlled_asset}.difference(assets))
    if unknown:
        raise ValueError(f"Unknown pair survival assets: {unknown}")
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("Pair survival multiplier must be between zero and one")

    anchor_index = assets.index(anchor_asset)
    controlled_index = assets.index(controlled_asset)
    adjusted = weights.copy()
    anchor_weight = float(adjusted[anchor_index])
    controlled_weight = float(adjusted[controlled_index])
    if anchor_weight < 0.0 or controlled_weight < 0.0:
        raise ValueError("Pair survival weights must be non-negative")
    pair_total = anchor_weight + controlled_weight
    maximum_controlled_weight = pair_total * multiplier
    reallocated = max(controlled_weight - maximum_controlled_weight, 0.0)
    adjusted[controlled_index] -= reallocated
    adjusted[anchor_index] += reallocated
    return adjusted, float(reallocated)


def filter_core_weights_by_trend(
    core_weights: dict[str, float],
    assets: list[str],
    trend_scores: np.ndarray,
    filtered_assets: list[str],
    minimum_score: float,
    weak_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Remove weak-trend assets and reallocate their core weight pro rata."""
    if len(trend_scores) != len(assets):
        raise ValueError("Core trend scores must align with the asset universe")
    unknown = sorted(set(filtered_assets).difference(core_weights))
    if unknown:
        raise ValueError(
            f"Core trend-filter assets must have strategic weights: {unknown}"
        )
    if any(asset not in assets for asset in core_weights):
        raise ValueError("Core trend weights contain assets outside the universe")
    if (
        any(weight < 0.0 for weight in core_weights.values())
        or not np.isclose(sum(core_weights.values()), 1.0)
    ):
        raise ValueError("Core trend weights must be non-negative and sum to one")
    weak_targets = weak_weights or {}
    unknown_weak = sorted(set(weak_targets).difference(filtered_assets))
    if unknown_weak:
        raise ValueError(
            f"Weak core weights must belong to filtered assets: {unknown_weak}"
        )
    if any(
        weight < 0.0 or weight > core_weights[asset]
        for asset, weight in weak_targets.items()
    ):
        raise ValueError(
            "Weak core weights must be between zero and their strategic weights"
        )

    weak_assets: list[str] = []
    for asset in filtered_assets:
        score = float(trend_scores[assets.index(asset)])
        if np.isfinite(score) and score <= minimum_score:
            weak_assets.append(asset)
    if not weak_assets:
        return dict(core_weights), 0

    fixed_total = sum(
        float(weak_targets.get(asset, 0.0)) for asset in weak_assets
    )
    flexible_assets = [
        asset for asset in core_weights if asset not in weak_assets
    ]
    flexible_total = sum(core_weights[asset] for asset in flexible_assets)
    if fixed_total >= 1.0 or flexible_total <= 1e-12:
        raise ValueError("Core trend filter cannot remove the entire core")
    scale = (1.0 - fixed_total) / flexible_total
    adjusted = {
        asset: (
            float(weak_targets.get(asset, 0.0))
            if asset in weak_assets
            else float(weight * scale)
        )
        for asset, weight in core_weights.items()
    }
    return (
        adjusted,
        len(weak_assets),
    )


def long_only_trend_defensive_weights(
    historical_returns: pd.DataFrame,
    assets: list[str],
    defensive_assets: list[str],
    cash_index: int,
    trend_scores: np.ndarray,
    volatility_days: int,
    fast_trend_assets: list[str] | None = None,
    fast_trend_days: int | None = None,
) -> tuple[np.ndarray, int]:
    """Allocate risk-off capital by equal volatility risk among positive trends."""
    if volatility_days <= 1:
        raise ValueError("Defensive trend volatility window must exceed one day")
    if len(trend_scores) != len(assets):
        raise ValueError("Defensive trend scores must align with the asset universe")
    if len(historical_returns) < volatility_days:
        raise ValueError("Insufficient history for defensive trend volatility")
    unknown = sorted(set(defensive_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown defensive trend assets: {unknown}")
    fast_assets = set(fast_trend_assets or [])
    unknown_fast = sorted(fast_assets.difference(defensive_assets))
    if unknown_fast:
        raise ValueError(
            f"Fast-trend assets must be in the defensive trend universe: {unknown_fast}"
        )
    if fast_assets and (fast_trend_days is None or fast_trend_days <= 1):
        raise ValueError("Fast-trend window must exceed one day")
    if fast_trend_days is not None and len(historical_returns) < fast_trend_days:
        raise ValueError("Insufficient history for defensive fast trend")

    def fast_trend_is_positive(asset: str) -> bool:
        if asset not in fast_assets:
            return True
        assert fast_trend_days is not None
        return float(
            (1.0 + historical_returns[asset].iloc[-fast_trend_days:]).prod() - 1.0
        ) > 0.0

    selected = [
        asset
        for asset in defensive_assets
        if float(trend_scores[assets.index(asset)]) > 0.0
        and fast_trend_is_positive(asset)
    ]
    target = np.zeros(len(assets), dtype=float)
    if not selected:
        target[cash_index] = 1.0
        return target, 0
    volatility = (
        historical_returns[selected]
        .iloc[-volatility_days:]
        .std(ddof=1)
        .to_numpy(dtype=float)
    )
    inverse_volatility = 1.0 / np.maximum(volatility, 1e-6)
    selected_weights = inverse_volatility / inverse_volatility.sum()
    for asset, weight in zip(selected, selected_weights, strict=True):
        target[assets.index(asset)] = float(weight)
    return target, len(selected)


def update_vix_recovery_bridge(
    current_active: bool,
    previous_backwardation: bool | None,
    current_backwardation: bool,
    base_risk_on: bool,
    recovery_confirmed: bool = True,
) -> bool:
    """Bridge a transient volatility recovery until the base model re-enters risk."""
    if base_risk_on or current_backwardation or not recovery_confirmed:
        return False
    if previous_backwardation:
        return True
    return current_active


def apply_asset_multiplier(
    weights: np.ndarray,
    assets: list[str],
    selected_assets: list[str],
    cash_index: int,
    multiplier: float,
) -> np.ndarray:
    """Scale named long sleeves and transfer released capital to cash."""
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("Asset multiplier must be between zero and one")
    adjusted = weights.copy()
    released = 0.0
    for asset in selected_assets:
        index = assets.index(asset)
        reduction = adjusted[index] * (1.0 - multiplier)
        adjusted[index] *= multiplier
        released += float(reduction)
    adjusted[cash_index] += released
    return adjusted / adjusted.sum()


def blend_equal_weight_core(
    optimized_weights: np.ndarray,
    cash_index: int,
    blend: float,
) -> np.ndarray:
    """Shrink noisy optimized weights toward an equal-weight risky-asset prior."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Strategic core blend must be between zero and one")
    core = np.full(len(optimized_weights), 1.0 / (len(optimized_weights) - 1))
    core[cash_index] = 0.0
    result = (1.0 - blend) * optimized_weights + blend * core
    return result / result.sum()


def blend_strategic_core(
    optimized_weights: np.ndarray,
    assets: list[str],
    cash_index: int,
    blend: float,
    core_weights: dict[str, float],
) -> np.ndarray:
    """Shrink optimized weights toward a configured strategic allocation."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Strategic core blend must be between zero and one")
    core = np.zeros(len(assets), dtype=float)
    for asset, weight in core_weights.items():
        core[assets.index(asset)] = float(weight)
    if core[cash_index] != 0.0:
        raise ValueError("Strategic risk core cannot contain the cash/financing sleeve")
    if not np.isclose(core.sum(), 1.0):
        raise ValueError("Strategic core weights must sum to one")
    result = (1.0 - blend) * optimized_weights + blend * core
    return result / result.sum()


def inverse_volatility_weights(
    covariance: np.ndarray,
    variance_floor: float = 1e-10,
) -> np.ndarray:
    """Allocate equal standalone volatility risk without estimating expected returns."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Covariance must be a non-empty square matrix")
    variances = np.maximum(np.diag(matrix), variance_floor)
    inverse_volatility = 1.0 / np.sqrt(variances)
    return inverse_volatility / inverse_volatility.sum()


def trend_volatility_exposure(
    asset_returns: pd.Series,
    trend_days: int,
    volatility_days: int,
    target_volatility: float,
    max_exposure: float,
    annualization: int = 252,
) -> tuple[float, bool, float]:
    """Size a long-only satellite from a causal price trend and realized volatility."""
    if trend_days <= 1 or volatility_days <= 1:
        raise ValueError("Satellite lookback windows must exceed one day")
    if target_volatility <= 0.0 or not 0.0 <= max_exposure <= 1.0:
        raise ValueError("Satellite risk settings are invalid")
    clean = asset_returns.dropna().clip(lower=-0.999999)
    if len(clean) < max(trend_days, volatility_days):
        return 0.0, False, float("nan")
    wealth_window = (1.0 + clean.iloc[-trend_days:]).cumprod()
    trend_active = bool(wealth_window.iloc[-1] > wealth_window.mean())
    log_window = np.log1p(clean.iloc[-volatility_days:])
    annual_volatility = float(log_window.std(ddof=1) * np.sqrt(annualization))
    if not trend_active or annual_volatility <= 1e-12:
        exposure = max_exposure if trend_active else 0.0
    else:
        exposure = min(max_exposure, target_volatility / annual_volatility)
    return float(exposure), trend_active, annual_volatility


def blend_satellite_allocation(
    base_weights: np.ndarray,
    asset_index: int,
    cash_index: int,
    satellite_share: float,
    satellite_exposure: float,
    total_asset_cap: float,
) -> np.ndarray:
    """Combine account-level sleeves, net overlapping assets, and cap concentration."""
    if asset_index == cash_index:
        raise ValueError("Satellite asset cannot be the cash sleeve")
    if not 0.0 <= satellite_share <= 1.0:
        raise ValueError("Satellite share must be between zero and one")
    if not 0.0 <= satellite_exposure <= 1.0:
        raise ValueError("Satellite exposure must be between zero and one")
    if not 0.0 <= total_asset_cap <= 1.0:
        raise ValueError("Total satellite-asset cap must be between zero and one")
    satellite = np.zeros(len(base_weights), dtype=float)
    satellite[asset_index] = satellite_exposure
    satellite[cash_index] = 1.0 - satellite_exposure
    combined = (1.0 - satellite_share) * base_weights + satellite_share * satellite
    excess = max(float(combined[asset_index]) - total_asset_cap, 0.0)
    combined[asset_index] -= excess
    combined[cash_index] += excess
    return combined / combined.sum()


def satellite_share_for_mode(
    configured_share: float,
    activation_mode: str,
    satellite_active: bool,
    base_risk_on: bool,
) -> float:
    """Select whether a satellite is permanent or only bridges slow risk re-entry."""
    if not 0.0 <= configured_share <= 1.0:
        raise ValueError("Satellite share must be between zero and one")
    if activation_mode == "always":
        return configured_share
    if activation_mode == "risk_off_bridge":
        return configured_share if satellite_active and not base_risk_on else 0.0
    raise ValueError(f"Unsupported satellite activation mode: {activation_mode}")


def compose_probability_allocation(
    defensive_weights: np.ndarray,
    growth_core_weights: np.ndarray,
    cash_index: int,
    growth_exposure: float,
) -> np.ndarray:
    """Map a favorable-regime probability to growth exposure without shorting defenses."""
    if growth_exposure < 0.0:
        raise ValueError("Growth exposure cannot be negative")
    if not np.isclose(defensive_weights.sum(), 1.0):
        raise ValueError("Defensive weights must sum to one")
    if not np.isclose(growth_core_weights.sum(), 1.0):
        raise ValueError("Growth core weights must sum to one")

    if growth_exposure <= 1.0:
        result = (
            growth_exposure * growth_core_weights
            + (1.0 - growth_exposure) * defensive_weights
        )
    else:
        result = growth_exposure * growth_core_weights
        result[cash_index] += 1.0 - growth_exposure
    return result / result.sum()


def apply_risk_off_growth_floor(
    defensive_weights: np.ndarray,
    growth_core_weights: np.ndarray,
    cash_index: int,
    account_share: float,
    risk_on: bool,
) -> tuple[np.ndarray, float]:
    """Retain a fixed growth sleeve during risk-off without altering risk-on targets."""
    if not 0.0 <= account_share <= 1.0:
        raise ValueError("Risk-off growth-floor share must be between zero and one")
    if risk_on:
        return defensive_weights, 0.0
    return (
        compose_probability_allocation(
            defensive_weights,
            growth_core_weights,
            cash_index,
            account_share,
        ),
        account_share,
    )


def risk_off_growth_floor_is_active(
    risk_on: bool,
    stress_guard_active: bool,
    disable_when_stressed: bool,
) -> bool:
    """Keep a growth floor from overriding an already-triggered stress guard."""
    return not risk_on and not (disable_when_stressed and stress_guard_active)


def turning_point_cycle(
    growth_returns: pd.Series,
    cash_returns: pd.Series,
    fast_days: int,
    slow_days: int,
) -> tuple[str, float, float]:
    """Classify Bull, Correction, Bear, and Rebound from causal excess trends."""
    if fast_days <= 0 or slow_days <= fast_days:
        raise ValueError("Turning-point windows require 0 < fast_days < slow_days")
    aligned = pd.concat(
        [
            growth_returns.rename("growth"),
            cash_returns.rename("cash"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < slow_days:
        return "insufficient", float("nan"), float("nan")

    def excess_return(window: int) -> float:
        sample = aligned.iloc[-window:]
        growth_wealth = float((1.0 + sample["growth"]).prod())
        cash_wealth = float((1.0 + sample["cash"]).prod())
        if cash_wealth <= 0.0:
            raise ValueError("Cash wealth must remain positive")
        return growth_wealth / cash_wealth - 1.0

    fast_return = excess_return(fast_days)
    slow_return = excess_return(slow_days)
    slow_positive = slow_return >= 0.0
    fast_positive = fast_return >= 0.0
    if slow_positive and fast_positive:
        state = "bull"
    elif slow_positive:
        state = "correction"
    elif fast_positive:
        state = "rebound"
    else:
        state = "bear"
    return state, fast_return, slow_return


def causal_realized_volatility_state(
    returns: pd.Series,
    window_days: int = 20,
    threshold_lookback_days: int = 756,
    minimum_threshold_observations: int = 504,
    low_quantile: float = 0.45,
    high_quantile: float = 0.90,
    annualization: int = 252,
    estimator: str = "standard",
) -> tuple[str, float]:
    if window_days <= 1:
        raise ValueError("Volatility window must exceed one day")
    if threshold_lookback_days < minimum_threshold_observations:
        raise ValueError(
            "Volatility threshold lookback cannot be shorter than its minimum"
        )
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("Volatility quantiles must be strictly ordered")
    clean = returns.dropna()
    if estimator == "standard":
        realized = (
            clean.rolling(window_days).std(ddof=1)
            * np.sqrt(float(annualization))
        ).dropna()
    elif estimator == "downside_semivolatility":
        downside_squared = clean.clip(upper=0.0).pow(2)
        realized = np.sqrt(
            downside_squared.rolling(window_days).mean()
            * float(annualization)
        ).dropna()
    else:
        raise ValueError(
            f"Unsupported causal volatility estimator: {estimator}"
        )
    if len(realized) < minimum_threshold_observations:
        return "insufficient", float("nan")
    history = realized.iloc[-threshold_lookback_days:]
    current = float(history.iloc[-1])
    low_threshold = float(history.quantile(low_quantile))
    high_threshold = float(history.quantile(high_quantile))
    if current <= low_threshold:
        return "low", current
    if current >= high_threshold:
        return "high", current
    return "medium", current


def realized_downside_variation_share(
    returns: pd.Series,
    window_days: int = 20,
) -> float:
    """Measure how much recent squared variation came from negative returns."""
    if window_days <= 1:
        raise ValueError("Downside-variation window must exceed one day")
    clean = returns.dropna()
    if len(clean) < window_days:
        return float("nan")
    sample = clean.iloc[-window_days:]
    total_variation = float(sample.pow(2).sum())
    if total_variation <= 0.0:
        return float("nan")
    downside_variation = float(sample.clip(upper=0.0).pow(2).sum())
    return downside_variation / total_variation


def realized_jump_variation_share(
    returns: pd.Series,
    window_days: int = 20,
) -> float:
    """Estimate the recent jump share using daily variance minus bipower variance.

    The estimate can be negative in finite samples. Keeping that value rather
    than flooring it at zero makes a zero-threshold gate conservative and
    avoids turning sampling noise into artificial evidence of jumps.
    """
    if window_days <= 1:
        raise ValueError("Jump-variation window must exceed one day")
    clean = returns.dropna().clip(lower=-0.999999)
    if len(clean) < window_days + 1:
        return float("nan")
    log_returns = np.log1p(clean)
    sample = log_returns.iloc[-window_days:]
    realized_variance = float(sample.pow(2).mean())
    if realized_variance <= 0.0:
        return float("nan")
    absolute_log_returns = log_returns.abs()
    bipower_variance = float(
        np.pi
        / 2.0
        * (
            absolute_log_returns
            * absolute_log_returns.shift(1)
        ).iloc[-window_days:].mean()
    )
    return (realized_variance - bipower_variance) / realized_variance


def trailing_compounded_return(
    returns: pd.Series,
    window_days: int,
) -> float:
    """Return the causal compounded return over completed observations."""
    if window_days <= 0:
        raise ValueError("Trailing-return window must be positive")
    clean = returns.dropna().clip(lower=-0.999999)
    if len(clean) < window_days:
        return float("nan")
    return float(np.expm1(np.log1p(clean.iloc[-window_days:]).sum()))


def volatility_regime_gated_relative_momentum_weight(
    momentum_weight: float,
    base_weight: float,
    volatility_state: str,
    allowed_states: list[str],
) -> tuple[float, bool]:
    if not 0.0 <= momentum_weight <= 1.0:
        raise ValueError("Momentum weight must be between zero and one")
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("Base weight must be between zero and one")
    valid_states = {"insufficient", "low", "medium", "high"}
    if volatility_state not in valid_states:
        raise ValueError(f"Unsupported volatility state: {volatility_state}")
    if not set(allowed_states).issubset(valid_states):
        raise ValueError("Allowed volatility states contain an unknown state")
    gated = volatility_state not in allowed_states
    return (base_weight if gated else momentum_weight), gated


def ensure_minimum_growth_exposure(
    weights: np.ndarray,
    growth_core_weights: np.ndarray,
    cash_index: int,
    minimum_exposure: float,
) -> tuple[np.ndarray, float]:
    """Raise growth to a floor while preserving the relative defensive allocation."""
    if weights.shape != growth_core_weights.shape:
        raise ValueError("Weights and growth core must have the same shape")
    if not 0 <= cash_index < len(weights):
        raise ValueError("Cash index is out of range")
    if not 0.0 <= minimum_exposure <= 1.0:
        raise ValueError("Minimum growth exposure must be between zero and one")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to one")
    if np.any(growth_core_weights < 0.0):
        raise ValueError("Growth core weights cannot be negative")
    if not np.isclose(growth_core_weights.sum(), 1.0):
        raise ValueError("Growth core weights must sum to one")

    growth_mask = growth_core_weights > 1e-12
    if growth_mask[cash_index]:
        raise ValueError("Cash cannot be part of the growth core")
    current_exposure = float(weights[growth_mask].sum())
    if current_exposure >= minimum_exposure - 1e-12:
        return weights.copy(), max(current_exposure, 0.0)
    if current_exposure < -1e-12 or current_exposure >= 1.0:
        raise ValueError("Current growth exposure is outside the supported range")

    result = weights.copy()
    defensive_scale = (1.0 - minimum_exposure) / (1.0 - current_exposure)
    result[~growth_mask] *= defensive_scale
    core = growth_core_weights[growth_mask]
    result[growth_mask] = minimum_exposure * core / core.sum()
    result /= result.sum()
    return result, float(result[growth_mask].sum())


def ensure_minimum_growth_group_exposure(
    weights: np.ndarray,
    incremental_core_weights: np.ndarray,
    growth_group_mask: np.ndarray,
    cash_index: int,
    minimum_exposure: float,
) -> tuple[np.ndarray, float]:
    """Raise total growth exposure while directing only the increment."""
    if (
        weights.shape != incremental_core_weights.shape
        or weights.shape != growth_group_mask.shape
    ):
        raise ValueError("Weights, incremental core, and group mask must align")
    if not 0 <= cash_index < len(weights):
        raise ValueError("Cash index is out of range")
    if not 0.0 <= minimum_exposure <= 1.0:
        raise ValueError("Minimum growth exposure must be between zero and one")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to one")
    if np.any(weights < -1e-12):
        raise ValueError("Portfolio weights cannot be negative")
    if np.any(incremental_core_weights < 0.0):
        raise ValueError("Incremental core weights cannot be negative")
    if not np.isclose(incremental_core_weights.sum(), 1.0):
        raise ValueError("Incremental core weights must sum to one")

    group_mask = np.asarray(growth_group_mask, dtype=bool)
    if group_mask[cash_index]:
        raise ValueError("Cash cannot be part of the growth group")
    if not group_mask.any():
        raise ValueError("Growth group cannot be empty")
    incremental_mask = incremental_core_weights > 1e-12
    if np.any(incremental_mask & ~group_mask):
        raise ValueError("Incremental core must remain inside the growth group")

    current_exposure = float(weights[group_mask].sum())
    if current_exposure >= minimum_exposure - 1e-12:
        return weights.copy(), max(current_exposure, 0.0)
    if current_exposure < -1e-12 or current_exposure >= 1.0:
        raise ValueError("Current growth exposure is outside the supported range")

    result = weights.copy()
    defensive_scale = (1.0 - minimum_exposure) / (1.0 - current_exposure)
    result[~group_mask] *= defensive_scale
    increment = minimum_exposure - current_exposure
    result[group_mask] += increment * incremental_core_weights[group_mask]
    result /= result.sum()
    return result, float(result[group_mask].sum())


def update_asymmetric_regime_state(
    current_state: bool | None,
    candidate_state: bool,
    model_refit: bool,
) -> bool:
    """Exit immediately, but re-enter risk only when the HMM is formally refit."""
    if current_state is None:
        return bool(candidate_state and model_refit)
    if current_state and not candidate_state:
        return False
    if not current_state and candidate_state:
        return bool(model_refit)
    return current_state


def paper_regime_candidate(
    leverage_enabled: bool,
    growth_excess_return: float,
    use_auxiliary_trend_filters: bool,
    trend_stress: bool,
    trend_score: float,
    activation_score: float,
) -> bool:
    """Build the paper growth signal, optionally retaining legacy trend overlays."""
    if not leverage_enabled or growth_excess_return <= 0.0:
        return False
    if not use_auxiliary_trend_filters:
        return True
    return not trend_stress and trend_score >= activation_score


def drawdown_conditioned_regime_candidate(
    leverage_enabled: bool,
    growth_excess_return: float,
    trend_stress: bool,
    trend_score: float,
    activation_score: float,
    drawdown: float,
    hmm_guard_drawdown: float,
) -> bool:
    """Let trend lead normally, requiring HMM confirmation after material losses."""
    trend_candidate = (
        leverage_enabled
        and not trend_stress
        and trend_score >= activation_score
    )
    if not trend_candidate:
        return False
    return drawdown > hmm_guard_drawdown or growth_excess_return > 0.0


def apply_risk_on_leverage(
    weights: np.ndarray,
    annual_covariance: np.ndarray,
    cash_index: int,
    target_volatility: float,
    max_gross_leverage: float,
    enabled: bool,
) -> tuple[np.ndarray, float]:
    """Scale a long risky mix above 100%, funding the excess through negative cash."""
    if not enabled:
        gross = float(np.delete(weights, cash_index).sum())
        return weights, gross
    risky = np.ones(len(weights), dtype=bool)
    risky[cash_index] = False
    risky_total = float(weights[risky].sum())
    if risky_total <= 1e-12:
        return weights, 0.0
    unit_mix = np.zeros(len(weights), dtype=float)
    unit_mix[risky] = weights[risky] / risky_total
    unit_volatility = float(
        np.sqrt(max(unit_mix @ annual_covariance @ unit_mix, 0.0))
    )
    if unit_volatility <= 1e-12:
        return weights, risky_total
    gross = min(max_gross_leverage, target_volatility / unit_volatility)
    gross = max(gross, 1.0)
    adjusted = np.zeros(len(weights), dtype=float)
    adjusted[risky] = unit_mix[risky] * gross
    adjusted[cash_index] = 1.0 - gross
    return adjusted, float(gross)


def financing_spread_cost(
    weights: np.ndarray,
    cash_index: int,
    annual_spread_bps: float,
    annualization: int = 252,
    short_borrow_spread_bps: float = 0.0,
) -> float:
    borrowed = max(-float(weights[cash_index]), 0.0)
    risky = np.ones(len(weights), dtype=bool)
    risky[cash_index] = False
    short_notional = float(np.maximum(-weights[risky], 0.0).sum())
    return (
        borrowed * annual_spread_bps
        + short_notional * short_borrow_spread_bps
    ) / 10_000.0 / annualization


def solve_allocation(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    previous_weights: np.ndarray,
    cash_index: int,
    risk_aversion: float,
    turnover_penalty: float,
    max_risky_weight: float,
) -> np.ndarray:
    assets = len(expected_returns)
    covariance = project_psd(covariance)
    weights = cp.Variable(assets)
    objective = cp.Maximize(
        expected_returns @ weights
        - risk_aversion * cp.quad_form(weights, covariance)
        - turnover_penalty * cp.norm1(weights - previous_weights)
    )
    constraints = [cp.sum(weights) == 1.0, weights >= 0.0]
    for index in range(assets):
        if index != cash_index:
            constraints.append(weights[index] <= max_risky_weight)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL")
    except cp.error.SolverError:
        problem.solve(solver="SCS", eps=1e-6)
    if weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Portfolio optimization failed: {problem.status}")
    result = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
    return result / result.sum()


def apply_volatility_target(
    weights: np.ndarray,
    annual_covariance: np.ndarray,
    cash_index: int,
    target_volatility: float,
) -> np.ndarray:
    predicted = float(np.sqrt(max(weights @ annual_covariance @ weights, 0.0)))
    if predicted <= target_volatility or predicted <= 1e-12:
        return weights
    multiplier = target_volatility / predicted
    adjusted = weights.copy()
    risky = np.ones(len(weights), dtype=bool)
    risky[cash_index] = False
    released = float(adjusted[risky].sum() * (1.0 - multiplier))
    adjusted[risky] *= multiplier
    adjusted[cash_index] += released
    return adjusted / adjusted.sum()


def apply_drawdown_overlay(
    weights: np.ndarray,
    cash_index: int,
    drawdown: float,
    tiers: list[dict[str, float]],
) -> tuple[np.ndarray, float]:
    multiplier = 1.0
    for tier in sorted(tiers, key=lambda item: item["threshold"]):
        if drawdown <= float(tier["threshold"]):
            multiplier = float(tier["risky_multiplier"])
            break
    if multiplier >= 1.0:
        return weights, multiplier
    adjusted = weights.copy()
    risky = np.ones(len(weights), dtype=bool)
    risky[cash_index] = False
    released = float(adjusted[risky].sum() * (1.0 - multiplier))
    adjusted[risky] *= multiplier
    adjusted[cash_index] += released
    return adjusted / adjusted.sum(), multiplier
