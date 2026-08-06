from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import annualized_bipower_volatility, project_psd


@dataclass(frozen=True)
class SurvivalResult:
    weights: np.ndarray
    pre_volatility: float
    post_volatility: float
    long_horizon_volatility: float
    stress_volatility_ratio: float
    target_volatility: float
    risk_multiplier: float
    novelty_percentile: float
    novelty_multiplier: float
    gross_before: float
    gross_after: float
    max_risk_share_before: float
    max_risk_share_after: float
    triggered: bool
    active: bool


@dataclass
class SurvivalOverlayState:
    """Keep the alpha target separate from a temporary survival overlay."""

    reference_weights: np.ndarray | None = None

    def begin_alpha_target(self, weights: np.ndarray) -> None:
        self.reference_weights = weights.copy()

    def target_input(self, current_weights: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.reference_weights is None:
            return current_weights.copy(), False
        return self.reference_weights.copy(), True

    def observe_result(
        self,
        input_weights: np.ndarray,
        survival_active: bool,
    ) -> None:
        if survival_active:
            if self.reference_weights is None:
                self.reference_weights = input_weights.copy()
        else:
            self.reference_weights = None


def multi_horizon_stress_covariance(
    historical_returns: pd.DataFrame,
    horizons: list[int],
    annualization: int = 252,
    jump_aware: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a long-only worst-horizon covariance without forecasting returns."""
    if not horizons or any(days <= 1 for days in horizons):
        raise ValueError("Survival horizons must contain values greater than one")
    ordered_horizons = sorted(set(int(days) for days in horizons))
    clean = historical_returns.dropna(how="any")
    minimum_history = max(ordered_horizons) + int(jump_aware)
    if len(clean) < minimum_history:
        raise ValueError("Insufficient history for survival covariance")

    volatility_estimates: list[np.ndarray] = []
    correlation_estimates: list[np.ndarray] = []
    for days in ordered_horizons:
        window = clean.iloc[-days:]
        volatility_estimates.append(
            window.std(ddof=1).to_numpy(dtype=float) * np.sqrt(annualization)
        )
        if jump_aware:
            volatility_estimates.append(
                np.asarray(
                    [
                        annualized_bipower_volatility(
                            clean[asset], days, annualization
                        )
                        for asset in clean
                    ],
                    dtype=float,
                )
            )
        correlation_estimates.append(window.corr().to_numpy(dtype=float))

    effective_volatility = np.maximum.reduce(volatility_estimates)
    effective_volatility = np.maximum(effective_volatility, 1e-6)
    stress_correlation = np.maximum.reduce(correlation_estimates)
    stress_correlation = np.nan_to_num(stress_correlation, nan=0.0)
    stress_correlation = np.clip(stress_correlation, -1.0, 1.0)
    np.fill_diagonal(stress_correlation, 1.0)
    covariance = project_psd(
        np.outer(effective_volatility, effective_volatility)
        * stress_correlation
    )
    return covariance, effective_volatility, stress_correlation


def feature_novelty_percentile(
    historical_features: pd.DataFrame,
    current_feature: pd.Series,
    lookback_days: int,
    shrinkage: float = 0.20,
) -> float:
    """Measure distribution shift without using returns or HMM state labels."""
    if lookback_days <= 2:
        raise ValueError("Novelty lookback must exceed two observations")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Novelty shrinkage must lie between zero and one")
    history = historical_features.reindex(columns=current_feature.index).dropna(
        how="any"
    )
    history = history.iloc[-lookback_days:]
    if len(history) < max(63, len(current_feature) * 2):
        return 1.0

    values = history.to_numpy(dtype=float)
    current = current_feature.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale = np.maximum(scale, 1e-12)
    standardized = (values - mean) / scale
    standardized_current = (current - mean) / scale
    covariance = np.cov(standardized, rowvar=False, ddof=1)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * np.eye(
        covariance.shape[0]
    )
    inverse = np.linalg.pinv(project_psd(shrunk), hermitian=True)
    historical_distance = np.einsum(
        "ij,jk,ik->i", standardized, inverse, standardized
    )
    current_distance = float(standardized_current @ inverse @ standardized_current)
    return float(np.mean(historical_distance <= current_distance))


def novelty_risk_multiplier(
    percentile: float,
    activation_percentile: float,
    minimum_multiplier: float,
) -> float:
    """Continuously reduce risk only in the empirical novelty tail."""
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("Novelty percentile must lie between zero and one")
    if not 0.0 < activation_percentile < 1.0:
        raise ValueError("Novelty activation percentile must lie inside (0, 1)")
    if not 0.0 < minimum_multiplier <= 1.0:
        raise ValueError("Minimum novelty multiplier must lie inside (0, 1]")
    severity = np.clip(
        (percentile - activation_percentile)
        / (1.0 - activation_percentile),
        0.0,
        1.0,
    )
    return float(1.0 - severity * (1.0 - minimum_multiplier))


def _risk_contribution_shares(
    risk_weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    variance = float(risk_weights @ covariance @ risk_weights)
    if variance <= 1e-16:
        return np.zeros_like(risk_weights)
    return risk_weights * (covariance @ risk_weights) / variance


def _reduce_concentrated_risk(
    weights: np.ndarray,
    risk_indices: list[int],
    cash_index: int,
    covariance: np.ndarray,
    maximum_risk_shares: np.ndarray,
) -> np.ndarray:
    adjusted = weights.copy()
    for _ in range(80):
        risk_weights = np.maximum(adjusted[risk_indices], 0.0)
        shares = _risk_contribution_shares(risk_weights, covariance)
        violations = shares - maximum_risk_shares
        if float(violations.max(initial=0.0)) <= 1e-5:
            break
        index = int(np.argmax(violations))
        account_index = risk_indices[index]
        old_weight = float(adjusted[account_index])
        if old_weight <= 1e-12:
            break
        reduction_factor = np.sqrt(
            max(float(maximum_risk_shares[index]), 1e-8)
            / max(float(shares[index]), 1e-8)
        )
        new_weight = old_weight * min(max(reduction_factor * 0.995, 0.0), 0.995)
        adjusted[account_index] = new_weight
        adjusted[cash_index] += old_weight - new_weight
    return adjusted


def apply_survival_governor(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    risk_assets: list[str],
    cash_index: int,
    horizons: list[int],
    target_volatility: float,
    maximum_risk_shares: dict[str, float] | None = None,
    novelty_percentile: float = 0.0,
    novelty_activation_percentile: float = 0.95,
    minimum_novelty_multiplier: float = 0.75,
    maximum_gross_when_novel: float = 1.0,
    annualization: int = 252,
    activation_stress_ratio: float | None = None,
    long_horizon_days: int = 252,
) -> SurvivalResult:
    """Apply a return-agnostic survival budget that can only remove risk."""
    if weights.ndim != 1 or len(weights) != len(assets):
        raise ValueError("Survival weights and assets must align")
    if target_volatility <= 0.0:
        raise ValueError("Survival target volatility must be positive")
    if maximum_gross_when_novel <= 0.0:
        raise ValueError("Novelty gross limit must be positive")
    if activation_stress_ratio is not None and activation_stress_ratio <= 1.0:
        raise ValueError("Survival stress-ratio activation must exceed one")
    if long_horizon_days <= 2:
        raise ValueError("Survival long horizon must exceed two observations")
    unknown = sorted(set(risk_assets).difference(assets))
    if unknown:
        raise ValueError(f"Unknown survival risk assets: {unknown}")
    if assets[cash_index] in risk_assets:
        raise ValueError("Cash cannot be a survival risk asset")

    risk_indices = [assets.index(asset) for asset in risk_assets]
    covariance, _, _ = multi_horizon_stress_covariance(
        historical_returns[risk_assets],
        horizons,
        annualization,
    )
    original = weights.copy()
    adjusted = weights.copy()
    original_risk_weights = np.maximum(original[risk_indices], 0.0)
    pre_volatility = float(
        np.sqrt(max(original_risk_weights @ covariance @ original_risk_weights, 0.0))
    )
    long_window = historical_returns[risk_assets].dropna(how="any").iloc[
        -long_horizon_days:
    ]
    if len(long_window) < long_horizon_days:
        raise ValueError("Insufficient history for survival long covariance")
    long_covariance = project_psd(
        long_window.cov().to_numpy(dtype=float) * annualization
    )
    long_horizon_volatility = float(
        np.sqrt(
            max(
                original_risk_weights
                @ long_covariance
                @ original_risk_weights,
                0.0,
            )
        )
    )
    stress_volatility_ratio = (
        pre_volatility / long_horizon_volatility
        if long_horizon_volatility > 1e-12
        else 1.0
    )
    original_shares = _risk_contribution_shares(
        original_risk_weights, covariance
    )
    max_share_before = float(original_shares.max(initial=0.0))
    gross_before = float(np.abs(np.delete(original, cash_index)).sum())

    novelty_multiplier = novelty_risk_multiplier(
        novelty_percentile,
        novelty_activation_percentile,
        minimum_novelty_multiplier,
    )
    novelty_active = novelty_percentile > novelty_activation_percentile
    covariance_shift_active = (
        activation_stress_ratio is None
        or stress_volatility_ratio >= activation_stress_ratio
    )
    triggered = bool(covariance_shift_active or novelty_active)
    if not triggered:
        return SurvivalResult(
            weights=original,
            pre_volatility=pre_volatility,
            post_volatility=pre_volatility,
            long_horizon_volatility=long_horizon_volatility,
            stress_volatility_ratio=stress_volatility_ratio,
            target_volatility=target_volatility,
            risk_multiplier=1.0,
            novelty_percentile=novelty_percentile,
            novelty_multiplier=novelty_multiplier,
            gross_before=gross_before,
            gross_after=gross_before,
            max_risk_share_before=max_share_before,
            max_risk_share_after=max_share_before,
            triggered=False,
            active=False,
        )
    if novelty_active and gross_before > maximum_gross_when_novel:
        gross_multiplier = maximum_gross_when_novel / gross_before
        risky_before = adjusted[risk_indices].copy()
        adjusted[risk_indices] *= gross_multiplier
        adjusted[cash_index] += float(
            risky_before.sum() - adjusted[risk_indices].sum()
        )

    configured_caps = maximum_risk_shares or {}
    unknown_caps = sorted(set(configured_caps).difference(risk_assets))
    if unknown_caps:
        raise ValueError(f"Unknown survival risk-share caps: {unknown_caps}")
    caps = np.asarray(
        [float(configured_caps.get(asset, 1.0)) for asset in risk_assets],
        dtype=float,
    )
    if np.any(caps <= 0.0) or np.any(caps > 1.0) or caps.sum() < 1.0:
        raise ValueError("Survival risk-share caps are infeasible")
    adjusted = _reduce_concentrated_risk(
        adjusted,
        risk_indices,
        cash_index,
        covariance,
        caps,
    )

    effective_target = target_volatility * novelty_multiplier
    concentrated_risk_weights = np.maximum(adjusted[risk_indices], 0.0)
    concentrated_volatility = float(
        np.sqrt(
            max(
                concentrated_risk_weights
                @ covariance
                @ concentrated_risk_weights,
                0.0,
            )
        )
    )
    volatility_multiplier = (
        min(1.0, effective_target / concentrated_volatility)
        if concentrated_volatility > 1e-12
        else 1.0
    )
    if volatility_multiplier < 1.0:
        risky_before = adjusted[risk_indices].copy()
        adjusted[risk_indices] *= volatility_multiplier
        adjusted[cash_index] += float(
            risky_before.sum() - adjusted[risk_indices].sum()
        )

    final_risk_weights = np.maximum(adjusted[risk_indices], 0.0)
    post_volatility = float(
        np.sqrt(max(final_risk_weights @ covariance @ final_risk_weights, 0.0))
    )
    final_shares = _risk_contribution_shares(final_risk_weights, covariance)
    max_share_after = float(final_shares.max(initial=0.0))
    gross_after = float(np.abs(np.delete(adjusted, cash_index)).sum())
    risk_before = float(original_risk_weights.sum())
    risk_after = float(final_risk_weights.sum())
    risk_multiplier = risk_after / risk_before if risk_before > 1e-12 else 1.0
    return SurvivalResult(
        weights=adjusted,
        pre_volatility=pre_volatility,
        post_volatility=post_volatility,
        long_horizon_volatility=long_horizon_volatility,
        stress_volatility_ratio=stress_volatility_ratio,
        target_volatility=effective_target,
        risk_multiplier=float(risk_multiplier),
        novelty_percentile=novelty_percentile,
        novelty_multiplier=novelty_multiplier,
        gross_before=gross_before,
        gross_after=gross_after,
        max_risk_share_before=max_share_before,
        max_risk_share_after=max_share_after,
        triggered=True,
        active=bool(
            np.max(np.abs(adjusted - original), initial=0.0) > 1e-10
        ),
    )
