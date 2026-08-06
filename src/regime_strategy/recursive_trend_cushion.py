from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Mapping


@dataclass(frozen=True)
class RecursiveTrendCushionParameters:
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05

    def __post_init__(self) -> None:
        if not -1.0 < self.floor_drawdown < 0.0:
            raise ValueError("floor_drawdown must be between -1 and 0")
        if self.bear_multiplier <= 0.0:
            raise ValueError("bear_multiplier must be positive")
        if self.bull_multiplier < self.bear_multiplier:
            raise ValueError("bull_multiplier cannot be below bear_multiplier")
        if not 0.0 < self.tier_size <= 1.0:
            raise ValueError("tier_size must be in (0, 1]")


@dataclass(frozen=True)
class RecursiveTrendCushionState:
    prior_equity: float
    prior_peak: float
    prior_drawdown: float
    floor_equity: float
    cushion: float
    multiplier: float
    requested_non_cash_cap: float
    accepted_non_cash_cap: float
    state: str
    dual_trend_positive: bool


def calculate_recursive_trend_cushion(
    *,
    prior_equity: float,
    prior_peak: float,
    dual_trend_positive: bool,
    parameters: RecursiveTrendCushionParameters = RecursiveTrendCushionParameters(),
) -> RecursiveTrendCushionState:
    if prior_equity <= 0.0 or prior_peak <= 0.0:
        raise ValueError("prior_equity and prior_peak must be positive")
    if prior_equity > prior_peak + 1e-10:
        raise ValueError("prior_peak must include prior_equity")

    floor_equity = prior_peak * (1.0 + parameters.floor_drawdown)
    cushion = max(prior_equity - floor_equity, 0.0)
    multiplier = (
        parameters.bull_multiplier
        if dual_trend_positive
        else parameters.bear_multiplier
    )
    requested = min(max(multiplier * cushion / prior_equity, 0.0), 1.0)
    if requested >= 1.0 - 1e-12:
        accepted = 1.0
    else:
        accepted = floor((requested + 1e-12) / parameters.tier_size) * parameters.tier_size
        accepted = min(max(accepted, 0.0), 1.0)
    state = "floor" if accepted <= 1e-12 else "controlled" if accepted < 1.0 else "normal"
    return RecursiveTrendCushionState(
        prior_equity=prior_equity,
        prior_peak=prior_peak,
        prior_drawdown=prior_equity / prior_peak - 1.0,
        floor_equity=floor_equity,
        cushion=cushion,
        multiplier=multiplier,
        requested_non_cash_cap=requested,
        accepted_non_cash_cap=accepted,
        state=state,
        dual_trend_positive=dual_trend_positive,
    )


def cap_total_non_cash(
    target: Mapping[str, float],
    cap: float,
    *,
    cash_key: str = "cash",
) -> dict[str, float]:
    if not 0.0 <= cap <= 1.0:
        raise ValueError("non-cash cap must be in [0, 1]")
    if cash_key not in target:
        raise ValueError(f"target is missing {cash_key}")
    adjusted = {asset: float(weight) for asset, weight in target.items()}
    if any(weight < -1e-12 for asset, weight in adjusted.items() if asset != cash_key):
        raise ValueError("non-cash target weights cannot be negative")
    total = sum(adjusted.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("target weights must sum to one")
    non_cash_total = sum(
        weight for asset, weight in adjusted.items() if asset != cash_key
    )
    if non_cash_total > cap + 1e-12:
        scale = cap / non_cash_total
        for asset in adjusted:
            if asset != cash_key:
                adjusted[asset] *= scale
        adjusted[cash_key] = 1.0 - sum(
            weight for asset, weight in adjusted.items() if asset != cash_key
        )
    return adjusted


def protected_account_target(
    *,
    staged_target: Mapping[str, float],
    r11_target: Mapping[str, float],
    state: RecursiveTrendCushionState,
    cash_key: str = "cash",
) -> dict[str, float]:
    base = r11_target if state.accepted_non_cash_cap < 1.0 - 1e-12 else staged_target
    return cap_total_non_cash(base, state.accepted_non_cash_cap, cash_key=cash_key)
