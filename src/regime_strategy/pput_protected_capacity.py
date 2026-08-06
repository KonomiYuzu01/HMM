from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

from regime_strategy.recursive_trend_cushion import (
    RecursiveTrendCushionParameters,
    RecursiveTrendCushionState,
    calculate_recursive_trend_cushion,
)


@dataclass(frozen=True)
class PputProtectedCapacityParameters:
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 20.0
    tier_size: float = 0.05
    normal_non_cash_cap: float = 1.20
    target_put_coverage: float = 0.06
    minimum_put_coverage: float = 0.05
    maximum_put_coverage: float = 0.07
    option_multiplier: int = 100

    def __post_init__(self) -> None:
        if not 1.0 <= self.normal_non_cash_cap <= 1.20:
            raise ValueError("normal_non_cash_cap must be in [1.0, 1.2]")
        if not (
            0.0
            < self.minimum_put_coverage
            <= self.target_put_coverage
            <= self.maximum_put_coverage
            <= 1.0
        ):
            raise ValueError("put coverage bounds are invalid")
        if self.option_multiplier <= 0:
            raise ValueError("option_multiplier must be positive")

    def cushion_parameters(self) -> RecursiveTrendCushionParameters:
        return RecursiveTrendCushionParameters(
            floor_drawdown=self.floor_drawdown,
            bull_multiplier=self.bull_multiplier,
            bear_multiplier=self.bear_multiplier,
            tier_size=self.tier_size,
        )


@dataclass(frozen=True)
class ProtectedCapacityState:
    cushion: RecursiveTrendCushionState
    hedge_confirmed: bool
    normal_cap_extended: bool
    accepted_non_cash_cap: float


@dataclass(frozen=True)
class PutContractMapping:
    account_equity: float
    underlying_price: float
    target_underlying_notional: float
    contract_underlying_notional: float
    raw_contracts: float
    contracts: int
    implemented_underlying_notional: float
    implemented_coverage: float
    coverage_qualified: bool


def calculate_pput_protected_capacity(
    *,
    prior_equity: float,
    prior_peak: float,
    dual_trend_positive: bool,
    hedge_confirmed: bool,
    parameters: PputProtectedCapacityParameters = PputProtectedCapacityParameters(),
) -> ProtectedCapacityState:
    cushion = calculate_recursive_trend_cushion(
        prior_equity=prior_equity,
        prior_peak=prior_peak,
        dual_trend_positive=dual_trend_positive,
        parameters=parameters.cushion_parameters(),
    )
    extended = bool(
        hedge_confirmed
        and dual_trend_positive
        and cushion.state == "normal"
    )
    cap = parameters.normal_non_cash_cap if extended else cushion.accepted_non_cash_cap
    return ProtectedCapacityState(
        cushion=cushion,
        hedge_confirmed=hedge_confirmed,
        normal_cap_extended=extended,
        accepted_non_cash_cap=cap,
    )


def map_put_contracts(
    *,
    account_equity: float,
    underlying_price: float,
    parameters: PputProtectedCapacityParameters = PputProtectedCapacityParameters(),
) -> PutContractMapping:
    if account_equity <= 0.0 or underlying_price <= 0.0:
        raise ValueError("account_equity and underlying_price must be positive")
    contract_notional = underlying_price * parameters.option_multiplier
    target_notional = account_equity * parameters.target_put_coverage
    raw = target_notional / contract_notional
    candidates = {
        max(0, floor(raw)),
        max(0, ceil(raw)),
    }
    qualified = [
        count
        for count in candidates
        if parameters.minimum_put_coverage
        <= count * contract_notional / account_equity
        <= parameters.maximum_put_coverage
    ]
    if qualified:
        count = min(
            qualified,
            key=lambda value: (
                abs(value * contract_notional / account_equity - parameters.target_put_coverage),
                value,
            ),
        )
    else:
        count = min(
            candidates,
            key=lambda value: (
                abs(value * contract_notional / account_equity - parameters.target_put_coverage),
                value,
            ),
        )
    implemented_notional = count * contract_notional
    coverage = implemented_notional / account_equity
    return PutContractMapping(
        account_equity=account_equity,
        underlying_price=underlying_price,
        target_underlying_notional=target_notional,
        contract_underlying_notional=contract_notional,
        raw_contracts=raw,
        contracts=count,
        implemented_underlying_notional=implemented_notional,
        implemented_coverage=coverage,
        coverage_qualified=(
            parameters.minimum_put_coverage
            <= coverage
            <= parameters.maximum_put_coverage
        ),
    )


def select_five_percent_otm_strike(
    *,
    underlying_price: float,
    listed_put_strikes: Iterable[float],
) -> float:
    if underlying_price <= 0.0:
        raise ValueError("underlying_price must be positive")
    strikes = sorted({float(strike) for strike in listed_put_strikes if strike > 0.0})
    target = underlying_price * 0.95
    eligible = [strike for strike in strikes if strike <= target]
    if not eligible:
        raise ValueError("no listed put strike at or below 95% of spot")
    return max(eligible)
