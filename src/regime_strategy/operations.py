from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import CustomBusinessDay

from .portfolio import (
    apply_account_stress_volatility_cap,
    apply_asset_aware_growth_risk_budget,
    apply_relative_asset_risk_cap,
    realized_stress_covariance,
)


class _UsEquityHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_US_EQUITY_BUSINESS_DAY = CustomBusinessDay(calendar=_UsEquityHolidayCalendar())


def next_us_equity_session(last_complete_session: pd.Timestamp) -> pd.Timestamp:
    """Return the next standard NYSE session, excluding regular full-day holidays."""
    return pd.Timestamp(last_complete_session).normalize() + _US_EQUITY_BUSINESS_DAY


def latest_completed_us_equity_session(
    now: pd.Timestamp | None = None,
    completion_time: time = time(16, 15),
) -> pd.Timestamp:
    """Return the latest U.S. equity session with a safely completed daily bar."""
    current = now or pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    candidate = current.tz_localize(None).normalize()
    is_session = _US_EQUITY_BUSINESS_DAY.is_on_offset(candidate)
    if (
        not is_session
        or current.time().replace(tzinfo=None) < completion_time
    ):
        candidate -= pd.Timedelta(days=1)
    return _US_EQUITY_BUSINESS_DAY.rollback(candidate).normalize()


def missing_recent_us_equity_sessions(
    observed_sessions: pd.Index,
    now: pd.Timestamp | None = None,
    session_count: int = 5,
    completion_time: time = time(16, 15),
) -> pd.DatetimeIndex:
    """Return missing sessions in the latest completed U.S. equity window."""
    if session_count <= 0:
        raise ValueError("Session count must be positive")
    expected_latest = latest_completed_us_equity_session(
        now,
        completion_time=completion_time,
    )
    expected = pd.date_range(
        end=expected_latest,
        periods=session_count,
        freq=_US_EQUITY_BUSINESS_DAY,
    ).normalize()
    observed = pd.DatetimeIndex(observed_sessions).tz_localize(None).normalize()
    return expected.difference(observed)


def production_signal_generation_status(
    now: pd.Timestamp | None = None,
    assumed_open: time = time(9, 30),
    completion_time: time = time(16, 15),
) -> str:
    """Block production snapshots while the current U.S. session is incomplete."""
    current = now or pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    current_date = current.tz_localize(None).normalize()
    current_time = current.time().replace(tzinfo=None)
    if (
        _US_EQUITY_BUSINESS_DAY.is_on_offset(current_date)
        and assumed_open <= current_time < completion_time
    ):
        return "INTRADAY_BLOCKED"
    return "AVAILABLE"


def next_session_execution_status(
    last_complete_session: pd.Timestamp,
    now: pd.Timestamp | None = None,
    assumed_open: time = time(9, 30),
) -> tuple[str, pd.Timestamp]:
    """Classify whether the next-session execution window is still available."""
    current = now or pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    execution_date = next_us_equity_session(last_complete_session)
    if current.date() < execution_date.date():
        return "UPCOMING", execution_date
    if current.date() > execution_date.date():
        return "MISSED", execution_date
    if current.time().replace(tzinfo=None) < assumed_open:
        return "READY_FOR_OPEN", execution_date
    return "MISSED", execution_date


def onboarding_targets(
    model_weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    cash_index: int,
    fast_days: int,
    slow_days: int,
    target_volatility: float,
    annualization: int = 252,
) -> dict[str, np.ndarray]:
    """Build model, growth-first, and risk-first targets for a zero position."""
    growth_first, *_ = apply_asset_aware_growth_risk_budget(
        model_weights,
        historical_returns,
        assets,
        growth_assets,
        cash_index,
        fast_days,
        slow_days,
        target_volatility,
        {growth_assets[1]: 0.50},
        annualization,
        activate_only_when_breached=False,
        cap_total_volatility=False,
    )
    volatility_capped, *_ = apply_asset_aware_growth_risk_budget(
        model_weights,
        historical_returns,
        assets,
        growth_assets,
        cash_index,
        fast_days,
        slow_days,
        target_volatility,
        {growth_assets[1]: 0.50},
        annualization,
        activate_only_when_breached=True,
        cap_total_volatility=True,
    )
    cash_first, _, _ = apply_relative_asset_risk_cap(
        model_weights,
        historical_returns,
        assets,
        growth_assets[0],
        growth_assets[1],
        cash_index,
        fast_days,
        slow_days,
        annualization,
        activation_volatility_cap=target_volatility,
    )
    account_risk_assets = [
        asset
        for asset in assets
        if asset != assets[cash_index] and growth_first[assets.index(asset)] > 1e-8
    ]
    account_capped, _, _ = apply_account_stress_volatility_cap(
        growth_first,
        historical_returns,
        assets,
        account_risk_assets,
        cash_index,
        fast_days,
        slow_days,
        target_volatility,
        annualization,
    )
    return {
        "model_replication": model_weights.copy(),
        "growth_first": growth_first,
        "account_volatility_capped": account_capped,
        "volatility_capped": volatility_capped,
        "cash_first": cash_first,
    }


def growth_risk_snapshot(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    assets: list[str],
    growth_assets: list[str],
    fast_days: int,
    slow_days: int,
    annualization: int = 252,
    risk_assets: list[str] | None = None,
) -> dict[str, float]:
    """Return total account volatility and Euler shares for the growth sleeves."""
    covariance_assets = risk_assets or growth_assets
    if not set(growth_assets).issubset(covariance_assets):
        raise ValueError("Risk assets must include all growth assets")
    covariance, effective_volatility = realized_stress_covariance(
        historical_returns[covariance_assets],
        fast_days,
        slow_days,
        annualization,
    )
    risk_weights = weights[[assets.index(asset) for asset in covariance_assets]]
    account_volatility = float(
        np.sqrt(max(risk_weights @ covariance @ risk_weights, 0.0))
    )
    component_risk = risk_weights * (covariance @ risk_weights)
    total_variance = float(component_risk.sum())
    risk_shares = (
        component_risk / total_variance
        if total_variance > 1e-16
        else np.zeros_like(component_risk)
    )
    volatility_by_asset = dict(
        zip(covariance_assets, effective_volatility, strict=True)
    )
    share_by_asset = dict(zip(covariance_assets, risk_shares, strict=True))
    result = {
        "account_realized_volatility": account_volatility,
        "qqq_effective_volatility": float(volatility_by_asset[growth_assets[0]]),
        "smh_effective_volatility": float(volatility_by_asset[growth_assets[1]]),
        "qqq_risk_share": float(share_by_asset[growth_assets[0]]),
        "smh_risk_share": float(share_by_asset[growth_assets[1]]),
    }
    if "GOLD" in share_by_asset:
        result["gold_risk_share"] = float(share_by_asset["GOLD"])
        result["gold_effective_volatility"] = float(volatility_by_asset["GOLD"])
    return result


def dollar_order_plan(
    target_weights: np.ndarray,
    current_dollars: np.ndarray,
    prices: np.ndarray,
    assets: list[str],
    account_value: float,
) -> pd.DataFrame:
    """Translate target weights into dollar and reference-share orders."""
    if account_value <= 0.0:
        raise ValueError("Account value must be positive")
    if not (
        len(target_weights) == len(current_dollars) == len(prices) == len(assets)
    ):
        raise ValueError("Order-plan inputs must align")
    if np.any(prices <= 0.0):
        raise ValueError("Reference prices must be positive")
    target_dollars = target_weights * account_value
    trade_dollars = target_dollars - current_dollars
    actions = np.where(
        trade_dollars > 0.01,
        "BUY",
        np.where(trade_dollars < -0.01, "SELL", "HOLD"),
    )
    reference_trade_shares = np.abs(trade_dollars) / prices
    whole_trade_shares = np.floor(reference_trade_shares)
    return pd.DataFrame(
        {
            "asset": assets,
            "target_weight": target_weights,
            "target_dollars": target_dollars,
            "current_dollars": current_dollars,
            "trade_dollars": trade_dollars,
            "action": actions,
            "reference_price": prices,
            "reference_trade_shares": reference_trade_shares,
            "whole_trade_shares": whole_trade_shares,
            "whole_share_trade_dollars": whole_trade_shares * prices,
            "rounding_residual_dollars": (
                np.abs(trade_dollars) - whole_trade_shares * prices
            ),
        }
    )
