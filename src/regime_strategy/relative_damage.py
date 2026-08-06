"""Causal account-level relative-damage concentration controls."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_relative_damage(
    closes: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """Return relative damage known before each execution date."""
    if lookback_days < 2:
        raise ValueError("lookback must be at least two sessions")
    missing = [
        asset for asset in ("QQQ", "SEMIS") if asset not in closes
    ]
    if missing:
        raise ValueError(f"Missing relative-damage assets: {missing}")
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    log_returns = np.log(selected.div(selected.shift(1)))
    relative = log_returns["SEMIS"] - log_returns["QQQ"]
    prior_log_return = (
        relative.rolling(
            lookback_days,
            min_periods=lookback_days,
        )
        .sum()
        .shift(1)
        .reindex(decision_index)
    )
    return pd.DataFrame(
        {
            "prior_relative_log_return": prior_log_return,
            "relative_loss": (
                1.0 - np.exp(prior_log_return)
            ).clip(lower=0.0),
        },
        index=decision_index,
    )


def completed_close_relative_damage(
    closes: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.Series:
    """Return damage through the latest completed close."""
    if len(closes) < lookback_days + 1:
        raise ValueError("insufficient closes for relative-damage lookback")
    next_date = closes.index[-1] + pd.Timedelta(days=1)
    extended = closes.copy()
    extended.loc[next_date] = closes.iloc[-1]
    return causal_relative_damage(
        extended,
        pd.DatetimeIndex([next_date]),
        lookback_days=lookback_days,
    ).iloc[0]


def apply_relative_damage_veto(
    weights: pd.DataFrame,
    relative_loss: pd.Series,
    maximum_share_permission: pd.Series,
    *,
    account_loss_budget: float,
    minimum_semis_growth_share: float,
    maximum_active_semis_growth_share: float,
    maximum_share_activation_multiple: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap damaged SMH by both loss budget and growth-sleeve share."""
    if not 0.0 < account_loss_budget < 1.0:
        raise ValueError("account loss budget must be in (0, 1)")
    if not 0.0 <= minimum_semis_growth_share <= 1.0:
        raise ValueError("minimum SMH share must be in [0, 1]")
    if not 0.0 < maximum_active_semis_growth_share <= 0.5:
        raise ValueError(
            "maximum active SMH growth share must be in (0, 0.5]"
        )
    if maximum_share_activation_multiple < 1.0:
        raise ValueError(
            "maximum share activation multiple must be at least one"
        )
    missing = [
        asset for asset in ("QQQ", "SEMIS") if asset not in weights
    ]
    if missing:
        raise ValueError(f"Missing target assets: {missing}")

    index = (
        weights.index.intersection(relative_loss.index)
        .intersection(maximum_share_permission.index)
    )
    adjusted = weights.loc[index].copy()
    loss = relative_loss.reindex(index).astype(float)
    share_permission = (
        maximum_share_permission.reindex(index).fillna(False).astype(bool)
    )
    growth_total = adjusted["QQQ"] + adjusted["SEMIS"]
    valid_growth = growth_total.gt(1e-12)
    base_share = adjusted["SEMIS"].div(
        growth_total.where(valid_growth)
    ).fillna(0.0)
    proposed_loss = adjusted["SEMIS"].clip(lower=0.0) * loss
    active = (
        proposed_loss.ge(account_loss_budget)
        & base_share.ge(minimum_semis_growth_share)
    ).fillna(False)
    maximum_share_activation_loss = (
        account_loss_budget * maximum_share_activation_multiple
    )
    maximum_share_active = (
        active
        & proposed_loss.ge(maximum_share_activation_loss)
        & share_permission
    ).fillna(False)
    allowed_by_loss_budget = (
        account_loss_budget / loss.where(loss.gt(0.0))
    )
    allowed_by_growth_share = (
        growth_total * maximum_active_semis_growth_share
    )
    allowed_semis = np.minimum(
        allowed_by_loss_budget,
        allowed_by_growth_share.where(
            maximum_share_active,
            adjusted["SEMIS"],
        ),
    )
    implemented_semis = adjusted["SEMIS"].where(
        ~active,
        np.minimum(adjusted["SEMIS"], allowed_semis),
    )
    removed_semis = adjusted["SEMIS"] - implemented_semis
    adjusted["SEMIS"] = implemented_semis
    adjusted["QQQ"] = adjusted["QQQ"] + removed_semis
    implemented_share = adjusted["SEMIS"].div(
        growth_total.where(valid_growth)
    ).fillna(0.0)
    implemented_loss = adjusted["SEMIS"].clip(lower=0.0) * loss
    growth_error = (
        adjusted["QQQ"] + adjusted["SEMIS"] - growth_total
    ).abs()
    if float(growth_error.max()) > 1e-12:
        raise AssertionError("Relative-damage veto changed growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Relative-damage veto created short growth")
    diagnostics = pd.DataFrame(
        {
            "relative_loss": loss,
            "proposed_account_relative_loss": proposed_loss,
            "base_semis_growth_share": base_share,
            "relative_damage_guard_active": active,
            "maximum_share_guard_permitted": share_permission,
            "maximum_share_guard_active": maximum_share_active,
            "maximum_active_semis_growth_share": (
                maximum_active_semis_growth_share
            ),
            "maximum_share_activation_multiple": (
                maximum_share_activation_multiple
            ),
            "maximum_share_activation_account_loss": (
                maximum_share_activation_loss
            ),
            "allowed_semis_by_loss_budget": allowed_by_loss_budget,
            "allowed_semis_by_growth_share": allowed_by_growth_share,
            "allowed_semis_weight": allowed_semis,
            "removed_semis_weight": removed_semis,
            "implemented_semis_growth_share": implemented_share,
            "implemented_account_relative_loss": implemented_loss,
            "growth_budget_error": growth_error,
        },
        index=index,
    )
    return adjusted, diagnostics
