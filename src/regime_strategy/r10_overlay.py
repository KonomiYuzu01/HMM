from __future__ import annotations

import numpy as np
import pandas as pd


BASE_ASSETS = (
    "SPX",
    "QQQ",
    "SEMIS",
    "BOND",
    "GOLD",
    "OIL",
    "USD",
    "CASH",
    "VIX_HEDGE",
)
R10_ASSETS = (*BASE_ASSETS, "GDE")


def risk_scaled_target(
    base_target: pd.Series,
    *,
    risk_multiplier: float,
) -> pd.Series:
    if risk_multiplier < 0.0:
        raise ValueError("risk_multiplier must be non-negative")
    missing = [asset for asset in BASE_ASSETS if asset not in base_target]
    if missing:
        raise ValueError(f"Missing base target assets: {missing}")
    target = base_target.loc[list(BASE_ASSETS)].astype(float).copy()
    non_cash = [asset for asset in BASE_ASSETS if asset != "CASH"]
    target.loc[non_cash] *= risk_multiplier
    target["CASH"] = 1.0 - float(target.loc[non_cash].sum())
    return target


def r10_center_target(
    base_target: pd.Series,
    *,
    substitution_fraction: float,
) -> pd.Series:
    if not 0.0 <= substitution_fraction <= 1.0:
        raise ValueError("substitution_fraction must be in [0, 1]")
    missing = [asset for asset in BASE_ASSETS if asset not in base_target]
    if missing:
        raise ValueError(f"Missing base target assets: {missing}")
    target = pd.Series(0.0, index=R10_ASSETS, dtype=float)
    target.loc[list(BASE_ASSETS)] = base_target.loc[
        list(BASE_ASSETS)
    ].astype(float)
    growth_weight = float(
        base_target[["SPX", "QQQ", "SEMIS"]].sum()
    )
    growth_gate = float(np.clip(growth_weight / 0.80, 0.0, 1.0))
    gde_weight = (
        substitution_fraction
        * max(float(base_target["GOLD"]), 0.0)
        * growth_gate
    )
    target["GOLD"] -= gde_weight
    target["GDE"] = gde_weight
    return target


def r10_band_targets(
    center_target: pd.Series,
    *,
    no_trade_band: float,
) -> tuple[pd.Series, pd.Series]:
    if no_trade_band < 0.0:
        raise ValueError("no_trade_band must be non-negative")
    center = center_target.reindex(R10_ASSETS, fill_value=0.0).astype(float)
    gold_sleeve = float(center["GOLD"] + center["GDE"])
    lower_gde = max(float(center["GDE"]) - no_trade_band, 0.0)
    upper_gde = min(
        float(center["GDE"]) + no_trade_band,
        gold_sleeve,
    )
    lower = center.copy()
    upper = center.copy()
    lower["GDE"] = lower_gde
    lower["GOLD"] = gold_sleeve - lower_gde
    upper["GDE"] = upper_gde
    upper["GOLD"] = gold_sleeve - upper_gde
    return lower, upper


def blend_rollout_target(
    r9_target: pd.Series,
    r10_target: pd.Series,
    *,
    rollout_share: float,
) -> pd.Series:
    if not 0.0 <= rollout_share <= 1.0:
        raise ValueError("rollout_share must be in [0, 1]")
    r9 = r9_target.reindex(R10_ASSETS, fill_value=0.0).astype(float)
    r10 = r10_target.reindex(R10_ASSETS, fill_value=0.0).astype(float)
    return (1.0 - rollout_share) * r9 + rollout_share * r10


def target_from_actual_gde(
    r9_target: pd.Series,
    r10_center: pd.Series,
    *,
    rollout_share: float,
    no_trade_band: float,
    actual_account_gde_weight: float,
) -> pd.Series:
    if actual_account_gde_weight < 0.0:
        raise ValueError("actual_account_gde_weight must be non-negative")
    center = blend_rollout_target(
        r9_target,
        r10_center,
        rollout_share=rollout_share,
    )
    if rollout_share <= 0.0:
        return center
    full_sleeve_actual = actual_account_gde_weight / rollout_share
    full_sleeve_center = float(r10_center["GDE"])
    full_sleeve_target = full_sleeve_center + float(
        np.clip(
            full_sleeve_actual - full_sleeve_center,
            -no_trade_band,
            no_trade_band,
        )
    )
    account_gde_target = rollout_share * full_sleeve_target
    gold_sleeve = float(center["GOLD"] + center["GDE"])
    center["GDE"] = account_gde_target
    center["GOLD"] = gold_sleeve - account_gde_target
    return center
