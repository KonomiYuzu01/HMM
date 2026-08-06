from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class MarketEnvironment(str, Enum):
    INCOMPLETE = "incomplete"
    HEALTHY = "healthy"
    ACUTE_LIQUIDITY_SHOCK = "acute_liquidity_shock"
    ORDINARY_CORRECTION = "ordinary_correction"
    STRUCTURAL_DAMAGE = "structural_damage"
    EARLY_REPAIR = "early_repair"


class EvidenceStatus(str, Enum):
    APPROVED = "approved"
    SHADOW = "shadow"
    REJECTED = "rejected"


def classify_market_environment(
    *,
    qqq_63d_return: float,
    semis_63d_return: float,
    semis_126d_return: float,
    vix_term_ratio: float,
    previous_growth_return: float,
    shock_threshold: float = -0.03,
) -> MarketEnvironment:
    values = (
        qqq_63d_return,
        semis_63d_return,
        semis_126d_return,
        vix_term_ratio,
        previous_growth_return,
    )
    if any(pd.isna(value) for value in values):
        return MarketEnvironment.INCOMPLETE
    qqq_weak = qqq_63d_return <= 0.0
    semis_weak = semis_63d_return <= 0.0
    semis_medium_weak = semis_126d_return <= 0.0
    stressed_term = vix_term_ratio > 1.0
    shock = previous_growth_return <= shock_threshold
    if qqq_weak and semis_weak and semis_medium_weak:
        return MarketEnvironment.STRUCTURAL_DAMAGE
    if (
        semis_medium_weak
        and not qqq_weak
        and not semis_weak
        and not stressed_term
    ):
        return MarketEnvironment.EARLY_REPAIR
    if shock and stressed_term:
        return MarketEnvironment.ACUTE_LIQUIDITY_SHOCK
    if qqq_weak or semis_weak or stressed_term:
        return MarketEnvironment.ORDINARY_CORRECTION
    return MarketEnvironment.HEALTHY


def validate_rule_capital_authority(
    status: EvidenceStatus | str,
    maximum_managed_capital_share: float,
) -> None:
    parsed = EvidenceStatus(status)
    if not 0.0 <= maximum_managed_capital_share <= 1.0:
        raise ValueError("Managed-capital share must be in [0, 1]")
    if (
        parsed is not EvidenceStatus.APPROVED
        and maximum_managed_capital_share > 0.0
    ):
        raise ValueError(
            "Shadow and rejected rules cannot control real capital"
        )


def _growth_weight(
    target: pd.Series,
    growth_assets: Sequence[str],
) -> float:
    missing = sorted(set(growth_assets).difference(target.index))
    if missing:
        raise ValueError(f"Target is missing growth assets: {missing}")
    return float(target.loc[list(growth_assets)].sum())


def apply_exit_authority(
    current: pd.Series,
    proposed: pd.Series,
    *,
    growth_assets: Sequence[str],
    status: EvidenceStatus | str,
) -> pd.Series:
    parsed = EvidenceStatus(status)
    if parsed is not EvidenceStatus.APPROVED:
        return current.copy()
    if _growth_weight(proposed, growth_assets) > _growth_weight(
        current,
        growth_assets,
    ) + 1e-12:
        raise ValueError("Exit authority cannot increase growth exposure")
    return proposed.copy()


def apply_entry_authority(
    current: pd.Series,
    proposed: pd.Series,
    *,
    growth_assets: Sequence[str],
    status: EvidenceStatus | str,
) -> pd.Series:
    parsed = EvidenceStatus(status)
    if parsed is not EvidenceStatus.APPROVED:
        return current.copy()
    if _growth_weight(proposed, growth_assets) < _growth_weight(
        current,
        growth_assets,
    ) - 1e-12:
        raise ValueError("Entry authority cannot reduce growth exposure")
    return proposed.copy()


def state_only_transition_target(
    current_target: pd.Series,
    *,
    proposal_present: bool,
    proposed_target: pd.Series | None = None,
) -> pd.Series:
    if not proposal_present:
        if proposed_target is not None and not np.allclose(
            current_target.to_numpy(dtype=float),
            proposed_target.reindex(current_target.index).to_numpy(
                dtype=float
            ),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "A state-only transition cannot change the target"
            )
        return current_target.copy()
    if proposed_target is None:
        raise ValueError("A declared proposal requires a target")
    return proposed_target.reindex(current_target.index).astype(float)


def validate_authority_registry(
    rules: Mapping[str, Mapping[str, object]],
) -> None:
    if not rules:
        raise ValueError("Authority registry cannot be empty")
    for name, specification in rules.items():
        try:
            status = EvidenceStatus(str(specification["status"]))
            share = float(
                specification["maximum_managed_capital_share"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid authority registration for {name}"
            ) from error
        validate_rule_capital_authority(status, share)
