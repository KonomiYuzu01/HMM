from __future__ import annotations

import pandas as pd
import pytest

from regime_strategy.decision_authority import (
    EvidenceStatus,
    MarketEnvironment,
    apply_entry_authority,
    apply_exit_authority,
    classify_market_environment,
    state_only_transition_target,
    validate_authority_registry,
    validate_rule_capital_authority,
)


GROWTH = ("QQQ", "SEMIS")


def target(growth: float) -> pd.Series:
    return pd.Series(
        {
            "QQQ": 0.75 * growth,
            "SEMIS": 0.25 * growth,
            "GOLD": 0.20,
            "CASH": 0.80 - growth,
        }
    )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            {
                "qqq_63d_return": 0.10,
                "semis_63d_return": 0.12,
                "semis_126d_return": 0.20,
                "vix_term_ratio": 0.90,
                "previous_growth_return": 0.01,
            },
            MarketEnvironment.HEALTHY,
        ),
        (
            {
                "qqq_63d_return": 0.10,
                "semis_63d_return": 0.12,
                "semis_126d_return": 0.20,
                "vix_term_ratio": 1.05,
                "previous_growth_return": -0.04,
            },
            MarketEnvironment.ACUTE_LIQUIDITY_SHOCK,
        ),
        (
            {
                "qqq_63d_return": -0.10,
                "semis_63d_return": -0.12,
                "semis_126d_return": -0.20,
                "vix_term_ratio": 1.05,
                "previous_growth_return": -0.04,
            },
            MarketEnvironment.STRUCTURAL_DAMAGE,
        ),
        (
            {
                "qqq_63d_return": 0.10,
                "semis_63d_return": 0.12,
                "semis_126d_return": -0.20,
                "vix_term_ratio": 0.95,
                "previous_growth_return": 0.01,
            },
            MarketEnvironment.EARLY_REPAIR,
        ),
    ],
)
def test_market_mechanisms_are_explicit(
    inputs: dict[str, float],
    expected: MarketEnvironment,
) -> None:
    assert classify_market_environment(**inputs) is expected


def test_exit_authority_cannot_increase_growth() -> None:
    with pytest.raises(ValueError, match="cannot increase"):
        apply_exit_authority(
            target(0.20),
            target(0.40),
            growth_assets=GROWTH,
            status=EvidenceStatus.APPROVED,
        )

    result = apply_exit_authority(
        target(0.40),
        target(0.20),
        growth_assets=GROWTH,
        status=EvidenceStatus.APPROVED,
    )
    assert result.equals(target(0.20))


def test_entry_authority_cannot_reduce_growth() -> None:
    with pytest.raises(ValueError, match="cannot reduce"):
        apply_entry_authority(
            target(0.40),
            target(0.20),
            growth_assets=GROWTH,
            status=EvidenceStatus.APPROVED,
        )

    result = apply_entry_authority(
        target(0.20),
        target(0.40),
        growth_assets=GROWTH,
        status=EvidenceStatus.APPROVED,
    )
    assert result.equals(target(0.40))


@pytest.mark.parametrize(
    "status",
    [EvidenceStatus.SHADOW, EvidenceStatus.REJECTED],
)
def test_unapproved_authority_fails_to_no_change(
    status: EvidenceStatus,
) -> None:
    current = target(0.20)
    proposal = target(0.40)

    result = apply_entry_authority(
        current,
        proposal,
        growth_assets=GROWTH,
        status=status,
    )

    assert result.equals(current)
    with pytest.raises(ValueError, match="cannot control real capital"):
        validate_rule_capital_authority(status, 0.01)


def test_state_transition_cannot_trade_without_a_proposal() -> None:
    current = target(0.20)

    assert state_only_transition_target(
        current,
        proposal_present=False,
    ).equals(current)
    with pytest.raises(ValueError, match="cannot change"):
        state_only_transition_target(
            current,
            proposal_present=False,
            proposed_target=target(0.40),
        )


def test_registry_rejects_shadow_capital() -> None:
    with pytest.raises(ValueError, match="cannot control real capital"):
        validate_authority_registry(
            {
                "unqualified_rule": {
                    "status": "shadow",
                    "maximum_managed_capital_share": 0.01,
                }
            }
        )
