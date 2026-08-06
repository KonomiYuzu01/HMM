from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r40_term_structure_capacity import (
    causal_term_permission,
    term_conditioned_weights,
)


def test_term_permission_is_lagged_and_proxy_neutral() -> None:
    index = pd.bdate_range("2026-01-01", periods=4)
    closes = pd.DataFrame(
        {"VIX": [20.0] * 4, "VIX3M": [20.0, 22.0, 22.0, 22.0]},
        index=index,
    )
    permission = causal_term_permission(closes, term_premium_threshold=1.05)
    assert not bool(permission.iloc[1])
    assert bool(permission.iloc[2])


def test_denied_days_cap_at_one_and_allowed_days_scale() -> None:
    index = pd.bdate_range("2026-01-01", periods=2)
    weights = pd.DataFrame(
        {"QQQ": [0.8, 0.8], "GOLD": [0.4, 0.4], "CASH": [-0.2, -0.2]},
        index=index,
    )
    result = term_conditioned_weights(
        weights,
        pd.Series([False, True], index=index),
        capacity_scale=1.02,
    )
    assert result.loc[index[0], ["QQQ", "GOLD"]].sum() == pytest.approx(1.0)
    assert result.loc[index[1], ["QQQ", "GOLD"]].sum() == pytest.approx(1.224)
    assert result.sum(axis=1).eq(1.0).all()


def test_invalid_term_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        causal_term_permission(
            pd.DataFrame({"VIX": [20.0], "VIX3M": [21.0]}),
            term_premium_threshold=0.99,
        )
