from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from tools.evaluate_r38_asymmetric_volatility_reentry import (
    asymmetric_stability_permission,
)


def test_acceleration_exit_is_immediate() -> None:
    raw = pd.Series(
        [True, True, True, True, True, True, False],
        dtype=bool,
    )
    accepted = asymmetric_stability_permission(
        raw,
        confirmation_days=5,
    )
    assert accepted.iloc[4]
    assert accepted.iloc[5]
    assert not accepted.iloc[6]


def test_reentry_requires_five_consecutive_stable_days() -> None:
    raw = pd.Series(
        [False, True, True, False, True, True, True, True, True],
        dtype=bool,
    )
    accepted = asymmetric_stability_permission(
        raw,
        confirmation_days=5,
    )
    assert not accepted.iloc[:-1].any()
    assert accepted.iloc[-1]


def test_confirmation_must_be_positive() -> None:
    raw = pd.Series([True], dtype=bool)
    try:
        asymmetric_stability_permission(
            raw,
            confirmation_days=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError")

