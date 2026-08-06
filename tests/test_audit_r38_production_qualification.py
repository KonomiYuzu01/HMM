from __future__ import annotations

import pandas as pd

from tools.audit_r38_production_qualification import (
    staged_formula_matches,
    target_columns_sum_to_one,
)


def _targets() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "r11_reference_target": [0.4, 0.4, 0.2],
            "r38_full_lower": [0.5, 0.4, 0.1],
            "r38_full_center": [0.5, 0.4, 0.1],
            "r38_full_upper": [0.5, 0.4, 0.1],
        },
        index=["QQQ", "SMH", "CASH"],
    )
    staged = 0.75 * frame["r11_reference_target"] + 0.25 * frame[
        "r38_full_center"
    ]
    frame["staged_account_lower"] = staged
    frame["staged_account_center"] = staged
    frame["staged_account_upper"] = staged
    return frame


def test_staged_formula_is_exactly_r11_plus_r38() -> None:
    targets = _targets()
    assert staged_formula_matches(targets, 0.25)
    targets.loc["QQQ", "staged_account_center"] += 0.001
    assert not staged_formula_matches(targets, 0.25)


def test_all_target_columns_must_sum_to_one() -> None:
    targets = _targets()
    assert target_columns_sum_to_one(targets)
    targets.loc["QQQ", "r38_full_lower"] += 0.01
    assert not target_columns_sum_to_one(targets)
