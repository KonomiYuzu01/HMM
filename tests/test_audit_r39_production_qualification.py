from __future__ import annotations

import pandas as pd

from tools.audit_r39_production_qualification import (
    non_growth_assets_unchanged,
    target_columns_sum_to_one,
)


def _targets() -> pd.DataFrame:
    frame = pd.DataFrame(
        index=["QQQ", "SEMIS", "GOLD", "CASH"]
    )
    for variant in ("lower", "center", "upper"):
        frame[f"r38_staged_account_{variant}"] = [
            0.10,
            0.40,
            0.20,
            0.30,
        ]
        frame[f"r39_account_{variant}"] = [
            0.20,
            0.30,
            0.20,
            0.30,
        ]
    return frame


def test_target_columns_sum_to_one() -> None:
    targets = _targets()
    assert target_columns_sum_to_one(targets)
    targets.loc["QQQ", "r39_account_center"] += 0.01
    assert not target_columns_sum_to_one(targets)


def test_only_growth_assets_may_change() -> None:
    targets = _targets()
    assert non_growth_assets_unchanged(targets)
    targets.loc["GOLD", "r39_account_center"] += 0.01
    assert not non_growth_assets_unchanged(targets)
