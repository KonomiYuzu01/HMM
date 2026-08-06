from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.build_r39_production_overlay import (
    apply_veto_to_target_columns,
)


def _targets() -> pd.DataFrame:
    index = ["QQQ", "SEMIS", "GOLD", "CASH"]
    frame = pd.DataFrame(index=index)
    for variant in ("lower", "center", "upper"):
        frame[f"staged_account_{variant}"] = [
            0.10,
            0.40,
            0.20,
            0.30,
        ]
    return frame


def test_all_variants_enforce_budget_share_cap_and_preserve_growth() -> None:
    targets = _targets()
    adjusted, diagnostics = apply_veto_to_target_columns(
        targets,
        0.10,
        maximum_share_permitted=True,
        account_loss_budget=0.03,
        minimum_semis_growth_share=0.60,
        maximum_active_semis_growth_share=0.50,
        maximum_share_activation_multiple=1.10,
    )
    for variant in ("lower", "center", "upper"):
        column = f"r39_account_{variant}"
        assert adjusted[column].sum() == 1.0
        assert adjusted.loc["SEMIS", column] == pytest.approx(0.25)
        assert adjusted.loc["QQQ", column] == pytest.approx(0.25)
        assert diagnostics[variant][
            "implemented_account_relative_loss"
        ] == pytest.approx(0.025)
        assert diagnostics[variant][
            "implemented_semis_growth_share"
        ] == pytest.approx(0.50)


def test_inactive_target_is_unchanged() -> None:
    targets = _targets()
    adjusted, diagnostics = apply_veto_to_target_columns(
        targets,
        0.02,
        maximum_share_permitted=True,
        account_loss_budget=0.03,
        minimum_semis_growth_share=0.60,
        maximum_active_semis_growth_share=0.50,
        maximum_share_activation_multiple=1.10,
    )
    for variant in ("lower", "center", "upper"):
        assert np.allclose(
            adjusted[f"r39_account_{variant}"],
            targets[f"staged_account_{variant}"],
        )
        assert not bool(
            diagnostics[variant]["relative_damage_guard_active"]
        )
