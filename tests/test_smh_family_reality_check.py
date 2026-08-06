from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_smh_causal_family_reality_check import (  # noqa: E402
    circular_family_reality_check,
)


def test_family_reality_check_centers_constant_alpha() -> None:
    relative = np.column_stack(
        [
            np.full(252, 0.001),
            np.full(252, 0.0005),
        ]
    )

    result = circular_family_reality_check(
        relative,
        selected_index=0,
        block_days=21,
        samples=99,
        seed=7,
    )

    assert result["selected_annualized_relative_log_return"] == pytest.approx(
        0.252
    )
    assert result["nominal_one_sided_p_value"] == 0.01
    assert result["familywise_reality_check_p_value"] == 0.01
