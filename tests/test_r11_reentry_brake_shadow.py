from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from tools.build_r11_reentry_brake_shadow import (
    shadow_state,
    validate_frozen_parameters,
)


CONFIG = Path(
    "config/paper_core_growth_gold20_r11_reentry_brake_shadow.yaml"
)


def test_shadow_configuration_is_frozen_and_not_production_eligible() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    validate_frozen_parameters(config)

    assert config["status"] == "production_shadow"
    assert config["qualification"]["production_eligible"] is False
    assert config["execution"]["orders_executable"] is False
    assert config["execution"]["real_capital_share"] == 0.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"released": True, "engaged": False, "active": False}, "released_today"),
        ({"released": False, "engaged": True, "active": True}, "limiting_reentry"),
        ({"released": False, "engaged": False, "active": True}, "armed"),
        ({"released": False, "engaged": False, "active": False}, "inactive"),
    ],
)
def test_shadow_state_is_unambiguous(
    values: dict[str, bool],
    expected: str,
) -> None:
    assert shadow_state(pd.Series(values)) == expected
