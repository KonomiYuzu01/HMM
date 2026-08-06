from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as final


def test_final_candidate_forwards_frozen_multipliers(
    monkeypatch,
) -> None:
    received: dict[str, float] = {}

    def fake_simulate(settings, scenario, **kwargs):
        del settings, scenario
        received.update(kwargs)
        return "trial", "diagnostics"

    monkeypatch.setattr(
        final.accel,
        "simulate_candidate",
        fake_simulate,
    )
    result = final.simulate_candidate({}, object())
    assert result == ("trial", "diagnostics")
    assert received["low_vol_active_multiplier"] == 1.375
    assert received["high_vol_active_multiplier"] == 1.30
    assert received["cash_floor"] == -0.20
    assert received["overlay_fraction"] == 0.10


def test_final_neighborhood_is_centered_on_1375() -> None:
    definitions = {
        label: (low, high)
        for label, low, high in final._definitions()
    }
    assert definitions["central"] == (1.375, 1.30)
    assert definitions["low1350"] == (1.35, 1.30)
    assert definitions["low1400"] == (1.40, 1.30)
    assert definitions["identity_r38"] == (1.30, 1.30)
    assert definitions["identity_fixed1375"] == (1.375, 1.375)

