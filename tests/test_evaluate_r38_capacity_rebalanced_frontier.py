from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tools.evaluate_r38_capacity_rebalanced_frontier as frontier


def test_central_candidate_forwards_frozen_capacity_shape(
    monkeypatch,
) -> None:
    received: dict[str, float] = {}

    def fake_simulate(settings, scenario, **kwargs):
        del settings, scenario
        received.update(kwargs)
        return "trial", "diagnostics"

    monkeypatch.setattr(
        frontier.r38,
        "simulate_candidate",
        fake_simulate,
    )
    result = frontier.simulate_candidate({}, object())
    assert result == ("trial", "diagnostics")
    assert received["base_multiplier"] == 1.070
    assert received["active_multiplier"] == 1.35
    assert received["shock_multiplier"] == 1.00
    assert received["cash_floor"] == -0.18
    assert received["overlay_fraction"] == 0.10


def test_identity_path_is_frozen_r38_parameters() -> None:
    definitions = {
        label: (active, cash, family)
        for label, active, cash, family
        in frontier._candidate_definitions()
    }
    assert definitions["identity_r38"] == (
        1.30,
        -0.20,
        "identity",
    )


def test_all_registered_paths_respect_declared_bounds() -> None:
    definitions = frontier._candidate_definitions()
    assert len(definitions) == len({row[0] for row in definitions})
    assert all(1.30 <= active <= 1.40 for _, active, _, _ in definitions)
    assert all(-0.20 <= cash <= -0.16 for _, _, cash, _ in definitions)

