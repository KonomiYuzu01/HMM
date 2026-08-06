from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import tools.evaluate_r38_active135_capacity_fill as active135


def test_candidate_forwards_only_the_frozen_active_change(
    monkeypatch,
) -> None:
    received: dict[str, float] = {}

    def fake_simulate(settings, scenario, **kwargs):
        del settings, scenario
        received.update(kwargs)
        return "trial", "diagnostics"

    monkeypatch.setattr(
        active135.r38,
        "simulate_candidate",
        fake_simulate,
    )
    result = active135.simulate_candidate({}, object())
    assert result == ("trial", "diagnostics")
    assert received == {
        "base_multiplier": 1.070,
        "active_multiplier": 1.35,
        "shock_multiplier": 1.00,
        "cash_floor": -0.20,
        "overlay_fraction": 0.10,
    }


def test_identity_neighbor_is_original_r38() -> None:
    identity = {
        label: (base, active, shock, cash, overlay)
        for (
            label,
            _,
            base,
            active,
            shock,
            cash,
            overlay,
        ) in active135._definitions()
    }["active130"]
    assert identity == (1.070, 1.30, 1.00, -0.20, 0.10)


def test_two_sided_neighborhoods_cover_all_five_families() -> None:
    definitions = active135._definitions()
    families = [row[1] for row in definitions]
    assert set(families) == {
        "base",
        "active",
        "shock",
        "cash",
        "overlay",
    }
    assert all(families.count(family) == 2 for family in set(families))

