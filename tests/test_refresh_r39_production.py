from __future__ import annotations

from tools.refresh_r39_production import active_strategy_for_refresh


def test_r39_is_active_only_when_both_qualification_layers_pass() -> None:
    assert (
        active_strategy_for_refresh(
            research_qualified=True,
            production_qualified=True,
        )
        == "R39"
    )
    assert (
        active_strategy_for_refresh(
            research_qualified=False,
            production_qualified=True,
        )
        == "R38"
    )
    assert (
        active_strategy_for_refresh(
            research_qualified=True,
            production_qualified=False,
        )
        == "R38"
    )
