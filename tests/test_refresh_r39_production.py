from __future__ import annotations

import json
import sys

from tools import refresh_r39_production
from tools.refresh_r39_production import (
    active_strategy_for_refresh,
    r40_maintenance_steps,
)


def test_r39_is_active_only_when_both_qualification_layers_pass() -> None:
    assert (
        active_strategy_for_refresh(
            research_qualified=True,
            production_qualified=True,
            successor_promotion_allowed=True,
        )
        == "R39"
    )
    assert (
        active_strategy_for_refresh(
            research_qualified=False,
            production_qualified=True,
            successor_promotion_allowed=True,
        )
        == "R38"
    )
    assert (
        active_strategy_for_refresh(
            research_qualified=True,
            production_qualified=False,
            successor_promotion_allowed=True,
        )
        == "R38"
    )


def test_forward_freeze_blocks_r39_even_when_old_audits_pass() -> None:
    assert (
        active_strategy_for_refresh(
            research_qualified=True,
            production_qualified=True,
            successor_promotion_allowed=False,
        )
        == "R38"
    )


def test_r40_is_refreshed_even_while_successors_are_frozen() -> None:
    assert r40_maintenance_steps() == [
        ["tools/build_r40_account_protection.py"],
        ["tools/audit_r40_production_qualification.py"],
    ]


def test_frozen_refresh_runs_r40_maintenance_before_r38_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    governance = tmp_path / "governance.json"
    governance.write_text(
        json.dumps(
            {
                "successor_promotion_allowed": False,
                "operational_pass": True,
                "forward_sessions": 8,
                "minimum_forward_sessions": 63,
            }
        ),
        encoding="utf-8",
    )
    r38_metadata = tmp_path / "r38_metadata.json"
    r38_metadata.write_text(
        json.dumps({"price_as_of": "2026-08-07"}),
        encoding="utf-8",
    )
    steps: list[list[str]] = []
    monkeypatch.setattr(
        refresh_r39_production,
        "run_step",
        lambda command, check=True: steps.append(command) or 0,
    )
    monkeypatch.setattr(
        refresh_r39_production,
        "GOVERNANCE_SUMMARY",
        governance,
    )
    monkeypatch.setattr(
        refresh_r39_production,
        "R38_METADATA",
        r38_metadata,
    )
    monkeypatch.setattr(
        refresh_r39_production,
        "REPORT",
        tmp_path / "latest.json",
    )
    monkeypatch.setattr(sys, "argv", ["refresh_r39_production.py"])

    refresh_r39_production.main()

    assert steps == [
        ["tools/refresh_r38_production.py"],
        ["tools/build_r40_account_protection.py"],
        ["tools/audit_r40_production_qualification.py"],
        ["tools/export_r38_panel_snapshot.py"],
    ]
