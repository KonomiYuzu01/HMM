from __future__ import annotations

from tools.refresh_r38_production import refresh_steps


def test_r38_refresh_rebuilds_research_after_r11_refresh() -> None:
    assert refresh_steps(False) == [
        ["tools/refresh_r11_production.py"],
        ["tools/evaluate_r38_accelerating_volatility_capacity_fill_1375.py"],
        ["tools/build_r38_production_overlay.py"],
        ["tools/audit_r38_accelerating_volatility_1375_final_candidate.py"],
        ["tools/audit_r38_anti_overfit_governance.py"],
        ["tools/audit_r38_forward_review.py"],
        ["tools/export_r38_panel_snapshot.py"],
        ["tools/audit_r38_production_qualification.py"],
        ["tools/export_r38_panel_snapshot.py"],
        ["tools/audit_r38_production_qualification.py"],
    ]


def test_r38_reused_close_also_rebuilds_research() -> None:
    assert refresh_steps(True) == [
        ["tools/evaluate_r38_accelerating_volatility_capacity_fill_1375.py"],
        [
            "tools/build_r38_production_overlay.py",
            "--reuse-latest-complete-inputs",
        ],
        ["tools/audit_r38_accelerating_volatility_1375_final_candidate.py"],
        ["tools/audit_r38_anti_overfit_governance.py"],
        ["tools/audit_r38_forward_review.py"],
        ["tools/export_r38_panel_snapshot.py"],
        ["tools/audit_r38_production_qualification.py"],
        ["tools/export_r38_panel_snapshot.py"],
        ["tools/audit_r38_production_qualification.py"],
    ]
