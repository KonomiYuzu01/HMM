from tools.audit_r11_production_qualification import build_audit


def test_r11_production_qualification_has_no_blocking_failures() -> None:
    audit = build_audit()
    blocking = audit[audit["blocking"]]

    assert len(blocking) == 14
    assert blocking["status"].eq("pass").all()
    assert audit["status"].eq("warning").sum() == 2


def test_r11_production_qualification_covers_required_categories() -> None:
    audit = build_audit()

    assert {
        "selection",
        "executable_backtest",
        "independent_periods",
        "parameter_robustness",
        "multiple_comparisons",
        "implementation_uncertainty",
        "data_integrity",
        "production_output",
    }.issubset(set(audit["category"]))
