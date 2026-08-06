from tools.audit_r11_reentry_brake_shadow import build_audit


def test_shadow_release_is_observable_but_cannot_control_capital() -> None:
    audit = build_audit()

    assert len(audit) == 6
    assert audit["status"].eq("pass").all()
    assert {
        "selection",
        "point_estimates",
        "statistical_gate",
        "capital_isolation",
        "output",
        "forward_monitoring",
    } == set(audit["category"])
