from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_r11_decision_authority import build_audit


def test_all_decision_authority_requirements_pass() -> None:
    audit = build_audit()

    assert len(audit) == 10
    assert audit["status"].eq("pass").all()
