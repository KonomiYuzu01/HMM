import importlib.util
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / "evaluate_r10_combined_tail_capital.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_r10_combined_tail_capital",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selected_guard_accepts_post_trigger_cap() -> None:
    guard = MODULE.selected_guard(
        35.0,
        post_trigger_cap=0.15,
        relative_gap_trigger=-0.005,
        absolute_gap_trigger=-0.0275,
        minimum_semis_weight=0.55,
    )
    assert guard.post_trigger_cap == pytest.approx(0.15)
    assert guard.relative_gap_trigger == pytest.approx(-0.005)
    assert guard.absolute_gap_trigger == pytest.approx(-0.0275)
    assert guard.minimum_semis_weight == pytest.approx(0.55)
    assert guard.emergency_slippage_bps == pytest.approx(35.0)
    assert "cap15" in guard.name
    assert "gap275bp" in guard.name
    assert "min55" in guard.name
    assert "rel50bp" in guard.name
    assert "slip35" in guard.name
