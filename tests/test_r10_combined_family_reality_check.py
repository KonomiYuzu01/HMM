import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / "evaluate_r10_combined_family_reality_check.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_r10_combined_family_reality_check",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_relative_log_path_requires_complete_candidate() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    baseline = pd.Series([0.01, 0.02, -0.01], index=index)
    candidate = pd.Series([0.02, 0.01], index=index[:2])
    with pytest.raises(ValueError, match="full baseline"):
        MODULE.relative_log_path(candidate, baseline)


def test_register_path_deduplicates_identical_returns() -> None:
    arrays = {}
    names = {}
    relative = np.array([0.01, -0.02])
    first = MODULE.register_path(arrays, names, "first", relative)
    second = MODULE.register_path(arrays, names, "second", relative.copy())
    assert first == second
    assert len(arrays) == 1
    assert names[first] == ["first", "second"]
