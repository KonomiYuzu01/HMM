import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_tool_module(name: str):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_tool_module("evaluate_probabilistic_reentry")
evaluation = load_tool_module("evaluate_macro_conditioned_reentry")


def test_bond_gate_excludes_decision_day_price() -> None:
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "BOND": np.linspace(100.0, 120.0, len(dates)),
            "CASH": np.linspace(100.0, 101.0, len(dates)),
        },
        index=dates,
    )
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("BOND")] = 1.0

    original = evaluation.bond_disinflation_gate(prices)
    modified = evaluation.bond_disinflation_gate(changed)

    assert original.iloc[-1] == modified.iloc[-1]


def test_macro_gate_blocks_inverse_signal_when_bonds_are_weak() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    raw = pd.DataFrame(
        {
            "probability": [0.30, 0.30],
            "base_rate": [0.60, 0.60],
        },
        index=dates,
    )
    gate = pd.Series([True, False], index=dates)

    result = evaluation.macro_conditioned_inverse_predictions(raw, gate)

    assert result.iloc[0]["probability"] > result.iloc[0]["base_rate"]
    assert result.iloc[1]["probability"] == result.iloc[1]["base_rate"]
