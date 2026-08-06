from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r24_declared_cash_hard_limit import (
    DECLARED_CASH_FLOOR,
    enforce_daily_cash_target_limit,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2025-01-02", periods=3)
    weights = pd.DataFrame(
        {
            "QQQ": [0.70, 0.60, 0.40],
            "SEMIS": [0.60, 0.50, 0.30],
            "GOLD": [0.00, 0.05, 0.10],
            "CASH": [-0.30, -0.15, 0.20],
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return weights, daily


def test_hard_limit_enforces_declared_cash_floor() -> None:
    weights, daily = _inputs()
    limited, _, _ = enforce_daily_cash_target_limit(weights, daily)
    assert limited["CASH"].ge(DECLARED_CASH_FLOOR - 1e-12).all()


def test_hard_limit_preserves_total_weight() -> None:
    weights, daily = _inputs()
    limited, _, _ = enforce_daily_cash_target_limit(weights, daily)
    assert np.allclose(limited.sum(axis=1), weights.sum(axis=1))


def test_hard_limit_scales_non_cash_assets_proportionally() -> None:
    weights, daily = _inputs()
    limited, _, diagnostics = enforce_daily_cash_target_limit(
        weights,
        daily,
    )
    date = weights.index[0]
    scale = diagnostics.loc[date, "hard_limit_cash_scale"]
    for asset in ("QQQ", "SEMIS", "GOLD"):
        assert np.isclose(
            limited.loc[date, asset],
            weights.loc[date, asset] * scale,
        )


def test_hard_limit_creates_execution_update_when_active() -> None:
    weights, daily = _inputs()
    _, updated, diagnostics = enforce_daily_cash_target_limit(
        weights,
        daily,
    )
    active = diagnostics["hard_limit_active"]
    assert updated.loc[active, "turnover"].gt(0.0).all()
    assert updated.loc[~active, "turnover"].eq(0.0).all()
