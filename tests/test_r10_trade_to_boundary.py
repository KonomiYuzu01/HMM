import numpy as np
import pytest

from tools.evaluate_r10_trade_to_boundary import (
    ASSETS,
    boundary_target,
    material_risk_change,
)


def test_boundary_target_stops_at_remaining_turnover_boundary() -> None:
    current = np.array([0.0, 0.5, 0.0, 0.5])
    desired = np.array([0.0, 0.2, 0.3, 0.5])
    target, executed = boundary_target(current, desired, 0.10)
    remaining = 0.5 * np.abs(desired - target).sum()
    assert remaining == pytest.approx(0.10)
    assert executed == pytest.approx(0.20)


def test_boundary_target_does_not_trade_inside_region() -> None:
    current = np.array([0.5, 0.5])
    desired = np.array([0.49, 0.51])
    target, executed = boundary_target(current, desired, 0.02)
    assert np.allclose(target, current)
    assert executed == 0.0


def test_material_risk_change_ignores_growth_internal_rotation() -> None:
    current = np.zeros(len(ASSETS))
    desired = np.zeros(len(ASSETS))
    current[ASSETS.index("QQQ")] = 0.40
    current[ASSETS.index("SEMIS")] = 0.40
    desired[ASSETS.index("QQQ")] = 0.60
    desired[ASSETS.index("SEMIS")] = 0.20
    current[ASSETS.index("GOLD")] = desired[ASSETS.index("GOLD")] = 0.20
    assert not material_risk_change(current, desired)


def test_material_risk_change_detects_growth_reentry() -> None:
    current = np.zeros(len(ASSETS))
    desired = np.zeros(len(ASSETS))
    current[ASSETS.index("CASH")] = 1.0
    desired[ASSETS.index("QQQ")] = 0.20
    desired[ASSETS.index("CASH")] = 0.80
    assert material_risk_change(current, desired)
