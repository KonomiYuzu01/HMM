from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r40_highwater_satellite import (
    HighWaterSatellitePolicy,
    SatelliteCandidate,
)


class StubBasePolicy:
    def __init__(self, state: str = "normal") -> None:
        self.state = state

    def __call__(self, date, target, prior_equity, prior_peak):
        return target.copy(), False, {"state": self.state}


def base_target() -> pd.Series:
    return pd.Series(
        {
            "SPX": 0.20,
            "QQQ": 0.20,
            "SEMIS": 0.10,
            "BOND": 0.10,
            "GOLD": 0.10,
            "OIL": 0.0,
            "USD": 0.0,
            "CASH": 0.30,
            "VIX_HEDGE": 0.0,
        }
    )


def test_satellite_adds_fixed_growth_weight_and_preserves_total() -> None:
    date = pd.Timestamp("2026-01-05")
    candidate = SatelliteCandidate("test", 0.10, -0.02, 0.75)
    policy = HighWaterSatellitePolicy(
        StubBasePolicy(),  # type: ignore[arg-type]
        pd.Series(True, index=[date]),
        candidate,
    )
    target, force, metadata = policy(date, base_target(), 1.0, 1.0)
    assert abs(float(target.sum()) - 1.0) < 1e-12
    assert target["QQQ"] == pytest.approx(0.225)
    assert target["SEMIS"] == pytest.approx(0.175)
    assert target["CASH"] == pytest.approx(0.20)
    assert force
    assert metadata["satellite_enabled"] is True


def test_satellite_can_split_weight_across_growth_spx_and_gold() -> None:
    date = pd.Timestamp("2026-01-05")
    candidate = SatelliteCandidate(
        "test",
        0.10,
        -0.02,
        semis_share=0.20,
        spx_share=0.30,
        gold_share=0.40,
    )
    policy = HighWaterSatellitePolicy(
        StubBasePolicy(),  # type: ignore[arg-type]
        pd.Series(True, index=[date]),
        candidate,
    )
    target, force, metadata = policy(date, base_target(), 1.0, 1.0)
    assert abs(float(target.sum()) - 1.0) < 1e-12
    assert target["QQQ"] == pytest.approx(0.21)
    assert target["SEMIS"] == pytest.approx(0.12)
    assert target["SPX"] == pytest.approx(0.23)
    assert target["GOLD"] == pytest.approx(0.14)
    assert target["CASH"] == pytest.approx(0.20)
    assert force
    assert metadata["satellite_qqq_weight"] == pytest.approx(0.01)
    assert metadata["satellite_semis_weight"] == pytest.approx(0.02)
    assert metadata["satellite_spx_weight"] == pytest.approx(0.03)
    assert metadata["satellite_gold_weight"] == pytest.approx(0.04)


def test_account_drawdown_gate_disables_satellite_and_forces_exit() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    candidate = SatelliteCandidate("test", 0.10, -0.02, 0.50)
    policy = HighWaterSatellitePolicy(
        StubBasePolicy(),  # type: ignore[arg-type]
        pd.Series(True, index=dates),
        candidate,
    )
    enabled, _, _ = policy(dates[0], base_target(), 1.0, 1.0)
    disabled, force, metadata = policy(dates[1], base_target(), 0.97, 1.0)
    assert enabled["CASH"] == pytest.approx(0.20)
    assert disabled.equals(base_target())
    assert force
    assert metadata["satellite_enabled"] is False


def test_controlled_base_state_never_receives_satellite() -> None:
    date = pd.Timestamp("2026-01-05")
    candidate = SatelliteCandidate("test", 0.10, -0.02, 0.50)
    policy = HighWaterSatellitePolicy(
        StubBasePolicy("controlled"),  # type: ignore[arg-type]
        pd.Series(True, index=[date]),
        candidate,
    )
    target, force, metadata = policy(date, base_target(), 1.0, 1.0)
    assert target.equals(base_target())
    assert not force
    assert metadata["satellite_enabled"] is False


def test_false_permission_never_uses_satellite() -> None:
    date = pd.Timestamp("2026-01-05")
    candidate = SatelliteCandidate("test", 0.10, -0.02, 0.50)
    policy = HighWaterSatellitePolicy(
        StubBasePolicy(),  # type: ignore[arg-type]
        pd.Series(False, index=[date]),
        candidate,
    )
    target, force, metadata = policy(date, base_target(), 1.0, 1.0)
    assert target.equals(base_target())
    assert not force
    assert metadata["satellite_permission"] is False
