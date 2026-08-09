from __future__ import annotations

import pandas as pd
import pytest

from tools.audit_r38_forward_review import (
    execution_evidence,
    expected_forward_dates,
    forward_path_metrics,
)


def test_forward_path_enforces_relative_drawdown_tolerance() -> None:
    dates = pd.date_range("2026-07-29", periods=3, freq="B")
    metrics = forward_path_metrics(
        pd.Series([-0.01, 0.01, 0.0], index=dates),
        pd.Series([-0.014, 0.01, 0.0], index=dates),
        dates,
    )

    assert metrics["complete"] is True
    assert -0.005 < metrics["relative_max_drawdown_delta"] < 0.0
    assert (
        metrics["r38_max_drawdown"]
        < metrics["staged_max_drawdown"]
        < metrics["r11_max_drawdown"]
    )


def test_execution_evidence_passes_only_complete_real_records() -> None:
    dates = pd.date_range("2026-07-29", periods=2, freq="B")
    log = pd.DataFrame(
        {
            "date": dates,
            "signal_logged": [True, True],
            "positions_reconciled": [True, True],
            "realized_gde_one_way_cost_bps": [40.0, 74.9],
            "realized_financing_spread_bps": [100.0, 149.9],
            "unresolved_operational_incident": [False, False],
        }
    )

    evidence = execution_evidence(
        log,
        dates,
        maximum_gde_cost_bps=75.0,
        maximum_financing_spread_bps=150.0,
    )

    assert evidence["complete"] is True
    assert evidence["signals_complete"] is True
    assert evidence["positions_reconciled"] is True
    assert evidence["gde_cost_pass"] is True
    assert evidence["financing_cost_pass"] is True
    assert evidence["incident_free"] is True


def test_execution_evidence_fails_closed_when_a_session_is_missing() -> None:
    dates = pd.date_range("2026-07-29", periods=2, freq="B")
    log = pd.DataFrame(
        {
            "date": dates[:1],
            "signal_logged": [True],
            "positions_reconciled": [True],
            "realized_gde_one_way_cost_bps": [40.0],
            "realized_financing_spread_bps": [100.0],
            "unresolved_operational_incident": [False],
        }
    )

    evidence = execution_evidence(
        log,
        dates,
        maximum_gde_cost_bps=75.0,
        maximum_financing_spread_bps=150.0,
    )

    assert evidence["complete"] is False
    assert evidence["gde_cost_pass"] is False


def test_execution_evidence_rejects_duplicate_dates() -> None:
    dates = pd.date_range("2026-07-29", periods=2, freq="B")
    log = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "signal_logged": [True, True],
            "positions_reconciled": [True, True],
            "realized_gde_one_way_cost_bps": [40.0, 40.0],
            "realized_financing_spread_bps": [100.0, 100.0],
            "unresolved_operational_incident": [False, False],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        execution_evidence(
            log,
            dates,
            maximum_gde_cost_bps=75.0,
            maximum_financing_spread_bps=150.0,
        )


def test_expected_forward_dates_excludes_the_freeze_close() -> None:
    dates = pd.Series(pd.to_datetime(["2026-07-28", "2026-07-29", "2026-07-30"]))
    result = expected_forward_dates(
        dates,
        freeze_price_as_of="2026-07-28",
        current_price_as_of="2026-07-30",
    )

    assert result.tolist() == [pd.Timestamp("2026-07-29"), pd.Timestamp("2026-07-30")]
