from __future__ import annotations

import pandas as pd

from tools.audit_recursive_trend_cushion_qualified import audit_daily_path


def test_daily_audit_accepts_causal_prior_equity_path() -> None:
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    path = pd.DataFrame(
        {
            "net_return": [-0.10, 0.05, 0.02],
            "policy_prior_equity": [1.0, 0.9, 0.945],
            "policy_prior_peak": [1.0, 1.0, 1.0],
            "recursive_prior_drawdown": [0.0, -0.10, -0.055],
            "floor_equity": [0.8125, 0.8125, 0.8125],
            "implemented_cash_weight": [0.0, 0.5, 0.4],
            "accepted_non_cash_cap": [1.0, 0.5, 0.6],
            "state": ["normal", "controlled", "controlled"],
            "r38_incremental_enabled": [True, False, False],
        },
        index=index,
    )
    result = audit_daily_path(path)
    assert result["passed"]


def test_daily_audit_rejects_same_day_equity_leakage() -> None:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    path = pd.DataFrame(
        {
            "net_return": [-0.10, 0.0],
            "policy_prior_equity": [0.9, 0.9],
            "policy_prior_peak": [1.0, 1.0],
            "recursive_prior_drawdown": [-0.1, -0.1],
            "floor_equity": [0.8125, 0.8125],
            "implemented_cash_weight": [0.0, 0.5],
            "accepted_non_cash_cap": [1.0, 0.5],
            "state": ["normal", "controlled"],
            "r38_incremental_enabled": [True, False],
        },
        index=index,
    )
    result = audit_daily_path(path)
    assert not result["passed"]
    assert result["prior_equity_max_error"] > 0.0
