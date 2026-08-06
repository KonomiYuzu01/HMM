from __future__ import annotations

import pandas as pd
import pytest

import tools.evaluate_campaign_loss_governor as campaign
import tools.evaluate_reference_drawdown_governor as reference
from tools.evaluate_campaign_loss_governor import (
    CampaignLossCandidate,
    campaign_loss_state,
    causal_campaign_signals,
)


def test_campaign_signals_use_only_prior_close_information() -> None:
    index = pd.date_range("2024-01-01", periods=70, freq="B")
    closes = pd.DataFrame(
        {
            "QQQ": range(100, 170),
            "SEMIS": range(200, 270),
        },
        index=index,
        dtype=float,
    )
    reference = pd.DataFrame(
        {
            "equity": [1.0 + value / 1000 for value in range(70)],
            "drawdown": [0.0] * 70,
        },
        index=index,
    )
    original = causal_campaign_signals(closes, reference, sma_sessions=5)
    changed_closes = closes.copy()
    changed_reference = reference.copy()
    changed_closes.loc[index[-1], ["QQQ", "SEMIS"]] = [1.0, 1.0]
    changed_reference.loc[index[-1], "equity"] = 0.1
    changed = causal_campaign_signals(
        changed_closes,
        changed_reference,
        sma_sessions=5,
    )
    pd.testing.assert_series_equal(original.iloc[-1], changed.iloc[-1])


def test_false_bear_rally_does_not_release_defense() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    signals = pd.DataFrame(
        {
            "prior_reference_equity": [1.00, 0.89, 0.90, 0.91, 0.92],
            "prior_reference_drawdown": [0.00, -0.11, -0.10, -0.09, -0.08],
            "growth_above_sma": [True] * 5,
            "prior_growth_drawdown": [0.00, -0.20, -0.18, -0.16, -0.14],
        },
        index=index,
    )
    candidate = CampaignLossCandidate(
        "test",
        recovery_confirmation_days=2,
        ramp_stage_sessions=1,
    )
    diagnostics = campaign_loss_state(signals, candidate)
    assert diagnostics.loc[index[1], "state"] == "defense"
    assert (diagnostics.loc[index[1]:, "state"] == "defense").all()
    assert diagnostics.loc[index[1]:, "growth_cap"].eq(0.0).all()


def test_completed_recovery_starts_a_new_campaign_anchor() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    signals = pd.DataFrame(
        {
            "prior_reference_equity": [
                1.00,
                0.89,
                0.90,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.84,
            ],
            "prior_reference_drawdown": [0.00] * 10,
            "growth_above_sma": [True] * 10,
            "prior_growth_drawdown": [0.00] * 10,
        },
        index=index,
    )
    candidate = CampaignLossCandidate(
        "test",
        recovery_confirmation_days=1,
        ramp_stage_sessions=1,
    )
    diagnostics = campaign_loss_state(signals, candidate)
    assert diagnostics.loc[index[1], "entry"]
    assert diagnostics.loc[index[5], "state"] == "normal"
    assert diagnostics.loc[index[5], "campaign_peak"] == pytest.approx(0.93)
    assert not diagnostics.loc[index[6], "entry"]
    assert diagnostics.loc[index[9], "entry"]
    assert diagnostics.loc[index[9], "state"] == "defense"
    assert not diagnostics.loc[index[9], "r38_incremental_enabled"]


def test_campaign_signal_wrapper_remains_callable_when_installed() -> None:
    index = pd.date_range("2024-01-01", periods=70, freq="B")
    closes = pd.DataFrame(
        {"QQQ": range(100, 170), "SEMIS": range(200, 270)},
        index=index,
        dtype=float,
    )
    reference_path = pd.DataFrame(
        {"equity": [1.0] * 70, "drawdown": [0.0] * 70},
        index=index,
    )
    original = reference.causal_reference_signals
    try:
        reference.causal_reference_signals = campaign.causal_campaign_signals
        result = reference.causal_reference_signals(closes, reference_path)
    finally:
        reference.causal_reference_signals = original
    assert len(result) == len(index)
    assert "prior_growth_drawdown" in result


def test_reference_equity_is_filled_after_alignment_to_longer_price_history() -> None:
    price_index = pd.date_range("2023-01-02", periods=70, freq="B")
    reference_index = price_index[-10:]
    closes = pd.DataFrame(
        {"QQQ": range(100, 170), "SEMIS": range(200, 270)},
        index=price_index,
        dtype=float,
    )
    reference_path = pd.DataFrame(
        {"equity": [1.0] * 10, "drawdown": [0.0] * 10},
        index=reference_index,
    )
    result = causal_campaign_signals(closes, reference_path)
    assert result["prior_reference_equity"].notna().all()
    assert result.loc[price_index[0], "prior_reference_equity"] == pytest.approx(1.0)
