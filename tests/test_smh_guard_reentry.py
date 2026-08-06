from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_smh_dynamic_guard import (  # noqa: E402
    ASSETS,
    account_weighted_gap_loss,
    loss_trigger_source,
    relative_gap_filter_passed,
    restore_pair_mix,
    switch_guard_overflow,
    trigger_cap_for_loss,
)
from evaluate_smh_shock_guard import restore_asset_pair  # noqa: E402
from evaluate_smh_dynamic_guard_robustness import (  # noqa: E402
    event_bootstrap,
    event_clusters,
    event_cycles,
)


def weights(**overrides: float) -> np.ndarray:
    result = np.zeros(len(ASSETS), dtype=float)
    for asset, value in overrides.items():
        result[ASSETS.index(asset)] = value
    return result


def test_open_gap_guard_reinvests_temporary_cash() -> None:
    baseline = weights(QQQ=0.10, SEMIS=0.50, GOLD=0.20, CASH=0.20)
    guarded = weights(QQQ=0.10, SEMIS=0.15, GOLD=0.20, CASH=0.55)

    restored = restore_pair_mix(guarded, baseline, "CASH")

    np.testing.assert_allclose(restored, baseline)


def test_close_confirmed_guard_restores_semis_qqq_mix() -> None:
    baseline = weights(QQQ=0.20, SEMIS=0.60, GOLD=0.20)
    guarded = weights(QQQ=0.65, SEMIS=0.15, GOLD=0.20)

    restored = restore_asset_pair(guarded, baseline, "QQQ")

    np.testing.assert_allclose(restored, baseline)


def test_bridge_moves_only_temporary_cash_to_qqq() -> None:
    baseline = weights(QQQ=0.10, SEMIS=0.50, GOLD=0.20, CASH=0.20)
    cash_guarded = weights(QQQ=0.10, SEMIS=0.15, GOLD=0.20, CASH=0.55)

    bridged = switch_guard_overflow(
        cash_guarded,
        baseline,
        "CASH",
        "QQQ",
    )

    np.testing.assert_allclose(
        bridged,
        weights(QQQ=0.45, SEMIS=0.15, GOLD=0.20, CASH=0.20),
    )


def test_account_weighted_gap_loss_has_account_units() -> None:
    assert account_weighted_gap_loss(0.50, -0.04) == 0.02
    assert account_weighted_gap_loss(0.25, -0.04) == 0.01
    assert account_weighted_gap_loss(0.50, 0.04) == 0.0


def test_severe_weighted_gap_loss_selects_deeper_cap() -> None:
    assert trigger_cap_for_loss(0.021, 0.30, 0.022, 0.15) == 0.30
    assert trigger_cap_for_loss(0.022, 0.30, 0.022, 0.15) == 0.15


def test_prior_close_loss_is_an_alternative_trigger_source() -> None:
    assert loss_trigger_source(True, True, 0.01, 0.02) == "gap"
    assert loss_trigger_source(True, False, 0.02, 0.02) == "prior_close"
    assert loss_trigger_source(True, True, 0.02, 0.02) == "both"
    assert loss_trigger_source(False, False, 0.01, None) is None


def test_large_account_loss_can_bypass_relative_gap_filter() -> None:
    assert relative_gap_filter_passed(-0.021, -0.02, 0.020, 0.022)
    assert relative_gap_filter_passed(-0.010, -0.02, 0.022, 0.022)
    assert not relative_gap_filter_passed(-0.010, -0.02, 0.021, 0.022)


def test_event_cycles_merge_same_day_recovery_and_retrigger() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    candidate = pd.DataFrame(
        {
            "net_return": [-0.01, 0.01, -0.02, 0.01],
            "triggered": [True, False, True, False],
            "recovered": [False, False, True, True],
        },
        index=dates,
    )
    baseline = pd.DataFrame(
        {"net_return": [-0.02, 0.01, -0.03, 0.01]},
        index=dates,
    )

    cycles = event_cycles(candidate, baseline, "candidate")

    assert len(cycles) == 1
    assert cycles.iloc[0]["trigger_date"] == dates[0]
    assert cycles.iloc[0]["recovery_date"] == dates[3]


def test_event_clusters_merge_nearby_cycles_but_not_distant_ones() -> None:
    dates = pd.date_range("2026-01-01", periods=16, freq="B")
    candidate = pd.DataFrame(
        {
            "net_return": np.zeros(len(dates)),
            "triggered": [
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "recovered": [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        },
        index=dates,
    )
    baseline = pd.DataFrame(
        {"net_return": np.zeros(len(dates))},
        index=dates,
    )

    clusters = event_clusters(
        candidate,
        baseline,
        "candidate",
        maximum_gap_sessions=5,
    )

    assert len(clusters) == 2
    assert clusters.iloc[0]["trigger_date"] == dates[0]
    assert clusters.iloc[0]["recovery_date"] == dates[4]
    assert clusters.iloc[1]["trigger_date"] == dates[10]


def test_event_bootstrap_reports_small_sample_sign_test() -> None:
    cycles = pd.DataFrame(
        {"return_delta": [0.01, 0.02, 0.03, 0.04]}
    )

    result = event_bootstrap(cycles, samples=100, seed=7)

    assert result["positive_event_share"] == 1.0
    assert result["one_sided_sign_test_pvalue"] == 0.0625
    assert 0.0 < result["positive_event_rate_95pct_lower"] < 0.5
