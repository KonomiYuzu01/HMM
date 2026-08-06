import numpy as np
import pandas as pd

from regime_strategy.operations import (
    dollar_order_plan,
    growth_risk_snapshot,
    latest_completed_us_equity_session,
    missing_recent_us_equity_sessions,
    next_session_execution_status,
    next_us_equity_session,
    onboarding_targets,
    production_signal_generation_status,
)


def test_onboarding_targets_form_a_growth_to_safety_frontier() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.06, -0.06], 40),
            "CASH": np.zeros(80),
        }
    )
    model = np.array([0.4, 0.4, 0.2])
    targets = onboarding_targets(
        model, returns, assets, ["QQQ", "SEMIS"], 2, 20, 60, 0.20
    )

    assert all(np.isclose(target.sum(), 1.0) for target in targets.values())
    assert targets["growth_first"][0] > model[0]
    assert targets["growth_first"][1] < model[1]
    assert np.isclose(targets["growth_first"][:2].sum(), model[:2].sum())
    assert targets["volatility_capped"][:2].sum() < model[:2].sum()
    assert targets["account_volatility_capped"][:2].sum() < model[:2].sum()
    assert np.isclose(targets["cash_first"][0], model[0])
    assert targets["cash_first"][1] < model[1]


def test_growth_risk_snapshot_attributes_more_risk_to_high_volatility_smh() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.01, -0.01], 40),
            "SEMIS": np.tile([0.03, -0.03], 40),
            "CASH": np.zeros(80),
        }
    )
    snapshot = growth_risk_snapshot(
        np.array([0.4, 0.4, 0.2]),
        returns,
        assets,
        ["QQQ", "SEMIS"],
        20,
        60,
    )
    assert snapshot["account_realized_volatility"] > 0.20
    assert snapshot["smh_effective_volatility"] > snapshot["qqq_effective_volatility"]
    assert snapshot["smh_risk_share"] > snapshot["qqq_risk_share"]
    assert np.isclose(
        snapshot["smh_risk_share"] + snapshot["qqq_risk_share"], 1.0
    )


def test_growth_risk_snapshot_includes_gold_in_total_account_risk() -> None:
    assets = ["QQQ", "SEMIS", "GOLD", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.01, -0.01], 40),
            "SEMIS": np.tile([0.03, -0.03], 40),
            "GOLD": np.tile([-0.005, 0.005], 40),
            "CASH": np.zeros(80),
        }
    )
    snapshot = growth_risk_snapshot(
        np.array([0.4, 0.4, 0.2, 0.0]),
        returns,
        assets,
        ["QQQ", "SEMIS"],
        20,
        60,
        risk_assets=["QQQ", "SEMIS", "GOLD"],
    )
    assert "gold_risk_share" in snapshot
    assert np.isclose(
        snapshot["qqq_risk_share"]
        + snapshot["smh_risk_share"]
        + snapshot["gold_risk_share"],
        1.0,
    )


def test_dollar_order_plan_translates_zero_holdings_to_buys() -> None:
    plan = dollar_order_plan(
        np.array([0.5, 0.3, 0.2]),
        np.zeros(3),
        np.array([500.0, 250.0, 100.0]),
        ["QQQ", "SEMIS", "CASH"],
        700_000.0,
    )
    assert np.isclose(plan["target_dollars"].sum(), 700_000.0)
    assert plan["action"].eq("BUY").all()
    assert np.allclose(plan["reference_trade_shares"], [700.0, 840.0, 1400.0])
    assert np.allclose(plan["whole_trade_shares"], [700.0, 840.0, 1400.0])
    assert np.allclose(plan["rounding_residual_dollars"], 0.0)


def test_next_session_execution_status_rejects_an_intraday_late_signal() -> None:
    last_close = pd.Timestamp("2026-07-21")
    ready, execution_date = next_session_execution_status(
        last_close,
        pd.Timestamp("2026-07-22 08:00", tz="America/New_York"),
    )
    missed, _ = next_session_execution_status(
        last_close,
        pd.Timestamp("2026-07-22 15:00", tz="America/New_York"),
    )
    assert ready == "READY_FOR_OPEN"
    assert missed == "MISSED"
    assert execution_date == pd.Timestamp("2026-07-22")


def test_next_us_equity_session_skips_observed_independence_day() -> None:
    assert next_us_equity_session(pd.Timestamp("2026-07-02")) == pd.Timestamp(
        "2026-07-06"
    )


def test_next_us_equity_session_skips_good_friday() -> None:
    assert next_us_equity_session(pd.Timestamp("2026-04-02")) == pd.Timestamp(
        "2026-04-06"
    )


def test_latest_completed_session_changes_only_after_completion_buffer() -> None:
    before_close = latest_completed_us_equity_session(
        pd.Timestamp("2026-07-28 12:14", tz="America/New_York")
    )
    after_close = latest_completed_us_equity_session(
        pd.Timestamp("2026-07-28 18:00", tz="America/New_York")
    )

    assert before_close == pd.Timestamp("2026-07-27")
    assert after_close == pd.Timestamp("2026-07-28")


def test_latest_completed_session_skips_weekends_and_holidays() -> None:
    weekend = latest_completed_us_equity_session(
        pd.Timestamp("2026-07-05 18:00", tz="America/New_York")
    )
    holiday_morning = latest_completed_us_equity_session(
        pd.Timestamp("2026-07-06 08:00", tz="America/New_York")
    )

    assert weekend == pd.Timestamp("2026-07-02")
    assert holiday_morning == pd.Timestamp("2026-07-02")


def test_production_generation_is_blocked_only_during_live_session() -> None:
    assert (
        production_signal_generation_status(
            pd.Timestamp("2026-07-28 12:14", tz="America/New_York")
        )
        == "INTRADAY_BLOCKED"
    )
    assert (
        production_signal_generation_status(
            pd.Timestamp("2026-07-28 08:00", tz="America/New_York")
        )
        == "AVAILABLE"
    )


def test_recent_session_gap_detection_uses_exchange_holidays() -> None:
    observed = pd.to_datetime(
        [
            "2026-06-30",
            "2026-07-01",
            "2026-07-02",
            "2026-07-06",
            "2026-07-08",
        ]
    )
    missing = missing_recent_us_equity_sessions(
        observed,
        pd.Timestamp("2026-07-08 18:00", tz="America/New_York"),
    )

    assert missing.tolist() == [pd.Timestamp("2026-07-07")]
    assert (
        production_signal_generation_status(
            pd.Timestamp("2026-07-28 18:00", tz="America/New_York")
        )
        == "AVAILABLE"
    )
