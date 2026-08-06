from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.evaluate_r10_gde_capital_efficiency import (
    ASSETS,
    ALL_ASSETS,
    apply_gde_no_trade_band,
    causal_equity_excess_trend_fraction,
    causal_risk_budgeted_substitution_fraction,
    causal_calm_broad_bull_signal,
    substitution_target,
    synthetic_gde_interval_return,
)


def test_synthetic_gde_return_uses_financing_correct_proxy() -> None:
    returns = pd.Series({"SPX": 0.02, "GOLD": 0.01, "CASH": 0.001})
    observed = synthetic_gde_interval_return(returns)
    expected = 0.90 * 0.02 + 0.90 * 0.01 - 0.80 * 0.001
    assert abs(observed - expected) < 1e-12


def test_gold_substitution_preserves_account_weight() -> None:
    base = pd.Series(0.0, index=ASSETS)
    base["QQQ"] = 0.40
    base["SEMIS"] = 0.20
    base["GOLD"] = 0.20
    base["CASH"] = 0.20
    target, gde_weight = substitution_target(base, 0.50, "always")
    assert abs(gde_weight - 0.10) < 1e-12
    assert abs(float(target["GOLD"]) - 0.10) < 1e-12
    assert abs(float(target["GDE"]) - 0.10) < 1e-12
    assert abs(float(target.sum()) - 1.0) < 1e-12


def test_gde_band_trades_to_nearest_boundary_and_preserves_sleeve() -> None:
    target = pd.Series(0.0, index=ALL_ASSETS)
    target["GOLD"] = 0.12
    target["GDE"] = 0.08

    adjusted, implemented = apply_gde_no_trade_band(
        target,
        current_gde_weight=0.02,
        no_trade_band=0.01,
    )

    assert implemented == pytest.approx(0.07)
    assert adjusted["GDE"] == pytest.approx(0.07)
    assert adjusted["GOLD"] == pytest.approx(0.13)
    assert adjusted["GOLD"] + adjusted["GDE"] == pytest.approx(0.20)


def test_gde_band_does_not_trade_inside_band() -> None:
    target = pd.Series(0.0, index=ALL_ASSETS)
    target["GOLD"] = 0.12
    target["GDE"] = 0.08

    adjusted, implemented = apply_gde_no_trade_band(
        target,
        current_gde_weight=0.085,
        no_trade_band=0.01,
    )

    assert implemented == pytest.approx(0.085)
    assert adjusted["GDE"] == pytest.approx(0.085)
    assert adjusted["GOLD"] == pytest.approx(0.115)


def test_risk_budget_fraction_is_causal_to_same_day_close() -> None:
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    closes = pd.DataFrame(index=dates)
    for position, asset in enumerate(ASSETS):
        daily_return = 0.0001 * (position + 1)
        closes[asset] = (1.0 + daily_return) ** np.arange(len(dates))
    weights = pd.DataFrame(0.0, index=dates, columns=ASSETS)
    weights["QQQ"] = 0.40
    weights["SEMIS"] = 0.20
    weights["GOLD"] = 0.20
    weights["CASH"] = 0.20
    before = causal_risk_budgeted_substitution_fraction(
        weights,
        closes,
    )
    changed = closes.copy()
    changed.loc[dates[-1], "SPX"] *= 0.50
    after = causal_risk_budgeted_substitution_fraction(
        weights,
        changed,
    )

    assert before.loc[dates[-1]] == pytest.approx(
        after.loc[dates[-1]]
    )
    assert 0.0 <= before.loc[dates[-1]] <= 1.0


def test_equity_excess_trend_fraction_is_causal() -> None:
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    closes = pd.DataFrame(
        {
            "SPX": 100.0 * 1.002 ** np.arange(len(dates)),
            "CASH": 100.0 * 1.0001 ** np.arange(len(dates)),
        },
        index=dates,
    )
    before = causal_equity_excess_trend_fraction(
        closes,
        maximum_fraction=0.75,
    )
    changed = closes.copy()
    changed.loc[dates[-1], "SPX"] *= 0.50
    after = causal_equity_excess_trend_fraction(
        changed,
        maximum_fraction=0.75,
    )

    assert before.loc[dates[-1]] == pytest.approx(0.75)
    assert before.loc[dates[-1]] == pytest.approx(
        after.loc[dates[-1]]
    )


def test_growth_linked_gate_turns_off_when_growth_is_zero() -> None:
    base = pd.Series(0.0, index=ASSETS)
    base["GOLD"] = 0.20
    base["CASH"] = 0.80
    target, gde_weight = substitution_target(
        base,
        0.50,
        "growth_linked",
    )
    assert gde_weight == 0.0
    assert target["GOLD"] == 0.20
    assert target["GDE"] == 0.0


def test_calm_bull_signal_at_open_does_not_use_same_day_close() -> None:
    index = pd.date_range("2020-01-01", periods=230, freq="B")
    spx = pd.Series(range(100, 330), index=index, dtype=float)
    cash = pd.Series(range(100, 330), index=index, dtype=float) * 0.01 + 100
    closes = pd.DataFrame({"SPX": spx, "CASH": cash})
    before = causal_calm_broad_bull_signal(closes)
    changed = closes.copy()
    changed.loc[index[-1], "SPX"] *= 0.01
    after = causal_calm_broad_bull_signal(changed)
    assert bool(before.loc[index[-1]]) == bool(after.loc[index[-1]])


def test_calm_bull_market_gate_can_disable_substitution() -> None:
    base = pd.Series(0.0, index=ASSETS)
    base["QQQ"] = 0.40
    base["SEMIS"] = 0.20
    base["GOLD"] = 0.20
    base["CASH"] = 0.20
    target, gde_weight = substitution_target(
        base,
        0.50,
        "calm_broad_bull",
        market_gate=False,
    )
    assert gde_weight == 0.0
    assert target["GDE"] == 0.0


def test_base_plus_calm_keeps_floor_when_market_gate_is_off() -> None:
    base = pd.Series(0.0, index=ASSETS)
    base["QQQ"] = 0.40
    base["SEMIS"] = 0.40
    base["GOLD"] = 0.20
    target, gde_weight = substitution_target(
        base,
        0.75,
        "base_plus_calm",
        market_gate=False,
        minimum_substitution_fraction=0.25,
    )
    assert gde_weight == pytest.approx(0.05)
    assert target["GOLD"] == pytest.approx(0.15)
    assert target["GDE"] == pytest.approx(0.05)


def test_base_plus_calm_uses_ceiling_when_market_gate_is_on() -> None:
    base = pd.Series(0.0, index=ASSETS)
    base["QQQ"] = 0.40
    base["SEMIS"] = 0.40
    base["GOLD"] = 0.20
    _, gde_weight = substitution_target(
        base,
        0.75,
        "base_plus_calm",
        market_gate=True,
        minimum_substitution_fraction=0.25,
    )
    assert gde_weight == pytest.approx(0.15)


def test_base_plus_calm_rejects_floor_above_ceiling() -> None:
    base = pd.Series(0.0, index=ASSETS)
    with pytest.raises(ValueError, match="between zero"):
        substitution_target(
            base,
            0.25,
            "base_plus_calm",
            minimum_substitution_fraction=0.50,
        )


def test_frame_overrides_must_be_supplied_together() -> None:
    from tools.evaluate_r10_gde_capital_efficiency import (
        simulate_gde_substitution,
    )

    with pytest.raises(ValueError, match="supplied together"):
        simulate_gde_substitution(
            "unused",
            pd.DataFrame(),
            pd.DataFrame(),
            substitution_fraction=0.0,
            gate_mode="always",
            gde_return_mode="synthetic",
            start_date="2020-01-01",
            end_date=None,
            weights_override=pd.DataFrame(),
        )


def test_target_policy_uses_prior_equity_and_can_force_next_open_trade() -> None:
    from tools.evaluate_r10_gde_capital_efficiency import (
        simulate_gde_substitution,
    )

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    opens = pd.DataFrame(100.0, index=dates, columns=ASSETS)
    closes = opens.copy()
    closes.loc[dates[0], "QQQ"] = 90.0
    opens.loc[dates[1], "QQQ"] = 90.0
    closes.loc[dates[1], "QQQ"] = 80.0
    opens.loc[dates[2], "QQQ"] = 80.0
    closes.loc[dates[2], "QQQ"] = 70.0
    weights = pd.DataFrame(0.0, index=dates, columns=ASSETS)
    weights["QQQ"] = 1.0
    daily = pd.DataFrame(
        {"turnover": [1.0, 0.0, 0.0], "slippage_cost": 0.0},
        index=dates,
    )
    observed_prior_drawdowns: list[float] = []

    def policy(
        date: pd.Timestamp,
        target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | bool | str]]:
        del date
        prior_drawdown = prior_equity / prior_peak - 1.0
        observed_prior_drawdowns.append(prior_drawdown)
        if prior_drawdown <= -0.05:
            adjusted = pd.Series(0.0, index=ASSETS)
            adjusted["CASH"] = 1.0
            return adjusted, True, {"policy_state": "defense"}
        return target, False, {"policy_state": "normal"}

    result = simulate_gde_substitution(
        "unused",
        opens,
        closes,
        substitution_fraction=0.0,
        gate_mode="always",
        gde_return_mode="synthetic",
        start_date=str(dates[0].date()),
        end_date=None,
        base_one_way_cost_bps=0.0,
        gde_one_way_cost_bps=0.0,
        financing_spread_bps=0.0,
        weights_override=weights,
        daily_override=daily,
        target_policy=policy,
    )

    assert observed_prior_drawdowns == pytest.approx([0.0, -0.10, -0.10])
    assert result.loc[dates[0], "policy_state"] == "normal"
    assert result.loc[dates[1], "policy_state"] == "defense"
    assert bool(result.loc[dates[1], "policy_force_trade"])
    assert result.loc[dates[1], "net_return"] == pytest.approx(0.0)
    assert result.loc[dates[2], "net_return"] == pytest.approx(0.0)
    assert result.loc[dates[-1], "equity"] == pytest.approx(0.90)
    assert result.loc[dates[1], "policy_prior_drawdown"] == pytest.approx(-0.10)
