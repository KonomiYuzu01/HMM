import numpy as np
import pandas as pd

from regime_strategy.portfolio import (
    apply_account_stress_volatility_cap,
    apply_asset_aware_growth_risk_budget,
    apply_conditional_cash_funded_hedge,
    apply_daily_controlled_asset_risk_reduction,
    apply_daily_growth_risk_reduction,
    apply_daily_growth_risk_reallocation_target,
    apply_daily_growth_risk_target,
    apply_trend_cvar_risk_reduction,
    apply_relative_asset_risk_cap,
    annualized_bipower_volatility,
    annualized_downside_semivolatility,
    annualized_semiskew_effective_volatility,
    apply_asset_multiplier,
    apply_growth_stress_guard,
    apply_pair_survival_multiplier,
    apply_risk_off_growth_floor,
    apply_risk_on_leverage,
    apply_self_financing_overlay,
    blend_equal_weight_core,
    blend_satellite_allocation,
    blend_strategic_core,
    cap_asset_standalone_risk_contribution,
    cap_pair_asset_weight,
    cap_pair_standalone_risk_share,
    causal_realized_volatility_state,
    compose_probability_allocation,
    financing_spread_cost,
    filter_core_weights_by_trend,
    high_volatility_core_weights,
    implied_volatility_term_structure,
    incremental_growth_floor_term_structure_cap,
    incremental_leverage_term_structure_gate,
    idiosyncratic_downside_survival_multiplier,
    inverse_volatility_weights,
    joint_excess_trend_leverage_gate,
    long_only_trend_defensive_weights,
    drawdown_conditioned_regime_candidate,
    ensure_minimum_growth_exposure,
    ensure_minimum_growth_group_exposure,
    multi_horizon_trend_forecast,
    paper_regime_candidate,
    reallocate_pair_weights,
    realized_downside_variation_share,
    realized_jump_variation_share,
    realized_volatility_stress,
    relative_momentum_pair_weight,
    relative_momentum_spread_overlay_weights,
    residual_momentum_beta_pair_weight,
    realized_stress_covariance,
    risk_off_growth_floor_is_active,
    satellite_share_for_mode,
    self_financing_overlay_share_for_mode,
    solve_allocation,
    transaction_cost,
    time_series_momentum_overlay_weights,
    trailing_compounded_return,
    volatility_regime_gated_relative_momentum_weight,
    trend_volatility_exposure,
    turning_point_cycle,
    update_asymmetric_regime_state,
    update_vix_recovery_bridge,
    volatility_managed_relative_momentum_weight,
    zero_entry_allocation_method,
)


def test_core_trend_filter_reallocates_filtered_weight_pro_rata() -> None:
    adjusted, removed = filter_core_weights_by_trend(
        {"QQQ": 0.40, "SEMIS": 0.40, "GOLD": 0.20},
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        np.array([0.5, 0.7, -0.1, 0.0]),
        ["GOLD"],
        0.0,
    )

    assert removed == 1
    assert adjusted == {"QQQ": 0.50, "SEMIS": 0.50, "GOLD": 0.0}


def test_core_trend_filter_keeps_asset_without_finite_signal() -> None:
    weights = {"QQQ": 0.40, "SEMIS": 0.40, "GOLD": 0.20}
    adjusted, removed = filter_core_weights_by_trend(
        weights,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        np.array([0.5, 0.7, np.nan, 0.0]),
        ["GOLD"],
        0.0,
    )

    assert removed == 0
    assert adjusted == weights


def test_core_trend_filter_can_keep_a_permanent_insurance_floor() -> None:
    adjusted, removed = filter_core_weights_by_trend(
        {"QQQ": 0.40, "SEMIS": 0.40, "GOLD": 0.20},
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        np.array([0.5, 0.7, -0.1, 0.0]),
        ["GOLD"],
        0.0,
        {"GOLD": 0.10},
    )

    assert removed == 1
    assert adjusted == {"QQQ": 0.45, "SEMIS": 0.45, "GOLD": 0.10}


def test_jump_variation_share_separates_isolated_jump_from_diffusion() -> None:
    diffusion = pd.Series(np.tile([0.01, -0.01], 11))
    isolated_jump = diffusion.copy()
    isolated_jump.iloc[-1] = 0.10

    assert realized_jump_variation_share(diffusion, 20) < 0.0
    assert realized_jump_variation_share(isolated_jump, 20) > 0.0


def test_trailing_compounded_return_uses_only_requested_window() -> None:
    returns = pd.Series([0.50, 0.10, -0.10])

    assert np.isclose(
        trailing_compounded_return(returns, 2),
        (1.10 * 0.90) - 1.0,
    )


def test_optimizer_respects_long_only_caps_and_budget() -> None:
    expected = np.array([0.08, 0.04, 0.02])
    covariance = np.diag([0.04, 0.02, 0.001])
    previous = np.array([0.0, 0.0, 1.0])
    weights = solve_allocation(expected, covariance, previous, 2, 4.0, 0.001, 0.4)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert weights[0] <= 0.400001
    assert weights[1] <= 0.400001


def test_cost_uses_total_traded_notional() -> None:
    previous = np.array([0.5, 0.5])
    target = np.array([0.6, 0.4])
    cost, turnover = transaction_cost(previous, target, 10.0)
    assert np.isclose(turnover, 0.1)
    assert np.isclose(cost, 0.0002)


def test_pair_standalone_risk_share_cap_preserves_growth_exposure() -> None:
    weights = np.array([0.20, 0.65, 0.10, 0.05])
    adjusted, reallocated, before, after, maximum_weight = (
        cap_pair_standalone_risk_share(
            weights,
            ["QQQ", "SEMIS", "GOLD", "CASH"],
            "QQQ",
            "SEMIS",
            anchor_volatility=0.20,
            controlled_volatility=0.40,
            maximum_controlled_risk_share=0.60,
        )
    )
    assert before > 0.60
    assert np.isclose(after, 0.60)
    assert np.isclose(maximum_weight, 0.36428571428571427)
    assert np.isclose(reallocated, 0.28571428571428575)
    assert np.isclose(adjusted[:2].sum(), weights[:2].sum())
    assert np.isclose(adjusted.sum(), weights.sum())
    assert np.allclose(adjusted[2:], weights[2:])


def test_pair_standalone_risk_share_cap_does_not_force_a_tilt() -> None:
    weights = np.array([0.60, 0.20, 0.10, 0.10])
    adjusted, reallocated, before, after, _ = cap_pair_standalone_risk_share(
        weights,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        "QQQ",
        "SEMIS",
        anchor_volatility=0.20,
        controlled_volatility=0.30,
        maximum_controlled_risk_share=0.60,
    )
    assert np.allclose(adjusted, weights)
    assert np.isclose(reallocated, 0.0)
    assert np.isclose(after, before)


def test_asset_standalone_risk_contribution_moves_only_excess_to_cash() -> None:
    weights = np.array([0.60, 0.20, 0.10, 0.10])
    adjusted, released, before, after = cap_asset_standalone_risk_contribution(
        weights,
        asset_index=2,
        cash_index=3,
        effective_volatility=0.30,
        maximum_standalone_contribution=0.015,
    )
    assert np.isclose(before, 0.03)
    assert np.isclose(after, 0.015)
    assert np.isclose(released, 0.05)
    assert np.allclose(adjusted, [0.60, 0.20, 0.05, 0.15])
    assert np.isclose(adjusted.sum(), weights.sum())


def test_idiosyncratic_survival_responds_only_to_unexplained_downside() -> None:
    index = pd.bdate_range("2023-01-02", periods=147)
    anchor = pd.Series(
        0.001 * np.sin(np.arange(147) * np.pi / 4.0),
        index=index,
    )
    residual = pd.Series(
        np.resize(np.array([-0.001, 0.001]), 147),
        index=index,
    )
    controlled = 1.3 * anchor + residual
    calm_multiplier, calm_score, calm_beta = (
        idiosyncratic_downside_survival_multiplier(
            anchor,
            controlled,
            beta_days=126,
            shock_horizons=[5, 10, 21],
            soft_z_score=-1.0,
            hard_z_score=-2.5,
        )
    )
    assert np.isclose(calm_multiplier, 1.0)
    assert calm_score > -1.0
    assert np.isclose(calm_beta, 1.3, atol=0.01)

    shocked = controlled.copy()
    shocked.iloc[-10:] -= 0.005
    shock_multiplier, shock_score, _ = (
        idiosyncratic_downside_survival_multiplier(
            anchor,
            shocked,
            beta_days=126,
            shock_horizons=[5, 10, 21],
            soft_z_score=-1.0,
            hard_z_score=-2.5,
        )
    )
    assert np.isclose(shock_multiplier, 0.0)
    assert shock_score < -2.5


def test_pair_survival_multiplier_preserves_pair_and_account_exposure() -> None:
    weights = np.array([0.30, 0.50, 0.20])
    adjusted, reallocated = apply_pair_survival_multiplier(
        weights,
        ["QQQ", "SEMIS", "CASH"],
        "QQQ",
        "SEMIS",
        0.40,
    )
    assert np.isclose(reallocated, 0.18)
    assert np.allclose(adjusted, [0.48, 0.32, 0.20])
    assert np.isclose(adjusted[:2].sum(), weights[:2].sum())
    assert np.isclose(adjusted.sum(), weights.sum())

    repeated, repeated_reallocated = apply_pair_survival_multiplier(
        adjusted,
        ["QQQ", "SEMIS", "CASH"],
        "QQQ",
        "SEMIS",
        0.40,
    )
    assert np.allclose(repeated, adjusted)
    assert np.isclose(repeated_reallocated, 0.0)


def test_vix_term_structure_uses_the_economic_inversion_boundary() -> None:
    contango_ratio, contango = implied_volatility_term_structure(18.0, 21.0)
    backwardation_ratio, backwardation = implied_volatility_term_structure(28.0, 24.0)
    assert np.isclose(contango_ratio, 18.0 / 21.0)
    assert not contango
    assert np.isclose(backwardation_ratio, 28.0 / 24.0)
    assert backwardation


def test_vix_term_structure_rejects_invalid_levels() -> None:
    for spot, three_month in [(0.0, 20.0), (20.0, -1.0), (np.nan, 20.0)]:
        try:
            implied_volatility_term_structure(spot, three_month)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid implied-volatility levels must fail")


def test_vix_term_structure_can_veto_only_incremental_leverage() -> None:
    assert incremental_leverage_term_structure_gate(False, True)
    assert not incremental_leverage_term_structure_gate(True, True)
    assert incremental_leverage_term_structure_gate(True, False)


def test_vix_term_structure_can_cap_only_incremental_growth_floor() -> None:
    assert np.isclose(
        incremental_growth_floor_term_structure_cap(0.30, 0.20, False),
        0.30,
    )
    assert np.isclose(
        incremental_growth_floor_term_structure_cap(0.30, 0.20, True),
        0.20,
    )
    for cap in (-0.01, 0.31):
        try:
            incremental_growth_floor_term_structure_cap(0.30, cap, True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid backwardation floor cap must fail")


def test_conditional_vix_hedge_is_cash_funded_and_reversible() -> None:
    weights = np.array([0.55, 0.43, 0.0, 0.02])
    active = apply_conditional_cash_funded_hedge(
        weights,
        hedge_index=2,
        cash_index=3,
        account_share=0.02,
        active=True,
    )
    assert np.allclose(active, [0.55, 0.43, 0.02, 0.0])
    inactive = apply_conditional_cash_funded_hedge(
        active,
        hedge_index=2,
        cash_index=3,
        account_share=0.02,
        active=False,
    )
    assert np.allclose(inactive, weights)


def test_conditional_vix_hedge_can_use_a_small_negative_cash_balance() -> None:
    weights = np.array([0.60, 0.40, 0.0, 0.0])
    target = apply_conditional_cash_funded_hedge(
        weights,
        hedge_index=2,
        cash_index=3,
        account_share=0.02,
        active=True,
    )
    assert np.allclose(target, [0.60, 0.40, 0.02, -0.02])
    assert np.isclose(target.sum(), 1.0)


def test_vix_recovery_bridge_persists_only_until_risk_reentry() -> None:
    assert not update_vix_recovery_bridge(False, None, True, False)
    assert update_vix_recovery_bridge(False, True, False, False)
    assert update_vix_recovery_bridge(True, False, False, False)
    assert not update_vix_recovery_bridge(True, False, False, True)
    assert not update_vix_recovery_bridge(True, False, True, False)
    assert not update_vix_recovery_bridge(False, True, False, False, False)


def test_trend_forecast_and_stress_guard_have_expected_direction() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    returns = pd.DataFrame(
        {"SPX": np.full(260, -0.001), "CASH": np.full(260, 0.0001)}, index=dates
    )
    forecast, score = multi_horizon_trend_forecast(
        returns, [21, 63, 126, 252], [0.1, 0.2, 0.3, 0.4], 0.08
    )
    assert forecast[0] < 0.0 < forecast[1]
    assert score[0] < score[1]

    weights, stressed = apply_growth_stress_guard(
        np.array([0.6, 0.4]),
        returns,
        ["SPX", "CASH"],
        ["SPX"],
        1,
        200,
        20,
        0.25,
        0.25,
        "SPX",
    )
    assert stressed
    assert np.allclose(weights, [0.15, 0.85])


def test_equal_weight_core_blend_preserves_budget() -> None:
    optimized = np.array([0.4, 0.2, 0.4])
    blended = blend_equal_weight_core(optimized, cash_index=2, blend=0.5)
    assert np.allclose(blended, [0.45, 0.35, 0.2])
    assert np.isclose(blended.sum(), 1.0)


def test_joint_excess_trend_gate_requires_every_growth_sleeve_to_be_positive() -> None:
    returns = pd.DataFrame(
        {
            "QQQ": np.full(252, 0.0010),
            "SEMIS": np.full(252, 0.0015),
            "CASH": np.full(252, 0.0001),
        }
    )
    active, scores = joint_excess_trend_leverage_gate(
        returns,
        ["QQQ", "SEMIS"],
        "CASH",
    )
    assert active
    assert np.all(scores > 0.0)

    returns["SEMIS"] = -0.0005
    active, scores = joint_excess_trend_leverage_gate(
        returns,
        ["QQQ", "SEMIS"],
        "CASH",
    )
    assert not active
    assert scores[0] > 0.0 > scores[1]


def test_joint_excess_trend_gate_is_conservative_with_insufficient_history() -> None:
    returns = pd.DataFrame(
        {
            "QQQ": np.full(251, 0.0010),
            "SEMIS": np.full(251, 0.0015),
            "CASH": np.full(251, 0.0001),
        }
    )
    active, scores = joint_excess_trend_leverage_gate(
        returns,
        ["QQQ", "SEMIS"],
        "CASH",
    )
    assert not active
    assert np.isnan(scores).all()


def test_joint_excess_trend_gate_requires_fast_and_slow_trends() -> None:
    returns = pd.DataFrame(
        {
            "QQQ": np.r_[np.full(189, 0.0010), np.full(63, -0.0015)],
            "SEMIS": np.r_[np.full(189, 0.0015), np.full(63, -0.0015)],
            "CASH": np.full(252, 0.0001),
        }
    )
    active, scores = joint_excess_trend_leverage_gate(
        returns,
        ["QQQ", "SEMIS"],
        "CASH",
        [63, 252],
    )
    assert not active
    assert scores.shape == (4,)
    assert np.all(scores[:2] < 0.0)
    assert np.all(scores[2:] > 0.0)


def test_time_series_momentum_overlay_is_directional_and_self_financing() -> None:
    dates = pd.bdate_range("2020-01-01", periods=800)
    cycle = 0.0005 * np.sin(np.arange(800) / 7.0)
    returns = pd.DataFrame(
        {
            "UP": 0.0008 + cycle,
            "DOWN": -0.0006 + cycle,
            "CASH": np.full(800, 0.0001),
        },
        index=dates,
    )
    overlay, signals, target_volatility = time_series_momentum_overlay_weights(
        returns,
        ["UP", "DOWN", "CASH"],
        ["UP", "DOWN"],
        2,
        [21, 63, 252],
        756,
        0.10,
    )
    assert signals[0] > 0.0 > signals[1]
    assert overlay[0] > 0.0 > overlay[1]
    assert np.isclose(overlay.sum(), 0.0)
    assert np.isclose(target_volatility, 0.10)


def test_self_financing_overlay_respects_total_gross_limit() -> None:
    base = np.array([0.40, 0.40, 0.20])
    overlay = np.array([1.00, 1.00, -2.00])
    combined, applied_share, gross = apply_self_financing_overlay(
        base,
        overlay,
        2,
        0.20,
        1.10,
    )
    assert np.isclose(combined.sum(), 1.0)
    assert np.isclose(applied_share, 0.15)
    assert np.isclose(gross, 1.10)


def test_self_financing_overlay_can_be_restricted_to_risk_off() -> None:
    assert np.isclose(
        self_financing_overlay_share_for_mode(0.20, "risk_off", False),
        0.20,
    )
    assert np.isclose(
        self_financing_overlay_share_for_mode(0.20, "risk_off", True),
        0.0,
    )
    assert np.isclose(
        self_financing_overlay_share_for_mode(0.20, "always", True),
        0.20,
    )


def test_financing_cost_can_charge_short_borrow() -> None:
    weights = np.array([-0.20, 0.50, 0.70])
    cost = financing_spread_cost(
        weights,
        cash_index=2,
        annual_spread_bps=100.0,
        short_borrow_spread_bps=100.0,
    )
    assert np.isclose(cost, 0.20 * 0.01 / 252.0)


def test_strategic_core_and_leverage_use_negative_cash_funding() -> None:
    optimized = np.array([0.4, 0.2, 0.0, 0.0, 0.0, 0.4])
    assets = ["SPX", "BOND", "GOLD", "OIL", "USD", "CASH"]
    blended = blend_strategic_core(
        optimized,
        assets,
        cash_index=5,
        blend=0.5,
        core_weights={"SPX": 0.8, "BOND": 0.1, "GOLD": 0.1},
    )
    leveraged, gross = apply_risk_on_leverage(
        blended,
        np.diag([0.04, 0.01, 0.02, 0.04, 0.01, 0.0001]),
        cash_index=5,
        target_volatility=0.30,
        max_gross_leverage=2.0,
        enabled=True,
    )
    assert np.isclose(leveraged.sum(), 1.0)
    assert 1.0 < gross <= 2.0
    assert leveraged[5] < 0.0
    assert financing_spread_cost(leveraged, 5, 100.0) > 0.0


def test_named_growth_sleeves_can_be_fully_defunded() -> None:
    assets = ["QQQ", "SEMIS", "GOLD", "CASH"]
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    defensive = apply_asset_multiplier(
        weights, assets, ["QQQ", "SEMIS"], cash_index=3, multiplier=0.0
    )
    assert np.allclose(defensive, [0.0, 0.0, 0.2, 0.8])


def test_probability_allocation_preserves_budget_and_funds_leverage_with_cash() -> None:
    defensive = np.array([0.0, 0.0, 0.3, 0.7])
    growth_core = np.array([0.5, 0.5, 0.0, 0.0])

    reduced = compose_probability_allocation(defensive, growth_core, 3, 0.75)
    leveraged = compose_probability_allocation(defensive, growth_core, 3, 1.25)

    assert np.allclose(reduced, [0.375, 0.375, 0.075, 0.175])
    assert np.allclose(leveraged, [0.625, 0.625, 0.0, -0.25])
    assert np.isclose(reduced.sum(), 1.0)
    assert np.isclose(leveraged.sum(), 1.0)


def test_risk_off_growth_floor_preserves_a_small_growth_sleeve() -> None:
    defensive = np.array([0.0, 0.0, 0.3, 0.7])
    growth_core = np.array([0.5, 0.5, 0.0, 0.0])
    risk_off, active_share = apply_risk_off_growth_floor(
        defensive, growth_core, 3, 0.2, False
    )
    risk_on, inactive_share = apply_risk_off_growth_floor(
        defensive, growth_core, 3, 0.2, True
    )
    assert np.allclose(risk_off, [0.1, 0.1, 0.24, 0.56])
    assert np.isclose(active_share, 0.2)
    assert np.allclose(risk_on, defensive)
    assert inactive_share == 0.0


def test_risk_off_growth_floor_does_not_override_stress_guard() -> None:
    assert risk_off_growth_floor_is_active(False, False, True)
    assert not risk_off_growth_floor_is_active(True, False, True)
    assert not risk_off_growth_floor_is_active(False, True, True)
    assert risk_off_growth_floor_is_active(False, True, False)


def test_turning_point_cycle_distinguishes_all_four_states() -> None:
    cash = pd.Series(np.zeros(252))
    bull = pd.Series(np.full(252, 0.001))
    correction = pd.Series(
        np.concatenate([np.full(232, 0.001), np.full(20, -0.005)])
    )
    bear = pd.Series(np.full(252, -0.001))
    rebound = pd.Series(
        np.concatenate([np.full(232, -0.001), np.full(20, 0.01)])
    )

    assert turning_point_cycle(bull, cash, 20, 252)[0] == "bull"
    assert turning_point_cycle(correction, cash, 20, 252)[0] == "correction"
    assert turning_point_cycle(bear, cash, 20, 252)[0] == "bear"
    assert turning_point_cycle(rebound, cash, 20, 252)[0] == "rebound"


def test_turning_point_cycle_requires_only_past_window_observations() -> None:
    growth = pd.Series(np.full(251, -0.001))
    cash = pd.Series(np.zeros(251))
    state, fast_return, slow_return = turning_point_cycle(
        growth, cash, 20, 252
    )
    assert state == "insufficient"
    assert np.isnan(fast_return)
    assert np.isnan(slow_return)


def test_causal_realized_volatility_state_detects_recent_high_and_low() -> None:
    high_history = pd.Series(
        np.concatenate(
            [
                np.tile([0.001, -0.001], 390),
                np.tile([0.015, -0.015], 10),
            ]
        )
    )
    low_history = pd.Series(
        np.concatenate(
            [
                np.tile([0.015, -0.015], 390),
                np.tile([0.001, -0.001], 10),
            ]
        )
    )

    high_state, high_volatility = causal_realized_volatility_state(
        high_history
    )
    low_state, low_volatility = causal_realized_volatility_state(
        low_history
    )

    assert high_state == "high"
    assert low_state == "low"
    assert high_volatility > low_volatility


def test_downside_semivolatility_does_not_penalize_upside_variation() -> None:
    history = pd.Series(
        np.concatenate(
            [
                np.tile([0.003, -0.003], 390),
                np.tile([0.030, 0.001], 10),
            ]
        )
    )

    standard_state, _ = causal_realized_volatility_state(history)
    downside_state, downside_volatility = (
        causal_realized_volatility_state(
            history,
            estimator="downside_semivolatility",
        )
    )

    assert standard_state == "high"
    assert downside_state == "low"
    assert np.isclose(downside_volatility, 0.0)


def test_downside_variation_share_distinguishes_good_and_bad_variation() -> None:
    good_variation = pd.Series(np.tile([0.01, 0.04], 10))
    bad_variation = pd.Series(
        np.concatenate([np.full(19, 0.006), [-0.10]])
    )

    assert np.isclose(
        realized_downside_variation_share(good_variation),
        0.0,
    )
    assert realized_downside_variation_share(bad_variation) > 0.90
    assert np.isnan(
        realized_downside_variation_share(good_variation.iloc[:-1])
    )


def test_high_volatility_gate_neutralizes_relative_momentum_tilt() -> None:
    allowed_weight, allowed_gate = (
        volatility_regime_gated_relative_momentum_weight(
            0.80,
            0.50,
            "medium",
            ["insufficient", "low", "medium"],
        )
    )
    gated_weight, gated = volatility_regime_gated_relative_momentum_weight(
        0.80,
        0.50,
        "high",
        ["insufficient", "low", "medium"],
    )

    assert allowed_weight == 0.80
    assert not allowed_gate
    assert gated_weight == 0.50
    assert gated


def test_high_volatility_gate_accepts_risk_balanced_fallback_weight() -> None:
    gated_weight, gated = volatility_regime_gated_relative_momentum_weight(
        0.80,
        0.35,
        "high",
        ["insufficient", "low", "medium"],
    )

    assert gated_weight == 0.35
    assert gated


def test_minimum_growth_exposure_preserves_defensive_mix() -> None:
    weights = np.array([0.05, 0.05, 0.30, 0.60])
    core = np.array([0.50, 0.50, 0.0, 0.0])
    adjusted, exposure = ensure_minimum_growth_exposure(
        weights, core, cash_index=3, minimum_exposure=0.35
    )

    assert np.isclose(exposure, 0.35)
    assert np.allclose(adjusted[:2], [0.175, 0.175])
    assert np.isclose(adjusted[2] / adjusted[3], weights[2] / weights[3])
    assert np.isclose(adjusted.sum(), 1.0)


def test_minimum_growth_exposure_is_a_floor_not_a_cap() -> None:
    weights = np.array([0.30, 0.30, 0.10, 0.30])
    core = np.array([0.50, 0.50, 0.0, 0.0])
    adjusted, exposure = ensure_minimum_growth_exposure(
        weights, core, cash_index=3, minimum_exposure=0.35
    )
    assert np.allclose(adjusted, weights)
    assert np.isclose(exposure, 0.60)


def test_growth_group_floor_directs_only_increment_to_qqq() -> None:
    weights = np.array([0.10, 0.10, 0.20, 0.60])
    qqq_increment = np.array([1.0, 0.0, 0.0, 0.0])
    growth_group = np.array([True, True, False, False])

    adjusted, exposure = ensure_minimum_growth_group_exposure(
        weights,
        qqq_increment,
        growth_group,
        cash_index=3,
        minimum_exposure=0.40,
    )

    assert np.isclose(exposure, 0.40)
    assert np.allclose(adjusted[:2], [0.30, 0.10])
    assert np.isclose(adjusted[2] / adjusted[3], weights[2] / weights[3])
    assert np.isclose(adjusted.sum(), 1.0)


def test_realized_volatility_stress_uses_existing_threshold() -> None:
    calm = pd.Series(np.tile([0.001, -0.001], 20))
    volatile = pd.Series(np.tile([0.03, -0.03], 20))

    assert not realized_volatility_stress(calm, 20, 0.30)
    assert realized_volatility_stress(volatile, 20, 0.30)


def test_asymmetric_regime_state_exits_immediately_and_reenters_on_refit() -> None:
    assert update_asymmetric_regime_state(None, True, True)
    assert not update_asymmetric_regime_state(True, False, False)
    assert not update_asymmetric_regime_state(False, True, False)
    assert update_asymmetric_regime_state(False, True, True)
    assert update_asymmetric_regime_state(True, True, False)


def test_paper_signal_can_exclude_redundant_auxiliary_trend_filters() -> None:
    assert paper_regime_candidate(True, 0.10, False, True, -1.0, 0.0)
    assert not paper_regime_candidate(True, 0.10, True, True, 1.0, 0.0)
    assert not paper_regime_candidate(True, -0.01, False, False, 1.0, 0.0)


def test_inverse_volatility_weights_reduce_the_high_volatility_sleeve() -> None:
    weights = inverse_volatility_weights(np.diag([0.04, 0.16]))
    assert np.allclose(weights, [2.0 / 3.0, 1.0 / 3.0])
    assert np.isclose(weights.sum(), 1.0)


def test_realized_stress_covariance_reacts_fast_and_recovers_slowly() -> None:
    calm = np.tile([0.002, -0.002], 30)
    shock = np.tile([0.04, -0.04], 10)
    fast_shock = pd.DataFrame(
        {
            "QQQ": np.concatenate([calm, shock]),
            "SEMIS": np.concatenate([calm * 1.5, shock * 2.0]),
        }
    )
    _, shock_volatility = realized_stress_covariance(fast_shock, 20, 60)
    slow_memory = pd.DataFrame(
        {
            "QQQ": np.concatenate([shock, calm[:40]]),
            "SEMIS": np.concatenate([shock * 2.0, calm[:40] * 1.5]),
        }
    )
    _, memory_volatility = realized_stress_covariance(slow_memory, 20, 60)
    calm_only = pd.DataFrame({"QQQ": calm, "SEMIS": calm * 1.5})
    _, calm_volatility = realized_stress_covariance(calm_only, 20, 60)

    assert np.all(shock_volatility > calm_volatility)
    assert np.all(memory_volatility > calm_volatility)
    assert shock_volatility[1] > shock_volatility[0]
    assert memory_volatility[1] > memory_volatility[0]


def test_asset_aware_budget_reduces_smh_without_relevering() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    dates = pd.bdate_range("2025-01-01", periods=80)
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.06, -0.06], 40),
            "CASH": np.full(80, 0.0001),
        },
        index=dates,
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, core, volatility, unit_volatility, growth_cap = (
        apply_asset_aware_growth_risk_budget(
            original,
            returns,
            assets,
            ["QQQ", "SEMIS"],
            2,
            20,
            60,
            0.20,
            {"SEMIS": 0.50},
        )
    )
    assert core[0] > core[1]
    assert core[1] <= 0.50
    assert volatility[1] > volatility[0]
    assert growth_cap < original[:2].sum()
    assert adjusted[1] < original[1]
    assert np.isclose(adjusted[:2].sum(), growth_cap)
    assert unit_volatility > 0.20
    assert np.isclose(adjusted.sum(), 1.0)


def test_tail_only_asset_budget_preserves_a_mix_below_the_risk_cap() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.002, -0.002], 40),
            "SEMIS": np.tile([0.004, -0.004], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, core, _, unit_volatility, growth_cap = (
        apply_asset_aware_growth_risk_budget(
            original,
            returns,
            assets,
            ["QQQ", "SEMIS"],
            2,
            20,
            60,
            0.20,
            {"SEMIS": 0.50},
            activate_only_when_breached=True,
        )
    )
    assert np.allclose(adjusted, original)
    assert np.allclose(core, [0.5, 0.5])
    assert np.isclose(growth_cap, 0.8)
    assert unit_volatility < 0.20


def test_tail_only_asset_budget_activates_above_the_risk_cap() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.06, -0.06], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, core, _, _, growth_cap = apply_asset_aware_growth_risk_budget(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        60,
        0.20,
        {"SEMIS": 0.50},
        activate_only_when_breached=True,
    )
    assert core[0] > core[1]
    assert growth_cap < 0.8
    assert adjusted[1] < original[1]
    assert adjusted[2] > original[2]


def test_asset_budget_can_reallocate_risk_without_cutting_growth_total() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.06, -0.06], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, core, _, _, growth_cap = apply_asset_aware_growth_risk_budget(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        60,
        0.20,
        {"SEMIS": 0.50},
        activate_only_when_breached=True,
        cap_total_volatility=False,
    )
    assert core[0] > core[1]
    assert adjusted[0] > original[0]
    assert adjusted[1] < original[1]
    assert np.isclose(adjusted[:2].sum(), original[:2].sum())
    assert np.isclose(adjusted[2], original[2])
    assert np.isclose(growth_cap, 0.8)


def test_daily_risk_overlay_can_only_reduce_existing_growth_weights() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.04, -0.04], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    reduced, realized, multiplier = apply_daily_growth_risk_reduction(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        60,
        0.20,
    )
    assert realized > 0.20
    assert 0.0 < multiplier < 1.0
    assert np.all(reduced[:2] < original[:2])
    assert reduced[2] > original[2]
    assert np.isclose(reduced.sum(), 1.0)


def test_daily_risk_overlay_can_use_conservative_stress_correlation() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    early = np.tile([0.015, -0.015], 10)
    recent = np.tile([0.02, -0.02], 10)
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate([early, recent]),
            "SEMIS": np.concatenate([-early, recent]),
            "CASH": np.zeros(40),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    _, slow_realized, slow_multiplier = apply_daily_growth_risk_reduction(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        40,
        0.20,
        correlation_estimator="slow",
    )
    stressed, stress_realized, stress_multiplier = apply_daily_growth_risk_reduction(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        40,
        0.20,
        correlation_estimator="stress_max",
    )
    assert stress_realized > slow_realized
    assert stress_multiplier < slow_multiplier
    assert stressed[0] < original[0]
    assert stressed[1] < original[1]


def test_realized_risk_can_use_conservative_downside_correlation() -> None:
    rows: list[list[float]] = []
    for shock in np.linspace(0.005, 0.05, 20):
        rows.extend(
            [
                [0.02, -0.02],
                [-0.02, 0.02],
                [-shock, -1.2 * shock],
            ]
        )
    returns = pd.DataFrame(rows, columns=["QQQ", "SEMIS"])
    regular, _ = realized_stress_covariance(
        returns,
        20,
        60,
        correlation_estimator="stress_max",
    )
    downside, _ = realized_stress_covariance(
        returns,
        20,
        60,
        correlation_estimator="downside_stress_max",
    )
    assert downside[0, 1] > regular[0, 1]
    assert np.allclose(np.diag(downside), np.diag(regular))


def test_trend_cvar_overlay_reduces_growth_only_in_downtrend() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    quiet = np.tile([0.005, -0.005], 450)
    bear = np.tile([-0.04, 0.025], 100)
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate([quiet, bear]),
            "SEMIS": np.concatenate([quiet, bear]),
            "CASH": np.zeros(1100),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    reduced, cvar, multiplier, bear_regime = apply_trend_cvar_risk_reduction(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        200,
        1000,
        0.005,
        0.20,
    )
    assert bear_regime
    assert cvar > 0.0
    assert 0.0 < multiplier < 1.0
    assert np.all(reduced[:2] < original[:2])
    assert reduced[2] > original[2]
    assert np.isclose(reduced.sum(), 1.0)


def test_trend_cvar_overlay_does_not_cut_growth_in_uptrend() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    quiet = np.tile([0.005, -0.005], 450)
    bull = np.tile([0.04, -0.025], 100)
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate([quiet, bull]),
            "SEMIS": np.concatenate([quiet, bull]),
            "CASH": np.zeros(1100),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, cvar, multiplier, bear_regime = apply_trend_cvar_risk_reduction(
        original,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        200,
        1000,
        0.005,
        0.20,
    )
    assert not bear_regime
    assert np.isnan(cvar)
    assert multiplier == 1.0
    assert np.allclose(adjusted, original)


def test_jump_aware_stress_volatility_forgets_isolated_old_jump() -> None:
    quiet = np.tile([0.001, -0.001], 40).astype(float)
    quiet[35] = -0.12
    returns = pd.DataFrame({"QQQ": quiet, "SEMIS": quiet * 1.2})
    _, standard_volatility = realized_stress_covariance(
        returns, 20, 60, volatility_estimator="standard"
    )
    _, jump_aware_volatility = realized_stress_covariance(
        returns, 20, 60, volatility_estimator="jump_aware"
    )
    assert np.all(jump_aware_volatility < standard_volatility)


def test_jump_aware_stress_volatility_retains_persistent_variation() -> None:
    persistent = np.tile([0.02, -0.02], 40).astype(float)
    returns = pd.DataFrame({"QQQ": persistent, "SEMIS": persistent * 1.2})
    _, jump_aware_volatility = realized_stress_covariance(
        returns, 20, 60, volatility_estimator="jump_aware"
    )
    assert np.all(jump_aware_volatility > 0.20)


def test_daily_risk_target_restores_only_to_capped_scheduled_mix() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    quiet_returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.004, -0.004], 40),
            "SEMIS": np.tile([0.006, -0.006], 40),
            "CASH": np.zeros(80),
        }
    )
    current = np.array([0.20, 0.20, 0.60])
    desired = np.array([0.40, 0.40, 0.20])
    restored, realized, multiplier = apply_daily_growth_risk_target(
        current,
        desired,
        quiet_returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        60,
        0.20,
    )
    assert realized < 0.20
    assert np.isclose(multiplier, 1.0)
    assert np.allclose(restored, desired)


def test_daily_risk_target_cannot_exceed_current_growth_plus_cash() -> None:
    assets = ["QQQ", "SEMIS", "GOLD", "CASH"]
    quiet_returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.004, -0.004], 40),
            "SEMIS": np.tile([0.006, -0.006], 40),
            "GOLD": np.zeros(80),
            "CASH": np.zeros(80),
        }
    )
    current = np.array([0.25, 0.25, 0.51, -0.01])
    desired = np.array([0.40, 0.40, 0.20, 0.00])
    restored, _, _ = apply_daily_growth_risk_target(
        current,
        desired,
        quiet_returns,
        assets,
        ["QQQ", "SEMIS"],
        3,
        20,
        60,
        0.20,
    )
    assert np.isclose(restored[:2].sum(), 0.49)
    assert np.isclose(restored[3], 0.0)
    assert np.isclose(restored.sum(), 1.0)


def test_daily_risk_reallocation_preserves_more_lower_volatility_asset() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.01, -0.01], 40),
            "SEMIS": np.tile([0.04, -0.04], 40),
            "CASH": np.zeros(80),
        }
    )
    current = np.array([0.25, 0.25, 0.50])
    desired = np.array([0.40, 0.40, 0.20])
    adjusted, realized, multiplier = apply_daily_growth_risk_reallocation_target(
        current,
        desired,
        returns,
        assets,
        ["QQQ", "SEMIS"],
        2,
        20,
        60,
        0.20,
        {"SEMIS": 0.50},
    )
    assert realized > 0.20
    assert 0.0 < multiplier < 1.0
    assert adjusted[0] > adjusted[1]
    assert adjusted[1] < desired[1]
    assert np.isclose(adjusted.sum(), 1.0)


def test_account_stress_cap_scales_growth_and_gold_to_joint_budget() -> None:
    assets = ["QQQ", "SEMIS", "GOLD", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.05, -0.05], 40),
            "GOLD": np.tile([-0.01, 0.01], 40),
            "CASH": np.zeros(80),
        }
    )
    weights = np.array([0.45, 0.25, 0.20, 0.10])
    adjusted, realized, multiplier = apply_account_stress_volatility_cap(
        weights,
        returns,
        assets,
        ["QQQ", "SEMIS", "GOLD"],
        3,
        20,
        60,
        0.20,
    )
    assert realized > 0.20
    assert 0.0 < multiplier < 1.0
    assert np.allclose(adjusted[:3], weights[:3] * multiplier)
    assert adjusted[3] > weights[3]
    covariance, _ = realized_stress_covariance(
        returns[["QQQ", "SEMIS", "GOLD"]], 20, 60
    )
    assert np.isclose(
        np.sqrt(adjusted[:3] @ covariance @ adjusted[:3]),
        0.20,
    )


def test_controlled_daily_risk_reduction_preserves_anchor_first() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.015, -0.015], 40),
            "SEMIS": np.tile([0.045, -0.045], 40),
            "CASH": np.zeros(80),
        },
        index=index,
    )
    original = np.array([0.40, 0.40, 0.20])
    adjusted, realized, multiplier = apply_daily_controlled_asset_risk_reduction(
        original,
        returns,
        assets,
        "QQQ",
        "SEMIS",
        2,
        20,
        60,
        0.20,
    )
    assert realized > 0.20
    assert np.isclose(adjusted[0], original[0])
    assert adjusted[1] < original[1]
    assert adjusted[2] > original[2]
    assert 0.0 < multiplier < 1.0
    covariance, _ = realized_stress_covariance(
        returns[["QQQ", "SEMIS"]],
        20,
        60,
        correlation_estimator="stress_max",
    )
    assert np.isclose(
        np.sqrt(adjusted[:2] @ covariance @ adjusted[:2]),
        0.20,
    )


def test_controlled_daily_risk_reduction_can_reduce_anchor_last() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.06, -0.06], 40),
            "SEMIS": np.tile([0.09, -0.09], 40),
            "CASH": np.zeros(80),
        },
        index=index,
    )
    original = np.array([0.40, 0.20, 0.40])
    adjusted, _, _ = apply_daily_controlled_asset_risk_reduction(
        original,
        returns,
        assets,
        "QQQ",
        "SEMIS",
        2,
        20,
        60,
        0.20,
    )
    assert adjusted[1] < 1e-10
    assert adjusted[0] < original[0]
    assert adjusted[2] > original[2]


def test_relative_asset_risk_cap_preserves_anchor_and_moves_excess_to_cash() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.02, -0.02], 40),
            "SEMIS": np.tile([0.06, -0.06], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, volatility, multiplier = apply_relative_asset_risk_cap(
        original,
        returns,
        assets,
        "QQQ",
        "SEMIS",
        2,
        20,
        60,
    )
    assert np.isclose(adjusted[0], original[0])
    assert adjusted[1] < original[1]
    assert adjusted[2] > original[2]
    assert volatility[1] > volatility[0]
    assert 0.0 < multiplier < 1.0
    assert np.isclose(adjusted[0] * volatility[0], adjusted[1] * volatility[1])
    assert np.isclose(adjusted.sum(), 1.0)


def _monthly_factor_returns(
    beta: float,
    residual_by_month: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    month_ends = pd.date_range("2020-01-31", periods=len(residual_by_month), freq="ME")
    anchor_monthly = np.tile([0.04, -0.02, 0.03, -0.01], 10)[
        : len(residual_by_month)
    ]
    controlled_monthly = beta * anchor_monthly + residual_by_month
    daily_dates: list[pd.Timestamp] = []
    anchor_daily: list[float] = []
    controlled_daily: list[float] = []
    for month_end, anchor_return, controlled_return in zip(
        month_ends, anchor_monthly, controlled_monthly, strict=True
    ):
        daily_dates.append(month_end)
        anchor_daily.append(float(anchor_return))
        controlled_daily.append(float(controlled_return))
    next_month = month_ends[-1] + pd.offsets.MonthEnd(1)
    daily_dates.append(next_month - pd.Timedelta(days=10))
    anchor_daily.append(0.0)
    controlled_daily.append(0.0)
    return (
        pd.Series(anchor_daily, index=daily_dates),
        pd.Series(controlled_daily, index=daily_dates),
    )


def test_negative_residual_momentum_shrinks_high_beta_sleeve() -> None:
    residual = np.zeros(36)
    residual[-11:] = -0.01
    anchor, controlled = _monthly_factor_returns(2.0, residual)
    weight, score, beta = residual_momentum_beta_pair_weight(
        anchor, controlled, 36, 12, 1, 0.50
    )
    assert score < 0.0
    assert np.isclose(beta, 2.0, atol=0.02)
    assert np.isclose(weight, 1.0 / (1.0 + beta))


def test_positive_residual_momentum_preserves_strategic_sleeve() -> None:
    residual = np.zeros(36)
    residual[-11:] = 0.01
    anchor, controlled = _monthly_factor_returns(2.0, residual)
    weight, score, beta = residual_momentum_beta_pair_weight(
        anchor, controlled, 36, 12, 1, 0.50
    )
    assert score > 0.0
    assert np.isclose(beta, 2.0, atol=0.02)
    assert np.isclose(weight, 0.50)


def test_residual_momentum_ignores_incomplete_current_month() -> None:
    residual = np.zeros(36)
    residual[-11:] = -0.01
    anchor, controlled = _monthly_factor_returns(2.0, residual)
    base = residual_momentum_beta_pair_weight(
        anchor, controlled, 36, 12, 1, 0.50
    )
    controlled.iloc[-1] = 10.0
    shocked = residual_momentum_beta_pair_weight(
        anchor, controlled, 36, 12, 1, 0.50
    )
    assert np.allclose(base, shocked)


def test_relative_asset_risk_cap_can_ignore_a_low_risk_entry() -> None:
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.002, -0.002], 40),
            "SEMIS": np.tile([0.006, -0.006], 40),
            "CASH": np.zeros(80),
        }
    )
    original = np.array([0.4, 0.4, 0.2])
    adjusted, _, multiplier = apply_relative_asset_risk_cap(
        original,
        returns,
        assets,
        "QQQ",
        "SEMIS",
        2,
        20,
        60,
        activation_volatility_cap=0.20,
    )
    assert np.allclose(adjusted, original)
    assert np.isclose(multiplier, 1.0)


def test_zero_entry_method_uses_cash_only_after_relative_weakness() -> None:
    assert zero_entry_allocation_method(-0.01) == "relative_asset_cap"
    assert zero_entry_allocation_method(0.0) == "total_volatility"
    assert zero_entry_allocation_method(0.01) == "total_volatility"
    assert zero_entry_allocation_method(float("nan")) == "total_volatility"


def test_daily_bipower_proxy_downweights_an_isolated_jump() -> None:
    returns = pd.Series(np.full(100, 0.001))
    returns.iloc[-10] = 0.20
    standard = float(returns.iloc[-20:].std(ddof=1) * np.sqrt(252))
    bipower = annualized_bipower_volatility(returns, 20)
    assert 0.0 < bipower < standard


def test_downside_semivolatility_ignores_an_upside_jump() -> None:
    calm = pd.Series(np.tile([0.01, -0.01], 30))
    upside = calm.copy()
    upside.iloc[-2] = 0.15
    standard = float(upside.std(ddof=1) * np.sqrt(252))
    downside = annualized_downside_semivolatility(upside, 60)
    calm_downside = annualized_downside_semivolatility(calm, 60)
    assert np.isclose(downside, calm_downside)
    assert downside < standard


def test_downside_semivolatility_reacts_to_a_downside_jump() -> None:
    calm = pd.Series(np.tile([0.01, -0.01], 30))
    downside_jump = calm.copy()
    downside_jump.iloc[-2] = -0.15
    calm_downside = annualized_downside_semivolatility(calm, 60)
    stressed = annualized_downside_semivolatility(downside_jump, 60)
    assert stressed > calm_downside * 2.0


def test_downside_stress_covariance_uses_the_same_windows() -> None:
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.01, -0.01], 40),
            "SEMIS": np.tile([0.02, -0.02], 40),
        }
    )
    covariance, volatility = realized_stress_covariance(
        returns,
        20,
        60,
        volatility_estimator="downside",
    )
    assert covariance.shape == (2, 2)
    assert np.all(volatility > 0.0)
    assert volatility[1] > volatility[0]


def test_semiskew_volatility_matches_symmetric_semivolatility() -> None:
    symmetric = pd.Series(np.tile([0.01, -0.01], 30))
    downside = annualized_downside_semivolatility(symmetric, 60)
    semiskew = annualized_semiskew_effective_volatility(symmetric, 60)
    assert np.isclose(semiskew, downside)


def test_semiskew_volatility_rewards_upside_and_penalizes_downside_jumps() -> None:
    calm = pd.Series(np.tile([0.01, -0.01], 30))
    upside = calm.copy()
    upside.iloc[-2] = 0.15
    downside = calm.copy()
    downside.iloc[-2] = -0.15
    calm_risk = annualized_semiskew_effective_volatility(calm, 60)
    upside_risk = annualized_semiskew_effective_volatility(upside, 60)
    downside_risk = annualized_semiskew_effective_volatility(downside, 60)
    assert upside_risk < calm_risk
    assert downside_risk > calm_risk


def test_smh_high_volatility_budget_only_changes_the_tail_state() -> None:
    base = np.array([0.5, 0.5])
    covariance = np.diag([0.04, 0.16])
    calm = pd.Series(
        np.concatenate(
            [np.tile([0.02, -0.02], 470), np.tile([0.001, -0.001], 80)]
        )
    )
    calm_weights, calm_state, calm_vol, calm_threshold = high_volatility_core_weights(
        covariance, base, calm, 60, 1008, 0.90
    )
    assert not calm_state and calm_vol < calm_threshold
    assert np.allclose(calm_weights, base)

    stressed = pd.Series(
        np.concatenate(
            [np.tile([0.001, -0.001], 520), np.tile([0.04, -0.04], 30)]
        )
    )
    stressed_weights, stressed_state, stressed_vol, stressed_threshold = (
        high_volatility_core_weights(covariance, base, stressed, 60, 1008, 0.90)
    )
    assert stressed_state and stressed_vol >= stressed_threshold
    assert np.allclose(stressed_weights, [2.0 / 3.0, 1.0 / 3.0])


def test_relative_momentum_tilts_without_using_the_skipped_month() -> None:
    anchor = pd.Series(np.zeros(320))
    positive_tilt = pd.Series(np.full(320, 0.001))
    tilt_weight, score = relative_momentum_pair_weight(
        anchor,
        positive_tilt,
        [126, 252],
        [0.5, 0.5],
        21,
        0.5,
        0.2,
    )
    assert score > 0.0 and 0.5 < tilt_weight <= 0.7

    changed_skip_month = positive_tilt.copy()
    changed_skip_month.iloc[-21:] = -0.10
    changed_weight, changed_score = relative_momentum_pair_weight(
        anchor,
        changed_skip_month,
        [126, 252],
        [0.5, 0.5],
        21,
        0.5,
        0.2,
    )
    assert np.isclose(changed_weight, tilt_weight)
    assert np.isclose(changed_score, score)


def test_relative_momentum_volatility_management_caps_active_risk() -> None:
    anchor = pd.Series(np.zeros(126))
    volatile_tilt = pd.Series(np.tile([0.04, -0.04], 63))
    adjusted, spread_volatility, multiplier = (
        volatility_managed_relative_momentum_weight(
            0.9,
            0.5,
            anchor,
            volatile_tilt,
            126,
            0.12,
        )
    )
    assert spread_volatility > 0.12
    assert 0.5 < adjusted < 0.9
    assert 0.0 < multiplier < 1.0
    assert np.isclose((adjusted - 0.5) * spread_volatility, 0.12)


def test_relative_momentum_volatility_management_never_amplifies_tilt() -> None:
    anchor = pd.Series(np.zeros(126))
    calm_tilt = pd.Series(np.tile([0.005, -0.005], 63))
    adjusted, spread_volatility, multiplier = (
        volatility_managed_relative_momentum_weight(
            0.9,
            0.5,
            anchor,
            calm_tilt,
            126,
            0.12,
        )
    )
    assert spread_volatility < 0.12
    assert adjusted == 0.9
    assert multiplier == 1.0


def test_relative_momentum_spread_is_directional_and_self_financing() -> None:
    anchor = pd.Series(np.tile([0.01, -0.01], 63))
    tilt = anchor + 0.001
    assets = ["QQQ", "SEMIS", "CASH"]
    overlay, spread_volatility, active_risk = (
        relative_momentum_spread_overlay_weights(
            anchor,
            tilt,
            assets,
            "QQQ",
            "SEMIS",
            2,
            0.50,
            126,
            0.12,
        )
    )
    assert overlay[1] > 0.0 > overlay[0]
    assert np.isclose(overlay.sum(), 0.0)
    assert spread_volatility > 0.0
    assert np.isclose(active_risk, 0.06)


def test_relative_momentum_pair_overlay_preserves_total_exposure() -> None:
    original = np.array([0.4, 0.35, 0.1, 0.15])
    adjusted, pair_total = reallocate_pair_weights(
        original,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        "QQQ",
        "SEMIS",
        0.7,
    )
    assert np.isclose(pair_total, 0.75)
    assert np.allclose(adjusted, [0.225, 0.525, 0.1, 0.15])
    assert np.isclose(adjusted.sum(), original.sum())


def test_pair_concentration_cap_reallocates_excess_to_anchor() -> None:
    original = np.array([0.20, 0.65, 0.10, 0.05])
    adjusted, reallocated = cap_pair_asset_weight(
        original,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        "QQQ",
        "SEMIS",
        0.50,
    )
    assert np.isclose(reallocated, 0.15)
    assert np.allclose(adjusted, [0.35, 0.50, 0.10, 0.05])
    assert np.isclose(adjusted.sum(), original.sum())


def test_defensive_trend_selects_positive_assets_by_inverse_volatility() -> None:
    assets = ["BOND", "GOLD", "OIL", "USD", "CASH"]
    returns = pd.DataFrame(
        {
            "BOND": np.tile([0.01, -0.01], 40),
            "GOLD": np.tile([0.02, -0.02], 40),
            "OIL": np.tile([0.03, -0.03], 40),
            "USD": np.tile([0.005, -0.005], 40),
            "CASH": np.zeros(80),
        }
    )
    target, selected = long_only_trend_defensive_weights(
        returns,
        assets,
        ["BOND", "GOLD", "OIL", "USD"],
        4,
        np.array([1.0, 1.0, -1.0, -1.0, 0.0]),
        60,
    )
    assert selected == 2
    assert target[0] > target[1] > 0.0
    assert target[2] == target[3] == target[4] == 0.0
    assert np.isclose(target.sum(), 1.0)

    cash_only, selected = long_only_trend_defensive_weights(
        returns,
        assets,
        ["BOND", "GOLD", "OIL", "USD"],
        4,
        np.array([-1.0, -1.0, -1.0, -1.0, 0.0]),
        60,
    )
    assert selected == 0
    assert np.allclose(cash_only, [0.0, 0.0, 0.0, 0.0, 1.0])


def test_defensive_trend_fast_filter_only_vetoes_configured_assets() -> None:
    assets = ["QQQ", "GOLD", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.r_[np.full(59, 0.002), -0.20],
            "GOLD": np.r_[np.full(59, 0.002), -0.20],
            "CASH": np.zeros(60),
        }
    )

    target, selected = long_only_trend_defensive_weights(
        returns,
        assets,
        ["QQQ", "GOLD"],
        2,
        np.array([1.0, 1.0, 0.0]),
        20,
        fast_trend_assets=["QQQ"],
        fast_trend_days=21,
    )

    assert selected == 1
    assert np.allclose(target, [0.0, 1.0, 0.0])


def test_drawdown_conditioned_hmm_veto_is_asymmetric() -> None:
    common = {
        "leverage_enabled": True,
        "trend_stress": False,
        "trend_score": 0.2,
        "activation_score": 0.0,
        "hmm_guard_drawdown": -0.10,
    }
    assert drawdown_conditioned_regime_candidate(
        growth_excess_return=-0.01,
        drawdown=-0.05,
        **common,
    )
    assert not drawdown_conditioned_regime_candidate(
        growth_excess_return=-0.01,
        drawdown=-0.10,
        **common,
    )
    assert drawdown_conditioned_regime_candidate(
        growth_excess_return=0.01,
        drawdown=-0.10,
        **common,
    )


def test_smh_satellite_uses_trend_volatility_and_account_cap() -> None:
    uptrend = pd.Series(np.linspace(0.0001, 0.003, 260))
    downtrend = -uptrend
    exposure, active, volatility = trend_volatility_exposure(
        uptrend, 200, 60, 0.20, 1.0
    )
    down_exposure, down_active, _ = trend_volatility_exposure(
        downtrend, 200, 60, 0.20, 1.0
    )
    assert active and volatility > 0.0
    assert 0.0 < exposure <= 1.0
    assert not down_active and down_exposure == 0.0

    combined = blend_satellite_allocation(
        np.array([0.5, 0.5, 0.0]),
        asset_index=1,
        cash_index=2,
        satellite_share=0.2,
        satellite_exposure=0.5,
        total_asset_cap=0.45,
    )
    assert np.allclose(combined, [0.4, 0.45, 0.15])
    assert np.isclose(combined.sum(), 1.0)


def test_smh_bridge_only_activates_before_the_base_strategy_reenters() -> None:
    assert np.isclose(satellite_share_for_mode(0.2, "always", False, True), 0.2)
    assert np.isclose(
        satellite_share_for_mode(0.2, "risk_off_bridge", True, False), 0.2
    )
    assert satellite_share_for_mode(0.2, "risk_off_bridge", False, False) == 0.0
    assert satellite_share_for_mode(0.2, "risk_off_bridge", True, True) == 0.0
