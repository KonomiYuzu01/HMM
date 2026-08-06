import numpy as np
import pandas as pd

from regime_strategy.survival import (
    SurvivalOverlayState,
    apply_survival_governor,
    feature_novelty_percentile,
    multi_horizon_stress_covariance,
    novelty_risk_multiplier,
)


def test_multi_horizon_covariance_retains_a_longer_horizon_shock() -> None:
    shock = np.tile([0.04, -0.04], 33)
    calm = np.r_[np.tile([0.002, -0.002], 30), 0.002]
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate([shock, calm]),
            "GOLD": np.concatenate([shock * 0.8, calm * -0.5]),
        }
    )
    _, volatility, correlation = multi_horizon_stress_covariance(
        returns,
        [20, 60, 126],
    )

    recent_volatility = (
        returns.iloc[-20:].std(ddof=1).to_numpy() * np.sqrt(252.0)
    )
    assert np.all(volatility > recent_volatility)
    assert correlation[0, 1] > returns.iloc[-20:].corr().iloc[0, 1]


def test_feature_novelty_flags_an_observation_outside_the_training_cloud() -> None:
    rng = np.random.default_rng(7)
    history = pd.DataFrame(
        rng.normal(size=(300, 4)),
        columns=["a", "b", "c", "d"],
    )
    ordinary = feature_novelty_percentile(
        history,
        pd.Series([0.1, -0.2, 0.2, 0.0], index=history.columns),
        252,
    )
    outlier = feature_novelty_percentile(
        history,
        pd.Series([6.0, 6.0, -6.0, 6.0], index=history.columns),
        252,
    )

    assert ordinary < 0.95
    assert outlier > 0.99
    assert novelty_risk_multiplier(0.90, 0.95, 0.75) == 1.0
    assert np.isclose(novelty_risk_multiplier(1.0, 0.95, 0.75), 0.75)


def test_survival_governor_caps_total_risk_and_concentration_to_cash() -> None:
    rng = np.random.default_rng(42)
    common = rng.normal(0.0, 0.02, 300)
    returns = pd.DataFrame(
        {
            "QQQ": common + rng.normal(0.0, 0.01, 300),
            "SEMIS": 2.0 * common + rng.normal(0.0, 0.02, 300),
            "GOLD": rng.normal(0.0, 0.018, 300),
        }
    )
    original = np.array([0.05, 0.85, 0.20, -0.10])
    result = apply_survival_governor(
        original,
        returns,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        ["QQQ", "SEMIS", "GOLD"],
        cash_index=3,
        horizons=[20, 60, 126],
        target_volatility=0.20,
        maximum_risk_shares={"QQQ": 0.50, "SEMIS": 0.50, "GOLD": 0.30},
    )

    assert result.active
    assert result.post_volatility <= 0.20001
    assert result.max_risk_share_after <= 0.5001
    assert result.weights[1] < original[1]
    assert result.weights[3] > original[3]
    assert np.isclose(result.weights.sum(), 1.0)


def test_survival_novelty_brake_removes_incremental_leverage() -> None:
    returns = pd.DataFrame(
        {
            "QQQ": np.tile([0.01, -0.01], 150),
            "GOLD": np.tile([-0.005, 0.005], 150),
        }
    )
    original = np.array([0.70, 0.40, -0.10])
    result = apply_survival_governor(
        original,
        returns,
        ["QQQ", "GOLD", "CASH"],
        ["QQQ", "GOLD"],
        cash_index=2,
        horizons=[20, 60, 126],
        target_volatility=1.0,
        novelty_percentile=0.975,
        novelty_activation_percentile=0.95,
        minimum_novelty_multiplier=0.75,
        maximum_gross_when_novel=1.0,
    )

    assert result.gross_before > 1.0
    assert result.gross_after <= 1.000001
    assert result.weights[2] >= -1e-12
    assert result.novelty_multiplier < 1.0


def test_survival_covariance_shift_trigger_sleeps_in_calm_conditions() -> None:
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(300, 2)),
        columns=["QQQ", "GOLD"],
    )
    original = np.array([0.70, 0.30, 0.0])
    result = apply_survival_governor(
        original,
        returns,
        ["QQQ", "GOLD", "CASH"],
        ["QQQ", "GOLD"],
        cash_index=2,
        horizons=[20, 60, 126],
        target_volatility=0.05,
        activation_stress_ratio=np.sqrt(2.0),
        long_horizon_days=252,
    )

    assert not result.triggered
    assert not result.active
    assert np.allclose(result.weights, original)


def test_survival_covariance_shift_trigger_activates_after_recent_shock() -> None:
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0, 0.005, size=(280, 2))
    shock = rng.normal(0.0, 0.04, size=(20, 2))
    returns = pd.DataFrame(
        np.vstack([calm, shock]),
        columns=["QQQ", "GOLD"],
    )
    original = np.array([0.70, 0.30, 0.0])
    result = apply_survival_governor(
        original,
        returns,
        ["QQQ", "GOLD", "CASH"],
        ["QQQ", "GOLD"],
        cash_index=2,
        horizons=[20, 60, 126],
        target_volatility=0.15,
        activation_stress_ratio=np.sqrt(2.0),
        long_horizon_days=252,
    )

    assert result.stress_volatility_ratio > np.sqrt(2.0)
    assert result.triggered
    assert result.active
    assert result.post_volatility <= 0.15001


def test_survival_overlay_state_restores_the_separate_alpha_target() -> None:
    state = SurvivalOverlayState()
    alpha_target = np.array([0.70, 0.30, 0.0])
    reduced_target = np.array([0.35, 0.15, 0.50])

    state.begin_alpha_target(alpha_target)
    stressed_input, recovering = state.target_input(reduced_target)
    assert recovering
    assert np.allclose(stressed_input, alpha_target)

    state.observe_result(stressed_input, survival_active=True)
    recovery_input, recovering = state.target_input(reduced_target)
    assert recovering
    assert np.allclose(recovery_input, alpha_target)

    state.observe_result(recovery_input, survival_active=False)
    ordinary_input, recovering = state.target_input(reduced_target)
    assert not recovering
    assert np.allclose(ordinary_input, reduced_target)
