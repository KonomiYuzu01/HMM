import numpy as np
import pandas as pd

from regime_strategy.backtest import (
    RegimeBacktester,
    adaptive_hmm_refit_policy,
    adaptive_hmm_refit_stress,
    should_update_hmm_templates,
)


def test_template_update_schedule_preserves_default_and_supports_refit_only() -> None:
    assert should_update_hmm_templates(False, {})
    assert should_update_hmm_templates(True, {"template_update_on_refit_only": True})
    assert not should_update_hmm_templates(
        False, {"template_update_on_refit_only": True}
    )


def test_adaptive_hmm_refit_stress_uses_only_prior_prices() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    calm = np.full(99, 0.001)
    returns = np.concatenate([[0.0], calm])
    prices = pd.DataFrame(
        {
            "QQQ": 100.0 * np.exp(np.cumsum(returns)),
            "SEMIS": 100.0 * np.exp(np.cumsum(returns)),
        },
        index=index,
    )
    date = index[-1]
    baseline = adaptive_hmm_refit_stress(
        prices, date, ["QQQ", "SEMIS"], 21, 63
    )
    changed = prices.copy()
    changed.loc[date, "SEMIS"] *= 2.0
    revised = adaptive_hmm_refit_stress(
        changed, date, ["QQQ", "SEMIS"], 21, 63
    )
    assert baseline == revised

    shocked = prices.copy()
    shocked.loc[index[-10] : index[-2], "SEMIS"] *= np.exp(
        np.linspace(0.0, 0.5, 9)
    )
    assert adaptive_hmm_refit_stress(
        shocked, date, ["QQQ", "SEMIS"], 21, 63
    )


def test_transition_refit_fires_once_on_each_state_change() -> None:
    config = {"mode": "transition", "calm_refit_days": 63}
    assert adaptive_hmm_refit_policy(False, None, config) == (63, False)
    assert adaptive_hmm_refit_policy(True, False, config) == (63, True)
    assert adaptive_hmm_refit_policy(True, True, config) == (63, False)
    assert adaptive_hmm_refit_policy(False, True, config) == (63, True)


def test_turning_point_reentry_can_be_limited_to_one_trade_per_rebound() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.01)]
            ),
            "SEMIS": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.01)]
            ),
            "CASH": np.zeros(252),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 20,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": True,
            "state_account_shares": {"rebound": 0.35},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
        }
    }
    weights = np.array([0.05, 0.05, 0.90])

    first_target, first_diagnostics = (
        backtester._turning_point_reentry_target(
            weights, returns, assets, 2, config
        )
    )
    backtester.turning_point_reentry_episode_fired = True
    second_target, second_diagnostics = (
        backtester._turning_point_reentry_target(
            weights, returns, assets, 2, config
        )
    )

    assert np.isclose(first_target[:2].sum(), 0.35)
    assert first_diagnostics["turning_point_reentry_active"] == 1
    assert np.allclose(second_target, weights)
    assert second_diagnostics["turning_point_reentry_active"] == 0


def test_turning_point_reentry_cannot_override_an_inactive_base_floor() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.01)]
            ),
            "SEMIS": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.01)]
            ),
            "CASH": np.zeros(252),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 20,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": False,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"rebound": 0.40},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
        }
    }

    blocked_target, blocked_diagnostics = (
        backtester._turning_point_reentry_target(
            np.array([0.05, 0.05, 0.90]),
            returns,
            assets,
            2,
            config,
        )
    )
    allowed_target, allowed_diagnostics = (
        backtester._turning_point_reentry_target(
            np.array([0.10, 0.10, 0.80]),
            returns,
            assets,
            2,
            config,
        )
    )

    assert np.allclose(blocked_target, [0.05, 0.05, 0.90])
    assert blocked_diagnostics["turning_point_base_floor_gate_pass"] == 0
    assert blocked_diagnostics["turning_point_reentry_active"] == 0
    assert np.isclose(allowed_target[:2].sum(), 0.40)
    assert allowed_diagnostics["turning_point_base_floor_gate_pass"] == 1
    assert allowed_diagnostics["turning_point_reentry_active"] == 1


def test_turning_point_can_separate_signal_and_incremental_floor_assets() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.full(252, 0.001),
            "SEMIS": np.concatenate(
                [np.full(232, -0.003), np.full(20, 0.01)]
            ),
            "CASH": np.zeros(252),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 20,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"bull": 0.40, "rebound": 0.40},
            "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            "signal_core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "growth_exposure_assets": ["QQQ", "SEMIS"],
        }
    }

    target, diagnostics = backtester._turning_point_reentry_target(
        np.array([0.10, 0.10, 0.80]),
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_state"] == 3
    assert diagnostics["turning_point_reentry_active"] == 1
    assert np.allclose(target, [0.30, 0.10, 0.60])


def test_turning_point_reentry_can_veto_high_volatility_rebound() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.full(757, -0.003),
            np.tile([0.015, -0.005], 31),
            np.array([0.015]),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": False,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"rebound": 0.40},
            "volatility_conditioning": {
                "enabled": True,
                "state_account_shares": {
                    "rebound": {
                        "low": 0.40,
                        "medium": 0.0,
                        "high": 0.0,
                    }
                },
            },
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
        }
    }
    weights = np.array([0.10, 0.10, 0.80])

    target, diagnostics = backtester._turning_point_reentry_target(
        weights,
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_state"] == 3
    assert diagnostics["turning_point_volatility_state"] == 2
    assert diagnostics["turning_point_reentry_active"] == 0
    assert np.allclose(target, weights)


def test_turning_point_reentry_can_veto_high_volatility_bear() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.full(757, -0.0003),
            np.tile([0.005, -0.015], 32),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": False,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"bear": 0.40},
            "volatility_conditioning": {
                "enabled": True,
                "state_override_maximum_slow_excess_return": {
                    "bear": -0.20,
                },
                "state_account_shares": {
                    "bear": {
                        "low": 0.40,
                        "medium": 0.40,
                        "high": 0.0,
                    }
                },
            },
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
        }
    }
    weights = np.array([0.10, 0.10, 0.80])

    target, diagnostics = backtester._turning_point_reentry_target(
        weights,
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_state"] == 2
    assert diagnostics["turning_point_volatility_state"] == 2
    assert diagnostics["turning_point_volatility_override_gate_pass"] == 1
    assert diagnostics["turning_point_reentry_active"] == 0
    assert np.allclose(target, weights)


def test_turning_point_reentry_keeps_floor_in_shallow_high_volatility_bear() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.full(757, -0.0001),
            np.full(43, -0.0001),
            np.tile([0.02, -0.02], 10),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": False,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"bear": 0.40},
            "volatility_conditioning": {
                "enabled": True,
                "state_override_maximum_slow_excess_return": {
                    "bear": -0.20,
                },
                "state_account_shares": {
                    "bear": {
                        "low": 0.40,
                        "medium": 0.40,
                        "high": 0.0,
                    }
                },
            },
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
        }
    }
    weights = np.array([0.10, 0.10, 0.80])

    target, diagnostics = backtester._turning_point_reentry_target(
        weights,
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_state"] == 2
    assert diagnostics["turning_point_volatility_state"] == 2
    assert diagnostics["turning_point_volatility_override_gate_pass"] == 0
    assert diagnostics["turning_point_reentry_active"] == 1
    assert np.isclose(target[:2].sum(), 0.40)


def test_turning_point_zero_entry_bridge_starts_with_qqq_only() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.full(232, -0.001),
            np.full(20, 0.002),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "activate_only_when_risk_off": True,
            "one_shot_per_rebound": False,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {"rebound": 0.40},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {
                    "bull": 0.20,
                    "rebound": 0.20,
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    weights = np.array([0.0, 0.0, 1.0])

    target, diagnostics = backtester._turning_point_reentry_target(
        weights,
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_base_floor_gate_pass"] == 0
    assert diagnostics["turning_point_zero_entry_state"] == 3
    assert diagnostics["turning_point_zero_entry_active"] == 1
    assert diagnostics["turning_point_reentry_active"] == 1
    assert np.allclose(target, [0.20, 0.0, 0.80])


def test_zero_entry_bridge_can_require_consecutive_market_signals() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [np.full(232, -0.001), np.full(20, 0.002)]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {
                    "bull": 0.20,
                    "rebound": 0.20,
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
                "path_conditioning": {
                    "enabled": True,
                    "minimum_confirmation_days": 3,
                },
            },
        }
    }
    weights = np.array([0.0, 0.0, 1.0])

    results = [
        backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
            signal_date=pd.Timestamp(f"2025-01-0{day}"),
        )
        for day in (2, 3, 6)
    ]

    for target, diagnostics in results[:2]:
        assert np.allclose(target, weights)
        assert diagnostics["turning_point_zero_entry_path_gate_pass"] == 0
    third_target, third_diagnostics = results[2]
    assert np.allclose(third_target, [0.20, 0.0, 0.80])
    assert (
        third_diagnostics["turning_point_zero_entry_market_signal_active"]
        == 1
    )
    assert third_diagnostics["turning_point_zero_entry_signal_streak"] == 3
    assert third_diagnostics["turning_point_zero_entry_path_gate_pass"] == 1


def test_zero_entry_path_streak_is_idempotent_per_decision_date() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    assets = ["QQQ", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.002)]
            ),
            "CASH": np.zeros(252),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 1.0},
            "zero_entry_bridge": {
                "enabled": True,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"rebound": 0.20},
                "core_weights": {"QQQ": 1.0},
                "path_conditioning": {
                    "enabled": True,
                    "minimum_confirmation_days": 2,
                },
            },
        }
    }

    diagnostics = [
        backtester._turning_point_reentry_target(
            np.array([0.0, 1.0]),
            returns,
            assets,
            1,
            config,
            signal_date=pd.Timestamp("2025-01-02"),
        )[1]
        for _ in range(2)
    ]

    assert [
        item["turning_point_zero_entry_signal_streak"]
        for item in diagnostics
    ] == [1, 1]


def test_zero_entry_bridge_can_inherit_relative_momentum_allocation() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    assets = ["QQQ", "SEMIS", "CASH"]
    returns = pd.DataFrame(
        {
            "QQQ": np.concatenate(
                [np.full(232, -0.001), np.full(20, 0.002)]
            ),
            "SEMIS": np.concatenate(
                [np.full(126, 0.002), np.full(106, -0.001), np.full(20, 0.004)]
            ),
            "CASH": np.zeros(252),
        }
    )
    config = {
        "relative_momentum_core": {
            "enabled": True,
            "anchor_asset": "QQQ",
            "tilt_asset": "SEMIS",
            "horizons": [126],
            "horizon_weights": [1.0],
            "skip_days": 0,
            "base_tilt_weight": 0.50,
            "max_tilt": 0.50,
        },
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {
                    "bull": 0.20,
                    "rebound": 0.20,
                },
                "allocation_method": "relative_momentum",
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        },
    }

    target, diagnostics = backtester._turning_point_reentry_target(
        np.array([0.0, 0.0, 1.0]),
        returns,
        assets,
        2,
        config,
    )

    assert np.isclose(target[:2].sum(), 0.20)
    assert target[1] > target[0]
    assert diagnostics["turning_point_zero_entry_tilt_weight"] > 0.50


def test_zero_entry_bridge_can_veto_parent_correction_state() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.full(189, 0.003),
            np.full(43, -0.004),
            np.full(20, 0.002),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "disallowed_parent_states": ["correction"],
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }

    target, diagnostics = backtester._turning_point_reentry_target(
        np.array([0.0, 0.0, 1.0]),
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_state"] == 1
    assert diagnostics["turning_point_zero_entry_state"] == 0
    assert diagnostics["turning_point_zero_entry_active"] == 0
    assert np.allclose(target, [0.0, 0.0, 1.0])


def test_zero_entry_bridge_can_distinguish_constructive_high_volatility() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    quiet = np.tile([0.003, -0.003], 390)
    constructive = np.tile([0.01, 0.04], 10)
    destructive = np.concatenate([np.full(19, 0.006), [-0.10]])
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "low_quantile": 0.45,
                    "high_quantile": 0.90,
                    "state_account_shares": {
                        "bull": {"high": 0.0},
                    },
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "window_days": 20,
                    "maximum_downside_variation_share": 0.50,
                    "state_account_shares": {"bull": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    weights = np.array([0.0, 0.0, 1.0])

    def target_for(tail: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        growth = np.concatenate([quiet, tail])
        returns = pd.DataFrame(
            {
                "QQQ": growth,
                "SEMIS": growth,
                "CASH": np.zeros(len(growth)),
            }
        )
        return backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
        )

    constructive_target, constructive_diagnostics = target_for(
        constructive
    )
    destructive_target, destructive_diagnostics = target_for(destructive)

    assert constructive_diagnostics[
        "turning_point_zero_entry_volatility_state"
    ] == 2
    assert constructive_diagnostics[
        "turning_point_zero_entry_downside_variation_share"
    ] == 0.0
    assert constructive_diagnostics["turning_point_zero_entry_active"] == 1
    assert np.allclose(constructive_target, [0.20, 0.0, 0.80])
    assert destructive_diagnostics[
        "turning_point_zero_entry_downside_variation_share"
    ] > 0.90
    assert destructive_diagnostics["turning_point_zero_entry_active"] == 0
    assert np.allclose(destructive_target, weights)


def test_constructive_high_volatility_bridge_respects_deep_bear_guard() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.tile([0.003, -0.003], 274),
            np.full(232, -0.002),
            np.tile([0.005, 0.015], 10),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"rebound": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "state_account_shares": {
                        "rebound": {"high": 0.0},
                    },
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "window_days": 20,
                    "maximum_downside_variation_share": 0.50,
                    "minimum_slow_excess_return": -0.20,
                    "state_account_shares": {"rebound": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }

    target, diagnostics = backtester._turning_point_reentry_target(
        np.array([0.0, 0.0, 1.0]),
        returns,
        assets,
        2,
        config,
    )

    assert diagnostics["turning_point_zero_entry_state"] == 3
    assert diagnostics[
        "turning_point_zero_entry_downside_variation_share"
    ] == 0.0
    assert diagnostics["turning_point_zero_entry_active"] == 0
    assert np.allclose(target, [0.0, 0.0, 1.0])


def test_constructive_high_volatility_bridge_requires_vix_contango() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [
            np.tile([0.003, -0.003], 390),
            np.tile([0.01, 0.04], 10),
        ]
    )
    returns = pd.DataFrame(
        {
            "QQQ": growth,
            "SEMIS": growth,
            "CASH": np.zeros(len(growth)),
        }
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "state_account_shares": {
                        "bull": {"high": 0.0},
                    },
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "require_vix_contango": True,
                    "state_account_shares": {"bull": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    weights = np.array([0.0, 0.0, 1.0])

    backtester.previous_vix_backwardation = True
    blocked_target, blocked_diagnostics = (
        backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
        )
    )
    backtester.previous_vix_backwardation = False
    allowed_target, allowed_diagnostics = (
        backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
        )
    )

    assert blocked_diagnostics["turning_point_zero_entry_active"] == 0
    assert np.allclose(blocked_target, weights)
    assert allowed_diagnostics["turning_point_zero_entry_active"] == 1
    assert np.allclose(allowed_target, [0.20, 0.0, 0.80])


def test_constructive_high_volatility_bridge_can_require_jump_dominance() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    quiet = np.tile([0.003, -0.003], 390)
    diffusive = np.full(20, 0.03)
    jump = diffusive.copy()
    jump[-1] = 0.20
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "state_account_shares": {
                        "bull": {"low": 0.0, "medium": 0.0, "high": 0.0}
                    },
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "window_days": 20,
                    "minimum_jump_variation_share": 0.0,
                    "state_account_shares": {"bull": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    weights = np.array([0.0, 0.0, 1.0])

    def target_for(tail: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        growth = np.concatenate([quiet, tail])
        returns = pd.DataFrame(
            {"QQQ": growth, "SEMIS": growth, "CASH": np.zeros(len(growth))}
        )
        return backtester._turning_point_reentry_target(
            weights, returns, assets, 2, config
        )

    blocked_target, blocked_diagnostics = target_for(diffusive)
    allowed_target, allowed_diagnostics = target_for(jump)

    assert blocked_diagnostics["turning_point_zero_entry_jump_gate_pass"] == 0
    assert np.allclose(blocked_target, weights)
    assert allowed_diagnostics["turning_point_zero_entry_jump_gate_pass"] == 1
    assert allowed_diagnostics["turning_point_zero_entry_active"] == 1
    assert np.allclose(allowed_target, [0.20, 0.0, 0.80])


def test_constructive_high_volatility_bridge_can_require_relief() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "VIX_HEDGE", "CASH"]
    growth = np.concatenate(
        [np.tile([0.003, -0.003], 390), np.tile([0.01, 0.04], 10)]
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "state_account_shares": {"bull": {"high": 0.0}},
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "window_days": 20,
                    "relief_asset": "VIX_HEDGE",
                    "relief_window_days": 5,
                    "maximum_relief_return": 0.0,
                    "state_account_shares": {"bull": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    weights = np.array([0.0, 0.0, 0.0, 1.0])

    def target_for(vol_tail: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        returns = pd.DataFrame(
            {
                "QQQ": growth,
                "SEMIS": growth,
                "VIX_HEDGE": np.concatenate(
                    [np.zeros(len(growth) - 5), vol_tail]
                ),
                "CASH": np.zeros(len(growth)),
            }
        )
        return backtester._turning_point_reentry_target(
            weights, returns, assets, 3, config
        )

    blocked_target, blocked_diagnostics = target_for(np.full(5, 0.01))
    allowed_target, allowed_diagnostics = target_for(np.full(5, -0.01))

    assert blocked_diagnostics["turning_point_zero_entry_relief_gate_pass"] == 0
    assert np.allclose(blocked_target, weights)
    assert allowed_diagnostics["turning_point_zero_entry_relief_gate_pass"] == 1
    assert allowed_diagnostics["turning_point_zero_entry_active"] == 1
    assert np.allclose(allowed_target, [0.20, 0.0, 0.0, 0.80])


def test_constructive_relief_signal_excludes_decision_date_level() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = False
    backtester.turning_point_reentry_episode_fired = False
    assets = ["QQQ", "SEMIS", "CASH"]
    growth = np.concatenate(
        [np.tile([0.003, -0.003], 390), np.tile([0.01, 0.04], 10)]
    )
    returns = pd.DataFrame(
        {"QQQ": growth, "SEMIS": growth, "CASH": np.zeros(len(growth))}
    )
    config = {
        "turning_point_reentry": {
            "enabled": True,
            "fast_days": 63,
            "slow_days": 252,
            "minimum_current_growth_exposure": 0.20,
            "state_account_shares": {},
            "core_weights": {"QQQ": 0.50, "SEMIS": 0.50},
            "zero_entry_bridge": {
                "enabled": True,
                "fast_days": 20,
                "slow_days": 252,
                "maximum_current_growth_exposure": 0.20,
                "state_account_shares": {"bull": 0.20},
                "volatility_conditioning": {
                    "enabled": True,
                    "window_days": 20,
                    "threshold_lookback_days": 756,
                    "minimum_threshold_observations": 504,
                    "state_account_shares": {"bull": {"high": 0.0}},
                },
                "constructive_high_volatility_override": {
                    "enabled": True,
                    "window_days": 20,
                    "relief_signal": "VIX",
                    "relief_window_days": 5,
                    "maximum_relief_return": 0.0,
                    "state_account_shares": {"bull": 0.20},
                },
                "core_weights": {"QQQ": 1.0, "SEMIS": 0.0},
            },
        }
    }
    dates = pd.bdate_range("2024-01-01", periods=7)
    decision_date = dates[-1]
    weights = np.array([0.0, 0.0, 1.0])

    allowed_target, allowed_diagnostics = (
        backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
            pd.DataFrame(
                {"VIX": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 100.0]},
                index=dates,
            ),
            decision_date,
        )
    )
    blocked_target, blocked_diagnostics = (
        backtester._turning_point_reentry_target(
            weights,
            returns,
            assets,
            2,
            config,
            pd.DataFrame(
                {"VIX": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 1.0]},
                index=dates,
            ),
            decision_date,
        )
    )

    assert allowed_diagnostics["turning_point_zero_entry_relief_return"] < 0.0
    assert allowed_diagnostics["turning_point_zero_entry_relief_gate_pass"] == 1
    assert np.allclose(allowed_target, [0.20, 0.0, 0.80])
    assert blocked_diagnostics["turning_point_zero_entry_relief_return"] > 0.0
    assert blocked_diagnostics["turning_point_zero_entry_relief_gate_pass"] == 0
    assert np.allclose(blocked_target, weights)


def test_conditional_vix_hedge_excludes_decision_date_signals() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    dates = pd.bdate_range("2026-07-20", periods=3)
    prices = pd.DataFrame(
        {
            "VIX": [18.0, 19.0, 100.0],
            "VIX3M": [20.0, 21.0, 10.0],
        },
        index=dates,
    )
    weights = np.array([0.0, 1.0])
    config = {
        "conditional_vix_hedge": {
            "enabled": True,
            "asset": "VIX_HEDGE",
            "spot_signal": "VIX",
            "three_month_signal": "VIX3M",
            "account_share": 0.05,
        }
    }

    target, active, ratio = backtester._conditional_vix_hedge_target(
        dates[-1],
        prices,
        weights,
        ["VIX_HEDGE", "CASH"],
        1,
        config,
    )

    assert active is False
    assert np.isclose(ratio, 19.0 / 21.0)
    assert np.allclose(target, weights)


def test_qqq_first_reentry_uses_existing_strategic_share() -> None:
    weights = np.array([0.10, 0.10, 0.20, 0.60])
    target = RegimeBacktester._qqq_first_reentry_target(
        weights,
        ["QQQ", "SEMIS", "GOLD", "CASH"],
        3,
        {
            "paper_growth_assets": ["QQQ", "SEMIS"],
            "strategic_core_weights": {
                "QQQ": 0.40,
                "SEMIS": 0.40,
                "GOLD": 0.20,
            },
        },
        "QQQ",
    )

    assert np.allclose(target, [0.40, 0.0, 0.12, 0.48])
    assert np.isclose(target.sum(), 1.0)


def test_daily_hmm_target_cache_tracks_each_scheduled_state() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.paper_risk_on_state = True
    backtester.last_daily_hmm_signal = None
    backtester.last_daily_hmm_auxiliary_gate = None
    backtester.last_risk_on_target = None
    backtester.last_risk_off_target = None
    config = {"daily_hmm_state_filter": {"enabled": True}}
    risk_on_target = np.array([0.40, 0.40, 0.20])
    risk_off_target = np.array([0.0, 0.0, 1.0])

    diagnostics = {
        "leverage_signal_score": 0.10,
        "trend_stress": 0,
        "vix_backwardation": 0,
    }
    portfolio_config = {
        **config,
        "allow_leverage": True,
        "leverage_activation_score": 0.0,
    }
    backtester._cache_daily_hmm_target(
        risk_on_target, portfolio_config, diagnostics
    )
    backtester.paper_risk_on_state = False
    backtester._cache_daily_hmm_target(
        risk_off_target, portfolio_config, diagnostics
    )

    assert np.allclose(backtester.last_risk_on_target, risk_on_target)
    assert np.allclose(backtester.last_risk_off_target, risk_off_target)
    assert backtester.last_daily_hmm_signal is False
    assert backtester.last_daily_hmm_auxiliary_gate is True


def test_daily_hmm_recovery_quality_requires_cross_asset_confirmation() -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "BREADTH": np.linspace(100.0, 112.0, len(dates)),
            "SPX": np.linspace(100.0, 105.0, len(dates)),
            "CREDIT": np.linspace(100.0, 108.0, len(dates)),
            "BOND": np.linspace(100.0, 102.0, len(dates)),
        },
        index=dates,
    )
    config = {
        "recovery_quality": {
            "enabled": True,
            "lookback_days": 21,
            "minimum_confirmations": 2,
            "signals": {
                "breadth": {"numerator": "BREADTH", "denominator": "SPX"},
                "credit": {"numerator": "CREDIT", "denominator": "BOND"},
            },
        }
    }

    passed, diagnostics = RegimeBacktester._daily_hmm_recovery_quality(
        dates[-1] + pd.offsets.BDay(1),
        prices,
        config,
    )

    assert passed
    assert diagnostics["daily_hmm_recovery_quality_confirmations"] == 2
    assert diagnostics["daily_hmm_recovery_quality_pass"] == 1


def test_daily_hmm_soft_recovery_quality_records_evidence_without_blocking() -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "BREADTH": np.linspace(100.0, 90.0, len(dates)),
            "SPX": np.linspace(100.0, 105.0, len(dates)),
            "CREDIT": np.linspace(100.0, 108.0, len(dates)),
            "BOND": np.linspace(100.0, 102.0, len(dates)),
        },
        index=dates,
    )
    config = {
        "recovery_quality": {
            "enabled": True,
            "decision_mode": "soft",
            "lookback_days": 21,
            "minimum_confirmations": 2,
            "signals": {
                "breadth": {"numerator": "BREADTH", "denominator": "SPX"},
                "credit": {"numerator": "CREDIT", "denominator": "BOND"},
            },
        }
    }

    accepted, diagnostics = RegimeBacktester._daily_hmm_recovery_quality(
        dates[-1] + pd.offsets.BDay(1),
        prices,
        config,
    )

    assert accepted
    assert diagnostics["daily_hmm_recovery_quality_confirmations"] == 1
    assert diagnostics["daily_hmm_recovery_quality_pass"] == 0
    assert diagnostics["daily_hmm_recovery_quality_gate_applied"] == 0


def test_confidence_blended_reentry_shrinks_full_basket_using_all_votes() -> None:
    current = np.array([0.10, 0.10, 0.20, 0.60])
    risk_on = np.array([0.40, 0.40, 0.20, 0.00])

    weak, weak_fraction = RegimeBacktester._confidence_blended_reentry_target(
        current, risk_on, confirmations=0, signal_count=2
    )
    strong, strong_fraction = (
        RegimeBacktester._confidence_blended_reentry_target(
            current, risk_on, confirmations=2, signal_count=2
        )
    )

    assert np.isclose(weak_fraction, 0.40)
    assert np.isclose(strong_fraction, 0.80)
    assert np.allclose(weak, [0.22, 0.22, 0.20, 0.36])
    assert np.allclose(strong, [0.34, 0.34, 0.20, 0.12])


def test_entry_only_daily_hmm_preserves_scheduled_exit_authority() -> None:
    config = {
        "daily_hmm_state_filter": {
            "entry_only": True,
            "state_update_mode": "scheduled_only",
        }
    }

    assert not RegimeBacktester._daily_hmm_exit_allowed(True, config)
    assert RegimeBacktester._daily_hmm_exit_allowed(False, config)
    assert not RegimeBacktester._daily_hmm_updates_scheduled_state(config)


def test_daily_hmm_episode_memory_blocks_repeated_failed_reentry() -> None:
    backtester = RegimeBacktester.__new__(RegimeBacktester)
    backtester.daily_hmm_reentry_attempt_active = False
    backtester.daily_hmm_reentry_blocked = False
    config = {
        "daily_hmm_state_filter": {
            "episode_memory": {"enabled": True}
        }
    }

    backtester._observe_daily_hmm_transition(True, config)
    backtester._observe_daily_hmm_transition(False, config)

    assert backtester.daily_hmm_reentry_attempt_active is False
    assert backtester.daily_hmm_reentry_blocked is True
