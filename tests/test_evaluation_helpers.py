import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_tool_module(name: str):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_metrics = load_tool_module("evaluate_layer_stress_episodes").path_metrics
rolling_annualized_return = load_tool_module(
    "evaluate_rolling_benchmark_consistency"
).rolling_annualized_return
hierarchical_regimes = load_tool_module("evaluate_hierarchical_regime_portfolios")
regime_walkforward = load_tool_module("evaluate_regime_walkforward")


def test_path_metrics_includes_window_starting_equity_as_peak() -> None:
    total_return, max_drawdown = path_metrics(pd.Series([-0.10, -0.10]))

    assert np.isclose(total_return, -0.19)
    assert np.isclose(max_drawdown, -0.19)


def test_rolling_annualized_return_uses_compounded_return() -> None:
    daily_return = np.expm1(np.log1p(0.10) / 252.0)
    result = rolling_annualized_return(pd.Series([daily_return] * 252), 252)

    assert result.iloc[:-1].isna().all()
    assert np.isclose(result.iloc[-1], 0.10)


def test_hierarchical_regime_declining_crisis_overrides_risk_on() -> None:
    row = pd.Series(
        {
            "volatility_state": "high",
            "jump_ratio": 1.30,
            "vix_term_ratio": 0.90,
            "risk_on_votes": 3,
            "growth_fast_trend": -0.10,
            "growth_slow_trend": 0.20,
            "bond_trend": 0.01,
            "gold_trend": 0.01,
        }
    )

    assert hierarchical_regimes.classify_regime_row(row) == "crisis_decline"


def test_hierarchical_regime_separates_correction_and_rebound() -> None:
    base = {
        "volatility_state": "medium",
        "jump_ratio": 1.0,
        "vix_term_ratio": 0.9,
        "risk_on_votes": 0,
        "growth_slow_trend": 0.10,
        "bond_trend": 0.01,
        "gold_trend": 0.01,
    }

    assert (
        hierarchical_regimes.classify_regime_row(
            pd.Series({**base, "growth_fast_trend": -0.05})
        )
        == "correction"
    )
    assert (
        hierarchical_regimes.classify_regime_row(
            pd.Series({**base, "growth_fast_trend": 0.05})
        )
        == "rebound"
    )


def test_factorized_states_can_change_between_hmm_schedule_dates() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    schedule = pd.DataFrame(
        {
            "risk_on_votes": [0, 0],
            "dominant_template": [1, 1],
            "dominant_probability": [0.8, 0.8],
        },
        index=dates[[0, 2]],
    )
    signals = pd.DataFrame(
        {
            "realized_volatility": [0.20, 0.40, 0.20],
            "low_volatility_threshold": [0.15] * 3,
            "high_volatility_threshold": [0.30] * 3,
            "volatility_state": ["medium", "high", "medium"],
            "jump_ratio": [1.0, 1.4, 1.0],
            "vix_term_ratio": [0.9, 1.1, 0.9],
            "growth_fast_trend": [0.1, -0.1, 0.1],
            "growth_slow_trend": [-0.1] * 3,
            "bond_trend": [0.1] * 3,
            "gold_trend": [0.1] * 3,
            "stock_bond_correlation": [-0.1] * 3,
        },
        index=dates,
    )

    _, states = hierarchical_regimes.build_factorized_daily_states(
        schedule,
        signals,
        dates,
    )

    assert states.tolist() == [
        "rebound",
        "crisis_decline",
        "rebound",
    ]


def test_regime_simulation_respects_no_trade_band() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    state = pd.Series(["quiet_bull", "quiet_bull"], index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=hierarchical_regimes.ASSETS)
    mapping = dict(hierarchical_regimes.ECONOMIC_MAP)

    daily, simulated_weights = hierarchical_regimes.simulate_mapping(
        state,
        returns,
        mapping,
        apply_production_risk_controls=False,
    )

    assert daily.iloc[0]["turnover"] > 0.0
    assert daily.iloc[1]["turnover"] == 0.0
    assert np.isclose(simulated_weights.iloc[-1].sum(), 1.0)


def test_regime_tilt_preserves_anchor_cash_vix_and_gross() -> None:
    anchor = np.array([0.0, 0.28, 0.28, 0.0, 0.20, 0.0, 0.0, 0.20, 0.04])
    regime = np.array([0.0, 0.60, 0.20, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0])

    result = hierarchical_regimes.risk_budget_preserving_target(
        anchor,
        regime,
        0.25,
    )

    cash_index = hierarchical_regimes.ASSETS.index("CASH")
    vix_index = hierarchical_regimes.ASSETS.index("VIX_HEDGE")
    assert np.isclose(result[cash_index], anchor[cash_index])
    assert np.isclose(result[vix_index], anchor[vix_index])
    assert np.isclose(
        np.delete(result, [cash_index, vix_index]).sum(),
        0.76,
    )
    assert np.isclose(result.sum(), 1.0)


def test_zero_regime_tilt_exactly_reproduces_production_net_return() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    states = pd.Series(["quiet_bull", "normal_bull"], index=dates)
    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=hierarchical_regimes.ASSETS,
    )
    anchor = pd.DataFrame(
        [[0.0, 0.30, 0.30, 0.0, 0.20, 0.0, 0.0, 0.20, 0.0]] * 2,
        index=dates,
        columns=hierarchical_regimes.ASSETS,
    )
    production = pd.DataFrame(
        {
            "gross_return": [0.01, -0.02],
            "net_return": [0.009, -0.021],
        },
        index=dates,
    )

    daily, tilted_weights = hierarchical_regimes.simulate_anchor_tilt(
        states,
        returns,
        anchor,
        production,
        hierarchical_regimes.ECONOMIC_MAP,
        blend=0.0,
    )

    assert np.allclose(daily["net_return"], production["net_return"])
    assert np.allclose(daily["incremental_net_return"], 0.0)
    assert np.allclose(tilted_weights, anchor)


def test_moving_block_bootstrap_preserves_constant_alpha() -> None:
    result = hierarchical_regimes.moving_block_bootstrap_alpha(
        pd.Series([0.001] * 60),
        block_days=20,
        replications=100,
        seed=1,
    )

    assert np.isclose(result["annualized_arithmetic_alpha"], 0.252)
    assert np.isclose(result["ci_2_5"], 0.252)
    assert np.isclose(result["ci_97_5"], 0.252)
    assert result["probability_alpha_positive"] == 1.0


def test_regularized_blend_selects_smallest_near_best_candidate() -> None:
    metrics = pd.DataFrame(
        {
            "cagr": [0.10, 0.1090, 0.1100, 0.1110],
            "sharpe": [1.00, 1.01, 1.02, 1.03],
            "max_drawdown": [-0.10, -0.10, -0.10, -0.10],
        },
        index=[0.0, 0.125, 0.25, 0.50],
    )

    selected = regime_walkforward.select_regularized_blend(metrics)

    assert selected == 0.125
