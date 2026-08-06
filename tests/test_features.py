import numpy as np
import pandas as pd

from regime_strategy.features import build_causal_features, build_next_feature


def test_feature_at_t_does_not_use_price_at_t() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.linspace(0, 0.2, len(dates)))}, index=dates
    )
    baseline = build_causal_features(prices, ["SPX"], 20, 10)
    changed = prices.copy()
    target_date = baseline.index[-1]
    changed.loc[target_date, "SPX"] *= 1.5
    perturbed = build_causal_features(changed, ["SPX"], 20, 10)
    pd.testing.assert_series_equal(baseline.loc[target_date], perturbed.loc[target_date])


def test_next_feature_uses_latest_completed_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.linspace(0, 0.2, len(dates)))}, index=dates
    )
    feature = build_next_feature(prices, ["SPX"], 20, 10)
    expected = np.log(prices["SPX"]).diff().iloc[-1]
    assert np.isclose(feature["ret1__SPX"], expected)
    assert feature.name > prices.index[-1]


def test_vrp_feature_is_lagged_in_sample_and_current_for_next_session() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = pd.DataFrame(
        {
            "SPX": 100.0 * np.exp(np.linspace(0, 0.2, len(dates))),
            "VIX": np.linspace(20.0, 18.0, len(dates)),
        },
        index=dates,
    )
    specification = [
        {
            "name": "equity_vrp",
            "kind": "variance_risk_premium",
            "implied_signal": "VIX",
            "realized_asset": "SPX",
            "realized_days": 21,
        }
    ]
    baseline = build_causal_features(prices, ["SPX"], 20, 10, specification)
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("VIX")] = 40.0
    perturbed = build_causal_features(changed, ["SPX"], 20, 10, specification)
    pd.testing.assert_series_equal(baseline.iloc[-1], perturbed.iloc[-1])

    baseline_next = build_next_feature(prices, ["SPX"], 20, 10, specification)
    changed_next = build_next_feature(changed, ["SPX"], 20, 10, specification)
    assert changed_next["derived__equity_vrp"] > baseline_next["derived__equity_vrp"]


def test_feature_components_can_exclude_one_session_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.linspace(0, 0.2, len(dates)))},
        index=dates,
    )
    components = ["volatility", "momentum"]
    historical = build_causal_features(
        prices, ["SPX"], 20, 10, components=components
    )
    next_feature = build_next_feature(
        prices, ["SPX"], 20, 10, components=components
    )
    assert list(historical.columns) == ["vol20__SPX", "mean10__SPX"]
    assert list(next_feature.index) == ["vol20__SPX", "mean10__SPX"]


def test_downside_volatility_is_causal_and_responds_only_to_losses() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))},
        index=dates,
    )
    components = ["downside_volatility"]
    baseline = build_causal_features(
        prices, ["SPX"], 20, 10, components=components
    )
    baseline_next = build_next_feature(
        prices, ["SPX"], 20, 10, components=components
    )
    assert np.isclose(baseline_next["downvol20__SPX"], 0.0)

    changed = prices.copy()
    target_date = baseline.index[-1]
    changed.loc[target_date, "SPX"] *= 0.5
    perturbed = build_causal_features(
        changed, ["SPX"], 20, 10, components=components
    )
    pd.testing.assert_series_equal(
        baseline.loc[target_date], perturbed.loc[target_date]
    )
    changed_next = build_next_feature(
        changed, ["SPX"], 20, 10, components=components
    )
    assert changed_next["downvol20__SPX"] > 0.0


def test_volatility_acceleration_is_causal_and_responds_to_recent_shock() -> None:
    dates = pd.bdate_range("2024-01-01", periods=121)
    log_returns = np.tile([0.01, -0.01], 60)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])},
        index=dates,
    )
    components = ["volatility_acceleration"]
    baseline = build_causal_features(
        prices, ["SPX"], 60, 20, components=components
    )
    baseline_next = build_next_feature(
        prices, ["SPX"], 60, 20, components=components
    )
    name = "volaccel20_60__SPX"
    assert abs(baseline_next[name]) < 0.03

    changed = prices.copy()
    target_date = baseline.index[-1]
    changed.loc[target_date, "SPX"] *= 0.5
    perturbed = build_causal_features(
        changed, ["SPX"], 60, 20, components=components
    )
    pd.testing.assert_series_equal(
        baseline.loc[target_date], perturbed.loc[target_date]
    )
    changed_next = build_next_feature(
        changed, ["SPX"], 60, 20, components=components
    )
    assert changed_next[name] > baseline_next[name]


def test_new_risk_components_match_next_and_appended_historical_features() -> None:
    dates = pd.bdate_range("2024-01-01", periods=121)
    log_returns = 0.002 + 0.015 * np.sin(np.arange(120) / 3.0)
    prices = pd.DataFrame(
        {"SPX": 100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])},
        index=dates,
    )
    components = ["downside_volatility", "volatility_acceleration"]
    next_feature = build_next_feature(
        prices, ["SPX"], 60, 20, components=components
    )

    next_date = pd.bdate_range(prices.index[-1], periods=2)[-1]
    extended = pd.concat(
        [prices, pd.DataFrame({"SPX": [prices.iloc[-1, 0] * 3.0]}, index=[next_date])]
    )
    historical = build_causal_features(
        extended, ["SPX"], 60, 20, components=components
    )

    np.testing.assert_allclose(
        historical.loc[next_date].to_numpy(), next_feature.to_numpy()
    )
