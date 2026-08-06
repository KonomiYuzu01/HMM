from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_sector_hmm_shadow import (
    build_monthly_features,
    complete_monthly_prices,
    eligible_assets,
    one_way_turnover,
    simulate_monthly,
    state_conditioned_scores,
    top_k_equal_weight,
)


def test_complete_monthly_prices_drops_current_partial_month() -> None:
    index = pd.to_datetime(["2026-05-29", "2026-06-30", "2026-07-29"])
    daily = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=index)
    result = complete_monthly_prices(daily, as_of=pd.Timestamp("2026-07-30"))
    assert result.index.tolist() == [
        pd.Timestamp("2026-05-31"),
        pd.Timestamp("2026-06-30"),
    ]


def test_cpi_feature_uses_prior_month_release() -> None:
    index = pd.date_range("2024-01-31", periods=16, freq="ME")
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.power(1.01, np.arange(16)),
            "HYG": 80.0 * np.power(1.005, np.arange(16)),
            "IEF": 90.0 * np.power(1.002, np.arange(16)),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "DGS10": np.linspace(4.0, 4.5, 16),
            "DFII10": np.linspace(1.5, 1.8, 16),
            "CPIAUCSL": np.arange(300.0, 316.0),
            "T10Y2Y": np.linspace(-0.5, 0.5, 16),
            "DTWEXBGS": np.linspace(120.0, 118.0, 16),
        },
        index=index,
    )
    before = build_monthly_features(
        prices, macro, cpi_release_lag_months=1
    )
    changed = macro.copy()
    changed.loc[index[-1], "CPIAUCSL"] = 999.0
    after = build_monthly_features(
        prices, changed, cpi_release_lag_months=1
    )
    assert before.loc[index[-1], "cpi_yoy"] == pytest.approx(
        after.loc[index[-1], "cpi_yoy"]
    )
    assert before.loc[index[-1], "cpi_direction_3m"] == pytest.approx(
        after.loc[index[-1], "cpi_direction_3m"]
    )


def test_late_listing_is_ineligible_until_minimum_history() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {
            "XLK": np.arange(100.0, 106.0),
            "XLC": [np.nan, np.nan, np.nan, 50.0, 51.0, 52.0],
        },
        index=index,
    )
    assert eligible_assets(prices, ["XLK", "XLC"], index[-1], 4) == ["XLK"]
    assert eligible_assets(prices, ["XLK", "XLC"], index[-1], 3) == [
        "XLK",
        "XLC",
    ]


def test_top_k_weights_are_long_only_and_conserve_budget() -> None:
    weights = top_k_equal_weight(
        pd.Series({"XLK": 3.0, "XLF": 2.0, "XLE": 1.0, "XLU": 0.0}),
        3,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    assert weights.max() == pytest.approx(1.0 / 3.0)
    assert weights["XLU"] == 0.0


def test_state_score_does_not_use_unknown_forward_return() -> None:
    index = pd.date_range("2020-01-31", periods=10, freq="ME")
    features = pd.DataFrame({"feature": np.arange(10.0)}, index=index)
    forward = pd.DataFrame(
        {"XLK": np.linspace(-0.02, 0.03, 10)},
        index=index,
    )
    path = [np.zeros(10, dtype=int)]
    before = state_conditioned_scores(
        features,
        forward,
        path,
        ["XLK"],
        minimum_observations=2,
        shrinkage_observations=2.0,
    )
    changed = forward.copy()
    changed.iloc[-1, 0] = 10.0
    after = state_conditioned_scores(
        features,
        changed,
        path,
        ["XLK"],
        minimum_observations=2,
        shrinkage_observations=2.0,
    )
    assert before["XLK"] == pytest.approx(after["XLK"])


def test_turnover_and_cost_are_exact() -> None:
    signal_dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    weights = pd.DataFrame(
        {"XLK": [0.5, 0.0], "XLF": [0.5, 1.0]},
        index=signal_dates,
    )
    assert one_way_turnover(weights).tolist() == pytest.approx([1.0, 0.5])
    price_dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    prices = pd.DataFrame(
        {
            "XLK": [100.0, 110.0, 110.0],
            "XLF": [100.0, 100.0, 105.0],
        },
        index=price_dates,
    )
    simulation = simulate_monthly(weights, prices, one_way_cost_bps=10.0)
    assert simulation["gross_return"].tolist() == pytest.approx([0.05, 0.05])
    assert simulation["cost"].tolist() == pytest.approx([0.001, 0.0005])
