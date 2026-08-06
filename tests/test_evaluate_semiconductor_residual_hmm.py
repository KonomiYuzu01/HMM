from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_semiconductor_residual_hmm import (
    apply_weak_state_cap,
    build_causal_features,
    walk_forward_weak_probability,
)


def synthetic_closes(observations: int = 220) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=observations)
    angle = np.arange(observations, dtype=float)
    qqq = 100.0 * np.exp(0.0005 * angle + 0.01 * np.sin(angle / 11.0))
    semis = qqq * np.exp(0.0002 * angle + 0.03 * np.sin(angle / 7.0))
    return pd.DataFrame({"QQQ": qqq, "SEMIS": semis}, index=index)


def test_features_do_not_use_same_day_close() -> None:
    closes = synthetic_closes()
    original = build_causal_features(closes)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 2.0
    revised = build_causal_features(changed)
    pd.testing.assert_series_equal(original.iloc[-1], revised.iloc[-1])


def test_walk_forward_signal_does_not_use_future_prices() -> None:
    closes = synthetic_closes()
    first = walk_forward_weak_probability(
        closes,
        training_days=80,
        refit_days=20,
        horizon=5,
        seeds=(7,),
        iterations=30,
    )
    cutoff = first.index[-2]
    changed = closes.copy()
    changed.loc[changed.index > cutoff, "SEMIS"] *= np.linspace(
        1.0, 2.0, int((changed.index > cutoff).sum())
    )
    second = walk_forward_weak_probability(
        changed,
        training_days=80,
        refit_days=20,
        horizon=5,
        seeds=(7,),
        iterations=30,
    )
    np.testing.assert_allclose(
        first.loc[:cutoff, "weak_probability"],
        second.loc[:cutoff, "weak_probability"],
        rtol=0.0,
        atol=1e-12,
    )


def test_weak_state_cap_preserves_growth_budget() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    weights = pd.DataFrame(
        {
            "QQQ": [0.20, 0.20, 0.20, 0.20],
            "SEMIS": [0.60, 0.60, 0.60, 0.60],
            "CASH": [0.20, 0.20, 0.20, 0.20],
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    probability = pd.Series([0.20, 0.60], index=index[[0, 2]])
    adjusted, updated_daily, diagnostics = apply_weak_state_cap(
        weights, daily, probability
    )
    np.testing.assert_allclose(
        adjusted["QQQ"] + adjusted["SEMIS"],
        weights["QQQ"] + weights["SEMIS"],
    )
    assert adjusted.loc[index[0], "SEMIS"] == 0.60
    assert adjusted.loc[index[-1], "SEMIS"] == 0.40
    assert updated_daily.loc[index[-1], "turnover"] > 0.0
    assert diagnostics["growth_budget_error"].max() <= 1e-12
