from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_hmm_diagnostic_predictiveness import (
    causal_template_risk_on_probability,
    forward_outcomes,
    market_signals,
)


def sample_closes(observations: int = 100) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=observations)
    steps = np.arange(observations, dtype=float)
    return pd.DataFrame(
        {
            "QQQ": 100.0 * np.exp(0.001 * steps),
            "SEMIS": 100.0 * np.exp(0.002 * steps),
        },
        index=index,
    )


def test_market_signals_exclude_same_day_close() -> None:
    closes = sample_closes()
    original = market_signals(closes)
    changed = closes.copy()
    changed.iloc[-1, changed.columns.get_loc("SEMIS")] *= 2.0
    revised = market_signals(changed)
    pd.testing.assert_series_equal(original.iloc[-1], revised.iloc[-1])


def test_forward_outcome_starts_with_next_session_return() -> None:
    closes = sample_closes(40)
    outcomes = forward_outcomes(closes, horizon=5)
    expected_log_return = (0.001 + 0.002) / 2.0
    expected = np.exp(5.0 * expected_log_return) - 1.0
    assert np.isclose(
        outcomes.iloc[1]["forward_growth_return_21d"], expected
    )
    assert outcomes.iloc[-4:]["forward_growth_return_21d"].isna().all()


def test_template_risk_probability_uses_only_prior_actions() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "template_0_probability": [1.0, 1.0, 0.0, 0.0, 1.0],
            "template_1_probability": [0.0, 0.0, 1.0, 1.0, 0.0],
            "paper_risk_on_candidate": [1.0, 1.0, 0.0, 0.0, 1.0],
        },
        index=index,
    )
    original = causal_template_risk_on_probability(frame)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("paper_risk_on_candidate")] = 0.0
    revised = causal_template_risk_on_probability(changed)
    pd.testing.assert_series_equal(original, revised)
    assert original.iloc[1] > original.iloc[0]
    assert original.iloc[3] < original.iloc[2]
