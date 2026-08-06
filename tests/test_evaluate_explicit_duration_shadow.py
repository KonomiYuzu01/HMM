from __future__ import annotations

import pandas as pd

from tools.evaluate_explicit_duration_shadow import (
    causal_duration_predictions,
    empirical_survival_probability,
)


def test_empirical_survival_uses_completed_durations() -> None:
    assert empirical_survival_probability([1, 2, 4], 2) == (1.0 + 1.0) / (
        2.0 + 2.0
    )


def test_duration_predictions_do_not_use_the_future_run_length() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="W")
    before = causal_duration_predictions(
        pd.Series([1, 1, 0, 0, 1, 1, 1], index=index)
    )
    after = causal_duration_predictions(
        pd.Series([1, 1, 0, 0, 1, 1, 0], index=index)
    )
    assert before.iloc[:-1].equals(after.iloc[:-1])
