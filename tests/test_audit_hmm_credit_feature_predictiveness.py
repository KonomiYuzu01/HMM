import numpy as np
import pandas as pd

from tools.audit_hmm_credit_feature_predictiveness import (
    causal_rolling_percentile,
    forward_path_labels,
)


def test_forward_path_labels_use_only_subsequent_returns() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="B")
    returns = pd.Series([99.0, 0.01, -0.02, 0.03, 0.04, 0.05], index=index)

    labels = forward_path_labels(returns, days=3)

    assert np.isclose(labels.iloc[0]["forward_log_return"], 0.02)
    assert np.isclose(labels.iloc[0]["forward_worst_log_return"], -0.01)
    assert np.isnan(labels.iloc[-3]["forward_log_return"])


def test_causal_percentile_is_unchanged_by_future_values() -> None:
    original = pd.Series(np.arange(800, dtype=float))
    changed = original.copy()
    changed.iloc[700:] = -999.0

    before = causal_rolling_percentile(original)
    after = causal_rolling_percentile(changed)

    pd.testing.assert_series_equal(before.iloc[:700], after.iloc[:700])
