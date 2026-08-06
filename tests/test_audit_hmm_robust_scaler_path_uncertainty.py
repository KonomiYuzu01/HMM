import pandas as pd
import pytest

from tools.audit_hmm_robust_scaler_path_uncertainty import (
    paired_circular_block_bootstrap,
)


def test_paired_bootstrap_preserves_constant_log_advantage() -> None:
    baseline = pd.Series([0.0] * 20)
    candidate = pd.Series([0.001] * 20)

    result = paired_circular_block_bootstrap(
        baseline,
        candidate,
        block_days=5,
        simulations=100,
        seed=7,
    )

    expected = 252.0 * 0.0009995003330834232
    assert result["observed_annualized_relative_log_return"] == pytest.approx(expected)
    assert result["lower_95"] == pytest.approx(expected)
    assert result["upper_95"] == pytest.approx(expected)
    assert result["probability_positive"] == 1.0


def test_paired_bootstrap_rejects_short_sample() -> None:
    with pytest.raises(ValueError, match="complete block"):
        paired_circular_block_bootstrap(
            pd.Series([0.0, 0.0]),
            pd.Series([0.0, 0.0]),
            block_days=3,
            simulations=10,
        )
