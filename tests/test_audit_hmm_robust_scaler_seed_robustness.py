import pandas as pd

from tools.audit_hmm_robust_scaler_seed_robustness import seed_period_summary


def test_seed_period_summary_reports_positive_fraction_and_worst_values() -> None:
    metrics = pd.DataFrame(
        {
            "sample": ["normal"] * 3,
            "period": ["holdout"] * 3,
            "seed": [7, 42, 123],
            "cagr_delta": [0.01, -0.02, 0.03],
            "max_drawdown_delta": [0.0, -0.01, 0.02],
        }
    )

    result = seed_period_summary(metrics).iloc[0]

    assert result["seeds"] == 3
    assert result["positive_cagr_fraction"] == 2.0 / 3.0
    assert result["median_cagr_delta"] == 0.01
    assert result["minimum_cagr_delta"] == -0.02
    assert result["worst_max_drawdown_delta"] == -0.01
