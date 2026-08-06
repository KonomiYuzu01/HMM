import numpy as np

from tools.audit_hmm_robust_scaler_feature_leverage import relative_scale_factors


def test_relative_scale_factors_detect_outlier_heavy_feature() -> None:
    base = np.linspace(-1.0, 1.0, 101)
    outlier_heavy = base.copy()
    outlier_heavy[-1] = 100.0
    values = np.column_stack([base, outlier_heavy, base * 2.0])

    factors = relative_scale_factors(values)

    assert np.isclose(np.median(factors), 1.0)
    assert factors[1] > factors[0]
    assert np.isclose(factors[0], factors[2])
