import numpy as np

from tools.audit_hmm_feature_geometry import correlation_geometry


def test_correlation_geometry_identifies_independent_dimensions() -> None:
    result = correlation_geometry(np.eye(4))

    assert np.isclose(result["effective_rank"], 4.0)
    assert np.isclose(result["participation_ratio"], 4.0)
    assert result["components_80pct"] == 4
    assert result["pairs_abs_correlation_ge_0_8"] == 0


def test_correlation_geometry_identifies_redundant_pair() -> None:
    correlation = np.array(
        [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    result = correlation_geometry(correlation)

    assert result["effective_rank"] < 3.0
    assert result["pairs_abs_correlation_ge_0_8"] == 1
