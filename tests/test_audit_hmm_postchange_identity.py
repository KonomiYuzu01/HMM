import numpy as np
import pandas as pd

from tools.audit_hmm_postchange_identity import maximum_common_numeric_difference


def test_maximum_common_numeric_difference_ignores_unshared_labels() -> None:
    baseline = pd.DataFrame(
        {"value": [1.0, 2.0], "baseline_only": [3.0, 4.0]},
        index=["a", "b"],
    )
    candidate = pd.DataFrame(
        {"value": [1.0, 2.1], "candidate_only": [5.0, 6.0]},
        index=["a", "b"],
    )

    difference = maximum_common_numeric_difference(baseline, candidate)

    assert np.isclose(difference, 0.1)
