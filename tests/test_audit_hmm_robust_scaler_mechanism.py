import numpy as np
import pandas as pd

from tools.audit_hmm_robust_scaler_mechanism import aligned_weight_l1


def test_aligned_weight_l1_uses_common_dates_and_assets() -> None:
    baseline = pd.DataFrame(
        {"QQQ": [0.5, 0.4], "CASH": [0.5, 0.6]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    candidate = pd.DataFrame(
        {"QQQ": [0.3, 0.5], "CASH": [0.7, 0.5], "EXTRA": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    result = aligned_weight_l1(baseline, candidate)

    np.testing.assert_allclose(result.to_numpy(), [0.4, 0.2])
