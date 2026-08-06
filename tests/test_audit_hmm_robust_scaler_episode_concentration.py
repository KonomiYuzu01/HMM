import pandas as pd

from tools.audit_hmm_robust_scaler_episode_concentration import (
    contiguous_episode_ids,
)


def test_contiguous_episode_ids_split_on_inactive_days() -> None:
    active = pd.Series([False, True, True, False, True, False, True, True])

    result = contiguous_episode_ids(active)

    assert result.astype("object").where(result.notna(), None).tolist() == [
        None,
        1,
        1,
        None,
        2,
        None,
        3,
        3,
    ]
