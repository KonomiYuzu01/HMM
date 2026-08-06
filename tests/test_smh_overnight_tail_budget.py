from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_smh_overnight_tail_budget import (  # noqa: E402
    rolling_tail_estimate,
    semis_cap_from_tail_budget,
)


def test_tail_estimate_uses_only_prior_observations() -> None:
    losses = pd.Series([0.01, 0.02, 0.03, 0.04])

    estimate = rolling_tail_estimate(
        losses,
        lookback_days=3,
        tail_probability=0.5,
        estimator="quantile",
    )

    assert np.isnan(estimate.iloc[2])
    assert estimate.iloc[3] == 0.02


def test_tail_budget_converts_account_loss_to_weight_cap() -> None:
    assert semis_cap_from_tail_budget(0.02, 0.04, 0.20) == 0.50
    assert semis_cap_from_tail_budget(0.02, 0.20, 0.20) == 0.20
    assert semis_cap_from_tail_budget(0.02, np.nan, 0.20) == 1.0
