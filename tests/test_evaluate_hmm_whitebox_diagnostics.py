from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_hmm_whitebox_diagnostics import (
    causal_percentile,
    ensemble_diagnostics,
    run_lengths,
)


def test_run_lengths_reset_when_the_state_changes() -> None:
    values = pd.Series([1, 1, 0, 0, 0, 1])
    assert run_lengths(values).tolist() == [1, 2, 1, 2, 3, 1]


def test_causal_percentile_does_not_use_the_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    result = causal_percentile(values, minimum_history=3)
    changed = values.copy()
    changed.iloc[-1] = 1_000.0
    changed_result = causal_percentile(changed, minimum_history=3)
    assert result.iloc[:-1].equals(changed_result.iloc[:-1])
    assert result.iloc[-1] == changed_result.iloc[-1] == 1.0


def test_ensemble_confidence_combines_margin_and_vote_disagreement() -> None:
    index = pd.to_datetime(["2026-01-01"])
    members = {}
    for seed, vote in zip((7, 42, 123), (1, 1, 0), strict=True):
        members[f"seed_{seed}"] = pd.DataFrame(
            {
                "state_entropy": [0.0],
                "state_probability_margin": [0.9],
                "one_step_predictive_log_likelihood": [-1.0],
                "template_expected_nearest_distance": [0.1],
                "template_expected_distance_margin": [0.2],
                "paper_risk_on_candidate": [vote],
                "dominant_template": [0],
                "state_duration_evaluations": [1],
            },
            index=index,
        )
    result = ensemble_diagnostics(members)
    assert np.isclose(result.iloc[0]["risk_on_vote_margin"], 1.0 / 3.0)
    assert np.isclose(
        result.iloc[0]["uncertainty_capacity_confidence"], 0.3
    )
