from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from audit_strategy_library_frontier_spa import (
    circular_block_standard_error,
    pareto_mask,
    path_is_eligible_name,
    studentized_spa,
)


def test_path_name_filter_excludes_incomparable_samples() -> None:
    assert path_is_eligible_name(
        Path("output/example/normal_synthetic_candidate_daily.csv")
    )
    assert not path_is_eligible_name(
        Path("output/example/proxy_synthetic_candidate_daily.csv")
    )
    assert not path_is_eligible_name(
        Path("output/example/normal_live_candidate_daily.csv")
    )
    assert not path_is_eligible_name(
        Path("output/example/normal_synthetic_cost_stress_daily.csv")
    )


def test_circular_block_standard_error_is_positive() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 0.01, size=(500, 3))
    standard_error = circular_block_standard_error(values, 21)
    assert standard_error.shape == (3,)
    assert np.isfinite(standard_error).all()
    assert (standard_error > 0.0).all()


def test_studentized_spa_ranks_stronger_target_higher() -> None:
    rng = np.random.default_rng(2)
    common = rng.normal(0.0, 0.01, size=700)
    relative = np.column_stack(
        [
            common + 0.0015,
            common + rng.normal(0.0, 0.002, size=700) + 0.0002,
            rng.normal(-0.0002, 0.012, size=700),
        ]
    )
    spa, _ = studentized_spa(
        relative,
        {"strong": 0, "weak": 1},
        21,
        samples=300,
        seed=2,
        batch_size=30,
    )
    indexed = spa.set_index("target")
    assert (
        indexed.loc["strong", "studentized_statistic"]
        > indexed.loc["weak", "studentized_statistic"]
    )
    assert (
        indexed.loc["strong", "familywise_spa_p_value"]
        <= indexed.loc["weak", "familywise_spa_p_value"]
    )


def test_pareto_mask_keeps_only_undominated_points() -> None:
    cagr = np.array([0.20, 0.21, 0.22, 0.19])
    drawdown = np.array([-0.10, -0.12, -0.15, -0.09])
    assert pareto_mask(cagr, drawdown).tolist() == [
        True,
        True,
        True,
        True,
    ]
    dominated = pareto_mask(
        np.append(cagr, 0.18),
        np.append(drawdown, -0.16),
    )
    assert not bool(dominated[-1])
