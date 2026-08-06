from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_causal_delayed_grid import variants
from evaluate_smh_dynamic_guard import load_inputs, simulate


DESTINATION = Path("output/smh_causal_delayed_grid")
SELECTED = "causal_gap250bp_min60_cap25_confirm2_hold4"


def circular_family_reality_check(
    relative_log_returns: np.ndarray,
    selected_index: int,
    block_days: int,
    samples: int = 5_000,
    seed: int = 20_260_727,
    batch_size: int = 50,
) -> dict[str, float | int]:
    if relative_log_returns.ndim != 2:
        raise ValueError("Relative returns must be observations by candidates")
    observations, candidates = relative_log_returns.shape
    if not 0 <= selected_index < candidates:
        raise ValueError("Selected index is outside the candidate matrix")
    if not 1 <= block_days <= observations:
        raise ValueError("Block length must fit inside the sample")

    observed = relative_log_returns.mean(axis=0) * 252.0
    extended = np.vstack(
        [
            relative_log_returns,
            relative_log_returns[: block_days - 1],
        ]
    )
    cumulative = np.vstack(
        [
            np.zeros((1, candidates), dtype=float),
            np.cumsum(extended, axis=0),
        ]
    )
    full_block_sums = (
        cumulative[block_days : block_days + observations]
        - cumulative[:observations]
    )
    full_blocks, remainder = divmod(observations, block_days)
    if remainder:
        remainder_sums = (
            cumulative[remainder : remainder + observations]
            - cumulative[:observations]
        )
    else:
        remainder_sums = np.empty((0, candidates), dtype=float)

    rng = np.random.default_rng(seed + block_days)
    maximum_statistics = np.empty(samples, dtype=float)
    selected_statistics = np.empty(samples, dtype=float)
    blocks_per_sample = full_blocks + int(remainder > 0)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        starts = rng.integers(
            0,
            observations,
            size=(stop - start, blocks_per_sample),
        )
        totals = full_block_sums[starts[:, :full_blocks]].sum(axis=1)
        if remainder:
            totals += remainder_sums[starts[:, -1]]
        null_statistics = totals / observations * 252.0 - observed
        maximum_statistics[start:stop] = null_statistics.max(axis=1)
        selected_statistics[start:stop] = null_statistics[:, selected_index]

    selected_observed = float(observed[selected_index])
    return {
        "block_days": block_days,
        "bootstrap_samples": samples,
        "observations": observations,
        "unique_candidate_paths": candidates,
        "selected_annualized_relative_log_return": selected_observed,
        "nominal_one_sided_p_value": float(
            (
                np.count_nonzero(
                    selected_statistics >= selected_observed
                )
                + 1
            )
            / (samples + 1)
        ),
        "familywise_reality_check_p_value": float(
            (
                np.count_nonzero(
                    maximum_statistics >= selected_observed
                )
                + 1
            )
            / (samples + 1)
        ),
        "null_maximum_95pct": float(
            np.quantile(maximum_statistics, 0.95)
        ),
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    weights, base_daily, opens, closes = load_inputs()
    guards = variants()
    baseline, _ = simulate(
        weights,
        base_daily,
        opens,
        closes,
        guards[0],
    )
    baseline_log = np.log1p(
        baseline["net_return"].to_numpy(dtype=float)
    )

    arrays_by_digest: dict[str, np.ndarray] = {}
    names_by_digest: dict[str, list[str]] = {}
    selected_digest: str | None = None
    for guard in guards[1:]:
        candidate, _ = simulate(
            weights,
            base_daily,
            opens,
            closes,
            guard,
        )
        relative = (
            np.log1p(candidate["net_return"].to_numpy(dtype=float))
            - baseline_log
        )
        digest = sha256(np.round(relative, 14).tobytes()).hexdigest()
        arrays_by_digest.setdefault(digest, relative)
        names_by_digest.setdefault(digest, []).append(guard.name)
        if guard.name == SELECTED:
            selected_digest = digest

    if selected_digest is None:
        raise RuntimeError(f"Selected candidate not found: {SELECTED}")
    digests = list(arrays_by_digest)
    matrix = np.column_stack(
        [arrays_by_digest[digest] for digest in digests]
    )
    selected_index = digests.index(selected_digest)
    observed = matrix.mean(axis=0) * 252.0
    ranking = pd.DataFrame(
        {
            "stream_id": np.arange(len(digests)),
            "representative_variant": [
                names_by_digest[digest][0] for digest in digests
            ],
            "parameterization_count": [
                len(names_by_digest[digest]) for digest in digests
            ],
            "annualized_relative_log_return": observed,
            "includes_selected": [
                int(digest == selected_digest) for digest in digests
            ],
        }
    ).sort_values(
        "annualized_relative_log_return",
        ascending=False,
    )
    ranking.to_csv(DESTINATION / "family_unique_path_ranking.csv", index=False)

    rows = []
    for block_days in (21, 63, 126):
        rows.append(
            {
                **circular_family_reality_check(
                    matrix,
                    selected_index,
                    block_days,
                ),
                "parameterized_candidate_count": len(guards) - 1,
                "selected_rank": int(
                    ranking["includes_selected"].to_numpy().argmax() + 1
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        DESTINATION / "family_reality_check.csv",
        index=False,
    )
    print("Family reality check:")
    print(result.round(6).to_string(index=False))
    print(
        "\nUnique paths:",
        len(digests),
        "from",
        len(guards) - 1,
        "parameterizations",
    )
    print("Artifacts:", DESTINATION.resolve())


if __name__ == "__main__":
    main()
