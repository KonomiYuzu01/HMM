from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("output")
DESTINATION = ROOT / "drawdown_uncertainty_validation"
BASELINE = "paper_core_robust_vol_guarded_floor_ensemble"
CANDIDATE = "paper_core_zero_entry_growth_reallocation_ensemble"


def path_metrics(returns: np.ndarray) -> tuple[float, float]:
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    drawdown = np.concatenate(([1.0], wealth)) / running_peak - 1.0
    years = len(returns) / 252.0
    cagr = wealth[-1] ** (1.0 / years) - 1.0
    return float(cagr), float(drawdown.min())


def paired_circular_block_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    block_length: int,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    if len(candidate) != len(baseline):
        raise ValueError("Paired bootstrap inputs must have equal length")
    generator = np.random.default_rng(seed)
    observations = len(candidate)
    blocks = int(np.ceil(observations / block_length))
    offsets = np.arange(block_length)
    rows = []
    for _ in range(simulations):
        starts = generator.integers(0, observations, size=blocks)
        indices = ((starts[:, None] + offsets) % observations).ravel()[:observations]
        candidate_cagr, candidate_drawdown = path_metrics(candidate[indices])
        baseline_cagr, baseline_drawdown = path_metrics(baseline[indices])
        rows.append(
            {
                "candidate_cagr": candidate_cagr,
                "baseline_cagr": baseline_cagr,
                "candidate_max_drawdown": candidate_drawdown,
                "baseline_max_drawdown": baseline_drawdown,
            }
        )
    frame = pd.DataFrame(rows)
    frame["cagr_delta"] = frame["candidate_cagr"] - frame["baseline_cagr"]
    frame["max_drawdown_delta"] = (
        frame["candidate_max_drawdown"] - frame["baseline_max_drawdown"]
    )
    return frame


def strategy_summary(
    simulations: pd.DataFrame,
    block_length: int,
    prefix: str,
) -> dict[str, float | int | str]:
    cagr = simulations[f"{prefix}_cagr"]
    drawdown = simulations[f"{prefix}_max_drawdown"]
    return {
        "block_length": block_length,
        "strategy": prefix,
        "simulations": len(simulations),
        "median_cagr": float(cagr.median()),
        "cagr_5pct": float(cagr.quantile(0.05)),
        "cagr_95pct": float(cagr.quantile(0.95)),
        "median_max_drawdown": float(drawdown.median()),
        "max_drawdown_5pct": float(drawdown.quantile(0.05)),
        "max_drawdown_95pct": float(drawdown.quantile(0.95)),
        "probability_breach_18pct": float((drawdown < -0.18).mean()),
        "probability_breach_20pct": float((drawdown < -0.20).mean()),
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(
        ROOT / CANDIDATE / "daily_returns.csv", index_col=0, parse_dates=True
    ).loc["2015":"2025", "net_return"]
    baseline = pd.read_csv(
        ROOT / BASELINE / "daily_returns.csv", index_col=0, parse_dates=True
    ).loc["2015":"2025", "net_return"]
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    aligned.columns = ["candidate", "baseline"]

    summary_rows = []
    relative_rows = []
    for block_length in (21, 63, 126):
        simulations = paired_circular_block_bootstrap(
            aligned["candidate"].to_numpy(),
            aligned["baseline"].to_numpy(),
            block_length,
            simulations=5_000,
            seed=20260723 + block_length,
        )
        for prefix in ("baseline", "candidate"):
            summary_rows.append(
                strategy_summary(simulations, block_length, prefix)
            )
        relative_rows.append(
            {
                "block_length": block_length,
                "probability_candidate_cagr_higher": float(
                    (simulations["cagr_delta"] > 0.0).mean()
                ),
                "median_cagr_delta": float(simulations["cagr_delta"].median()),
                "cagr_delta_5pct": float(
                    simulations["cagr_delta"].quantile(0.05)
                ),
                "cagr_delta_95pct": float(
                    simulations["cagr_delta"].quantile(0.95)
                ),
                "probability_candidate_drawdown_better": float(
                    (simulations["max_drawdown_delta"] > 0.0).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index(["block_length", "strategy"])
    relative = pd.DataFrame(relative_rows).set_index("block_length")
    summary.to_csv(DESTINATION / "strategy_tail_summary.csv")
    relative.to_csv(DESTINATION / "paired_relative_summary.csv")
    print(summary.round(6).to_string())
    print("\nPaired candidate minus baseline:")
    print(relative.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
