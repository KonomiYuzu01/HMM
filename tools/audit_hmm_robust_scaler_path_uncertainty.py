from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_hmm_order_ensemble import DEFINITIONS, with_strategy_inputs
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_robust_scaler_path_uncertainty")
BASELINES = {
    "normal": Path("output/research_hmm_standard_restarts5"),
    "proxy": Path("output/research_hmm_standard_restarts5_20y_proxy"),
}
CANDIDATES = {
    "normal": Path("output/research_hmm_robust_restarts5"),
    "proxy": Path("output/research_hmm_robust_restarts5_20y_proxy"),
}
PERIODS = {
    "normal_complete": ("normal", "2015-01-01", "2025-12-31"),
    "normal_holdout": ("normal", "2022-01-01", "2025-12-31"),
    "proxy_complete": ("proxy", "2006-08-01", "2025-12-31"),
}
BLOCK_DAYS = (5, 21, 63)
SIMULATIONS = 10_000
SEED = 20_260_801


def paired_circular_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    block_days: int,
    simulations: int = SIMULATIONS,
    seed: int = SEED,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [baseline.rename("baseline"), candidate.rename("candidate")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < block_days:
        raise ValueError("Sample must contain at least one complete block")
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    count = len(relative)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    generator = np.random.default_rng(seed)
    estimates = np.empty(simulations, dtype=float)
    for simulation in range(simulations):
        starts = generator.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[simulation] = float(relative[indices].mean() * 252.0)
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "observations": count,
        "block_days": block_days,
        "simulations": simulations,
        "observed_annualized_relative_log_return": float(relative.mean() * 252.0),
        "bootstrap_median": float(median),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "probability_positive": float(np.mean(estimates > 0.0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    scenario = next(item for item in COST_SCENARIOS if item.name == "current_liquidity")
    paths: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sample in ("normal", "proxy"):
        sample_name = str(DEFINITIONS[sample]["sample"])
        baseline_settings = with_strategy_inputs(samples[sample_name], BASELINES[sample])
        candidate_settings = with_strategy_inputs(samples[sample_name], CANDIDATES[sample])
        baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
        candidate, _ = r39.simulate_candidate(candidate_settings, scenario)
        paths[sample] = (baseline, candidate)

    rows: list[dict[str, float | int | str]] = []
    for period, (sample, start, end) in PERIODS.items():
        baseline, candidate = paths[sample]
        common = baseline.index.intersection(candidate.index)
        selected = common[(common >= start) & (common <= end)]
        for block_days in BLOCK_DAYS:
            rows.append(
                {
                    "period": period,
                    "sample": sample,
                    **paired_circular_block_bootstrap(
                        baseline.loc[selected, "net_return"],
                        candidate.loc[selected, "net_return"],
                        block_days=block_days,
                    ),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "bootstrap.csv", index=False)

    complete = results[results["period"].isin(["normal_complete", "proxy_complete"])]
    proxy = results[results["period"].eq("proxy_complete")]
    holdout = results[results["period"].eq("normal_holdout")]
    gates = {
        "all_observed_relative_returns_positive": bool(
            results["observed_annualized_relative_log_return"].gt(0.0).all()
        ),
        "complete_probability_at_least_95pct": bool(
            complete["probability_positive"].ge(0.95).all()
        ),
        "proxy_lower_95_nonnegative": bool(proxy["lower_95"].ge(0.0).all()),
        "holdout_probability_at_least_50pct": bool(
            holdout["probability_positive"].ge(0.5).all()
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "gates": gates,
        "path_uncertainty_support": bool(all(gates.values())),
        "decision": (
            "retain_as_shadow_candidate"
            if all(gates.values())
            else "insufficient_path_uncertainty_support"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results.round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
