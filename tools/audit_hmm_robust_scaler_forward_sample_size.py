from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tools.evaluate_hmm_order_ensemble import DEFINITIONS, with_strategy_inputs
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_robust_scaler_forward_sample_size")
DIRECTORIES = {
    "normal": (
        Path("output/research_hmm_standard_restarts5"),
        Path("output/research_hmm_robust_restarts5"),
    ),
    "proxy": (
        Path("output/research_hmm_standard_restarts5_20y_proxy"),
        Path("output/research_hmm_robust_restarts5_20y_proxy"),
    ),
}
BLOCK_DAYS = 21


def required_blocks_for_mean(
    mean: float,
    standard_deviation: float,
    *,
    significance: float = 0.05,
    power: float = 0.80,
) -> int | None:
    if not 0.0 < significance < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("Significance and power must lie inside (0, 1)")
    effect = abs(float(mean))
    scale = float(standard_deviation)
    if effect <= 0.0:
        return None
    if scale <= 0.0:
        return 1
    critical = norm.ppf(1.0 - significance / 2.0) + norm.ppf(power)
    return max(1, int(math.ceil(np.square(critical * scale / effect))))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    scenario = next(item for item in COST_SCENARIOS if item.name == "current_liquidity")
    definitions = deepcopy(DEFINITIONS)
    summaries: dict[str, dict[str, float | int | None]] = {}
    for sample, (baseline_directory, candidate_directory) in DIRECTORIES.items():
        sample_name = str(definitions[sample]["sample"])
        baseline_settings = with_strategy_inputs(samples[sample_name], baseline_directory)
        candidate_settings = with_strategy_inputs(samples[sample_name], candidate_directory)
        baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
        candidate, _ = r39.simulate_candidate(candidate_settings, scenario)
        common = baseline.index.intersection(candidate.index)
        relative = (
            np.log1p(candidate.loc[common, "net_return"])
            - np.log1p(baseline.loc[common, "net_return"])
        )
        complete_count = len(relative) // BLOCK_DAYS * BLOCK_DAYS
        complete = relative.iloc[:complete_count]
        blocks = complete.groupby(np.arange(complete_count) // BLOCK_DAYS).sum()
        block_mean = float(blocks.mean())
        block_standard_deviation = float(blocks.std(ddof=1))
        block_standard_error = block_standard_deviation / math.sqrt(len(blocks))
        required_blocks = required_blocks_for_mean(
            block_mean, block_standard_deviation
        )
        annual_factor = 252.0 / BLOCK_DAYS
        summaries[sample] = {
            "observations": len(relative),
            "complete_blocks": len(blocks),
            "observed_annualized_relative_log_return": float(relative.mean() * 252.0),
            "block_mean": block_mean,
            "block_standard_deviation": block_standard_deviation,
            "normal_approx_annualized_lower_95": float(
                (block_mean - 1.96 * block_standard_error) * annual_factor
            ),
            "normal_approx_annualized_upper_95": float(
                (block_mean + 1.96 * block_standard_error) * annual_factor
            ),
            "required_blocks_for_80pct_power": required_blocks,
            "required_years_for_80pct_power": (
                float(required_blocks * BLOCK_DAYS / 252.0)
                if required_blocks is not None
                else None
            ),
        }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_planning_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "block_days": BLOCK_DAYS,
        "significance_two_sided": 0.05,
        "power": 0.80,
        "warning": "Historical effect size is uncertain; required years are planning estimates, not guarantees.",
        "samples": summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
