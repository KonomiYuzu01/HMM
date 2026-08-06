from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_hmm_order_ensemble import DEFINITIONS, with_strategy_inputs
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_robust_scaler_seed_robustness")
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
SEEDS = (7, 42, 123)


def seed_period_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"sample", "period", "seed", "cagr_delta", "max_drawdown_delta"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Seed metrics are missing columns: {sorted(required - set(metrics))}")
    return (
        metrics.groupby(["sample", "period"])
        .agg(
            seeds=("seed", "nunique"),
            positive_cagr_fraction=("cagr_delta", lambda values: values.ge(0.0).mean()),
            median_cagr_delta=("cagr_delta", "median"),
            minimum_cagr_delta=("cagr_delta", "min"),
            worst_max_drawdown_delta=("max_drawdown_delta", "min"),
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    scenario = next(item for item in COST_SCENARIOS if item.name == "current_liquidity")
    definitions = deepcopy(DEFINITIONS)
    rows: list[dict[str, object]] = []
    for sample, (baseline_directory, candidate_directory) in DIRECTORIES.items():
        sample_name = str(definitions[sample]["sample"])
        periods = definitions[sample]["periods"]
        assert isinstance(periods, dict)
        for seed in SEEDS:
            member = f"seed_{seed}"
            baseline_settings = with_strategy_inputs(
                samples[sample_name], baseline_directory / "members" / member
            )
            candidate_settings = with_strategy_inputs(
                samples[sample_name], candidate_directory / "members" / member
            )
            baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
            candidate, _ = r39.simulate_candidate(candidate_settings, scenario)
            common = baseline.index.intersection(candidate.index)
            for period, (start, end) in periods.items():
                selected = common[(common >= start) & (common <= end)]
                rows.append(
                    {
                        "sample": sample,
                        "seed": seed,
                        "period": period,
                        **metric_delta(
                            baseline.loc[selected, "net_return"],
                            candidate.loc[selected, "net_return"],
                        ),
                    }
                )

    metrics = pd.DataFrame(rows)
    summary_frame = seed_period_summary(metrics)
    metrics.to_csv(OUTPUT / "metrics_by_seed.csv", index=False)
    summary_frame.to_csv(OUTPUT / "period_summary.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "seeds": list(SEEDS),
        "all_core_periods_majority_positive": bool(
            summary_frame.loc[
                ~summary_frame["period"].str.startswith("post_holdout"),
                "positive_cagr_fraction",
            ]
            .ge(2.0 / 3.0)
            .all()
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.round(6).to_string(index=False))
    print("\nSeed summary:")
    print(summary_frame.round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
