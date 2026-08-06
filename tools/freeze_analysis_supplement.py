from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DESTINATION = Path("output/forward_monitoring")
FILES = [
    Path("data/adjusted_open_close_2011_present.csv"),
    Path("output/open_execution_validation/metrics.csv"),
    Path("output/open_execution_validation/growth_primary_daily.csv"),
    Path("output/open_execution_validation/gold_20_unlevered_daily.csv"),
    Path("output/open_execution_validation/gold_20_daily_asset_cap_daily.csv"),
    Path("output/open_execution_validation/gold_20_jump_aware_daily_cap_daily.csv"),
    Path("tools/evaluate_account_tail_budget.py"),
    Path("tools/evaluate_layer_ablation.py"),
    Path("tools/evaluate_layer_stress_episodes.py"),
    Path("tools/evaluate_rolling_benchmark_consistency.py"),
    Path("tests/test_evaluation_helpers.py"),
    Path("output/layer_ablation/metrics.csv"),
    Path("output/layer_ablation/incremental_effects.csv"),
    Path("output/layer_ablation/implementation.csv"),
    Path("output/layer_ablation/period_metrics.csv"),
    Path("output/layer_ablation/start_year_metrics.csv"),
    Path("output/layer_ablation/start_year_checks.csv"),
    Path("output/layer_ablation/decision.md"),
    Path("output/account_tail_budget/summary.csv"),
    Path("output/account_tail_budget/weights.csv"),
    Path("output/account_tail_budget/bootstrap_summary.csv"),
    Path("output/account_tail_budget/monte_carlo_stability.csv"),
    Path("output/account_tail_budget/decision.md"),
    Path("output/layer_stress_episodes/episode_metrics.csv"),
    Path("output/layer_stress_episodes/worst_rolling_returns.csv"),
    Path("output/layer_stress_episodes/decision.md"),
    Path("output/rolling_benchmark_consistency/summary.csv"),
    Path("output/rolling_benchmark_consistency/rolling_detail.csv"),
    Path("output/rolling_benchmark_consistency/decision.md"),
    Path("output/strategy_improvement_report.md"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-07-22-v2-supplement")
    args = parser.parse_args()
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing analysis files: {missing}")
    manifest = {
        "release": args.release,
        "parent_release": "2026-07-22-v1",
        "analysis_as_of": "2026-07-22",
        "files": {str(path): sha256(path) for path in FILES},
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / f"analysis_manifest_{args.release}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                f"Frozen supplement {args.release} conflicts with current files; "
                "use a new release name instead of overwriting history"
            )
    else:
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"Artifacts: {path.resolve()}")


if __name__ == "__main__":
    main()
