from __future__ import annotations

import hashlib
import json
from pathlib import Path


RELEASE = "2026-07-25-r8-v3-monitoring-v3"
PARENT_RELEASE = "2026-07-25-r8-v3-monitoring-v2"
DESTINATION = Path("output/forward_monitoring")
FILES = [
    Path("output/regime_strategy_research_2026-07-25/r8_monitoring_policy.md"),
    Path(
        "output/regime_strategy_research_2026-07-25/"
        "r8_long_history_robustness/fixed_metrics.csv"
    ),
    Path(
        "output/regime_strategy_research_2026-07-25/"
        "r8_long_history_robustness/rolling_summary.csv"
    ),
    Path(
        "output/regime_strategy_research_2026-07-25/"
        "r8_long_history_robustness/stress_period_metrics.csv"
    ),
    Path(
        "output/regime_strategy_research_2026-07-25/"
        "r8_factorial_attribution/annual_layer_attribution.csv"
    ),
    Path(
        "output/regime_strategy_research_2026-07-25/"
        "r8_factorial_attribution/layer_summary.csv"
    ),
    Path("tools/evaluate_r8_long_history_robustness.py"),
    Path("tools/evaluate_r8_factorial_attribution.py"),
    Path("tools/export_r8_effective_parameters.py"),
    Path("output/current_operational_panel/strategy_parameters.csv"),
    Path("output/regime_strategy_research_2026-07-25/final_decision.md"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing monitoring files: {missing}")
    manifest = {
        "release": RELEASE,
        "parent_release": PARENT_RELEASE,
        "analysis_as_of": "2026-07-25",
        "files": {str(path): sha256(path) for path in FILES},
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / f"analysis_manifest_{RELEASE}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                f"Frozen supplement {RELEASE} conflicts; use a new release"
            )
    else:
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"Artifacts: {path.resolve()}")


if __name__ == "__main__":
    main()
