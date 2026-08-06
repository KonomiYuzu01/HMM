from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DESTINATION = Path("output/forward_monitoring")
CORE_FILES = [
    Path("src/regime_strategy/model.py"),
    Path("src/regime_strategy/features.py"),
    Path("src/regime_strategy/portfolio.py"),
    Path("src/regime_strategy/backtest.py"),
    Path("src/regime_strategy/ensemble.py"),
    Path("src/regime_strategy/data.py"),
]
CONFIG_FILES = [
    Path("config/paper_core_tail_only_asset_risk_vix_hedge.yaml"),
    Path("config/paper_core_growth_gold20_daily_risk_ensemble.yaml"),
]
EVIDENCE_FILES = [
    Path("output/lev110_trendlev_industrymom_vixhedge4_validation/acceptance.csv"),
    Path(
        "output/lev110_trendlev_industrymom_vixhedge4_validation/"
        "family_reality_check.csv"
    ),
    Path(
        "output/lev110_trendlev_industrymom_vixhedge4_validation/"
        "metrics_by_period_and_cost.csv"
    ),
    Path(
        "output/lev110_trendlev_industrymom_vixhedge4_validation/"
        "tail_bootstrap.csv"
    ),
    Path("output/strategy_replacement_report_2026-07-23.md"),
    Path("output/current_operational_panel/panel.md"),
    Path("output/current_operational_panel/order_plan.csv"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-07-23-v2")
    args = parser.parse_args()
    files = [*CORE_FILES, *CONFIG_FILES, *EVIDENCE_FILES]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing release files: {missing}")
    manifest = {
        "release": args.release,
        "forward_signal_date": "2026-07-22",
        "first_execution_date": "2026-07-23",
        "files": {str(path): sha256(path) for path in files},
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / f"freeze_manifest_{args.release}.json"
    if path.exists():
        stored = path.read_text(encoding="utf-8")
        if stored != serialized:
            raise RuntimeError(
                f"Frozen release {args.release} conflicts with current files; "
                "use a new release name instead of overwriting history"
            )
    else:
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"Artifacts: {path.resolve()}")


if __name__ == "__main__":
    main()
