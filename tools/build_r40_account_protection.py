from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/paper_core_growth_gold20_r40_recursive_trend_cushion.yaml"
R38 = ROOT / "output/paper_core_growth_gold20_r38_convex_overlay"
R39_REFRESH = ROOT / "output/r39_production_refresh/latest.json"
RESEARCH = ROOT / "output/recursive_trend_cushion_production_candidate"
OUTPUT = ROOT / "output/paper_core_growth_gold20_r40_recursive_trend_cushion"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    r38_metadata = json.loads((R38 / "run_metadata.json").read_text())
    refresh = json.loads(R39_REFRESH.read_text())
    research = json.loads((RESEARCH / "summary.json").read_text())
    research_audit = json.loads((RESEARCH / "independent_audit.json").read_text())
    diagnostics = pd.read_csv(
        R38 / "next_signal_diagnostics.csv", index_col=0
    ).iloc[:, 0]
    price_as_of = str(r38_metadata["price_as_of"])
    expected = str(r38_metadata["expected_price_as_of"])
    if not (
        r38_metadata["data_is_fresh"]
        and price_as_of == expected == str(refresh["price_as_of"])
        and refresh["status"] == "complete"
        and refresh["active_strategy"] == "R38"
    ):
        raise RuntimeError("R40 parent is not a same-date fresh qualified R38 snapshot")
    candidate = research["candidate"]
    expected_candidate = {
        "floor_drawdown": float(config["parameters"]["floor_drawdown"]),
        "bull_multiplier": float(config["parameters"]["dual_trend_positive_multiplier"]),
        "bear_multiplier": float(config["parameters"]["other_trend_multiplier"]),
        "tier_size": float(config["parameters"]["non_cash_cap_tier"]),
    }
    if any(float(candidate[key]) != value for key, value in expected_candidate.items()):
        raise RuntimeError("Frozen R40 parameters do not match the research candidate")
    if not research["research_pass"] or not research_audit["audit_pass"]:
        raise RuntimeError("R40 research or independent audit did not pass")

    specification = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": price_as_of,
        "expected_price_as_of": expected,
        "data_is_fresh": True,
        "parent_strategy": "R38",
        "parent_release": r38_metadata["release"],
        "parameters": expected_candidate,
        "dual_trend_positive": bool(int(float(diagnostics["trend_permission"]))),
        "account_state_location": "private authenticated holdings payload",
        "first_activation_rule": "initialize high-water to confirmed managed-account equity",
        "orders_executable": False,
        "order_blockers": [
            "account-specific R40 state must be confirmed in the private panel",
            "broker orders require a separate human-reviewed execution step",
        ],
        "source_hashes": {
            str(CONFIG): sha256(CONFIG),
            str(R38 / "run_metadata.json"): sha256(R38 / "run_metadata.json"),
            str(RESEARCH / "summary.json"): sha256(RESEARCH / "summary.json"),
            str(RESEARCH / "independent_audit.json"): sha256(RESEARCH / "independent_audit.json"),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "account_protection_spec.json").write_text(
        json.dumps(specification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(specification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
