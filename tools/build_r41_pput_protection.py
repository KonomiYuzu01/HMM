from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from regime_strategy.pput_protected_capacity import (
    PputProtectedCapacityParameters,
    map_put_contracts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/paper_core_growth_gold20_r41_pput_protected_capacity.yaml"
R40_SPEC = ROOT / "output/paper_core_growth_gold20_r40_recursive_trend_cushion/account_protection_spec.json"
R40_AUDIT = ROOT / "output/r40_production_qualification_audit/summary.json"
RESEARCH = ROOT / "output/r41_pput_production_candidate"
OUTPUT = ROOT / "output/paper_core_growth_gold20_r41_pput_protected_capacity"
EXAMPLE_EQUITY = 500_000.0
EXAMPLE_SPYM_PRICE = 90.52


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    r40 = json.loads(R40_SPEC.read_text())
    r40_audit = json.loads(R40_AUDIT.read_text())
    research = json.loads((RESEARCH / "summary.json").read_text())
    research_audit = json.loads((RESEARCH / "independent_audit.json").read_text())
    values = config["parameters"]
    parameters = PputProtectedCapacityParameters(
        floor_drawdown=float(values["floor_drawdown"]),
        bull_multiplier=float(values["dual_trend_positive_multiplier"]),
        bear_multiplier=float(values["other_trend_multiplier"]),
        tier_size=float(values["non_cash_cap_tier"]),
        normal_non_cash_cap=float(values["normal_non_cash_cap"]),
        target_put_coverage=float(values["target_put_coverage"]),
        minimum_put_coverage=float(values["minimum_put_coverage"]),
        maximum_put_coverage=float(values["maximum_put_coverage"]),
        option_multiplier=int(config["protection"]["option_multiplier"]),
    )
    expected_candidate = {
        "overlay_notional": parameters.target_put_coverage,
        "annual_implementation_drag": float(values["research_annual_implementation_drag"]),
        "normal_non_cash_cap": parameters.normal_non_cash_cap,
        "floor_drawdown": parameters.floor_drawdown,
        "bull_multiplier": parameters.bull_multiplier,
        "bear_multiplier": parameters.bear_multiplier,
        "tier_size": parameters.tier_size,
    }
    if any(
        abs(float(research["candidate"][key]) - value) > 1e-12
        for key, value in expected_candidate.items()
    ):
        raise RuntimeError("Frozen R41 parameters do not match the qualified candidate")
    if not research["research_pass"] or not research_audit["audit_pass"]:
        raise RuntimeError("R41 research or independent audit did not pass")
    if not r40_audit["production_qualification_pass"]:
        raise RuntimeError("R41 requires a qualified R40 fail-closed parent")
    if not (
        r40["data_is_fresh"]
        and r40["price_as_of"] == r40["expected_price_as_of"]
        and r40["price_as_of"] == r40_audit["price_as_of"]
    ):
        raise RuntimeError("R41 parent is not a same-date fresh R40 snapshot")
    frozen_history = ROOT / config["validation"]["frozen_pput_history"]
    if sha256(frozen_history) != config["validation"]["frozen_pput_sha256"]:
        raise RuntimeError("Frozen PPUT history hash does not match the config")
    mapping = map_put_contracts(
        account_equity=EXAMPLE_EQUITY,
        underlying_price=EXAMPLE_SPYM_PRICE,
        parameters=parameters,
    )
    specification = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": r40["price_as_of"],
        "expected_price_as_of": r40["expected_price_as_of"],
        "data_is_fresh": True,
        "parent_strategy": "R40",
        "parent_release": r40["release"],
        "parameters": expected_candidate,
        "protection": config["protection"],
        "example_contract_mapping": {
            "account_equity": mapping.account_equity,
            "spym_price": mapping.underlying_price,
            "contracts": mapping.contracts,
            "implemented_coverage": mapping.implemented_coverage,
            "coverage_qualified": mapping.coverage_qualified,
            "quote_evidence_as_of": "2026-08-06 12:59 ET; delayed Cboe chain",
        },
        "production_eligible": False,
        "orders_executable": False,
        "order_blockers": [
            "confirmed live SPYM put holding is required before the 120% cap can activate",
            "live option bid-ask and account option permission require human review",
            "automatic broker orders and automatic exercise are disabled",
        ],
        "source_hashes": {
            str(CONFIG): sha256(CONFIG),
            str(R40_SPEC): sha256(R40_SPEC),
            str(R40_AUDIT): sha256(R40_AUDIT),
            str(RESEARCH / "summary.json"): sha256(RESEARCH / "summary.json"),
            str(RESEARCH / "independent_audit.json"): sha256(RESEARCH / "independent_audit.json"),
            str(frozen_history): sha256(frozen_history),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protection_spec.json").write_text(
        json.dumps(specification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(specification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
