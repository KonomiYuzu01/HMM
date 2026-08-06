from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from regime_strategy.pput_protected_capacity import (
    PputProtectedCapacityParameters,
    calculate_pput_protected_capacity,
    map_put_contracts,
    select_five_percent_otm_strike,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "output/paper_core_growth_gold20_r41_pput_protected_capacity/protection_spec.json"
RESEARCH = ROOT / "output/r41_pput_production_candidate"
AUDIT_OUTPUT = ROOT / "output/r41_production_qualification_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    research = json.loads((RESEARCH / "summary.json").read_text())
    independent = json.loads((RESEARCH / "independent_audit.json").read_text())
    candidate = research["candidate"]
    parameters = PputProtectedCapacityParameters(
        floor_drawdown=float(candidate["floor_drawdown"]),
        bull_multiplier=float(candidate["bull_multiplier"]),
        bear_multiplier=float(candidate["bear_multiplier"]),
        tier_size=float(candidate["tier_size"]),
        normal_non_cash_cap=float(candidate["normal_non_cash_cap"]),
        target_put_coverage=float(candidate["overlay_notional"]),
    )
    protected = calculate_pput_protected_capacity(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
        hedge_confirmed=True,
        parameters=parameters,
    )
    unhedged = calculate_pput_protected_capacity(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
        hedge_confirmed=False,
        parameters=parameters,
    )
    controlled = calculate_pput_protected_capacity(
        prior_equity=85.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        hedge_confirmed=True,
        parameters=parameters,
    )
    floor_state = calculate_pput_protected_capacity(
        prior_equity=81.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        hedge_confirmed=True,
        parameters=parameters,
    )
    mapping = map_put_contracts(
        account_equity=500_000.0,
        underlying_price=90.52,
        parameters=parameters,
    )
    strike = select_five_percent_otm_strike(
        underlying_price=90.52,
        listed_put_strikes=[85.0, 86.0, 87.0, 88.0, 89.0, 90.0],
    )
    basis_risk = str(spec["protection"]["basis_risk"])
    gates = {
        "research_pass": bool(research["research_pass"]),
        "independent_research_audit_pass": bool(independent["audit_pass"]),
        "same_date_parent_pass": spec["price_as_of"] == spec["expected_price_as_of"],
        "fresh_parent_pass": bool(spec["data_is_fresh"]),
        "modern_cagr_target_pass": 0.24 <= float(research["modern_cagr"]) <= 0.25,
        "modern_drawdown_pass": float(research["modern_max_drawdown"]) >= -0.20,
        "pre2008_drawdown_pass": float(research["pre2008_max_drawdown"]) >= -0.20,
        "qualified_hedge_extends_to_120_pass": protected.accepted_non_cash_cap == 1.20,
        "missing_hedge_fails_closed_pass": unhedged.accepted_non_cash_cap == 1.0,
        "controlled_formula_pass": abs(controlled.accepted_non_cash_cap - 0.90) < 1e-12,
        "floor_exhaustion_pass": floor_state.accepted_non_cash_cap == 0.0,
        "integer_contract_coverage_pass": mapping.contracts == 3 and mapping.coverage_qualified,
        "five_percent_otm_strike_pass": strike == 85.0,
        "basis_risk_disclosed_pass": all(
            word in basis_risk for word in ("American-style", "PPUT", "SPX")
        ),
        "frozen_pput_hash_pass": research["pput_sha256"] == sha256(
            ROOT / research["pput_path"]
        ),
        "automatic_broker_orders_disabled": not spec["orders_executable"],
    }
    passed = bool(all(gates.values()))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "production_qualification_pass": passed,
        "production_eligible": passed,
        "production_changed": passed,
        "orders_generated": False,
        "price_as_of": spec["price_as_of"],
        "release": spec["release"],
        "gates": gates,
        "modern_cagr": research["modern_cagr"],
        "modern_max_drawdown": research["modern_max_drawdown"],
        "proxy_cagr": research["proxy_cagr"],
        "proxy_max_drawdown": research["proxy_max_drawdown"],
        "historical_pre2008_max_drawdown": research["pre2008_max_drawdown"],
        "example_500k_contracts": mapping.contracts,
        "example_500k_coverage": mapping.implemented_coverage,
        "activation_rule": "R41 activates only after a confirmed qualified SPYM long-put holding; otherwise R40 remains active.",
        "limitation": "The 20% ceiling is historical close-to-close evidence, not a guarantee against future gaps, option-basis differences, or execution slippage.",
    }
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    (AUDIT_OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("R41 production qualification failed")


if __name__ == "__main__":
    main()
