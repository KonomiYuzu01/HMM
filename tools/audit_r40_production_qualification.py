from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from regime_strategy.recursive_trend_cushion import (
    RecursiveTrendCushionParameters,
    calculate_recursive_trend_cushion,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/paper_core_growth_gold20_r40_recursive_trend_cushion"
RESEARCH = ROOT / "output/recursive_trend_cushion_production_candidate"
AUDIT_OUTPUT = ROOT / "output/r40_production_qualification_audit"


def main() -> None:
    spec = json.loads((OUTPUT / "account_protection_spec.json").read_text())
    research = json.loads((RESEARCH / "summary.json").read_text())
    independent = json.loads((RESEARCH / "independent_audit.json").read_text())
    parameters = RecursiveTrendCushionParameters(**spec["parameters"])
    high_water = calculate_recursive_trend_cushion(
        prior_equity=100.0,
        prior_peak=100.0,
        dual_trend_positive=True,
        parameters=parameters,
    )
    bear_drawdown = calculate_recursive_trend_cushion(
        prior_equity=85.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        parameters=parameters,
    )
    floor_state = calculate_recursive_trend_cushion(
        prior_equity=81.0,
        prior_peak=100.0,
        dual_trend_positive=False,
        parameters=parameters,
    )
    gates = {
        "research_pass": bool(research["research_pass"]),
        "independent_research_audit_pass": bool(independent["audit_pass"]),
        "same_date_parent_pass": spec["price_as_of"] == spec["expected_price_as_of"],
        "fresh_parent_pass": bool(spec["data_is_fresh"]),
        "high_water_full_participation_pass": high_water.accepted_non_cash_cap == 1.0,
        "bear_tier_formula_pass": abs(bear_drawdown.accepted_non_cash_cap - 0.40) < 1e-12,
        "floor_exhaustion_pass": floor_state.accepted_non_cash_cap == 0.0,
        "private_state_required_pass": spec["account_state_location"] == "private authenticated holdings payload",
        "automatic_broker_orders_disabled": not spec["orders_executable"],
    }
    passed = all(gates.values())
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "production_qualification_pass": passed,
        "production_eligible": passed,
        "orders_generated": False,
        "price_as_of": spec["price_as_of"],
        "release": spec["release"],
        "gates": gates,
        "historical_pre2008_max_drawdown": independent["central_complete_pre2008_mdd"],
        "modern_cagr": independent["central_modern_cagr"],
        "modern_cagr_delta": independent["central_modern_cagr_delta"],
        "proxy_cagr_delta": independent["central_proxy_cagr_delta"],
        "limitation": "The 20% ceiling is a historical close-to-close result, not a guarantee against future gaps or execution slippage.",
    }
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    (AUDIT_OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("R40 production qualification failed")


if __name__ == "__main__":
    main()
