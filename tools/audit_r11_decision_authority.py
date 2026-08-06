from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_strategy.decision_authority import validate_authority_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config/paper_core_growth_gold20_r11_decision_authority.yaml"
)
OUTPUT = ROOT / "output/paper_core_growth_gold20_r11_decision_authority"
AUDIT_OUTPUT = ROOT / "output/r11_decision_authority_audit"


@dataclass(frozen=True)
class AuditRecord:
    problem: str
    requirement: str
    status: str
    evidence: str


def build_audit() -> pd.DataFrame:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_authority_registry(config["authorities"])
    payload = json.loads(
        (OUTPUT / "decision_authority.json").read_text(encoding="utf-8")
    )
    proof = pd.read_csv(
        OUTPUT / "target_authority_proof.csv",
        index_col="asset",
    )
    cross_asset = pd.read_csv(
        ROOT / "output/r11_cross_asset_shock_validation/acceptance.csv"
    )
    entry_metrics = pd.read_csv(
        ROOT / "output/r11_entry_permission/metrics_by_period.csv"
    )
    rows: list[AuditRecord] = []

    def add(
        problem: str,
        requirement: str,
        passed: bool,
        evidence: str,
    ) -> None:
        rows.append(
            AuditRecord(
                problem=problem,
                requirement=requirement,
                status="pass" if passed else "fail",
                evidence=evidence,
            )
        )

    add(
        "1_market_mechanisms",
        "Market environment is explicit and informational only",
        bool(
            payload["market_environment"]
            in {
                "healthy",
                "acute_liquidity_shock",
                "ordinary_correction",
                "structural_damage",
                "early_repair",
                "incomplete",
            }
            and payload["market_environment_informational_only"]
        ),
        f"environment={payload['market_environment']}",
    )
    add(
        "1_market_mechanisms",
        "Classification uses no date after the production price date",
        bool(
            payload["environment_inputs"]["signal_max_date"]
            <= payload["price_as_of"]
        ),
        (
            f"signal={payload['environment_inputs']['signal_max_date']}; "
            f"price={payload['price_as_of']}"
        ),
    )
    add(
        "2_exit_vs_entry",
        "Approved exit and rejected entry rules have separate authority types",
        bool(
            config["authorities"]["smh_gap_exit"]["kind"]
            == "exit_only"
            and all(
                config["authorities"][name]["kind"] == "entry_only"
                for name in (
                    "selective_reentry_brake",
                    "medium_damage_budget10",
                    "hierarchical_permission_10_35_full",
                )
            )
        ),
        "smh_gap_exit=exit_only; three experiments=entry_only",
    )
    add(
        "2_exit_vs_entry",
        "Every unapproved rule controls zero real capital",
        bool(
            all(
                float(rule["maximum_managed_capital_share"]) == 0.0
                for rule in config["authorities"].values()
                if rule["status"] != "approved"
            )
            and float(payload["unapproved_real_capital_share"]) == 0.0
        ),
        f"blocked={','.join(payload['blocked_authorities'])}",
    )
    add(
        "3_signal_vs_trade",
        "State changes cannot emit orders",
        bool(
            not config["execution"]["state_changes_can_emit_orders"]
            and not payload["state_change_orders_allowed"]
        ),
        "state_change_orders_allowed=false",
    )
    maximum_difference = float(proof["difference"].abs().max())
    add(
        "3_signal_vs_trade",
        "Authority layer does not alter the approved R11 economic target",
        maximum_difference <= 1e-12,
        f"maximum absolute target difference={maximum_difference:.3e}",
    )
    add(
        "3_signal_vs_trade",
        "All final and base target portfolios sum to 100%",
        bool(
            abs(proof["r11_approved_base_target"].sum() - 1.0)
            <= 1e-12
            and abs(
                proof["authority_controlled_final_target"].sum() - 1.0
            )
            <= 1e-12
        ),
        (
            f"base={proof['r11_approved_base_target'].sum():.12f}; "
            "final="
            f"{proof['authority_controlled_final_target'].sum():.12f}"
        ),
    )
    cross_asset_pass = bool(
        cross_asset.loc[
            cross_asset["gate"].eq("all_gates_pass"),
            "passed",
        ].iloc[0]
    )
    add(
        "4_rare_event_evidence",
        "Failed independent cross-asset evidence is recorded as a blocker",
        bool(
            not cross_asset_pass
            and config["authorities"]["selective_reentry_brake"][
                "status"
            ]
            == "rejected"
        ),
        "cross_asset_all_gates_pass=false; selective_reentry_brake=rejected",
    )
    complete_rows = entry_metrics[
        entry_metrics["rollout_share"].eq(0.25)
        & entry_metrics["period"].isin(
            [
                "complete_2015_2026",
                "complete_2006_2026",
                "live_2022_2026",
            ]
        )
    ]
    add(
        "4_rare_event_evidence",
        "Negative broad permission results cannot be promoted",
        bool(
            not complete_rows.empty
            and complete_rows["cagr_delta"].lt(0.0).all()
            and config["authorities"][
                "hierarchical_permission_10_35_full"
            ]["status"]
            == "rejected"
        ),
        (
            "maximum staged CAGR delta="
            f"{complete_rows['cagr_delta'].max():.6f}"
        ),
    )
    add(
        "4_rare_event_evidence",
        "Production remains executable only if the underlying R11 allows it",
        not bool(payload["orders_executable"]),
        f"orders_executable={payload['orders_executable']}",
    )
    return pd.DataFrame(asdict(row) for row in rows)


def main() -> None:
    audit = build_audit()
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT / "requirements.csv", index=False)
    print(audit.to_string(index=False))
    failures = audit[audit["status"].eq("fail")]
    if not failures.empty:
        raise SystemExit(1)
    print(
        "\nR11 decision-authority audit: PASS "
        "(economic target unchanged; unapproved research fails closed)"
    )


if __name__ == "__main__":
    main()
