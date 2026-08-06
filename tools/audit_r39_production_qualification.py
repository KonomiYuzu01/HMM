from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "config/paper_core_growth_gold20_r39_relative_damage_veto.yaml"
)
PRODUCTION = (
    ROOT / "output/paper_core_growth_gold20_r39_relative_damage_veto"
)
FINAL_AUDIT = ROOT / "output/r39_final_candidate_audit"
PARENT_QUALIFICATION = (
    ROOT / "output/r38_production_qualification_audit"
)
PARENT_PRODUCTION = (
    ROOT / "output/paper_core_growth_gold20_r38_convex_overlay"
)
OUTPUT = ROOT / "output/r39_production_qualification_audit"
VARIANTS = ("lower", "center", "upper")


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    passed: bool
    evidence: str
    source: str


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def target_columns_sum_to_one(
    targets: pd.DataFrame,
    tolerance: float = 1e-12,
) -> bool:
    return bool(
        np.allclose(
            targets[
                [f"r39_account_{variant}" for variant in VARIANTS]
            ].sum(axis=0),
            1.0,
            atol=tolerance,
        )
    )


def non_growth_assets_unchanged(
    targets: pd.DataFrame,
    tolerance: float = 1e-12,
) -> bool:
    assets = [
        asset
        for asset in targets.index
        if asset not in ("QQQ", "SEMIS")
    ]
    return all(
        np.allclose(
            targets.loc[
                assets,
                f"r38_staged_account_{variant}",
            ],
            targets.loc[assets, f"r39_account_{variant}"],
            atol=tolerance,
        )
        for variant in VARIANTS
    )


def build_audit() -> pd.DataFrame:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = dict(config["relative_damage_concentration_veto"])
    targets = pd.read_csv(
        PRODUCTION / "next_target_weights.csv",
        index_col="asset",
    )
    diagnostics = pd.read_csv(
        PRODUCTION / "next_signal_diagnostics.csv",
        index_col=0,
    )["value"]
    metadata = json.loads(
        (PRODUCTION / "run_metadata.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (FINAL_AUDIT / "summary.json").read_text(encoding="utf-8")
    )
    parent = json.loads(
        (PARENT_QUALIFICATION / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    parent_targets = pd.read_csv(
        PARENT_PRODUCTION / "next_target_weights.csv",
        index_col="asset",
    )
    rows: list[AuditRecord] = []

    def add(
        category: str,
        requirement: str,
        passed: bool,
        evidence: str,
        source: str,
    ) -> None:
        rows.append(
            AuditRecord(
                category,
                requirement,
                bool(passed),
                evidence,
                source,
            )
        )

    add(
        "research",
        "Final R39 candidate qualification passed",
        bool(final["qualification_pass"]),
        (
            f"{final['requirements_passed']}/"
            f"{final['requirements']} requirements"
        ),
        "r39_final_candidate_audit/summary.json",
    )
    add(
        "parent",
        "Parent R38 production qualification remains valid",
        bool(parent["production_qualification_pass"]),
        (
            f"{parent['requirements_passed']}/"
            f"{parent['requirements']} requirements"
        ),
        "r38_production_qualification_audit/summary.json",
    )
    parent_targets_match = all(
        np.allclose(
            targets[f"r38_staged_account_{variant}"],
            parent_targets[f"staged_account_{variant}"],
            atol=1e-12,
        )
        for variant in VARIANTS
    )
    add(
        "parent",
        "Embedded R38 staged targets exactly match the qualified parent",
        parent_targets_match,
        f"variants={len(VARIANTS)}",
        "R38 and R39 next_target_weights.csv",
    )
    fixed = bool(
        int(policy["lookback_trading_days"]) == 21
        and float(policy["account_relative_loss_budget"]) == 0.03
        and float(policy["minimum_smh_share_of_growth_sleeve"])
        == 0.60
        and float(
            policy["maximum_active_smh_share_of_growth_sleeve"]
        )
        == 0.50
        and float(policy["maximum_share_activation_multiple"])
        == 1.10
        and bool(
            policy[
                "maximum_share_requires_volatility_acceleration_clear"
            ]
        )
        and str(policy["overflow_asset"]) == "QQQ"
        and bool(policy["preserve_qqq_plus_smh_growth_budget"])
    )
    add(
        "configuration",
        "Frozen R39 production parameters match the qualified candidate",
        fixed,
        (
            "lookback=21; budget=3%; trigger share=60%; "
            "hard trigger=1.1x after volatility acceleration clears; "
            "active cap=50%; overflow=QQQ"
        ),
        str(CONFIG.relative_to(ROOT)),
    )
    add(
        "targets",
        "Every R39 target variant sums to 100%",
        target_columns_sum_to_one(targets),
        "; ".join(
            f"{variant}={targets[f'r39_account_{variant}'].sum():.12f}"
            for variant in VARIANTS
        ),
        "R39 next_target_weights.csv",
    )
    add(
        "scope",
        "R39 changes only QQQ and SMH",
        non_growth_assets_unchanged(targets),
        "all non-growth target differences are zero",
        "R39 next_target_weights.csv",
    )
    growth_error = float(diagnostics["growth_budget_error"])
    add(
        "risk_invariant",
        "QQQ plus SMH growth exposure is exactly conserved",
        bool(growth_error <= 1e-12),
        f"growth budget error={growth_error:.3e}",
        "R39 next_signal_diagnostics.csv",
    )
    proposed = float(diagnostics["proposed_account_relative_loss"])
    implemented = float(
        diagnostics["implemented_account_relative_loss"]
    )
    budget = float(diagnostics["account_relative_loss_budget"])
    active = bool(int(float(diagnostics["relative_damage_veto_active"])))
    maximum_share_active = bool(
        int(float(diagnostics["maximum_share_guard_active"]))
    )
    add(
        "risk_budget",
        "Active current target satisfies the declared account loss budget",
        bool(active and proposed > budget and implemented <= budget + 1e-12),
        (
            f"active={active}; proposed={proposed:.6f}; "
            f"implemented={implemented:.6f}; budget={budget:.6f}"
        ),
        "R39 diagnostics and production YAML",
    )
    implemented_share = float(
        diagnostics["implemented_semis_growth_share"]
    )
    maximum_active_share = float(
        diagnostics["maximum_active_smh_share_of_growth_sleeve"]
    )
    add(
        "risk_concentration",
        "Active current target keeps SMH at or below QQQ",
        bool(
            active
            and maximum_share_active
            and implemented_share <= maximum_active_share + 1e-12
            and maximum_active_share <= 0.50
        ),
        (
            f"active={active}; hard cap={maximum_share_active}; "
            f"SMH growth share={implemented_share:.6f}; "
            f"cap={maximum_active_share:.6f}"
        ),
        "R39 diagnostics and production YAML",
    )
    before_smh = float(
        targets.loc["SEMIS", "r38_staged_account_center"]
    )
    after_smh = float(targets.loc["SEMIS", "r39_account_center"])
    before_qqq = float(
        targets.loc["QQQ", "r38_staged_account_center"]
    )
    after_qqq = float(targets.loc["QQQ", "r39_account_center"])
    add(
        "transfer",
        "Removed SMH weight is transferred exactly to QQQ",
        bool(
            abs((before_smh - after_smh) - (after_qqq - before_qqq))
            <= 1e-12
        ),
        (
            f"SMH removed={before_smh - after_smh:.6f}; "
            f"QQQ added={after_qqq - before_qqq:.6f}"
        ),
        "R39 next_target_weights.csv",
    )
    add(
        "data_freshness",
        "R39 price, expected, and diagnostic dates are synchronized",
        bool(
            metadata["data_is_fresh"]
            and metadata["price_as_of"]
            == metadata["expected_price_as_of"]
            == diagnostics["price_as_of"]
        ),
        (
            f"price={metadata['price_as_of']}; "
            f"expected={metadata['expected_price_as_of']}"
        ),
        "R39 metadata and diagnostics",
    )
    add(
        "execution",
        "R39 inherits the parent next execution session",
        bool(metadata["next_session"] == diagnostics["next_session"]),
        f"next_session={metadata['next_session']}",
        "R39 metadata and diagnostics",
    )
    recorded_hashes = dict(metadata["source_hashes"])
    source_hashes_match = all(
        Path(path).exists()
        and file_digest(Path(path)) == str(digest)
        for path, digest in recorded_hashes.items()
    )
    add(
        "provenance",
        "All recorded configuration and parent source hashes match",
        bool(source_hashes_match),
        f"sources={len(recorded_hashes)}",
        "R39 run_metadata.json source_hashes",
    )
    add(
        "execution_safety",
        "Orders remain blocked without complete real positions",
        bool(
            not metadata["orders_executable"]
            and int(float(diagnostics["orders_executable"])) == 0
            and any(
                "complete real positions" in blocker
                for blocker in metadata["order_blockers"]
            )
        ),
        (
            f"orders_executable={metadata['orders_executable']}; "
            f"blockers={'; '.join(metadata['order_blockers'])}"
        ),
        "R39 run_metadata.json",
    )
    return pd.DataFrame(asdict(row) for row in rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    audit.to_csv(OUTPUT / "requirements.csv", index=False)
    passed = bool(audit["passed"].all())
    summary = {
        "release": str(
            yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["release"]
        ),
        "production_qualification_pass": passed,
        "requirements": len(audit),
        "requirements_passed": int(audit["passed"].sum()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    if not passed:
        raise SystemExit(1)
    print("\nR39 production qualification: PASS")
    print(f"Artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()
