from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/r38_production_qualification_audit"
CONFIG = (
    ROOT / "config/paper_core_growth_gold20_r38_convex_overlay.yaml"
)
PRODUCTION = (
    ROOT / "output/paper_core_growth_gold20_r38_convex_overlay"
)
FINAL_AUDIT = (
    ROOT
    / "output/r38_accelerating_volatility_1375_final_candidate_audit"
)
ANTI_OVERFIT_GOVERNANCE = (
    ROOT / "output/r38_anti_overfit_governance/summary.json"
)
PANEL = ROOT / "strategy-panel"


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    passed: bool
    evidence: str
    source: str


def staged_formula_matches(
    targets: pd.DataFrame,
    rollout_share: float,
    tolerance: float = 1e-12,
) -> bool:
    expected = (
        (1.0 - rollout_share)
        * targets["r11_reference_target"].astype(float)
        + rollout_share * targets["r38_full_center"].astype(float)
    )
    actual = targets["staged_account_center"].astype(float)
    return bool(np.allclose(expected, actual, atol=tolerance))


def target_columns_sum_to_one(
    targets: pd.DataFrame,
    tolerance: float = 1e-12,
) -> bool:
    columns = (
        "r38_full_lower",
        "r38_full_center",
        "r38_full_upper",
        "staged_account_lower",
        "staged_account_center",
        "staged_account_upper",
    )
    return bool(
        np.allclose(
            targets.loc[:, list(columns)].sum(axis=0),
            1.0,
            atol=tolerance,
        )
    )


def load_panel_payload(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    match = re.fullmatch(
        r"export const strategyLiveData = (.*) as const;\n?",
        source,
    )
    if match is None:
        raise ValueError("Panel live data does not have the expected format")
    return json.loads(match.group(1))


def build_audit(root: Path = ROOT) -> pd.DataFrame:
    config = yaml.safe_load(
        (root / CONFIG).read_text(encoding="utf-8")
    )
    production = root / PRODUCTION
    targets = pd.read_csv(
        production / "next_target_weights.csv",
        index_col="asset",
    )
    diagnostics = pd.read_csv(
        production / "next_signal_diagnostics.csv",
        index_col=0,
    )["value"]
    metadata = json.loads(
        (production / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    final = json.loads(
        (root / FINAL_AUDIT / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    governance = json.loads(
        (root / ANTI_OVERFIT_GOVERNANCE).read_text(encoding="utf-8")
    )
    panel = load_panel_payload(
        root / PANEL / "app/strategy-live-data.ts"
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
        "Final R38 candidate qualification passed",
        bool(final["qualification_pass"]),
        (
            f"{final['requirements_passed']}/"
            f"{final['requirements']} requirements"
        ),
        "r38_accelerating_volatility_1375_final_candidate_audit/summary.json",
    )
    add(
        "governance",
        "Anti-overfit freeze is operational and successor promotion is blocked",
        bool(
            governance["operational_pass"]
            and not governance["successor_promotion_allowed"]
            and float(governance["current_r38_share"]) == 0.25
        ),
        (
            f"forward={governance['forward_sessions']}/"
            f"{governance['minimum_forward_sessions']}; "
            f"R38 share={float(governance['current_r38_share']):.0%}; "
            "successors=blocked"
        ),
        "r38_anti_overfit_governance/summary.json",
    )
    risk = dict(config["risk_budget"])
    overlay = dict(config["semiconductor_overlay"])
    fixed_parameters = (
        float(risk["base_non_cash_multiplier"]) == 1.070
        and float(
            risk["stable_volatility_relative_multiplier"]
        )
        == 1.375
        and float(
            risk["accelerating_volatility_relative_multiplier"]
        )
        == 1.30
        and float(
            risk["one_session_shock_absolute_multiplier"]
        )
        == 1.00
        and float(risk["maximum_negative_cash_weight"]) == -0.20
        and float(risk["maximum_non_cash_weight"]) == 1.20
        and float(overlay["overlay_fraction"]) == 0.10
        and int(overlay["relative_momentum_lookback_trading_days"])
        == 126
        and int(overlay["update_interval_trading_days"]) == 21
    )
    add(
        "configuration",
        "Frozen production parameters match preregistered R38",
        fixed_parameters,
        (
            "base=1.070; stable=1.375; accelerating=1.30; "
            "shock=1.00; "
            "cash=-0.20; overlay=0.10; lookback/update=126/21"
        ),
        str(CONFIG.relative_to(ROOT)),
    )
    rollout_share = float(
        dict(config["rollout"])[
            "initial_r38_share_of_managed_capital"
        ]
    )
    add(
        "targets",
        "Every full and staged target sums to 100%",
        target_columns_sum_to_one(targets),
        "; ".join(
            f"{column}={targets[column].sum():.12f}"
            for column in (
                "r38_full_center",
                "staged_account_center",
            )
        ),
        "R38 next_target_weights.csv",
    )
    add(
        "rollout",
        "Staged target is exactly 25% R38 plus 75% R11",
        staged_formula_matches(targets, rollout_share),
        f"rollout_share={rollout_share:.2f}",
        "R38 next_target_weights.csv; production YAML",
    )
    full = targets["r38_full_center"].astype(float)
    non_cash = full.drop(index="CASH").sum()
    add(
        "risk_limits",
        "Full target enforces the cash and non-cash hard limits",
        bool(
            full["CASH"]
            >= float(risk["maximum_negative_cash_weight"]) - 1e-12
            and non_cash
            <= float(risk["maximum_non_cash_weight"]) + 1e-12
        ),
        (
            f"cash={full['CASH']:.6f}; "
            f"non_cash={non_cash:.6f}"
        ),
        "R38 next_target_weights.csv",
    )
    add(
        "causality",
        "Production diagnostics expose all causal control states",
        bool(
            set(
                (
                    "trend_permission",
                    "volatility_acceleration_block",
                    "state_active_multiplier",
                    "one_session_shock_active",
                    "accepted_absolute_multiplier",
                    "r38_semis_growth_share",
                    "smh_guard_active",
                    "cash_hard_limit_active",
                )
            ).issubset(diagnostics.index)
        ),
        (
            f"trend={diagnostics['trend_permission']}; "
            f"acceleration_block="
            f"{diagnostics['volatility_acceleration_block']}; "
            f"state_multiplier="
            f"{diagnostics['state_active_multiplier']}; "
            f"shock={diagnostics['one_session_shock_active']}; "
            f"multiplier={diagnostics['accepted_absolute_multiplier']}"
        ),
        "R38 next_signal_diagnostics.csv",
    )
    add(
        "data_freshness",
        "Core, GDE, expected, and Panel dates are synchronized",
        bool(
            metadata["data_is_fresh"]
            and metadata["price_as_of"]
            == metadata["expected_price_as_of"]
            == diagnostics["core_price_as_of"]
            == diagnostics["gde_price_as_of"]
            == panel["priceAsOf"]
        ),
        (
            f"price={metadata['price_as_of']}; "
            f"expected={metadata['expected_price_as_of']}; "
            f"panel={panel['priceAsOf']}"
        ),
        "R38 metadata, diagnostics, and Panel live data",
    )
    generation_window_status = str(
        metadata["generation_window_status"]
    )
    generation_mode = str(
        metadata.get("generation_mode", "STANDARD_REFRESH")
    )
    execution_window_status = str(
        metadata.get("execution_window_status", "UPCOMING")
    )
    intraday_reuse_is_safe = bool(
        generation_window_status == "AVAILABLE"
        or (
            generation_mode == "REUSED_LATEST_COMPLETE_INPUTS"
            and execution_window_status == "MISSED"
            and not metadata["orders_executable"]
        )
    )
    add(
        "generation_safety",
        "Intraday deployment reuses only the latest complete close and cannot create orders",
        bool(
            intraday_reuse_is_safe
            and panel.get("generationMode") == generation_mode
            and panel.get("executionWindowStatus")
            == execution_window_status
        ),
        (
            f"window={generation_window_status}; "
            f"mode={generation_mode}; "
            f"execution={execution_window_status}; "
            f"orders_executable={metadata['orders_executable']}"
        ),
        "R38 run_metadata.json and Panel live data",
    )
    add(
        "execution_safety",
        "Orders remain blocked until complete real positions exist",
        bool(
            not metadata["orders_executable"]
            and "missing complete real positions"
            in metadata["order_blockers"]
            and int(float(diagnostics["orders_executable"])) == 0
        ),
        (
            f"orders_executable={metadata['orders_executable']}; "
            f"blockers={'; '.join(metadata['order_blockers'])}"
        ),
        "R38 run_metadata.json",
    )
    panel_target = panel["stagedTarget"]
    panel_to_internal = {
        "QQQ": "QQQ",
        "SMH": "SEMIS",
        "GLD": "GOLD",
        "GDE": "GDE",
        "cash": "CASH",
    }
    panel_match = all(
        abs(
            float(panel_target[key])
            - float(
                targets.loc[
                    panel_to_internal[key],
                    "staged_account_center",
                ]
            )
        )
        <= 1e-12
        for key in ("QQQ", "SMH", "GLD", "GDE", "cash")
    )
    add(
        "panel",
        "Private Panel displays the exact staged production target",
        bool(
            panel["release"] == config["release"]
            and panel_match
            and (root / PANEL / "dist/server/index.js").exists()
            and (root / PANEL / "public/og-r38.png").exists()
        ),
        (
            f"release={panel['release']}; "
            f"target_match={panel_match}; build_present="
            f"{(root / PANEL / 'dist/server/index.js').exists()}"
        ),
        "strategy-live-data.ts and built site",
    )
    return pd.DataFrame(asdict(row) for row in rows)


def write_report(
    audit: pd.DataFrame,
    output: Path = OUTPUT,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "requirements.csv", index=False)
    passed = bool(audit["passed"].all())
    summary = {
        "release": str(
            yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
                "release"
            ]
        ),
        "production_qualification_pass": passed,
        "requirements": len(audit),
        "requirements_passed": int(audit["passed"].sum()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# R38 production qualification audit",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Requirements: {int(audit['passed'].sum())}/{len(audit)}",
        "",
        "| Category | Requirement | Pass | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.category} | {row.requirement} | "
            f"{'yes' if row.passed else 'no'} | {row.evidence} |"
        )
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    audit = build_audit()
    write_report(audit)
    print(audit.to_string(index=False))
    if not audit["passed"].all():
        raise SystemExit(1)
    print("\nR38 production qualification: PASS")
    print(f"Artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()
