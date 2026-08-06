from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config/paper_core_growth_gold20_r11_reentry_brake_shadow.yaml"
)
EVALUATION = ROOT / "output/r11_selective_shock_memory"
SHADOW = (
    ROOT
    / "output/paper_core_growth_gold20_r11_reentry_brake_shadow"
)
OUTPUT = ROOT / "output/r11_reentry_brake_shadow_audit"


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    status: str
    evidence: str


def build_audit() -> pd.DataFrame:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    metrics = pd.read_csv(EVALUATION / "metrics_by_period.csv")
    selection = pd.read_csv(EVALUATION / "selection.csv")
    family = pd.read_csv(
        EVALUATION / "declared_family_reality_check.csv"
    )
    diagnostics = pd.read_csv(
        SHADOW / "next_signal_diagnostics.csv",
        index_col=0,
    ).iloc[:, 0]
    targets = pd.read_csv(
        SHADOW / "next_target_weights.csv",
        index_col="asset",
    )
    forward = pd.read_csv(
        ROOT
        / "output/forward_monitoring/"
        "r11_reentry_brake_shadow_log.csv"
    )
    candidate = "reentry_brake_semis_medium_break_cap35"
    rows: list[AuditRecord] = []

    def add(
        category: str,
        requirement: str,
        passed: bool,
        evidence: str,
    ) -> None:
        rows.append(
            AuditRecord(
                category,
                requirement,
                "pass" if passed else "fail",
                evidence,
            )
        )

    selected = selection[selection["selected"]]
    add(
        "selection",
        "Frozen shadow candidate matches the selected research path",
        bool(
            len(selected) == 1
            and selected.iloc[0]["candidate"] == candidate
            and bool(selected.iloc[0]["selection_gate"])
        ),
        (
            f"selected={selected.iloc[0]['candidate'] if len(selected) else ''}; "
            f"release={config['release']}"
        ),
    )
    full_periods = {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }
    full_rows = pd.concat(
        [
            metrics[
                metrics["candidate"].eq(candidate)
                & metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["rollout_share"].eq(1.0)
            ]
            for sample, period in full_periods.items()
        ],
        ignore_index=True,
    )
    add(
        "point_estimates",
        "All full windows improve CAGR, Sharpe, and maximum drawdown",
        bool(
            len(full_rows) == 6
            and full_rows["cagr_delta"].gt(0.0).all()
            and full_rows["sharpe_delta"].gt(0.0).all()
            and full_rows["max_drawdown_delta"].gt(0.0).all()
        ),
        (
            f"CAGR delta {full_rows['cagr_delta'].min():.6f} to "
            f"{full_rows['cagr_delta'].max():.6f}; "
            "worst MDD "
            f"{full_rows['candidate_max_drawdown'].min():.6f}"
        ),
    )
    family_p_max = float(
        family["familywise_reality_check_p_value"].max()
    )
    add(
        "statistical_gate",
        "Failed multiple-comparisons evidence blocks production eligibility",
        bool(
            family_p_max >= 0.05
            and not bool(
                config["qualification"][
                    "multiple_comparisons_gate_pass"
                ]
            )
            and not bool(
                config["qualification"]["production_eligible"]
            )
        ),
        f"maximum familywise p={family_p_max:.6f}; required <0.05",
    )
    add(
        "capital_isolation",
        "Shadow release controls no real capital and cannot emit orders",
        bool(
            float(diagnostics["real_capital_share"]) == 0.0
            and int(float(diagnostics["orders_executable"])) == 0
            and int(float(diagnostics["production_eligible"])) == 0
        ),
        (
            f"capital={diagnostics['real_capital_share']}; "
            f"orders={diagnostics['orders_executable']}; "
            f"eligible={diagnostics['production_eligible']}"
        ),
    )
    target_sums = targets[
        [
            "production_full_center",
            "shadow_full_center",
            "production_staged_center",
            "shadow_staged_center",
        ]
    ].sum()
    add(
        "output",
        "Production and shadow target portfolios each sum to 100%",
        bool((target_sums.sub(1.0).abs() <= 1e-12).all()),
        "; ".join(
            f"{column}={value:.12f}"
            for column, value in target_sums.items()
        ),
    )
    latest = forward.iloc[-1]
    add(
        "forward_monitoring",
        "Immutable forward log records the frozen release and signal date",
        bool(
            latest["release"] == config["release"]
            and latest["signal_date"] == diagnostics["price_as_of"]
            and not bool(latest["production_eligible"])
        ),
        (
            f"release={latest['release']}; "
            f"signal_date={latest['signal_date']}"
        ),
    )
    return pd.DataFrame(asdict(row) for row in rows)


def main() -> None:
    audit = build_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT / "requirements.csv", index=False)
    print(audit.to_string(index=False))
    failures = audit[audit["status"].eq("fail")]
    if not failures.empty:
        raise SystemExit(1)
    print(
        "\nShadow release audit: PASS "
        "(research state is observable and real capital remains isolated)"
    )


if __name__ == "__main__":
    main()
