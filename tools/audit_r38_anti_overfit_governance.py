from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/r38_anti_overfit_governance.yaml"
R38_CONFIG = ROOT / "config/paper_core_growth_gold20_r38_convex_overlay.yaml"
R38_METADATA = (
    ROOT / "output/paper_core_growth_gold20_r38_convex_overlay/run_metadata.json"
)
CORE_PRICES = ROOT / "data/adjusted_open_close_2011_present.csv"
SPA_AUDIT = ROOT / "output/strategy_library_frontier_spa/spa_target_audit.csv"
OUTPUT = ROOT / "output/r38_anti_overfit_governance"


@dataclass(frozen=True)
class GovernanceRecord:
    category: str
    requirement: str
    passed: bool
    required_for_operation: bool
    evidence: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_forward_sessions(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    freeze_price_as_of: str,
    current_price_as_of: str,
) -> int:
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if normalized.has_duplicates or not normalized.is_monotonic_increasing:
        raise ValueError("forward-session dates must be ordered and unique")
    freeze = pd.Timestamp(freeze_price_as_of)
    current = pd.Timestamp(current_price_as_of)
    return int(((normalized > freeze) & (normalized <= current)).sum())


def source_hashes_match(root: Path, expected: dict[str, str]) -> bool:
    return all(
        (root / relative).exists()
        and sha256(root / relative) == expected_hash
        for relative, expected_hash in expected.items()
    )


def spa_evidence_passes(
    frame: pd.DataFrame,
    *,
    target: str,
    required_block_days: set[int],
    minimum_unique_candidate_paths: int,
    maximum_p_value: float,
) -> bool:
    selected = frame.loc[frame["target"].astype(str).eq(target)].copy()
    if set(selected["block_days"].astype(int)) != required_block_days:
        return False
    return bool(
        selected["unique_candidate_paths"]
        .astype(int)
        .ge(minimum_unique_candidate_paths)
        .all()
        and selected["familywise_spa_p_value"]
        .astype(float)
        .le(maximum_p_value)
        .all()
    )


def build_audit(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    governance = yaml.safe_load((root / CONFIG.relative_to(ROOT)).read_text())
    strategy = yaml.safe_load((root / R38_CONFIG.relative_to(ROOT)).read_text())
    metadata = json.loads((root / R38_METADATA.relative_to(ROOT)).read_text())
    prices = pd.read_csv(root / CORE_PRICES.relative_to(ROOT), usecols=["date"])
    spa = pd.read_csv(root / SPA_AUDIT.relative_to(ROOT))
    freeze = dict(governance["freeze"])
    promotion = dict(governance["promotion"])
    multiple = dict(governance["multiple_testing"])
    frozen_hashes = {
        str(path): str(digest)
        for path, digest in dict(governance["frozen_source_sha256"]).items()
    }
    rollout = dict(strategy["rollout"])
    forward_sessions = completed_forward_sessions(
        prices["date"],
        freeze_price_as_of=str(freeze["price_as_of"]),
        current_price_as_of=str(metadata["price_as_of"]),
    )
    minimum_sessions = int(freeze["minimum_completed_forward_sessions"])
    current_share = float(rollout["initial_r38_share_of_managed_capital"])
    maximum_share = float(freeze["maximum_r38_share_before_review"])
    automatic_promotion = bool(promotion["automatic_successor_promotion"])
    rows: list[GovernanceRecord] = []

    def add(
        category: str,
        requirement: str,
        passed: bool,
        required_for_operation: bool,
        evidence: str,
    ) -> None:
        rows.append(
            GovernanceRecord(
                category,
                requirement,
                bool(passed),
                bool(required_for_operation),
                evidence,
            )
        )

    add(
        "freeze",
        "R38 release and frozen source hashes are unchanged",
        str(strategy["release"]) == str(governance["strategy_release"])
        and source_hashes_match(root, frozen_hashes),
        True,
        f"release={strategy['release']}; files={len(frozen_hashes)}",
    )
    add(
        "rollout",
        "R38 remains capped at 25% before forward review",
        current_share <= maximum_share and current_share == 0.25,
        True,
        f"current={current_share:.2%}; maximum={maximum_share:.2%}",
    )
    add(
        "sample_integrity",
        "All pre-freeze historical evidence is explicitly classified in-sample",
        bool(freeze["historical_samples_after_freeze_are_in_sample"]),
        True,
        "no historical period is treated as a new untouched holdout",
    )
    add(
        "sample_integrity",
        "Forward observations cannot be used for parameter tuning",
        bool(freeze["forward_sample_may_not_be_used_for_parameter_tuning"]),
        True,
        "forward evidence is go/no-go only",
    )
    add(
        "multiple_testing",
        "Full-library SPA covers the frozen R38 target at all required blocks",
        spa_evidence_passes(
            spa,
            target=str(multiple["target"]),
            required_block_days={
                int(value) for value in multiple["required_block_days"]
            },
            minimum_unique_candidate_paths=int(
                multiple["minimum_unique_candidate_paths"]
            ),
            maximum_p_value=float(multiple["maximum_familywise_spa_p_value"]),
        ),
        True,
        (
            f"target={multiple['target']}; "
            f"paths>={int(multiple['minimum_unique_candidate_paths'])}"
        ),
    )
    add(
        "successors",
        "R39-R41 cannot be promoted automatically",
        not automatic_promotion
        and set(promotion["blocked_successors"]) == {"R39", "R40", "R41"},
        True,
        "automatic promotion disabled; R39, R40, and R41 blocked",
    )
    add(
        "successors",
        "Future core must pass risk limits without optional overlays",
        bool(promotion["core_must_pass_without_optional_overlays"]),
        True,
        (
            f"central MDD <= {abs(float(promotion['central_maximum_drawdown_limit'])):.0%}; "
            f"stress/neighbor <= {abs(float(promotion['stressed_neighbor_maximum_drawdown_limit'])):.0%}"
        ),
    )
    add(
        "forward_evidence",
        "Minimum 63 completed forward sessions accumulated",
        forward_sessions >= minimum_sessions,
        False,
        f"completed={forward_sessions}; minimum={minimum_sessions}",
    )
    audit = pd.DataFrame(asdict(row) for row in rows)
    operational = audit.loc[audit["required_for_operation"], "passed"]
    operational_pass = bool(operational.all())
    forward_review_eligible = bool(forward_sessions >= minimum_sessions)
    successor_promotion_allowed = bool(
        operational_pass and forward_review_eligible and automatic_promotion
    )
    summary: dict[str, object] = {
        "status": "complete",
        "release": str(governance["release"]),
        "strategy_release": str(governance["strategy_release"]),
        "price_as_of": str(metadata["price_as_of"]),
        "operational_pass": operational_pass,
        "requirements": int(operational.size),
        "requirements_passed": int(operational.sum()),
        "forward_sessions": forward_sessions,
        "minimum_forward_sessions": minimum_sessions,
        "forward_review_eligible": forward_review_eligible,
        "current_r38_share": current_share,
        "maximum_r38_share_before_review": maximum_share,
        "successor_promotion_allowed": successor_promotion_allowed,
        "blocked_successors": list(promotion["blocked_successors"]),
        "historical_evidence_classification": "IN_SAMPLE_REUSED",
        "forward_evidence_use": "GO_NO_GO_ONLY",
    }
    return audit, summary


def write_report(
    audit: pd.DataFrame,
    summary: dict[str, object],
    output: Path = OUTPUT,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "requirements.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    audit, summary = build_audit()
    write_report(audit, summary)
    print(audit.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["operational_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
