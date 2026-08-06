from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output/r39_production_refresh/latest.json"
RESEARCH_SUMMARY = ROOT / "output/r39_final_candidate_audit/summary.json"
PRODUCTION_SUMMARY = ROOT / "output/r39_production_qualification_audit/summary.json"
R38_METADATA = ROOT / "output/paper_core_growth_gold20_r38_convex_overlay/run_metadata.json"
R39_METADATA = ROOT / "output/paper_core_growth_gold20_r39_relative_damage_veto/run_metadata.json"


def run_step(command: list[str], *, check: bool = True) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        check=check,
    )
    return completed.returncode


def qualified(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get(key, False))


def active_strategy_for_refresh(
    *,
    research_qualified: bool,
    production_qualified: bool,
) -> str:
    return "R39" if research_qualified and production_qualified else "R38"


def write_report(active_strategy: str, reason: str | None) -> None:
    metadata_path = R39_METADATA if active_strategy == "R39" else R38_METADATA
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "status": "complete",
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "requested_strategy": "R39",
                "active_strategy": active_strategy,
                "fallback_active": active_strategy != "R39",
                "fallback_reason": reason,
                "price_as_of": metadata["price_as_of"],
                "orders_executable": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh and qualify the staged R39 production config"
    )
    parser.add_argument(
        "--reuse-latest-complete-inputs",
        action="store_true",
    )
    arguments = parser.parse_args()
    parent = ["tools/refresh_r38_production.py"]
    if arguments.reuse_latest_complete_inputs:
        parent.append("--reuse-latest-complete-inputs")
    run_step(parent)
    run_step(["tools/evaluate_r39_relative_damage_concentration_veto.py"])
    research_return_code = run_step(
        ["tools/audit_r39_final_candidate.py"],
        check=False,
    )
    research_qualified = (
        research_return_code == 0
        and qualified(RESEARCH_SUMMARY, "qualification_pass")
    )
    if not research_qualified:
        reason = (
            "R39 research qualification failed; using the same-date qualified R38 snapshot."
        )
        run_step(["tools/export_r38_panel_snapshot.py"])
        write_report("R38", reason)
        return

    run_step(["tools/build_r39_production_overlay.py"])
    production_return_code = run_step(
        ["tools/audit_r39_production_qualification.py"],
        check=False,
    )
    production_qualified = (
        production_return_code == 0
        and qualified(PRODUCTION_SUMMARY, "production_qualification_pass")
    )
    active_strategy = active_strategy_for_refresh(
        research_qualified=research_qualified,
        production_qualified=production_qualified,
    )
    if active_strategy == "R39":
        run_step(["tools/export_r39_panel_snapshot.py"])
        write_report("R39", None)
        return
    reason = (
        "R39 production qualification failed; using the same-date qualified R38 snapshot."
    )
    run_step(["tools/export_r38_panel_snapshot.py"])
    write_report("R38", reason)


if __name__ == "__main__":
    main()
