from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        check=True,
    )


def refresh_steps(reuse_latest_complete_inputs: bool) -> list[list[str]]:
    steps: list[list[str]] = []
    if not reuse_latest_complete_inputs:
        steps.append(["tools/refresh_r11_production.py"])
    steps.extend(
        [
            [
                "tools/evaluate_r38_accelerating_volatility_capacity_fill_1375.py"
            ],
            ["tools/build_r38_production_overlay.py"],
            [
                "tools/audit_r38_accelerating_volatility_1375_final_candidate.py"
            ],
            ["tools/audit_r38_anti_overfit_governance.py"],
            ["tools/export_r38_panel_snapshot.py"],
            ["tools/audit_r38_production_qualification.py"],
            ["tools/export_r38_panel_snapshot.py"],
            ["tools/audit_r38_production_qualification.py"],
        ]
    )
    if reuse_latest_complete_inputs:
        steps[1].append("--reuse-latest-complete-inputs")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh and qualify the staged R38 production release"
    )
    parser.add_argument(
        "--reuse-latest-complete-inputs",
        action="store_true",
        help=(
            "During an incomplete U.S. session, reuse synchronized inputs "
            "from the latest completed session instead of downloading "
            "intraday data."
        ),
    )
    arguments = parser.parse_args()

    for command in refresh_steps(arguments.reuse_latest_complete_inputs):
        run_step(command)


if __name__ == "__main__":
    main()
