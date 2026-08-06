from __future__ import annotations

from pathlib import Path

from tools import evaluate_r11_selected_audit as audit


audit.OUTPUT = Path("output/r11_final_candidate_audit")
audit.RISK_MULTIPLIER = 1.065
audit.GDE_FRACTION = 0.10


if __name__ == "__main__":
    audit.main()
