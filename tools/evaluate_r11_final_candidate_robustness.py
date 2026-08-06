from __future__ import annotations

from pathlib import Path

from tools import evaluate_r10_selected_robustness as robustness


robustness.OUTPUT = Path("output/r11_final_candidate_robustness")
robustness.SELECTED_PATHS = Path("output/r11_confirmatory_financing_mix")
robustness.SUBSTITUTION_FRACTION = 0.10
robustness.GDE_NO_TRADE_BAND = 0.02
robustness.RISK_MULTIPLIER = 1.065
robustness.CANDIDATE_PATH_TEMPLATE = (
    "{sample}_risk1.065_gde10_daily.csv"
)


if __name__ == "__main__":
    robustness.main()
