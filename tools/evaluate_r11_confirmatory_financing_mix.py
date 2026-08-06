from __future__ import annotations

from pathlib import Path

from tools import evaluate_r11_levered_diversified_strategy as research


research.OUTPUT = Path("output/r11_confirmatory_financing_mix")
research.RISK_MULTIPLIERS = (1.060, 1.065, 1.070)
research.GDE_FRACTIONS = (0.05, 0.10, 0.15)


if __name__ == "__main__":
    research.main()
