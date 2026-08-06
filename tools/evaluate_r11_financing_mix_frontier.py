from __future__ import annotations

from pathlib import Path

from tools import evaluate_r11_levered_diversified_strategy as research


research.OUTPUT = Path("output/r11_financing_mix_frontier")
research.RISK_MULTIPLIERS = (1.050, 1.055, 1.060)
research.GDE_FRACTIONS = (0.10, 0.20, 0.30)


if __name__ == "__main__":
    research.main()
