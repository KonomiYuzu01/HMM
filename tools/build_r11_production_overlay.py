from pathlib import Path

from tools import build_r10_production_overlay as production_builder


production_builder.DEFAULT_CONFIG = Path(
    "config/paper_core_growth_gold20_r11_diversified_financing.yaml"
)


if __name__ == "__main__":
    production_builder.main()
