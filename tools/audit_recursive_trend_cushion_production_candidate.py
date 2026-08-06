from pathlib import Path

from tools.audit_recursive_trend_cushion_qualified import main as audit


if __name__ == "__main__":
    audit(
        Path("output/recursive_trend_cushion_production_candidate"),
        floor_drawdown=-0.19,
    )
