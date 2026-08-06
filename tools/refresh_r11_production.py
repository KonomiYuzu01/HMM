from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from regime_strategy.operations import (
    latest_completed_us_equity_session,
    production_signal_generation_status,
)
from tools.evaluate_individual_stock_overlay import load_stock_prices
from tools.evaluate_open_execution import load_open_close
from tools.evaluate_retail_alternatives import download_gde_open_close


ROOT = Path(__file__).resolve().parents[1]
R9_CONFIG = "config/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble.yaml"
R9_OUTPUT = ROOT / "output/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
R11_OUTPUT = ROOT / "output/paper_core_growth_gold20_r11_diversified_financing"
REPORT = ROOT / "output/r11_production_refresh/latest.json"


def run_step(arguments: list[str]) -> None:
    environment = os.environ.copy()
    python_path = [str(ROOT / "src"), str(ROOT), str(ROOT / "tools")]
    existing = environment.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    now_et = pd.Timestamp.now(tz="America/New_York")
    status = production_signal_generation_status(now_et)
    if status != "AVAILABLE":
        raise RuntimeError(
            "R11 production refresh is blocked while the U.S. session is "
            "still trading; rerun after 16:15 America/New_York"
        )

    expected = latest_completed_us_equity_session(now_et)
    load_open_close(refresh=True)
    download_gde_open_close()
    stock_prices = load_stock_prices(refresh=True)
    run_step(
        [
            "-m",
            "regime_strategy.ensemble_cli",
            "--config",
            R9_CONFIG,
            "--refresh",
        ]
    )
    run_step(["tools/build_r11_production_overlay.py"])
    run_step(["tools/build_r11_reentry_brake_shadow.py"])
    run_step(["tools/build_r11_decision_authority.py"])
    run_step(["tools/audit_r11_decision_authority.py"])
    run_step(["tools/evaluate_hierarchical_regime_portfolios.py"])
    run_step(["tools/export_stock_overlay_panel_data.py"])

    r9 = metadata(R9_OUTPUT / "run_metadata.json")
    r11 = metadata(R11_OUTPUT / "run_metadata.json")
    dates = {
        "expected": expected.date().isoformat(),
        "R9": str(r9["price_as_of"]),
        "R11": str(r11["price_as_of"]),
    }
    if len(set(dates.values())) != 1:
        raise RuntimeError(f"Production refresh date mismatch: {dates}")
    stock_price_as_of = stock_prices.index[-1].date().isoformat()
    if stock_price_as_of != dates["expected"]:
        raise RuntimeError(
            "Individual-stock price data is stale: "
            f"expected {dates['expected']}, observed {stock_price_as_of}"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "status": "complete",
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "price_as_of": expected.date().isoformat(),
                "R9_data_is_fresh": bool(r9["data_is_fresh"]),
                "R11_data_is_fresh": bool(r11["data_is_fresh"]),
                "stock_price_as_of": stock_price_as_of,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
