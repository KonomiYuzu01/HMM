from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from regime_strategy.forward_monitoring import append_immutable


DESTINATION = Path("output/forward_monitoring")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-name",
        default="production_baseline",
    )
    parser.add_argument(
        "--strategy-output",
        default="output/paper_core_growth_gold20_daily_risk_ensemble",
    )
    parser.add_argument(
        "--panel-output",
        default="output/current_operational_panel",
    )
    args = parser.parse_args()

    strategy_output = Path(args.strategy_output)
    panel_output = Path(args.panel_output)
    metadata = json.loads(
        (strategy_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    config_path = Path(metadata["config_path"])
    targets = pd.read_csv(
        strategy_output / "next_target_weights.csv", index_col=0
    )["ensemble_current_sleeve_weight"]
    diagnostics = pd.read_csv(
        strategy_output / "next_signal_diagnostics.csv", index_col=0
    )
    scenarios = pd.read_csv(panel_output / "scenario_weights.csv", index_col=0)
    risks = pd.read_csv(panel_output / "risk_scenarios.csv", index_col=0)
    orders = pd.read_csv(panel_output / "order_plan.csv")
    onboarding = scenarios.loc["account_volatility_capped"]
    onboarding_risk = risks.loc["account_volatility_capped"]
    execution_statuses = orders["execution_status"].astype(str).unique()
    executable_values = orders["executable"].astype(str).unique()
    if len(execution_statuses) != 1 or len(executable_values) != 1:
        raise RuntimeError("Panel order execution metadata is internally inconsistent")

    # The panel is authoritative for holidays; infer its date from the status-bearing order file
    # and the standard next-session rule encoded in the generated panel text.
    panel_text = (panel_output / "panel.md").read_text(encoding="utf-8")
    execution_line = next(
        line for line in panel_text.splitlines() if line.startswith("- 理论执行日：")
    )
    execution_date_text = execution_line.split("：", 1)[1].strip()

    row: dict[str, object] = {
        "strategy_name": args.strategy_name,
        "signal_date": metadata["price_as_of"],
        "execution_date": execution_date_text,
        "generated_at_utc": metadata["generated_at_utc"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "execution_status": execution_statuses[0],
        "executable": executable_values[0],
        "member_actions": "|".join(diagnostics.loc["action"].astype(str)),
        "model_qqq": float(targets["QQQ"]),
        "model_smh": float(targets["SEMIS"]),
        "model_gld": float(targets["GOLD"]),
        "model_bil": float(targets["CASH"]),
        "model_vixy": float(targets.get("VIX_HEDGE", 0.0)),
        "onboarding_qqq": float(onboarding["QQQ"]),
        "onboarding_smh": float(onboarding["SEMIS"]),
        "onboarding_gld": float(onboarding["GOLD"]),
        "onboarding_bil": float(onboarding["CASH"]),
        "onboarding_vixy": float(onboarding.get("VIX_HEDGE", 0.0)),
        "onboarding_stress_volatility": float(
            onboarding_risk["account_realized_volatility"]
        ),
        "onboarding_smh_risk_share": float(onboarding_risk["smh_risk_share"]),
        "onboarding_gold_risk_share": float(onboarding_risk["gold_risk_share"]),
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    append_immutable(
        DESTINATION / "signal_log.csv",
        row,
        keys=["strategy_name", "signal_date"],
    )
    (DESTINATION / f"{args.strategy_name}_{row['signal_date']}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(pd.Series(row).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
