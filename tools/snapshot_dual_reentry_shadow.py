from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from regime_strategy.forward_monitoring import append_immutable


OUTPUT = Path("output")
DESTINATION = OUTPUT / "forward_monitoring"
PRODUCTION = OUTPUT / "paper_core_growth_gold20_daily_risk_netted_ensemble"
CANDIDATE = (
    OUTPUT
    / "paper_core_growth_gold20_dual_reentry_inverse_momentum_netted_ensemble"
)
VALIDATION = OUTPUT / "dual_reentry_candidate_validation"
STRATEGY_NAME = "dual_reentry_inverse_momentum_shadow_v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(directory: Path) -> tuple[dict[str, object], pd.Series, str]:
    metadata = json.loads(
        (directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    targets = pd.read_csv(
        directory / "next_target_weights.csv",
        index_col=0,
    )["ensemble_current_sleeve_weight"]
    diagnostics = pd.read_csv(
        directory / "next_signal_diagnostics.csv",
        index_col=0,
    )
    actions = "|".join(diagnostics.loc["action"].astype(str))
    return metadata, targets, actions


def main() -> None:
    production_metadata, production_target, production_actions = (
        load_snapshot(PRODUCTION)
    )
    candidate_metadata, candidate_target, candidate_actions = load_snapshot(
        CANDIDATE
    )
    signal_date = str(candidate_metadata["price_as_of"])
    if signal_date != str(production_metadata["price_as_of"]):
        raise RuntimeError(
            "Production and shadow snapshots must share the same signal date"
        )

    metrics = pd.read_csv(
        VALIDATION / "metrics.csv",
        index_col=[0, 1, 2],
    )
    family = pd.read_csv(
        VALIDATION / "family_reality_check.csv",
        index_col=0,
    )
    delays = pd.read_csv(
        VALIDATION / "execution_delay_relative_metrics.csv",
        index_col=[0, 1, 2, 3],
    )
    episodes = pd.read_csv(
        VALIDATION / "zero_entry_episode_summary.csv",
        index_col=0,
    )

    baseline = metrics.loc[
        ("normal", "baseline", "complete_2015_2025")
    ]
    candidate = metrics.loc[
        ("normal", "candidate", "complete_2015_2025")
    ]
    point_gate = bool(
        candidate["cagr"] >= baseline["cagr"] + 0.01
        and candidate["max_drawdown"] >= -0.18
        and candidate["sharpe"] > baseline["sharpe"]
    )
    selection_p_max = float(
        family["candidate_selection_adjusted_p_value"].max()
    )
    candidate_count = int(family["candidate_count"].max())
    statistical_gate = selection_p_max <= 0.10
    one_day_vs_inverse = delays.loc[
        (
            "dual_reentry",
            "inverse_gate",
            1,
            "complete_2015_2025",
        )
    ]
    execution_gate = bool(
        one_day_vs_inverse["cagr_delta"] > 0.0
        and one_day_vs_inverse["sharpe_delta"] >= 0.0
    )
    episode_gate = bool(
        episodes.loc[
            "normal_2015_2025",
            "annualized_after_largest",
        ]
        > 0.0
    )
    eligible = bool(
        point_gate
        and statistical_gate
        and execution_gate
        and episode_gate
    )

    production_config = Path(str(production_metadata["config_path"]))
    candidate_config = Path(str(candidate_metadata["config_path"]))
    shared_assets = production_target.index.intersection(
        candidate_target.index
    )
    target_difference = (
        candidate_target.loc[shared_assets]
        - production_target.loc[shared_assets]
    )
    row: dict[str, object] = {
        "strategy_name": STRATEGY_NAME,
        "signal_date": signal_date,
        "candidate_generated_at_utc": candidate_metadata[
            "generated_at_utc"
        ],
        "production_config_path": str(production_config),
        "production_config_sha256": sha256(production_config),
        "candidate_config_path": str(candidate_config),
        "candidate_config_sha256": sha256(candidate_config),
        "production_actions": production_actions,
        "candidate_actions": candidate_actions,
        "production_qqq": float(production_target["QQQ"]),
        "production_smh": float(production_target["SEMIS"]),
        "production_gold": float(production_target["GOLD"]),
        "production_cash": float(production_target["CASH"]),
        "candidate_qqq": float(candidate_target["QQQ"]),
        "candidate_smh": float(candidate_target["SEMIS"]),
        "candidate_gold": float(candidate_target["GOLD"]),
        "candidate_cash": float(candidate_target["CASH"]),
        "target_one_way_turnover_difference": (
            0.5 * float(target_difference.abs().sum())
        ),
        "candidate_cagr_delta": float(
            candidate["cagr"] - baseline["cagr"]
        ),
        "candidate_sharpe_delta": float(
            candidate["sharpe"] - baseline["sharpe"]
        ),
        "candidate_max_drawdown": float(candidate["max_drawdown"]),
        "selection_adjusted_p_value_max": selection_p_max,
        "candidate_family_count": candidate_count,
        "point_gate_pass": point_gate,
        "statistical_gate_pass": statistical_gate,
        "execution_gate_pass": execution_gate,
        "episode_concentration_gate_pass": episode_gate,
        "production_eligible": eligible,
    }

    DESTINATION.mkdir(parents=True, exist_ok=True)
    append_immutable(
        DESTINATION / "dual_reentry_shadow_log.csv",
        row,
        keys=["strategy_name", "signal_date"],
    )
    snapshot_path = (
        DESTINATION / f"{STRATEGY_NAME}_{signal_date}.json"
    )
    serialized = json.dumps(row, indent=2, sort_keys=True) + "\n"
    if snapshot_path.exists():
        if snapshot_path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                "Immutable shadow snapshot conflicts with stored JSON"
            )
    else:
        snapshot_path.write_text(serialized, encoding="utf-8")

    print(pd.Series(row).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
