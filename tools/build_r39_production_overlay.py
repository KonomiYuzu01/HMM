from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_strategy.operations import (
    latest_completed_us_equity_session,
    missing_recent_us_equity_sessions,
)
from regime_strategy.relative_damage import (
    apply_relative_damage_veto,
    completed_close_relative_damage,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "config/paper_core_growth_gold20_r39_relative_damage_veto.yaml"
)
TARGET_VARIANTS = ("lower", "center", "upper")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def apply_veto_to_target_columns(
    parent_targets: pd.DataFrame,
    relative_loss: float,
    maximum_share_permitted: bool,
    *,
    account_loss_budget: float,
    minimum_semis_growth_share: float,
    maximum_active_semis_growth_share: float,
    maximum_share_activation_multiple: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    output = pd.DataFrame(index=parent_targets.index)
    diagnostics: dict[str, pd.Series] = {}
    for variant in TARGET_VARIANTS:
        source = f"staged_account_{variant}"
        target = parent_targets[source].astype(float)
        frame = target.to_frame().T
        frame.index = pd.Index([variant])
        loss = pd.Series(relative_loss, index=frame.index)
        permission = pd.Series(
            maximum_share_permitted,
            index=frame.index,
        )
        adjusted, detail = apply_relative_damage_veto(
            frame,
            loss,
            permission,
            account_loss_budget=account_loss_budget,
            minimum_semis_growth_share=minimum_semis_growth_share,
            maximum_active_semis_growth_share=(
                maximum_active_semis_growth_share
            ),
            maximum_share_activation_multiple=(
                maximum_share_activation_multiple
            ),
        )
        output[f"r39_account_{variant}"] = adjusted.loc[variant]
        diagnostics[variant] = detail.loc[variant]
    return output, diagnostics


def validate_parent(
    parent_metadata: dict[str, object],
    parent_qualification: dict[str, object],
    core: pd.DataFrame,
    now_et: pd.Timestamp,
) -> pd.Timestamp:
    expected = latest_completed_us_equity_session(now_et)
    price_as_of = pd.Timestamp(
        str(parent_metadata["price_as_of"])
    ).normalize()
    parent_expected = pd.Timestamp(
        str(parent_metadata["expected_price_as_of"])
    ).normalize()
    if not bool(
        parent_metadata["data_is_fresh"]
        and parent_qualification["production_qualification_pass"]
    ):
        raise RuntimeError("Parent R38 production state is not qualified")
    if not (
        price_as_of
        == parent_expected
        == expected
        == core.index.max().normalize()
    ):
        raise RuntimeError(
            "R39 inputs are not synchronized to the latest complete close"
        )
    missing = missing_recent_us_equity_sessions(core.index, now_et)
    if len(missing):
        formatted = ",".join(
            date.date().isoformat() for date in missing
        )
        raise RuntimeError(f"R39 core input has session gaps: {formatted}")
    return price_as_of


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the staged R39 account-level production target"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])
    policy = dict(config["relative_damage_concentration_veto"])
    parent_config_path = _read_path(data["parent_r38_config"])
    parent_output = _read_path(data["parent_r38_output"])
    parent_qualification_path = _read_path(
        data["parent_r38_qualification"]
    )
    core_path = _read_path(data["core_open_close"])
    output = _read_path(data["output_dir"])

    parent_config = yaml.safe_load(
        parent_config_path.read_text(encoding="utf-8")
    )
    parent_metadata = json.loads(
        (parent_output / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    parent_qualification = json.loads(
        (parent_qualification_path / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    parent_targets = pd.read_csv(
        parent_output / "next_target_weights.csv",
        index_col="asset",
    )
    parent_diagnostics = pd.read_csv(
        parent_output / "next_signal_diagnostics.csv",
        index_col=0,
    )["value"]
    core = pd.read_csv(
        core_path,
        index_col=0,
        parse_dates=True,
    )
    now_et = pd.Timestamp.now(tz="America/New_York")
    price_as_of = validate_parent(
        parent_metadata,
        parent_qualification,
        core,
        now_et,
    )
    closes = core.loc[
        :price_as_of,
        ["close_QQQ", "close_SEMIS"],
    ].rename(
        columns={
            "close_QQQ": "QQQ",
            "close_SEMIS": "SEMIS",
        }
    )
    lookback = int(policy["lookback_trading_days"])
    damage = completed_close_relative_damage(
        closes,
        lookback_days=lookback,
    )
    account_loss_budget = float(
        policy["account_relative_loss_budget"]
    )
    minimum_share = float(
        policy["minimum_smh_share_of_growth_sleeve"]
    )
    maximum_active_share = float(
        policy["maximum_active_smh_share_of_growth_sleeve"]
    )
    maximum_share_activation_multiple = float(
        policy["maximum_share_activation_multiple"]
    )
    guarded, diagnostics = apply_veto_to_target_columns(
        parent_targets,
        float(damage["relative_loss"]),
        maximum_share_permitted=not bool(
            int(float(parent_diagnostics["volatility_acceleration_block"]))
        ),
        account_loss_budget=account_loss_budget,
        minimum_semis_growth_share=minimum_share,
        maximum_active_semis_growth_share=maximum_active_share,
        maximum_share_activation_multiple=(
            maximum_share_activation_multiple
        ),
    )

    target_output = pd.DataFrame(index=parent_targets.index)
    target_output.index.name = "asset"
    target_output["ticker"] = parent_targets["ticker"]
    target_output["r11_reference_target"] = parent_targets[
        "r11_reference_target"
    ]
    target_output["r38_full_center"] = parent_targets[
        "r38_full_center"
    ]
    for variant in TARGET_VARIANTS:
        target_output[f"r38_staged_account_{variant}"] = (
            parent_targets[f"staged_account_{variant}"]
        )
        target_output[f"r39_account_{variant}"] = guarded[
            f"r39_account_{variant}"
        ]
    sums = target_output[
        [f"r39_account_{variant}" for variant in TARGET_VARIANTS]
    ].sum()
    if not np.allclose(sums, 1.0, atol=1e-12):
        raise AssertionError(f"R39 target sums are invalid: {sums}")

    center = diagnostics["center"]
    growth_before = float(
        parent_targets.loc[
            ["QQQ", "SEMIS"],
            "staged_account_center",
        ].sum()
    )
    growth_after = float(
        target_output.loc[
            ["QQQ", "SEMIS"],
            "r39_account_center",
        ].sum()
    )
    blockers = list(parent_metadata["order_blockers"])
    if "R39 requires complete real positions" not in blockers:
        blockers.append("R39 requires complete real positions")
    diagnostic_output = pd.Series(
        {
            "release": config["release"],
            "status": config["status"],
            "parent_release": parent_config["release"],
            "price_as_of": price_as_of.date().isoformat(),
            "next_session": parent_metadata["next_session"],
            "lookback_trading_days": lookback,
            "prior_relative_log_return": float(
                damage["prior_relative_log_return"]
            ),
            "relative_loss": float(damage["relative_loss"]),
            "account_relative_loss_budget": account_loss_budget,
            "minimum_smh_share_of_growth_sleeve": minimum_share,
            "maximum_active_smh_share_of_growth_sleeve": (
                maximum_active_share
            ),
            "maximum_share_activation_multiple": (
                maximum_share_activation_multiple
            ),
            "maximum_share_guard_active": int(
                bool(center["maximum_share_guard_active"])
            ),
            "maximum_share_guard_permitted": int(
                bool(center["maximum_share_guard_permitted"])
            ),
            "relative_damage_veto_active": int(
                bool(center["relative_damage_guard_active"])
            ),
            "proposed_account_relative_loss": float(
                center["proposed_account_relative_loss"]
            ),
            "implemented_account_relative_loss": float(
                center["implemented_account_relative_loss"]
            ),
            "base_semis_growth_share": float(
                center["base_semis_growth_share"]
            ),
            "implemented_semis_growth_share": float(
                center["implemented_semis_growth_share"]
            ),
            "removed_semis_weight": float(
                center["removed_semis_weight"]
            ),
            "growth_budget_before": growth_before,
            "growth_budget_after": growth_after,
            "growth_budget_error": abs(growth_after - growth_before),
            "orders_executable": 0,
            "order_blockers": "; ".join(blockers),
        },
        name="value",
    )
    metadata = {
        "release": config["release"],
        "status": config["status"],
        "parent_release": parent_config["release"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": price_as_of.date().isoformat(),
        "expected_price_as_of": parent_metadata[
            "expected_price_as_of"
        ],
        "data_is_fresh": True,
        "generation_window_status": parent_metadata[
            "generation_window_status"
        ],
        "generation_mode": "PARENT_R38_LATEST_COMPLETE",
        "execution_window_status": parent_metadata[
            "execution_window_status"
        ],
        "next_session": parent_metadata["next_session"],
        "orders_executable": False,
        "order_blockers": blockers,
        "source_hashes": {
            str(config_path): file_sha256(config_path),
            str(parent_config_path): file_sha256(parent_config_path),
            str(parent_output / "next_target_weights.csv"): file_sha256(
                parent_output / "next_target_weights.csv"
            ),
            str(parent_output / "run_metadata.json"): file_sha256(
                parent_output / "run_metadata.json"
            ),
            str(parent_output / "next_signal_diagnostics.csv"): (
                file_sha256(
                    parent_output / "next_signal_diagnostics.csv"
                )
            ),
            str(core_path): file_sha256(core_path),
        },
        "risk_policy": {
            "lookback_trading_days": lookback,
            "account_relative_loss_budget": account_loss_budget,
            "minimum_smh_share_of_growth_sleeve": minimum_share,
            "maximum_active_smh_share_of_growth_sleeve": (
                maximum_active_share
            ),
            "maximum_share_activation_multiple": (
                maximum_share_activation_multiple
            ),
            "maximum_share_requires_volatility_acceleration_clear": bool(
                policy[
                    "maximum_share_requires_volatility_acceleration_clear"
                ]
            ),
            "overflow_asset": policy["overflow_asset"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    target_output.to_csv(output / "next_target_weights.csv")
    diagnostic_output.to_csv(output / "next_signal_diagnostics.csv")
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(target_output.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostic_output.to_string())
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()
