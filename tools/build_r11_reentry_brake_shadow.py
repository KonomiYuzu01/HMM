from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_smh_dynamic_guard import simulate as simulate_smh_guard
from regime_strategy.forward_monitoring import append_immutable
from regime_strategy.r10_overlay import (
    BASE_ASSETS,
    R10_ASSETS,
    blend_rollout_target,
    r10_center_target,
)
from tools.build_r10_production_overlay import TICKERS
from tools.evaluate_r11_diversified_capital_grid import (
    load_strategy_inputs,
    scale_non_cash_weights,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    guard_for_multiplier,
)
from tools.evaluate_r11_selective_shock_memory import (
    ShockMemoryCandidate,
    apply_selective_shock_memory,
    build_causal_signals,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "config/paper_core_growth_gold20_r11_reentry_brake_shadow.yaml"
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def scalar_diagnostics(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, index_col=0)
    return {
        str(index): str(value)
        for index, value in frame.iloc[:, 0].items()
    }


def validate_frozen_parameters(config: dict[str, object]) -> None:
    brake = dict(config["reentry_brake"])
    expected = {
        "previous_account_growth_loss_trigger": -0.03,
        "minimum_previous_growth_weight": 0.50,
        "qqq_63d_return_maximum": 0.0,
        "semis_63d_return_maximum": 0.0,
        "semis_126d_return_maximum": 0.0,
        "vix_to_vix3m_minimum_exclusive": 1.0,
        "growth_weight_cap_when_engaged": 0.35,
        "overflow_asset": "CASH",
    }
    observed = {key: brake.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "Shadow configuration no longer matches the frozen research "
            f"candidate: expected={expected}, observed={observed}"
        )
    qualification = dict(config["qualification"])
    if bool(qualification["production_eligible"]):
        raise ValueError(
            "The shadow candidate cannot become production eligible by "
            "editing its configuration; rerun the qualification audit."
        )


def shadow_state(diagnostics: pd.Series) -> str:
    if bool(diagnostics["released"]):
        return "released_today"
    if bool(diagnostics["engaged"]):
        return "limiting_reentry"
    if bool(diagnostics["active"]):
        return "armed"
    return "inactive"


def main(config_path: Path = DEFAULT_CONFIG) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_parameters(config)
    data = {
        key: ROOT / str(value)
        for key, value in dict(config["data"]).items()
    }
    r11_config = yaml.safe_load(
        data["r11_config"].read_text(encoding="utf-8")
    )
    r11_metadata = json.loads(
        (data["r11_output"] / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    r11_targets = pd.read_csv(
        data["r11_output"] / "next_target_weights.csv",
        index_col="asset",
    )
    r11_diagnostics = scalar_diagnostics(
        data["r11_output"] / "next_signal_diagnostics.csv"
    )
    r9_weights, r9_daily = load_strategy_inputs(data["r9_output"])
    open_close = pd.read_csv(
        data["core_open_close"],
        index_col=0,
        parse_dates=True,
    )
    opens = open_close[
        [f"open_{asset}" for asset in BASE_ASSETS]
    ].copy()
    closes = open_close[
        [f"close_{asset}" for asset in BASE_ASSETS]
    ].copy()
    opens.columns = list(BASE_ASSETS)
    closes.columns = list(BASE_ASSETS)
    market_prices = pd.read_csv(
        data["market_prices"],
        index_col=0,
        parse_dates=True,
    )

    risk_multiplier = float(
        dict(r11_config["risk_budget"])[
            "target_non_cash_multiplier"
        ]
    )
    current_cost = [
        scenario
        for scenario in COST_SCENARIOS
        if scenario.name == "current_liquidity"
    ][0]
    levered = scale_non_cash_weights(r9_weights, risk_multiplier)
    guard_daily, guard_weights = simulate_smh_guard(
        levered,
        r9_daily,
        opens,
        closes,
        guard_for_multiplier(
            risk_multiplier,
            current_cost.emergency_slippage_bps,
        ),
        cost_bps=current_cost.base_one_way_cost_bps,
        start_date="2015-01-01",
    )

    price_as_of = pd.Timestamp(r11_metadata["price_as_of"])
    next_session = pd.Timestamp(r11_metadata["next_session"])
    if guard_weights.index.max() != price_as_of:
        raise RuntimeError(
            "R11 shadow history does not end on the production signal date"
        )
    base_next = r11_targets["r11_guard_adjusted_base"].reindex(
        BASE_ASSETS
    ).astype(float)
    extended_weights = guard_weights.copy()
    extended_weights.loc[next_session, list(BASE_ASSETS)] = base_next
    extended_daily = guard_daily.copy()
    extended_daily.loc[next_session] = extended_daily.iloc[-1]
    extended_daily.loc[next_session, "turnover"] = (
        1.0 if r11_diagnostics["r9_action"] == "REBALANCE" else 0.0
    )
    flat_close = closes.loc[price_as_of, list(BASE_ASSETS)]
    extended_closes = closes.copy()
    extended_closes.loc[next_session, list(BASE_ASSETS)] = flat_close
    extended_market = market_prices.copy()
    extended_market.loc[next_session] = market_prices.loc[price_as_of]

    signals = build_causal_signals(
        extended_weights,
        extended_closes,
        extended_market,
    )
    candidate = ShockMemoryCandidate(
        "semis_medium_break",
        0.35,
        "reentry_brake",
    )
    shadow_weights, _, diagnostics = apply_selective_shock_memory(
        extended_weights,
        extended_daily,
        signals,
        candidate,
    )
    latest_diagnostics = diagnostics.loc[next_session]
    shadow_base = shadow_weights.loc[
        next_session,
        list(BASE_ASSETS),
    ]
    substitution_fraction = float(
        dict(r11_config["capital_efficiency"])[
            "maximum_gold_sleeve_substitution_fraction"
        ]
    )
    shadow_full = r10_center_target(
        shadow_base,
        substitution_fraction=substitution_fraction,
    )
    r9_reference = r11_targets["r9_reference_target"].reindex(
        R10_ASSETS
    ).astype(float)
    rollout_share = float(
        dict(r11_config["rollout"])[
            "initial_r11_share_of_managed_capital"
        ]
    )
    shadow_staged = blend_rollout_target(
        r9_reference,
        shadow_full,
        rollout_share=rollout_share,
    )
    production_full = r11_targets["r11_full_center"].reindex(
        R10_ASSETS
    ).astype(float)
    production_staged = r11_targets[
        "staged_account_center"
    ].reindex(R10_ASSETS).astype(float)

    output = data["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    target_output = pd.DataFrame(index=R10_ASSETS)
    target_output.index.name = "asset"
    target_output["ticker"] = [
        TICKERS[asset] for asset in target_output.index
    ]
    target_output["production_full_center"] = production_full
    target_output["shadow_full_center"] = shadow_full
    target_output["full_difference"] = shadow_full - production_full
    target_output["production_staged_center"] = production_staged
    target_output["shadow_staged_center"] = shadow_staged
    target_output["staged_difference"] = (
        shadow_staged - production_staged
    )
    target_output.to_csv(output / "next_target_weights.csv")

    family = pd.read_csv(
        data["evaluation_output"]
        / "declared_family_reality_check.csv"
    )
    family_p_max = float(
        family["familywise_reality_check_p_value"].max()
    )
    qualification = dict(config["qualification"])
    state = shadow_state(latest_diagnostics)
    diagnostics_output = pd.Series(
        {
            "release": config["release"],
            "status": config["status"],
            "price_as_of": price_as_of.date().isoformat(),
            "next_session": next_session.date().isoformat(),
            "shadow_state": state,
            "triggered": int(bool(latest_diagnostics["triggered"])),
            "active": int(bool(latest_diagnostics["active"])),
            "engaged": int(bool(latest_diagnostics["engaged"])),
            "released": int(bool(latest_diagnostics["released"])),
            "previous_growth_weight": float(
                latest_diagnostics["previous_growth_weight"]
            ),
            "previous_weighted_growth_return": float(
                latest_diagnostics[
                    "previous_weighted_growth_return"
                ]
            ),
            "qqq_63d_return": float(
                latest_diagnostics["qqq_63d_return"]
            ),
            "semis_63d_return": float(
                latest_diagnostics["semis_63d_return"]
            ),
            "semis_126d_return": float(
                latest_diagnostics["semis_126d_return"]
            ),
            "vix_term_ratio": float(
                latest_diagnostics["vix_term_ratio"]
            ),
            "baseline_growth_weight": float(
                latest_diagnostics["baseline_growth_weight"]
            ),
            "shadow_growth_weight": float(
                latest_diagnostics["implemented_growth_weight"]
            ),
            "point_estimate_gate_pass": int(
                bool(qualification["point_estimate_gate_pass"])
            ),
            "multiple_comparisons_gate_pass": int(
                bool(qualification["multiple_comparisons_gate_pass"])
            ),
            "familywise_p_value_max": family_p_max,
            "production_eligible": 0,
            "real_capital_share": 0.0,
            "orders_executable": 0,
            "order_blocker": (
                "multiple-comparisons qualification failed; "
                "shadow observation only"
            ),
        },
        name="value",
    )
    diagnostics_output.to_csv(output / "next_signal_diagnostics.csv")

    source_paths = [
        config_path,
        data["r11_config"],
        data["core_open_close"],
        data["market_prices"],
        data["r11_output"] / "next_target_weights.csv",
        data["r11_output"] / "next_signal_diagnostics.csv",
        data["evaluation_output"]
        / "declared_family_reality_check.csv",
    ]
    metadata = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": price_as_of.date().isoformat(),
        "next_session": next_session.date().isoformat(),
        "shadow_state": state,
        "production_eligible": False,
        "orders_executable": False,
        "source_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in source_paths
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    row = {
        "strategy_name": str(config["name"]),
        "signal_date": price_as_of.date().isoformat(),
        "next_session": next_session.date().isoformat(),
        "release": str(config["release"]),
        "config_sha256": file_sha256(config_path),
        "shadow_state": state,
        "triggered": bool(latest_diagnostics["triggered"]),
        "active": bool(latest_diagnostics["active"]),
        "engaged": bool(latest_diagnostics["engaged"]),
        "released": bool(latest_diagnostics["released"]),
        "production_growth_weight": float(
            production_full.loc[["SPX", "QQQ", "SEMIS"]].sum()
        ),
        "shadow_growth_weight": float(
            shadow_full.loc[["SPX", "QQQ", "SEMIS"]].sum()
        ),
        "target_one_way_turnover_difference": (
            0.5
            * float(
                (shadow_staged - production_staged).abs().sum()
            )
        ),
        "familywise_p_value_max": family_p_max,
        "production_eligible": False,
    }
    append_immutable(
        data["forward_log"],
        row,
        keys=["strategy_name", "signal_date"],
    )
    print(target_output.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostics_output.to_string())
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()
