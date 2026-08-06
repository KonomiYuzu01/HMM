from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from regime_strategy.operations import (
    latest_completed_us_equity_session,
    missing_recent_us_equity_sessions,
    next_session_execution_status,
    next_us_equity_session,
    production_signal_generation_status,
)
from regime_strategy.r10_overlay import (
    BASE_ASSETS,
    R10_ASSETS,
    blend_rollout_target,
    r10_band_targets,
    r10_center_target,
    target_from_actual_gde,
)
from tools.build_r10_production_overlay import TICKERS, selected_guard
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
from tools.evaluate_r38_accelerating_volatility_capacity_fill import (
    accelerating_volatility_schedule,
)
import tools.evaluate_r38_convex_semiconductor_overlay as r38


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "config/paper_core_growth_gold20_r38_convex_overlay.yaml"
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def validate_generation_request(
    input_dates: dict[str, pd.Timestamp],
    generation_status: str,
    reuse_latest_complete_inputs: bool,
) -> None:
    if len(set(input_dates.values())) != 1:
        formatted = ", ".join(
            f"{name}={value.date()}"
            for name, value in input_dates.items()
        )
        raise RuntimeError(
            f"R38 production inputs are not synchronized: {formatted}"
        )
    if (
        generation_status != "AVAILABLE"
        and not reuse_latest_complete_inputs
    ):
        raise RuntimeError(
            "R38 production generation is blocked during the incomplete "
            "U.S. trading session"
        )


def next_r38_base_target(
    r9_output: Path,
    core_open_close: Path,
    r9_target: pd.Series,
    r9_action: str,
    config: dict[str, object],
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    risk = dict(config["risk_budget"])
    overlay = dict(config["semiconductor_overlay"])
    weights = pd.read_csv(
        r9_output / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        r9_output / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    frame = pd.read_csv(
        core_open_close,
        index_col=0,
        parse_dates=True,
    )
    opens = frame[
        [f"open_{asset}" for asset in BASE_ASSETS]
    ].copy()
    closes = frame[
        [f"close_{asset}" for asset in BASE_ASSETS]
    ].copy()
    opens.columns = list(BASE_ASSETS)
    closes.columns = list(BASE_ASSETS)
    latest = weights.index.intersection(opens.index).max()
    next_date = next_us_equity_session(latest)

    extended_weights = weights.copy()
    extended_weights.loc[next_date, list(BASE_ASSETS)] = (
        r9_target.reindex(BASE_ASSETS).to_numpy(dtype=float)
    )
    extended_daily = daily.copy()
    extended_daily.loc[next_date, "turnover"] = (
        1.0 if r9_action == "REBALANCE" else 0.0
    )
    flat_prices = closes.loc[latest, list(BASE_ASSETS)]
    extended_opens = opens.copy()
    extended_closes = closes.copy()
    extended_opens.loc[next_date, list(BASE_ASSETS)] = (
        flat_prices.to_numpy(dtype=float)
    )
    extended_closes.loc[next_date, list(BASE_ASSETS)] = (
        flat_prices.to_numpy(dtype=float)
    )

    base_multiplier = float(risk["base_non_cash_multiplier"])
    stable_multiplier = float(
        risk["stable_volatility_relative_multiplier"]
    )
    accelerating_multiplier = float(
        risk["accelerating_volatility_relative_multiplier"]
    )
    shock_multiplier = float(
        risk["one_session_shock_absolute_multiplier"]
    )
    cash_floor = float(risk["maximum_negative_cash_weight"])
    r38._configure(
        base_multiplier=base_multiplier,
        active_multiplier=stable_multiplier,
        shock_multiplier=shock_multiplier,
    )
    scheduled, execution_daily, risk_diagnostics = (
        accelerating_volatility_schedule(
            extended_weights,
            extended_daily,
            extended_closes,
            cash_floor=cash_floor,
            low_vol_active_multiplier=stable_multiplier,
            high_vol_active_multiplier=accelerating_multiplier,
            base_multiplier=base_multiplier,
            shock_multiplier=shock_multiplier,
        )
    )
    tilted, tilted_daily, overlay_diagnostics = (
        r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution_daily,
            extended_closes,
            rebalance_days=int(
                overlay["update_interval_trading_days"]
            ),
            max_tilt=float(overlay["overlay_fraction"]),
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
        extended_opens,
        extended_closes,
        selected_guard(config, strategy_label="R38"),
        cost_bps=float(
            dict(config["execution"])[
                "ordinary_assets_one_way_cost_bps"
            ]
        ),
        start_date="2015-01-01",
    )
    limited, _, limit_diagnostics = enforce_daily_cash_target_limit(
        guard_weights,
        guard_daily,
        cash_floor=cash_floor,
    )
    selected = limited.loc[next_date, list(BASE_ASSETS)].astype(float)
    diagnostics = pd.concat(
        [
            risk_diagnostics.loc[next_date],
            overlay_diagnostics.loc[next_date],
            guard_daily.loc[next_date].add_prefix("smh_guard_"),
            limit_diagnostics.loc[next_date],
        ]
    )
    if selected["CASH"] < cash_floor - 1e-12:
        raise AssertionError("R38 production target exceeded cash floor")
    if abs(float(selected.sum()) - 1.0) > 1e-12:
        raise AssertionError("R38 production base target does not sum to one")
    if float(diagnostics["growth_budget_error"]) > 1e-12:
        raise AssertionError("R38 production overlay changed growth budget")
    if (
        float(diagnostics["overlay_absolute_deviation"])
        > float(overlay["overlay_fraction"]) + 1e-12
    ):
        raise AssertionError("R38 production overlay exceeded its budget")
    return selected, diagnostics, next_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the staged R38 production target"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--actual-gde-weight", type=float)
    parser.add_argument(
        "--reuse-latest-complete-inputs",
        action="store_true",
        help=(
            "Allow an intraday build only when every input is synchronized "
            "to the latest completed U.S. equity session."
        ),
    )
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])
    capital = dict(config["capital_efficiency"])
    rollout = dict(config["rollout"])
    r9_output = _read_path(data["r9_output"])
    r11_output = _read_path(data["r11_output"])
    core_open_close = _read_path(data["core_open_close"])
    gde_open_close = _read_path(data["gde_open_close"])
    output = _read_path(data["output_dir"])

    now_et = pd.Timestamp.now(tz="America/New_York")
    generation_status = production_signal_generation_status(now_et)
    expected_as_of = latest_completed_us_equity_session(now_et)
    r9_metadata = json.loads(
        (r9_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    r11_metadata = json.loads(
        (r11_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    core_frame = pd.read_csv(
        core_open_close,
        index_col=0,
        parse_dates=True,
    )
    gde_frame = pd.read_csv(
        gde_open_close,
        index_col=0,
        parse_dates=True,
    )
    input_dates = {
        "expected": expected_as_of,
        "R9": pd.Timestamp(r9_metadata["price_as_of"]).normalize(),
        "R11": pd.Timestamp(r11_metadata["price_as_of"]).normalize(),
        "core": core_frame.index.max().normalize(),
        "GDE": gde_frame.index.max().normalize(),
    }
    if str(config.get("status", "")).lower() == "staged_production":
        validate_generation_request(
            input_dates,
            generation_status,
            arguments.reuse_latest_complete_inputs,
        )
    for name, frame in (("core", core_frame), ("GDE", gde_frame)):
        missing = missing_recent_us_equity_sessions(frame.index, now_et)
        if len(missing):
            formatted = ",".join(
                date.date().isoformat() for date in missing
            )
            raise RuntimeError(
                f"R38 {name} input has recent session gaps: {formatted}"
            )

    r9_next = pd.read_csv(
        r9_output / "next_target_weights.csv",
        index_col=0,
    )
    r9_target = r9_next[
        "ensemble_current_sleeve_weight"
    ].reindex(BASE_ASSETS).astype(float)
    r9_diagnostics = pd.read_csv(
        r9_output / "next_signal_diagnostics.csv",
        index_col=0,
    )
    actions = sorted(set(r9_diagnostics.loc["action"].astype(str)))
    if len(actions) != 1:
        raise RuntimeError(f"R9 members disagree on action: {actions}")
    r38_base, diagnostics, next_date = next_r38_base_target(
        r9_output,
        core_open_close,
        r9_target,
        actions[0],
        config,
    )
    execution_window_status, execution_date = (
        next_session_execution_status(
            input_dates["core"],
            now=now_et,
        )
    )
    if execution_date.normalize() != next_date.normalize():
        raise RuntimeError(
            "R38 execution date does not match the next strategy session"
        )
    full_center = r10_center_target(
        r38_base,
        substitution_fraction=float(
            capital["maximum_gold_sleeve_substitution_fraction"]
        ),
    )
    full_lower, full_upper = r10_band_targets(
        full_center,
        no_trade_band=float(
            capital["account_weight_no_trade_band"]
        ),
    )
    r11_targets = pd.read_csv(
        r11_output / "next_target_weights.csv",
        index_col=0,
    )
    r11_reference = r11_targets["r11_full_center"].reindex(
        R10_ASSETS
    ).astype(float)
    rollout_share = float(
        rollout["initial_r38_share_of_managed_capital"]
    )
    staged_center = blend_rollout_target(
        r11_reference,
        full_center,
        rollout_share=rollout_share,
    )
    staged_lower = blend_rollout_target(
        r11_reference,
        full_lower,
        rollout_share=rollout_share,
    )
    staged_upper = blend_rollout_target(
        r11_reference,
        full_upper,
        rollout_share=rollout_share,
    )
    actual_target = (
        None
        if arguments.actual_gde_weight is None
        else target_from_actual_gde(
            r11_reference,
            full_center,
            rollout_share=rollout_share,
            no_trade_band=float(
                capital["account_weight_no_trade_band"]
            ),
            actual_account_gde_weight=arguments.actual_gde_weight,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    target_output = pd.DataFrame(index=R10_ASSETS)
    target_output.index.name = "asset"
    target_output["ticker"] = [
        TICKERS[asset] for asset in target_output.index
    ]
    target_output["r11_reference_target"] = r11_reference
    target_output["r38_risk_adjusted_base"] = r38_base.reindex(
        R10_ASSETS,
        fill_value=0.0,
    )
    target_output["r38_full_lower"] = full_lower
    target_output["r38_full_center"] = full_center
    target_output["r38_full_upper"] = full_upper
    target_output["staged_account_lower"] = staged_lower
    target_output["staged_account_center"] = staged_center
    target_output["staged_account_upper"] = staged_upper
    if actual_target is not None:
        target_output["position_aware_target"] = actual_target
    target_output.to_csv(output / "next_target_weights.csv")

    blockers = ["missing complete real positions"]
    if arguments.actual_gde_weight is None:
        blockers.append("missing actual GDE weight")
    diagnostic_output = pd.Series(
        {
            "release": config["release"],
            "status": config["status"],
            "core_price_as_of": input_dates["core"].date().isoformat(),
            "gde_price_as_of": input_dates["GDE"].date().isoformat(),
            "next_session": next_date.date().isoformat(),
            "r9_action": actions[0],
            "trend_permission": int(
                bool(diagnostics["trend_permission"])
            ),
            "one_session_shock_active": int(
                bool(diagnostics["pulse_veto_active"])
            ),
            "volatility_acceleration_block": int(
                bool(diagnostics["volatility_acceleration_block"])
            ),
            "state_active_multiplier": float(
                diagnostics["state_active_multiplier"]
            ),
            "accepted_absolute_multiplier": float(
                diagnostics["accepted_absolute_multiplier"]
            ),
            "cash_hard_limit_active": int(
                bool(diagnostics["hard_limit_active"])
            ),
            "r38_pre_gde_cash_weight": float(r38_base["CASH"]),
            "base_semis_growth_share": float(
                diagnostics["base_semis_growth_share"]
            ),
            "full_signal_semis_growth_share": float(
                diagnostics["full_signal_semis_growth_share"]
            ),
            "r38_semis_growth_share": float(
                diagnostics["semis_growth_share"]
            ),
            "overlay_fraction": float(diagnostics["overlay_fraction"]),
            "smh_guard_triggered": int(
                bool(diagnostics["smh_guard_triggered"])
            ),
            "smh_guard_active": int(
                bool(diagnostics["smh_guard_active"])
            ),
            "smh_signal_gap": float(
                diagnostics["smh_guard_trigger_semis_gap"]
            ),
            "smh_relative_signal_gap": float(
                diagnostics["smh_guard_trigger_relative_gap"]
            ),
            "initial_r38_rollout_share": rollout_share,
            "full_r38_gde_center": float(full_center["GDE"]),
            "staged_account_gde_center": float(staged_center["GDE"]),
            "orders_executable": 0,
            "order_blockers": "; ".join(blockers),
        },
        name="value",
    )
    diagnostic_output.to_csv(output / "next_signal_diagnostics.csv")
    metadata = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": input_dates["core"].date().isoformat(),
        "expected_price_as_of": expected_as_of.date().isoformat(),
        "data_is_fresh": True,
        "generation_window_status": generation_status,
        "generation_mode": (
            "REUSED_LATEST_COMPLETE_INPUTS"
            if arguments.reuse_latest_complete_inputs
            else "STANDARD_REFRESH"
        ),
        "execution_window_status": execution_window_status,
        "next_session": next_date.date().isoformat(),
        "orders_executable": False,
        "order_blockers": blockers,
        "source_hashes": {
            str(config_path): file_sha256(config_path),
            str(core_open_close): file_sha256(core_open_close),
            str(gde_open_close): file_sha256(gde_open_close),
            str(r9_output / "next_target_weights.csv"): file_sha256(
                r9_output / "next_target_weights.csv"
            ),
            str(r11_output / "next_target_weights.csv"): file_sha256(
                r11_output / "next_target_weights.csv"
            ),
        },
        "evidence": config["validation"],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sums = target_output[
        [
            "r38_full_lower",
            "r38_full_center",
            "r38_full_upper",
            "staged_account_lower",
            "staged_account_center",
            "staged_account_upper",
        ]
    ].sum()
    if not np.allclose(sums, 1.0, atol=1e-12):
        raise AssertionError(f"R38 target sums are invalid: {sums}")
    print(target_output.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostic_output.to_string())
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()
