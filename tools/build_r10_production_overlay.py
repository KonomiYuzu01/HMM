from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_strategy.operations import (
    latest_completed_us_equity_session,
    missing_recent_us_equity_sessions,
    next_us_equity_session,
    production_signal_generation_status,
)
from regime_strategy.r10_overlay import (
    BASE_ASSETS,
    R10_ASSETS,
    blend_rollout_target,
    r10_band_targets,
    r10_center_target,
    risk_scaled_target,
    target_from_actual_gde,
)
from tools.evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    simulate as simulate_guard,
)


DEFAULT_CONFIG = Path(
    "config/paper_core_growth_gold20_r10_capital_efficient_guard.yaml"
)
TICKERS = {
    "SPX": "SPY",
    "QQQ": "QQQ",
    "SEMIS": "SMH",
    "BOND": "IEF",
    "GOLD": "GLD",
    "OIL": "DBC",
    "USD": "UUP",
    "CASH": "BIL / cash",
    "VIX_HEDGE": "VIXY",
    "GDE": "GDE",
}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def selected_guard(
    config: dict[str, object],
    *,
    strategy_label: str = "R10",
) -> OpenGapGuard:
    guard = dict(config["smh_gap_guard"])
    return OpenGapGuard(
        name=f"{strategy_label.lower()}_production_smh_gap_guard",
        absolute_gap_trigger=float(guard["absolute_smh_gap_trigger"]),
        relative_gap_trigger=float(
            guard["relative_smh_minus_qqq_gap_trigger"]
        ),
        post_trigger_cap=float(guard["post_trigger_smh_cap"]),
        trigger_delay_days=int(guard["execution_delay_trading_days"]),
        minimum_semis_weight=float(
            guard["minimum_pretrigger_smh_weight"]
        ),
        recovery_signal="relative",
        recovery_confirmations=int(guard["recovery_confirmations"]),
        maximum_hold_days=int(guard["maximum_hold_trading_days"]),
        overflow_asset=str(guard["overflow_asset"]),
        emergency_slippage_bps=float(
            guard["modeled_emergency_slippage_bps"]
        ),
    )


def next_guard_target(
    r9_output: Path,
    core_open_close: Path,
    base_reference_target: pd.Series,
    guard: OpenGapGuard,
    r9_action: str,
    risk_multiplier: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
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
    non_cash_assets = [
        asset for asset in BASE_ASSETS if asset != "CASH"
    ]
    extended_weights.loc[:, non_cash_assets] *= risk_multiplier
    extended_weights.loc[:, "CASH"] = (
        1.0 - extended_weights.loc[:, non_cash_assets].sum(axis=1)
    )
    scaled_reference_target = risk_scaled_target(
        base_reference_target,
        risk_multiplier=risk_multiplier,
    )
    extended_weights.loc[next_date, list(BASE_ASSETS)] = (
        scaled_reference_target.reindex(BASE_ASSETS).to_numpy(dtype=float)
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
    guard_daily, guard_weights = simulate_guard(
        extended_weights,
        extended_daily,
        extended_opens,
        extended_closes,
        guard,
        start_date="2015-01-01",
    )
    diagnostics = guard_daily.loc[next_date]
    event_or_active = bool(
        diagnostics[
            [
                "triggered",
                "active",
                "recovered",
                "deepened",
                "bridged",
            ]
        ]
        .astype(bool)
        .any()
    )
    selected_target = (
        guard_weights.loc[next_date, list(BASE_ASSETS)]
        if event_or_active
        else scaled_reference_target.reindex(BASE_ASSETS).astype(float)
    )
    return selected_target, diagnostics, next_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a staged production target and guard state"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--actual-gde-weight",
        type=float,
        help=(
            "Actual GDE weight as a share of all R9-managed capital. "
            "This refines the target but does not make orders executable."
        ),
    )
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    strategy_label = str(config.get("strategy_label", "R10")).upper()
    strategy_prefix = strategy_label.lower()
    data = dict(config["data"])
    capital = dict(config["capital_efficiency"])
    risk_budget = dict(config.get("risk_budget", {}))
    rollout = dict(config["rollout"])
    risk_multiplier = float(
        risk_budget.get("target_non_cash_multiplier", 1.0)
    )
    r9_output = Path(str(data["r9_output"]))
    core_open_close = Path(str(data["core_open_close"]))
    gde_open_close = Path(str(data["gde_open_close"]))
    output = Path(str(data["output_dir"]))
    now_et = pd.Timestamp.now(tz="America/New_York")
    generation_status = production_signal_generation_status(now_et)
    if (
        str(config.get("status", "")).lower() == "staged_production"
        and generation_status != "AVAILABLE"
    ):
        raise RuntimeError(
            f"{strategy_label} production generation is blocked during the "
            "incomplete U.S. trading session"
        )

    r9_metadata = json.loads(
        (r9_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    r9_as_of = pd.Timestamp(r9_metadata["price_as_of"]).normalize()
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
    core_as_of = core_frame.index.max().normalize()
    gde_as_of = gde_frame.index.max().normalize()
    expected_as_of = latest_completed_us_equity_session(now_et)
    input_dates = {
        "expected": expected_as_of,
        "R9": r9_as_of,
        "core": core_as_of,
        "GDE": gde_as_of,
    }
    if len(set(input_dates.values())) != 1:
        formatted = ", ".join(
            f"{name}={value.date()}"
            for name, value in input_dates.items()
        )
        raise RuntimeError(
            f"{strategy_label} production inputs are not synchronized: "
            f"{formatted}"
        )
    recent_gaps = {
        "core": missing_recent_us_equity_sessions(
            core_frame.index,
            now_et,
        ),
        "GDE": missing_recent_us_equity_sessions(
            gde_frame.index,
            now_et,
        ),
    }
    missing_descriptions = [
        f"{name}={','.join(date.date().isoformat() for date in missing)}"
        for name, missing in recent_gaps.items()
        if len(missing)
    ]
    if missing_descriptions:
        raise RuntimeError(
            f"{strategy_label} production inputs have recent session gaps: "
            + "; ".join(missing_descriptions)
        )

    output.mkdir(parents=True, exist_ok=True)

    next_frame = pd.read_csv(
        r9_output / "next_target_weights.csv",
        index_col=0,
    )
    r9_target = next_frame[
        "ensemble_current_sleeve_weight"
    ].reindex(BASE_ASSETS).astype(float)
    diagnostics_frame = pd.read_csv(
        r9_output / "next_signal_diagnostics.csv",
        index_col=0,
    )
    actions = sorted(
        set(diagnostics_frame.loc["action"].astype(str))
    )
    if len(actions) != 1:
        raise RuntimeError(f"R9 members disagree on action: {actions}")
    r9_action = actions[0]
    guard_target, guard_diagnostics, next_date = next_guard_target(
        r9_output,
        core_open_close,
        r9_target,
        selected_guard(config, strategy_label=strategy_label),
        r9_action,
        risk_multiplier=risk_multiplier,
    )
    center = r10_center_target(
        guard_target,
        substitution_fraction=float(
            capital["maximum_gold_sleeve_substitution_fraction"]
        ),
    )
    lower, upper = r10_band_targets(
        center,
        no_trade_band=float(
            capital["account_weight_no_trade_band"]
        ),
    )
    rollout_key = (
        f"initial_{strategy_prefix}_share_of_managed_capital"
    )
    if rollout_key not in rollout:
        rollout_key = "initial_r10_share_of_managed_capital"
    rollout_share = float(rollout[rollout_key])
    staged_center = blend_rollout_target(
        r9_target,
        center,
        rollout_share=rollout_share,
    )
    staged_lower = blend_rollout_target(
        r9_target,
        lower,
        rollout_share=rollout_share,
    )
    staged_upper = blend_rollout_target(
        r9_target,
        upper,
        rollout_share=rollout_share,
    )
    actual_target = (
        None
        if arguments.actual_gde_weight is None
        else target_from_actual_gde(
            r9_target,
            center,
            rollout_share=rollout_share,
            no_trade_band=float(
                capital["account_weight_no_trade_band"]
            ),
            actual_account_gde_weight=arguments.actual_gde_weight,
        )
    )

    target_output = pd.DataFrame(index=R10_ASSETS)
    target_output.index.name = "asset"
    target_output["ticker"] = [
        TICKERS[asset] for asset in target_output.index
    ]
    target_output["r9_reference_target"] = r9_target.reindex(
        R10_ASSETS,
        fill_value=0.0,
    )
    target_output[
        f"{strategy_prefix}_guard_adjusted_base"
    ] = guard_target.reindex(
        R10_ASSETS,
        fill_value=0.0,
    )
    target_output[f"{strategy_prefix}_full_lower"] = lower
    target_output[f"{strategy_prefix}_full_center"] = center
    target_output[f"{strategy_prefix}_full_upper"] = upper
    target_output["staged_account_lower"] = staged_lower
    target_output["staged_account_center"] = staged_center
    target_output["staged_account_upper"] = staged_upper
    if actual_target is not None:
        target_output["position_aware_target"] = actual_target
    target_output.to_csv(output / "next_target_weights.csv")

    blockers = ["missing complete real positions"]
    if gde_as_of.normalize() != core_as_of.normalize():
        blockers.append(
            f"stale GDE quote ({gde_as_of.date()} vs {core_as_of.date()})"
        )
    if arguments.actual_gde_weight is None:
        blockers.append("missing actual GDE weight")
    orders_executable = False
    validation = dict(config["validation"])
    strict_family = dict(
        validation.get("strict_smh_family_reality_check", {})
    )
    frontier_family = dict(
        validation.get("frontier_family_reality_check", {})
    )
    diagnostics = pd.Series(
        {
            "release": config["release"],
            "status": config["status"],
            "core_price_as_of": core_as_of.date().isoformat(),
            "gde_price_as_of": gde_as_of.date().isoformat(),
            "next_session": next_date.date().isoformat(),
            "r9_action": r9_action,
            "smh_guard_triggered": int(
                bool(guard_diagnostics["triggered"])
            ),
            "smh_guard_active": int(bool(guard_diagnostics["active"])),
            "smh_guard_recovered": int(
                bool(guard_diagnostics["recovered"])
            ),
            "smh_signal_gap": float(
                guard_diagnostics["trigger_semis_gap"]
            ),
            "smh_relative_signal_gap": float(
                guard_diagnostics["trigger_relative_gap"]
            ),
            "target_non_cash_multiplier": risk_multiplier,
            f"full_{strategy_prefix}_gde_center": float(center["GDE"]),
            f"full_{strategy_prefix}_gde_lower": float(lower["GDE"]),
            f"full_{strategy_prefix}_gde_upper": float(upper["GDE"]),
            f"initial_{strategy_prefix}_rollout_share": rollout_share,
            "staged_account_gde_center": float(staged_center["GDE"]),
            "staged_account_gde_lower": float(staged_lower["GDE"]),
            "staged_account_gde_upper": float(staged_upper["GDE"]),
            "orders_executable": int(orders_executable),
            "order_blockers": "; ".join(blockers),
            "strict_smh_family_pass": int(
                bool(strict_family.get("pass_at_5pct", False))
            ),
            "frontier_family_pass": int(
                bool(frontier_family.get("pass_at_5pct", False))
            ),
        },
        name="value",
    )
    diagnostics.to_csv(output / "next_signal_diagnostics.csv")
    metadata = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": core_as_of.date().isoformat(),
        "expected_price_as_of": expected_as_of.date().isoformat(),
        "data_is_fresh": True,
        "generation_window_status": generation_status,
        "next_session": next_date.date().isoformat(),
        "orders_executable": orders_executable,
        "order_blockers": blockers,
        "source_hashes": {
            str(config_path): file_sha256(config_path),
            str(core_open_close): file_sha256(core_open_close),
            str(gde_open_close): file_sha256(gde_open_close),
            str(r9_output / "next_target_weights.csv"): file_sha256(
                r9_output / "next_target_weights.csv"
            ),
            str(
                r9_output / "next_signal_diagnostics.csv"
            ): file_sha256(
                r9_output / "next_signal_diagnostics.csv"
            ),
        },
        "evidence": validation,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(target_output.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostics.to_string())
    print(f"\nArtifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
