from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import time
import json
from pathlib import Path

import pandas as pd
import yaml

from .backtest import RegimeBacktester
from .data import load_prices
from .ensemble import combine_model_sleeves, write_ensemble_report
from .operations import (
    latest_completed_us_equity_session,
    production_signal_generation_status,
)


def _deep_merge(
    base: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_ensemble_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent_path = config.pop("parent_config", None)
    if parent_path is None:
        return config
    parent = _load_ensemble_config(Path(str(parent_path)))
    return _deep_merge(parent, config)


def _member_specs(
    random_seeds: list[int],
    model_lookback_days: list[int] | None,
) -> list[tuple[str, int, int | None]]:
    seeds = [int(seed) for seed in random_seeds]
    if not model_lookback_days:
        return [(f"seed_{seed}", seed, None) for seed in seeds]
    lookbacks = [int(days) for days in model_lookback_days]
    if any(days <= 0 for days in lookbacks) or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("Model lookback days must be unique positive integers")
    return [
        (f"seed_{seed}_lookback_{lookback}", seed, lookback)
        for lookback in lookbacks
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robust multi-seed HMM ensemble")
    parser.add_argument("--config", default="config/paper_core_robust_ensemble.yaml")
    parser.add_argument("--refresh", action="store_true", help="Refresh adjusted prices")
    parser.add_argument("--cost-bps", type=float)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    ensemble_config = _load_ensemble_config(Path(args.config))
    base_config = yaml.safe_load(
        Path(ensemble_config["base_config"]).read_text(encoding="utf-8")
    )
    member_config = dict(ensemble_config["member_overrides"])
    outer_config = dict(ensemble_config["ensemble"])
    for key, value in ensemble_config.get("data_overrides", {}).items():
        base_config["data"][str(key)] = deepcopy(value)
    freshness_config = dict(
        ensemble_config.get("production_freshness", {})
    )
    freshness_enabled = bool(freshness_config.get("enabled", False))
    completion_time = time.fromisoformat(
        str(freshness_config.get("completion_time_et", "16:15"))
    )
    now_et = pd.Timestamp.now(tz="America/New_York")
    generation_status = production_signal_generation_status(
        now_et,
        completion_time=completion_time,
    )
    if freshness_enabled and generation_status != "AVAILABLE":
        raise RuntimeError(
            "Production signal generation is blocked during the incomplete "
            "U.S. trading session; rerun after 16:15 America/New_York"
        )
    cost_bps = float(
        args.cost_bps
        if args.cost_bps is not None
        else outer_config["cost_bps_per_dollar_traded"]
    )
    prices = load_prices(base_config["data"], refresh=args.refresh)
    expected_price_as_of = latest_completed_us_equity_session(
        now_et,
        completion_time=completion_time,
    )
    observed_price_as_of = prices.index[-1].normalize()
    data_is_fresh = observed_price_as_of == expected_price_as_of
    if freshness_enabled and not data_is_fresh:
        raise RuntimeError(
            "Production prices are stale: expected "
            f"{expected_price_as_of.date()}, observed "
            f"{observed_price_as_of.date()}"
        )

    members = {}
    backtesters = {}
    member_specs = _member_specs(
        list(ensemble_config["random_seeds"]),  # type: ignore[arg-type]
        (
            list(ensemble_config["model_lookback_days"])  # type: ignore[arg-type]
            if "model_lookback_days" in ensemble_config
            else None
        ),
    )
    for member_name, seed, lookback_days in member_specs:
        config = deepcopy(base_config)
        config["model"]["random_seed"] = seed
        if lookback_days is not None:
            config["model"]["lookback_days"] = lookback_days
        config["portfolio"]["max_gross_leverage"] = float(
            member_config["max_gross_leverage"]
        )
        if "risk_off_growth_floor" in member_config:
            config["portfolio"]["risk_off_growth_floor"] = deepcopy(
                member_config["risk_off_growth_floor"]
            )
        for key, value in member_config.get("portfolio_overrides", {}).items():
            config["portfolio"][str(key)] = deepcopy(value)
        for key, value in member_config.get("model_overrides", {}).items():
            config["model"][str(key)] = deepcopy(value)
        for key, value in member_config.get("feature_overrides", {}).items():
            config["features"][str(key)] = deepcopy(value)
        for key, value in member_config.get("backtest_overrides", {}).items():
            config["backtest"][str(key)] = deepcopy(value)
        config["portfolio"]["cost_bps_per_dollar_traded"] = cost_bps
        backtester = RegimeBacktester(config)
        members[member_name] = backtester.run(prices)
        backtesters[member_name] = backtester

    result = combine_model_sleeves(
        members,
        int(outer_config["rebalance_every_days"]),
        float(outer_config["no_trade_turnover"]),
        cost_bps,
        prices.index
        if bool(outer_config.get("calendar_anchored_schedule", False))
        else None,
        bool(outer_config.get("net_member_trades", False)),
        (
            dict(outer_config["disagreement_risk_control"])
            if "disagreement_risk_control" in outer_config
            else None
        ),
    )
    output_dir = Path(args.output_dir or ensemble_config["output_dir"])
    metrics = write_ensemble_report(result, output_dir)
    for name, member_result in members.items():
        member_dir = output_dir / "members" / name
        member_dir.mkdir(parents=True, exist_ok=True)
        member_result.daily.to_csv(member_dir / "daily_returns.csv")
        member_result.weights.to_csv(member_dir / "weights.csv")
        member_result.regimes.to_csv(member_dir / "regimes.csv")

    targets = {}
    diagnostics = {}
    for name, backtester in backtesters.items():
        target, member_diagnostics = backtester.recommend_next(prices, members[name])
        targets[name] = target
        diagnostics[name] = member_diagnostics
    target_frame = pd.DataFrame(targets)
    target_frame["ensemble_equal_weight"] = target_frame.mean(axis=1)
    last_starting_shares = result.sleeve_weights.iloc[-1]
    last_member_returns = result.member_returns.iloc[-1]
    current_shares = last_starting_shares * (1.0 + last_member_returns)
    current_shares /= current_shares.sum()
    target_frame["ensemble_current_sleeve_weight"] = target_frame[
        list(members)
    ].mul(current_shares, axis=1).sum(axis=1)
    target_frame.to_csv(output_dir / "next_target_weights.csv", index_label="asset")
    pd.DataFrame(diagnostics).to_csv(output_dir / "next_signal_diagnostics.csv")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "price_as_of": prices.index[-1].date().isoformat(),
                "expected_price_as_of": (
                    expected_price_as_of.date().isoformat()
                ),
                "data_is_fresh": data_is_fresh,
                "generation_window_status": generation_status,
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "config_path": str(Path(args.config).resolve()),
                "refresh_requested": bool(args.refresh),
                "data_quality": prices.attrs.get("data_quality", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(metrics.round(4).to_string())
    print("\nCurrent-sleeve-weighted next target:")
    print(
        target_frame["ensemble_current_sleeve_weight"]
        .sort_values(ascending=False)
        .round(4)
    )
    print(f"\nArtifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
