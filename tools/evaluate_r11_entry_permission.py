from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_smh_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    ASSETS,
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    relative_log_return,
    scale_non_cash_weights,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r11_post_shock_cooldown import (
    GROWTH_ASSETS,
    ROLLOUT_SHARES,
    metric_delta,
    sample_definitions,
)
from tools.evaluate_r11_selective_shock_memory import (
    build_causal_signals,
    load_market_prices,
)


OUTPUT = Path("output/r11_entry_permission")
RISK_MULTIPLIER = 1.065
GDE_FRACTION = 0.10
SHORT_TREND_SESSIONS = 20
PREVIOUS_REENTRY_TRIALS = 48


@dataclass(frozen=True)
class EntryPermissionPolicy:
    name: str = "hierarchical_permission_10_35_full"
    damage_growth_budget: float = 0.10
    early_repair_growth_budget: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.damage_growth_budget <= 1.0:
            raise ValueError("damage_growth_budget must be in [0, 1]")
        if not 0.0 <= self.early_repair_growth_budget <= 1.0:
            raise ValueError(
                "early_repair_growth_budget must be in [0, 1]"
            )
        if self.early_repair_growth_budget < self.damage_growth_budget:
            raise ValueError(
                "early repair budget cannot be below damage budget"
            )


def build_entry_permission_signals(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    market_prices: pd.DataFrame,
) -> pd.DataFrame:
    signals = build_causal_signals(weights, closes, market_prices)
    market = market_prices.reindex(signals.index)
    signals["qqq_20d_return"] = (
        market["QQQ"].pct_change(SHORT_TREND_SESSIONS).shift(1)
    )
    signals["semis_20d_return"] = (
        market["SEMIS"].pct_change(SHORT_TREND_SESSIONS).shift(1)
    )
    return signals


def classify_growth_regime(signal: pd.Series) -> str:
    required = (
        "qqq_63d_return",
        "semis_63d_return",
        "semis_126d_return",
        "vix_term_ratio",
    )
    if any(pd.isna(signal.get(name)) for name in required):
        return "unknown"
    qqq_63 = float(signal["qqq_63d_return"])
    semis_63 = float(signal["semis_63d_return"])
    semis_126 = float(signal["semis_126d_return"])
    vix_ratio = float(signal["vix_term_ratio"])
    if semis_126 > 0.0:
        if qqq_63 > 0.0 and semis_63 > 0.0:
            return (
                "healthy"
                if vix_ratio <= 1.0
                else "liquidity_stress"
            )
        return "ordinary_correction"
    if qqq_63 > 0.0 and semis_63 > 0.0 and vix_ratio <= 1.0:
        return "early_repair"
    return "structural_damage"


def project_growth_addition(
    current: pd.Series,
    desired: pd.Series,
    maximum_growth_weight: float,
) -> pd.Series:
    current = current.loc[ASSETS].astype(float)
    projected = desired.loc[ASSETS].astype(float).copy()
    current_growth = float(current.loc[list(GROWTH_ASSETS)].sum())
    desired_growth = float(projected.loc[list(GROWTH_ASSETS)].sum())
    if desired_growth <= current_growth + 1e-12:
        return projected
    implemented_growth = float(
        np.clip(
            maximum_growth_weight,
            current_growth,
            desired_growth,
        )
    )
    if desired_growth > 0.0:
        projected.loc[list(GROWTH_ASSETS)] *= (
            implemented_growth / desired_growth
        )
    projected["CASH"] += desired_growth - implemented_growth
    return projected


def permitted_growth_ceiling(
    signal: pd.Series,
    damage_trough_growth: float,
    policy: EntryPermissionPolicy,
) -> tuple[float | None, str]:
    required = (
        "qqq_20d_return",
        "semis_20d_return",
        "qqq_63d_return",
        "semis_63d_return",
        "vix_term_ratio",
    )
    if any(pd.isna(signal.get(name)) for name in required):
        return (
            damage_trough_growth + policy.damage_growth_budget,
            "damage_probe",
        )
    normal_term = float(signal["vix_term_ratio"]) <= 1.0
    medium_recovery = bool(
        float(signal["qqq_63d_return"]) > 0.0
        and float(signal["semis_63d_return"]) > 0.0
        and normal_term
    )
    if medium_recovery:
        return None, "confirmed_recovery"
    short_recovery = bool(
        float(signal["qqq_20d_return"]) > 0.0
        and float(signal["semis_20d_return"]) > 0.0
        and normal_term
    )
    if short_recovery:
        return (
            damage_trough_growth + policy.early_repair_growth_budget,
            "early_repair",
        )
    return (
        damage_trough_growth + policy.damage_growth_budget,
        "damage_probe",
    )


def apply_entry_permission(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    signals: pd.DataFrame,
    policy: EntryPermissionPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = (
        weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    adjusted_daily = base_daily.loc[dates].copy()
    aligned_signals = signals.reindex(dates)
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    damage_active = False
    damage_trough_growth: float | None = None
    weight_rows: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | bool | str]] = []

    for date in dates:
        if previous_date is not None:
            overnight_asset_returns = (
                opens.loc[date, ASSETS].to_numpy(dtype=float)
                / closes.loc[previous_date, ASSETS].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )

        current_at_open = pd.Series(current.copy(), index=ASSETS)
        current_growth = float(
            current_at_open.loc[list(GROWTH_ASSETS)].sum()
        )
        signal = aligned_signals.loc[date]
        regime = classify_growth_regime(signal)
        state_entered = False
        state_released = False
        if regime != "unknown":
            damaged_now = regime in {
                "early_repair",
                "structural_damage",
            }
            if damaged_now and not damage_active:
                damage_active = True
                damage_trough_growth = current_growth
                state_entered = True
            elif not damaged_now and damage_active:
                damage_active = False
                damage_trough_growth = None
                state_released = True

        proposal = bool(float(base_daily.loc[date, "turnover"]) > 1e-14)
        desired_growth = current_growth
        implemented_growth = current_growth
        permission_tier = (
            "unrestricted" if not damage_active else "damage_probe"
        )
        entry_limited = False
        reduction_passed = False
        turnover = 0.0
        target = current_at_open.copy()
        if proposal:
            desired = weights.loc[date, ASSETS].astype(float)
            desired_growth = float(
                desired.loc[list(GROWTH_ASSETS)].sum()
            )
            if damage_active and desired_growth > current_growth + 1e-12:
                if damage_trough_growth is None:
                    raise RuntimeError(
                        "Active damage state requires a growth trough"
                    )
                (
                    maximum_growth,
                    permission_tier,
                ) = permitted_growth_ceiling(
                    signal,
                    damage_trough_growth,
                    policy,
                )
                target = (
                    desired.copy()
                    if maximum_growth is None
                    else project_growth_addition(
                        current_at_open,
                        desired,
                        maximum_growth,
                    )
                )
                implemented_growth = float(
                    target.loc[list(GROWTH_ASSETS)].sum()
                )
                entry_limited = (
                    implemented_growth < desired_growth - 1e-12
                )
            else:
                target = desired.copy()
                implemented_growth = desired_growth
                reduction_passed = desired_growth < current_growth - 1e-12
            turnover = 0.5 * float(
                np.abs(
                    target.to_numpy(dtype=float)
                    - current_at_open.to_numpy(dtype=float)
                ).sum()
            )
            if turnover > 1e-14:
                current = target.to_numpy(dtype=float)
            if (
                damage_active
                and implemented_growth < current_growth - 1e-12
            ):
                if damage_trough_growth is None:
                    raise RuntimeError(
                        "Active damage state requires a growth trough"
                    )
                damage_trough_growth = min(
                    damage_trough_growth,
                    implemented_growth,
                )

        original_turnover = float(base_daily.loc[date, "turnover"])
        adjusted_daily.loc[date, "turnover"] = turnover
        if "slippage_cost" in adjusted_daily:
            original_slippage = float(
                base_daily.loc[date, "slippage_cost"]
            )
            adjusted_daily.loc[date, "slippage_cost"] = (
                original_slippage * turnover / original_turnover
                if original_turnover > 1e-14
                else 0.0
            )
        weight_rows.append(current.copy())
        diagnostics.append(
            {
                "regime": regime,
                "damage_active": damage_active,
                "state_entered": state_entered,
                "state_released": state_released,
                "proposal": proposal,
                "executed_trade": turnover > 1e-14,
                "state_change_trade": bool(
                    (state_entered or state_released) and not proposal
                    and turnover > 1e-14
                ),
                "entry_limited": entry_limited,
                "reduction_passed": reduction_passed,
                "permission_tier": permission_tier,
                "growth_at_open": current_growth,
                "desired_growth_weight": desired_growth,
                "implemented_growth_weight": implemented_growth,
                "damage_trough_growth": (
                    np.nan
                    if damage_trough_growth is None
                    else damage_trough_growth
                ),
                "turnover": turnover,
                "qqq_63d_return": signal.get("qqq_63d_return", np.nan),
                "semis_63d_return": signal.get(
                    "semis_63d_return",
                    np.nan,
                ),
                "semis_126d_return": signal.get(
                    "semis_126d_return",
                    np.nan,
                ),
                "vix_term_ratio": signal.get("vix_term_ratio", np.nan),
            }
        )

        intraday_asset_returns = (
            closes.loc[date, ASSETS].to_numpy(dtype=float)
            / opens.loc[date, ASSETS].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        current = current * (1.0 + intraday_asset_returns) / (
            1.0 + intraday_return
        )
        previous_date = date

    return (
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
        adjusted_daily,
        pd.DataFrame(diagnostics, index=dates),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    policy = EntryPermissionPolicy()
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(
        PROXY_OPEN_CLOSE
    )
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(NORMAL_DIRECTORY)
    proxy_weights, proxy_daily = load_strategy_inputs(PROXY_DIRECTORY)
    samples = sample_definitions(
        normal_weights,
        normal_daily,
        proxy_weights,
        proxy_daily,
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )
    market_prices = load_market_prices()
    metric_rows: list[dict[str, object]] = []
    event_rows: list[pd.DataFrame] = []
    baseline_paths: dict[str, pd.Series] = {}
    candidate_paths: dict[str, pd.Series] = {}

    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        levered = scale_non_cash_weights(weights, RISK_MULTIPLIER)

        for scenario in COST_SCENARIOS:
            r9 = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=0.0,
                gate_mode="always",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
            )
            guard_daily, guard_weights = simulate_smh_guard(
                levered,
                daily,
                opens,
                closes,
                guard_for_multiplier(
                    RISK_MULTIPLIER,
                    scenario.emergency_slippage_bps,
                ),
                cost_bps=scenario.base_one_way_cost_bps,
                start_date=str(settings["start"]),
            )
            current = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=GDE_FRACTION,
                gate_mode="growth_linked",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
                weights_override=guard_weights,
                daily_override=guard_daily,
                extra_slippage=guard_daily["slippage_cost"],
                gde_no_trade_band=GDE_NO_TRADE_BAND,
            )
            signals = build_entry_permission_signals(
                guard_weights,
                closes,
                market_prices[sample],
            )
            (
                permission_weights,
                permission_daily,
                diagnostics,
            ) = apply_entry_permission(
                guard_weights,
                guard_daily,
                opens,
                closes,
                signals,
                policy,
            )
            candidate = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=GDE_FRACTION,
                gate_mode="growth_linked",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
                weights_override=permission_weights,
                daily_override=permission_daily,
                extra_slippage=permission_daily["slippage_cost"],
                gde_no_trade_band=GDE_NO_TRADE_BAND,
            )
            common = (
                r9.index.intersection(current.index)
                .intersection(candidate.index)
            )
            if scenario.name == "current_liquidity":
                baseline_paths[sample] = current.loc[common, "net_return"]
                candidate_paths[sample] = candidate.loc[
                    common,
                    "net_return",
                ]
                saved = diagnostics.loc[common].copy()
                saved.insert(0, "sample", sample)
                event_rows.append(saved)
                candidate.to_csv(
                    OUTPUT / f"{sample}_{policy.name}_daily.csv",
                    index_label="date",
                )
                permission_weights.to_csv(
                    OUTPUT / f"{sample}_{policy.name}_weights.csv",
                    index_label="date",
                )
                diagnostics.to_csv(
                    OUTPUT / f"{sample}_{policy.name}_diagnostics.csv",
                    index_label="date",
                )

            for rollout_share in ROLLOUT_SHARES:
                baseline = (
                    (1.0 - rollout_share)
                    * r9.loc[common, "net_return"]
                    + rollout_share
                    * current.loc[common, "net_return"]
                )
                trial = (
                    (1.0 - rollout_share)
                    * r9.loc[common, "net_return"]
                    + rollout_share
                    * candidate.loc[common, "net_return"]
                )
                for period, (start, end) in periods.items():
                    selected = common[
                        (common >= start) & (common <= end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "policy": policy.name,
                            "rollout_share": rollout_share,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected],
                                trial.loc[selected],
                            ),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.concat(event_rows).to_csv(
        OUTPUT / "diagnostics_all_samples.csv",
        index_label="date",
    )

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        relative = relative_log_return(
            baseline_paths[sample],
            candidate_paths[sample],
        )[:, None]
        for block_days in (21, 63, 126):
            check = circular_family_reality_check(
                relative,
                selected_index=0,
                block_days=block_days,
            )
            nominal = float(check["nominal_one_sided_p_value"])
            family_rows.append(
                {
                    "sample": sample,
                    "policy": policy.name,
                    "predeclared_candidates": 1,
                    "cumulative_reentry_trials": (
                        PREVIOUS_REENTRY_TRIALS + 1
                    ),
                    **check,
                    "cumulative_bonferroni_p_value": min(
                        1.0,
                        nominal * (PREVIOUS_REENTRY_TRIALS + 1),
                    ),
                }
            )
    family = pd.DataFrame(family_rows)
    family.to_csv(OUTPUT / "multiple_comparison_audit.csv", index=False)

    selected = metrics[
        metrics["rollout_share"].eq(0.25)
        & metrics["period"].isin(
            [
                "complete_2015_2026",
                "complete_2006_2026",
                "live_2022_2026",
            ]
        )
    ]
    print(
        selected[
            [
                "sample",
                "scenario",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "max_drawdown_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print()
    print(
        family[
            [
                "sample",
                "block_days",
                "nominal_one_sided_p_value",
                "cumulative_bonferroni_p_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
