from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_r10_gde_capital_efficiency import (
    ASSETS,
    NORMAL_OPEN_CLOSE,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from evaluate_r12_gold_regime import (
    COST_SCENARIOS,
    GDE_FRACTION,
    GDE_NO_TRADE_BAND,
    MACRO_CACHE,
    R11_PATHS,
    RISK_MULTIPLIER,
    RISK_ON_GROWTH_THRESHOLD,
)
from evaluate_r12_gold_regime import (
    circular_family_reality_check,
    compounded_return,
    gold_drawdown_episodes,
    load_macro_cache,
    load_strategy_inputs,
    metric_delta,
    relative_log_return,
    sample_definitions,
    scale_non_cash_weights,
)
from evaluate_r12_gold_regime import guard_for_scenario
from evaluate_smh_dynamic_guard import simulate as simulate_smh_guard


OUTPUT = Path("output/r12_gold_crash_guard_v2")
LOOKBACK_DAYS = 63
RECOVERY_LOOKBACK_DAYS = 10
RECOVERY_CONFIRMATIONS = 2
TOTAL_ATTEMPTED_CANDIDATES = 63
MINIMUM_BASE_GOLD_WEIGHT = 0.10


@dataclass(frozen=True)
class GoldCrashCandidate:
    drawdown_trigger: float
    floor_fraction: float

    @property
    def name(self) -> str:
        trigger = int(round(abs(self.drawdown_trigger) * 100))
        floor = int(round(self.floor_fraction * 100))
        return f"dd{trigger}_floor{floor}"


def candidate_family() -> tuple[GoldCrashCandidate, ...]:
    return tuple(
        GoldCrashCandidate(trigger, floor)
        for trigger in (-0.08, -0.12, -0.16)
        for floor in (0.25, 0.50, 0.75)
    )


def causal_gold_crash_signals(
    closes: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    recovery_lookback_days: int = RECOVERY_LOOKBACK_DAYS,
) -> pd.DataFrame:
    required = ["GOLD", "CASH"]
    missing = [asset for asset in required if asset not in closes]
    if missing:
        raise ValueError(f"Gold crash signals lack prices: {missing}")
    if lookback_days < 2 or recovery_lookback_days < 1:
        raise ValueError("Gold crash lookbacks must be positive")

    prior = closes[required].shift(1)
    relative_log_price = np.log(prior["GOLD"] / prior["CASH"])
    prior_high = prior["GOLD"].rolling(
        lookback_days,
        min_periods=lookback_days,
    ).max()
    known_macro = macro.reindex(closes.index).ffill().shift(1)

    result = pd.DataFrame(index=closes.index)
    result["drawdown_from_high"] = prior["GOLD"] / prior_high - 1.0
    result["fast_excess_trend"] = relative_log_price.diff(
        recovery_lookback_days
    )
    result["real_yield_change"] = known_macro["DFII10"].diff(
        lookback_days
    )
    result["dollar_change"] = np.log(
        known_macro["DTWEXBGS"]
    ).diff(lookback_days)
    result["macro_headwinds"] = (
        result["real_yield_change"].ge(0.0).astype(float)
        + result["dollar_change"].ge(0.0).astype(float)
    )
    available = result[
        [
            "drawdown_from_high",
            "fast_excess_trend",
            "real_yield_change",
            "dollar_change",
        ]
    ].notna().all(axis=1)
    result.loc[~available, "macro_headwinds"] = np.nan
    return result


def selected_cap_fraction(
    floor_fraction: float,
    macro_headwinds: float,
) -> float:
    if not 0.0 <= floor_fraction <= 1.0:
        raise ValueError("floor_fraction must be in [0, 1]")
    if macro_headwinds >= 2.0:
        return floor_fraction
    if macro_headwinds >= 1.0:
        return 0.5 * (1.0 + floor_fraction)
    return 1.0


def move_gold_to_cap(
    reference: np.ndarray,
    baseline: np.ndarray,
    cap_fraction: float,
    *,
    never_increase: bool,
) -> np.ndarray:
    adjusted = reference.copy()
    gold = ASSETS.index("GOLD")
    cash = ASSETS.index("CASH")
    target_gold = max(float(baseline[gold]), 0.0) * cap_fraction
    if never_increase:
        target_gold = min(target_gold, float(adjusted[gold]))
    transfer = float(adjusted[gold]) - target_gold
    adjusted[gold] = target_gold
    adjusted[cash] += transfer
    return adjusted


def simulate_gold_crash_guard(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    signals: pd.DataFrame,
    candidate: GoldCrashCandidate | None,
    *,
    cost_bps: float,
    emergency_slippage_bps: float,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = (
        weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    baseline_values = weights.loc[dates, ASSETS].to_numpy(dtype=float)
    base_trade_values = (
        base_daily.loc[dates, "turnover"].to_numpy(dtype=float) > 1e-14
    )
    base_slippage_values = (
        base_daily["slippage_cost"]
        .reindex(dates, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    open_values = opens.loc[dates, ASSETS].to_numpy(dtype=float)
    close_values = closes.loc[dates, ASSETS].to_numpy(dtype=float)
    overnight_returns = np.zeros_like(open_values)
    overnight_returns[1:] = open_values[1:] / close_values[:-1] - 1.0
    intraday_returns = close_values / open_values - 1.0
    aligned_signals = signals.reindex(dates)

    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    active = False
    active_cap_fraction = 1.0
    positive_recovery_days = 0
    rows: list[dict[str, object]] = []
    weight_rows: list[np.ndarray] = []

    for position, date in enumerate(dates):
        overnight_asset_returns = overnight_returns[position]
        overnight_portfolio_return = float(
            current @ overnight_asset_returns
        )
        current = current * (1.0 + overnight_asset_returns) / (
            1.0 + overnight_portfolio_return
        )

        base_trade = bool(base_trade_values[position])
        desired = baseline_values[position].copy() if base_trade else None
        signal = aligned_signals.loc[date]
        available = bool(pd.notna(signal["macro_headwinds"]))
        growth_weight = float(
            baseline_values[position][
                [
                    ASSETS.index("SPX"),
                    ASSETS.index("QQQ"),
                    ASSETS.index("SEMIS"),
                ]
            ].sum()
        )
        baseline_gold = float(
            baseline_values[position][ASSETS.index("GOLD")]
        )
        recovered = False
        released_for_defense = False
        triggered = False
        emergency_trade = False
        trade_reason = "base" if base_trade else "none"

        if active:
            released_for_defense = bool(
                base_trade
                and growth_weight < RISK_ON_GROWTH_THRESHOLD
            )
            if released_for_defense:
                reference = current if desired is None else desired
                desired = move_gold_to_cap(
                    reference,
                    baseline_values[position],
                    1.0,
                    never_increase=False,
                )
                active = False
                active_cap_fraction = 1.0
                positive_recovery_days = 0
                trade_reason = "defense_release"
            else:
                assert candidate is not None
                recovery_positive = bool(
                    available
                    and signal["fast_excess_trend"] > 0.0
                    and signal["drawdown_from_high"]
                    > 0.5 * candidate.drawdown_trigger
                )
                positive_recovery_days = (
                    positive_recovery_days + 1
                    if recovery_positive
                    else 0
                )
                recovered = (
                    positive_recovery_days >= RECOVERY_CONFIRMATIONS
                )
                if recovered:
                    reference = current if desired is None else desired
                    desired = move_gold_to_cap(
                        reference,
                        baseline_values[position],
                        1.0,
                        never_increase=False,
                    )
                    active = False
                    active_cap_fraction = 1.0
                    positive_recovery_days = 0
                    emergency_trade = True
                    trade_reason = "recovery"
                elif base_trade and desired is not None:
                    desired = move_gold_to_cap(
                        desired,
                        baseline_values[position],
                        active_cap_fraction,
                        never_increase=False,
                    )
        if candidate is None:
            should_trigger = False
        else:
            should_trigger = bool(
                not active
                and not recovered
                and available
                and growth_weight >= RISK_ON_GROWTH_THRESHOLD
                and baseline_gold >= MINIMUM_BASE_GOLD_WEIGHT
                and signal["drawdown_from_high"]
                <= candidate.drawdown_trigger
                and signal["fast_excess_trend"] < 0.0
                and signal["macro_headwinds"] >= 1.0
            )
        if should_trigger:
            assert candidate is not None
            active_cap_fraction = selected_cap_fraction(
                candidate.floor_fraction,
                float(signal["macro_headwinds"]),
            )
            reference = current if desired is None else desired
            desired = move_gold_to_cap(
                reference,
                baseline_values[position],
                active_cap_fraction,
                never_increase=not base_trade,
            )
            active = True
            positive_recovery_days = 0
            triggered = True
            emergency_trade = True
            trade_reason = "trigger"

        turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current).sum())
        )
        gold_turnover = (
            0.0
            if desired is None or not emergency_trade
            else abs(
                float(
                    desired[ASSETS.index("GOLD")]
                    - current[ASSETS.index("GOLD")]
                )
            )
        )
        if desired is not None and turnover > 0.0:
            current = desired
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0
        gold_slippage = (
            2.0
            * gold_turnover
            * emergency_slippage_bps
            / 10_000.0
        )
        slippage_cost = (
            float(base_slippage_values[position]) + gold_slippage
        )
        intraday_portfolio_return = float(
            current @ intraday_returns[position]
        )
        financing_cost = (
            max(-float(current[ASSETS.index("CASH")]), 0.0)
            * 100.0
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_portfolio_return)
            * (1.0 + intraday_portfolio_return)
            - 1.0
            - trading_cost
            - slippage_cost
            - financing_cost
        )
        rows.append(
            {
                "net_return": net_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "slippage_cost": slippage_cost,
                "gold_slippage_cost": gold_slippage,
                "base_trade": base_trade,
                "triggered": triggered,
                "recovered": recovered,
                "released_for_defense": released_for_defense,
                "active": active,
                "active_cap_fraction": active_cap_fraction,
                "trade_reason": trade_reason,
                "drawdown_from_high": signal["drawdown_from_high"],
                "fast_excess_trend": signal["fast_excess_trend"],
                "macro_headwinds": signal["macro_headwinds"],
                "baseline_gold_weight": baseline_gold,
                "implemented_gold_weight": float(
                    current[ASSETS.index("GOLD")]
                ),
            }
        )
        weight_rows.append(current.copy())

        gross_intraday_growth = 1.0 + intraday_portfolio_return
        current = current * (
            1.0 + intraday_returns[position]
        ) / gross_intraday_growth

    return (
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
    )


def select_candidate(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    periods = {
        ("proxy_synthetic", "early_2007_2014"),
        ("normal_synthetic", "development_2015_2021"),
    }
    selected = metrics.loc[
        (metrics["scenario"] == "current_liquidity")
        & metrics[["sample", "period"]].apply(tuple, axis=1).isin(periods)
    ]
    rows: list[dict[str, object]] = []
    for name, group in selected.groupby("candidate"):
        rows.append(
            {
                "candidate": name,
                "selection_periods": int(group.shape[0]),
                "minimum_cagr_delta": float(group["cagr_delta"].min()),
                "mean_cagr_delta": float(group["cagr_delta"].mean()),
                "minimum_sharpe_delta": float(group["sharpe_delta"].min()),
                "worst_max_drawdown_delta": float(
                    group["max_drawdown_delta"].min()
                ),
                "selection_gate": bool(
                    group.shape[0] == 2
                    and group["cagr_delta"].ge(-1e-12).all()
                    and group["cagr_delta"].gt(1e-12).any()
                    and group["sharpe_delta"].ge(0.0).all()
                    and group["max_drawdown_delta"].ge(-0.005).all()
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[ranking["selection_gate"]]
    pool = eligible if not eligible.empty else ranking
    pool = pool.sort_values(
        [
            "minimum_cagr_delta",
            "mean_cagr_delta",
            "minimum_sharpe_delta",
            "worst_max_drawdown_delta",
            "candidate",
        ],
        ascending=[False, False, False, False, True],
    )
    chosen = str(pool.iloc[0]["candidate"])
    ranking["selected"] = ranking["candidate"].eq(chosen)
    return chosen, ranking.sort_values("candidate")


def candidate_definition(name: str) -> GoldCrashCandidate:
    matches = [candidate for candidate in candidate_family() if candidate.name == name]
    if len(matches) != 1:
        raise ValueError(f"Gold crash candidate is not unique: {name}")
    return matches[0]


def production_gate(
    selected_name: str,
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    family: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    checks: list[dict[str, object]] = []
    selection_row = selection.loc[
        selection["candidate"].eq(selected_name)
    ].iloc[0]
    checks.append(
        {
            "requirement": "selection_periods",
            "pass": bool(selection_row["selection_gate"]),
            "detail": (
                f"minimum CAGR {selection_row['minimum_cagr_delta']:+.4%}; "
                f"minimum Sharpe "
                f"{selection_row['minimum_sharpe_delta']:+.4f}"
            ),
        }
    )
    required = [
        ("normal_synthetic", "holdout_2022_2025"),
        ("normal_synthetic", "recent_2026"),
        ("proxy_synthetic", "holdout_2022_2026"),
        ("normal_live", "live_2022_2026"),
    ]
    for sample, period in required:
        row = metrics.loc[
            metrics["candidate"].eq(selected_name)
            & metrics["sample"].eq(sample)
            & metrics["scenario"].eq("current_liquidity")
            & metrics["period"].eq(period)
        ].iloc[0]
        passed = bool(
            row["cagr_delta"] >= -1e-12
            and row["sharpe_delta"] >= 0.0
            and row["max_drawdown_delta"] >= -0.005
        )
        checks.append(
            {
                "requirement": f"{sample}:{period}",
                "pass": passed,
                "detail": (
                    f"CAGR {row['cagr_delta']:+.4%}; "
                    f"Sharpe {row['sharpe_delta']:+.4f}; "
                    f"MDD {row['max_drawdown_delta']:+.4%}"
                ),
            }
        )
    for sample, period in [
        ("normal_synthetic", "complete_2015_2026"),
        ("proxy_synthetic", "complete_2007_2026"),
        ("normal_live", "live_2022_2026"),
    ]:
        row = metrics.loc[
            metrics["candidate"].eq(selected_name)
            & metrics["sample"].eq(sample)
            & metrics["scenario"].eq("cost_stress")
            & metrics["period"].eq(period)
        ].iloc[0]
        passed = bool(
            row["cagr_delta"] > 0.0
            and row["sharpe_delta"] >= 0.0
            and row["candidate_max_drawdown"] >= -0.18
        )
        checks.append(
            {
                "requirement": f"{sample}:cost_stress",
                "pass": passed,
                "detail": (
                    f"CAGR {row['cagr_delta']:+.4%}; "
                    f"Sharpe {row['sharpe_delta']:+.4f}; "
                    f"MDD {row['candidate_max_drawdown']:.4%}"
                ),
            }
        )
    for sample in ("normal_synthetic", "proxy_synthetic"):
        rows = family.loc[family["sample"].eq(sample)]
        passed = bool(
            rows.shape[0] == 3
            and rows["all_attempts_adjusted_p_value"].lt(0.05).all()
        )
        checks.append(
            {
                "requirement": f"{sample}:all_attempts_adjusted",
                "pass": passed,
                "detail": "p="
                + ",".join(
                    f"{value:.4f}"
                    for value in rows["all_attempts_adjusted_p_value"]
                ),
            }
        )
    material = events.loc[
        events["gold_drawdown"].le(-0.20)
        & events["peak_date"].ge("2011-01-01")
    ]
    event_pass = bool(
        not material.empty
        and material["candidate_minus_baseline_return"].mean() > 0.0
        and material["candidate_minus_baseline_return"].ge(-0.005).all()
    )
    checks.append(
        {
            "requirement": "gold_bear_events",
            "pass": event_pass,
            "detail": (
                f"events={material.shape[0]}; mean="
                f"{material['candidate_minus_baseline_return'].mean():+.4%}"
            ),
        }
    )
    audit = pd.DataFrame(checks)
    return bool(audit["pass"].all()), audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    macro = load_macro_cache(MACRO_CACHE)
    normal_opens, normal_closes = load_adjusted_open_close(NORMAL_OPEN_CLOSE)
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(normal_opens, normal_closes)
    samples = sample_definitions(
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )
    candidates = candidate_family()
    metric_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    paths: dict[tuple[str, str, str], pd.Series] = {}
    reference_paths: dict[tuple[str, str], pd.Series] = {}
    diagnostics: dict[tuple[str, str, str], pd.DataFrame] = {}

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
        signals = causal_gold_crash_signals(closes, macro)
        levered = scale_non_cash_weights(weights, RISK_MULTIPLIER)

        for scenario in COST_SCENARIOS:
            smh_daily, smh_weights = simulate_smh_guard(
                levered,
                daily,
                opens,
                closes,
                guard_for_scenario(scenario),
                cost_bps=scenario.base_one_way_cost_bps,
                start_date=str(settings["start"]),
            )
            no_op_daily, no_op_weights = simulate_gold_crash_guard(
                smh_weights,
                smh_daily,
                opens,
                closes,
                signals,
                None,
                cost_bps=scenario.base_one_way_cost_bps,
                emergency_slippage_bps=scenario.emergency_slippage_bps,
                start_date=str(settings["start"]),
            )
            common = smh_daily.index.intersection(no_op_daily.index)
            reconstruction_rows.append(
                {
                    "sample": sample,
                    "scenario": scenario.name,
                    "engine": "gold_guard_no_op_vs_smh_guard",
                    "maximum_return_error": float(
                        (
                            no_op_daily.loc[common, "net_return"]
                            - smh_daily.loc[common, "net_return"]
                        ).abs().max()
                    ),
                    "maximum_weight_error": float(
                        (
                            no_op_weights.loc[common, ASSETS]
                            - smh_weights.loc[common, ASSETS]
                        ).abs().to_numpy().max()
                    ),
                }
            )
            baseline = simulate_gde_substitution(
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
                weights_override=smh_weights,
                daily_override=smh_daily,
                extra_slippage=smh_daily["slippage_cost"],
                gde_no_trade_band=GDE_NO_TRADE_BAND,
            )["net_return"]
            reference_paths[(sample, scenario.name)] = baseline
            if scenario.name == "current_liquidity":
                saved_path = (
                    R11_PATHS / f"{sample}_risk1.065_gde10_daily.csv"
                )
                saved = pd.read_csv(
                    saved_path,
                    index_col=0,
                    parse_dates=True,
                )["net_return"]
                aligned = pd.concat(
                    [baseline.rename("rebuilt"), saved.rename("saved")],
                    axis=1,
                    join="inner",
                ).dropna()
                reconstruction_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "engine": "r11_saved_path",
                        "maximum_return_error": float(
                            (aligned["rebuilt"] - aligned["saved"])
                            .abs().max()
                        ),
                        "maximum_weight_error": np.nan,
                    }
                )

            for candidate in candidates:
                gold_daily, gold_weights = simulate_gold_crash_guard(
                    smh_weights,
                    smh_daily,
                    opens,
                    closes,
                    signals,
                    candidate,
                    cost_bps=scenario.base_one_way_cost_bps,
                    emergency_slippage_bps=(
                        scenario.emergency_slippage_bps
                    ),
                    start_date=str(settings["start"]),
                )
                candidate_path = simulate_gde_substitution(
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
                    weights_override=gold_weights,
                    daily_override=gold_daily,
                    extra_slippage=gold_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )["net_return"]
                paths[(sample, scenario.name, candidate.name)] = candidate_path
                diagnostics[(sample, scenario.name, candidate.name)] = (
                    gold_daily
                )
                common_dates = baseline.index.intersection(
                    candidate_path.index
                )
                for period, (start, end) in periods.items():
                    selected_dates = common_dates[
                        (common_dates >= start) & (common_dates <= end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "candidate": candidate.name,
                            "drawdown_trigger": candidate.drawdown_trigger,
                            "floor_fraction": candidate.floor_fraction,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected_dates],
                                candidate_path.loc[selected_dates],
                            ),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    selected_name, selection = select_candidate(metrics)
    selected = candidate_definition(selected_name)
    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = reference_paths[(sample, "current_liquidity")]
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    paths[(sample, "current_liquidity", candidate.name)],
                )
                for candidate in candidates
            ]
        )
        selected_index = [candidate.name for candidate in candidates].index(
            selected_name
        )
        for block_days in (21, 63, 126):
            result = circular_family_reality_check(
                matrix,
                selected_index,
                block_days,
            )
            adjusted = min(
                1.0,
                float(result["familywise_reality_check_p_value"])
                * TOTAL_ATTEMPTED_CANDIDATES
                / len(candidates),
            )
            family_rows.append(
                {
                    "sample": sample,
                    "candidate": selected_name,
                    "crash_family_size": len(candidates),
                    "total_attempted_candidates": (
                        TOTAL_ATTEMPTED_CANDIDATES
                    ),
                    **result,
                    "all_attempts_adjusted_p_value": adjusted,
                }
            )
    family = pd.DataFrame(family_rows)

    events_rows: list[dict[str, object]] = []
    baseline = reference_paths[("proxy_synthetic", "current_liquidity")]
    candidate_path = paths[
        ("proxy_synthetic", "current_liquidity", selected_name)
    ]
    selected_diagnostics = diagnostics[
        ("proxy_synthetic", "current_liquidity", selected_name)
    ]
    for episode in gold_drawdown_episodes(
        proxy_closes["GOLD"].loc["2007-01-01":]
    ):
        peak = episode["peak_date"]
        trough = episode["trough_date"]
        dates = baseline.index[
            (baseline.index >= peak) & (baseline.index <= trough)
        ]
        if dates.empty:
            continue
        base_return = compounded_return(baseline.loc[dates])
        trial_return = compounded_return(candidate_path.loc[dates])
        event_diag = selected_diagnostics.reindex(dates)
        events_rows.append(
            {
                "peak_date": peak.date().isoformat(),
                "trough_date": trough.date().isoformat(),
                "recovery_or_end_date": episode[
                    "recovery_or_end_date"
                ].date().isoformat(),
                "gold_drawdown": episode["gold_drawdown"],
                "baseline_return": base_return,
                "candidate_return": trial_return,
                "candidate_minus_baseline_return": (
                    trial_return - base_return
                ),
                "trigger_count": int(event_diag["triggered"].sum()),
                "active_fraction": float(
                    event_diag["active"].mean()
                ),
            }
        )
    events = pd.DataFrame(events_rows)
    qualified, audit = production_gate(
        selected_name,
        selection,
        metrics,
        family,
        events,
    )
    normal_diagnostics = diagnostics[
        ("normal_synthetic", "current_liquidity", selected_name)
    ]
    current = normal_diagnostics.iloc[-1]
    current_target = {
        "date": normal_diagnostics.index[-1].date().isoformat(),
        "candidate": selected_name,
        "drawdown_from_63_day_high": float(
            current["drawdown_from_high"]
        ),
        "gold_relative_cash_10_day_log_return": float(
            current["fast_excess_trend"]
        ),
        "macro_headwinds": int(current["macro_headwinds"]),
        "active": bool(current["active"]),
        "baseline_gold_weight": float(
            current["baseline_gold_weight"]
        ),
        "implemented_gold_weight": float(
            current["implemented_gold_weight"]
        ),
        "active_cap_fraction": float(
            current["active_cap_fraction"]
        ),
    }

    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    selection.to_csv(OUTPUT / "candidate_selection.csv", index=False)
    family.to_csv(OUTPUT / "family_reality_check.csv", index=False)
    events.to_csv(OUTPUT / "selected_gold_events.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(
        OUTPUT / "baseline_reconstruction.csv",
        index=False,
    )
    audit.to_csv(OUTPUT / "production_gate.csv", index=False)
    (OUTPUT / "current_target.json").write_text(
        json.dumps(current_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected: {selected_name}")
    print(f"Qualified for R12: {qualified}")
    print(selection.loc[selection["selected"]].to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(current_target, ensure_ascii=False, indent=2))
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
