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
    MINIMUM_PREVIOUS_GROWTH_WEIGHT,
    ROLLOUT_SHARES,
    cap_growth_to_cash,
    causal_growth_shock_signals,
    compounded_return,
    metric_delta,
    sample_definitions,
    select_candidate,
)


OUTPUT = Path("output/r11_selective_shock_memory")
NORMAL_PRICES = Path("data/prices_vix_hedge.csv")
PROXY_PRICES = Path("data/prices_20y_proxy.csv")
RISK_MULTIPLIER = 1.065
GDE_FRACTION = 0.10
LOSS_TRIGGER = -0.03
TREND_SESSIONS = 63
RECOVERY_GROWTH_WEIGHT = 0.70
RECOVERY_CONFIRMATIONS = 2
NORMAL_VIX_TERM_RATIO = 1.0
PREVIOUS_FIXED_COOLDOWN_TRIALS = 27
GATE_MODES = (
    "qqq_weak",
    "either_weak",
    "both_weak",
    "semis_medium_break",
)
RELEASE_MODES = (
    "base_growth",
    "trend_recovery",
    "reentry_brake",
)


@dataclass(frozen=True)
class ShockMemoryCandidate:
    gate_mode: str
    growth_cap: float
    release_mode: str = "base_growth"

    def __post_init__(self) -> None:
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")
        if self.release_mode not in RELEASE_MODES:
            raise ValueError(f"Unknown release_mode: {self.release_mode}")
        if not 0.0 <= self.growth_cap <= 1.0:
            raise ValueError("growth_cap must be in [0, 1]")

    @property
    def name(self) -> str:
        cap = int(round(self.growth_cap * 100))
        return f"{self.release_mode}_{self.gate_mode}_cap{cap}"


def candidate_family() -> tuple[ShockMemoryCandidate, ...]:
    existing = tuple(
        ShockMemoryCandidate(gate_mode, cap, release_mode)
        for release_mode in RELEASE_MODES
        for gate_mode in GATE_MODES[:3]
        for cap in (0.35, 0.50)
    )
    structural = tuple(
        ShockMemoryCandidate(
            "semis_medium_break",
            cap,
            "reentry_brake",
        )
        for cap in (0.35, 0.50)
    )
    return existing + structural


def build_causal_signals(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    market_prices: pd.DataFrame,
) -> pd.DataFrame:
    required = ["QQQ", "SEMIS", "VIX", "VIX3M"]
    missing = [column for column in required if column not in market_prices]
    if missing:
        raise ValueError(f"Missing market signal columns: {missing}")
    signals = causal_growth_shock_signals(weights, closes)
    market = market_prices.reindex(signals.index)
    signals["qqq_63d_return"] = (
        market["QQQ"].pct_change(TREND_SESSIONS).shift(1)
    )
    signals["semis_63d_return"] = (
        market["SEMIS"].pct_change(TREND_SESSIONS).shift(1)
    )
    signals["semis_126d_return"] = (
        market["SEMIS"].pct_change(2 * TREND_SESSIONS).shift(1)
    )
    signals["vix_term_ratio"] = (
        market["VIX"] / market["VIX3M"]
    ).shift(1)
    return signals


def structural_condition(
    signal: pd.Series,
    gate_mode: str,
) -> bool:
    qqq = signal["qqq_63d_return"]
    semis = signal["semis_63d_return"]
    semis_medium = signal["semis_126d_return"]
    term_ratio = signal["vix_term_ratio"]
    if (
        pd.isna(qqq)
        or pd.isna(semis)
        or pd.isna(semis_medium)
        or pd.isna(term_ratio)
    ):
        return False
    qqq_weak = float(qqq) <= 0.0
    semis_weak = float(semis) <= 0.0
    if gate_mode == "qqq_weak":
        trend_weak = qqq_weak
    elif gate_mode == "either_weak":
        trend_weak = qqq_weak or semis_weak
    elif gate_mode == "both_weak":
        trend_weak = qqq_weak and semis_weak
    elif gate_mode == "semis_medium_break":
        trend_weak = bool(
            qqq_weak
            and semis_weak
            and float(semis_medium) <= 0.0
        )
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")
    return bool(
        trend_weak and float(term_ratio) > NORMAL_VIX_TERM_RATIO
    )


def apply_selective_shock_memory(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    signals: pd.DataFrame,
    candidate: ShockMemoryCandidate | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = weights.index.intersection(base_daily.index)
    adjusted_weights = weights.loc[dates, ASSETS].copy()
    adjusted_daily = base_daily.loc[dates].copy()
    aligned_signals = signals.reindex(dates)
    diagnostics: list[dict[str, float | int | bool]] = []
    active = False
    engaged = False
    confirmations = 0
    overlay_had_effect = False
    previous_baseline_growth: float | None = None

    for date in dates:
        signal = aligned_signals.loc[date]
        structural = bool(
            candidate is not None
            and structural_condition(signal, candidate.gate_mode)
        )
        triggered = bool(
            structural
            and pd.notna(signal["previous_growth_weight"])
            and pd.notna(signal["previous_weighted_growth_return"])
            and float(signal["previous_growth_weight"])
            >= MINIMUM_PREVIOUS_GROWTH_WEIGHT
            and float(signal["previous_weighted_growth_return"])
            <= LOSS_TRIGGER
        )
        baseline_growth = float(
            adjusted_weights.loc[date, list(GROWTH_ASSETS)].sum()
        )
        released = False
        recovery_observation = False
        base_trade = bool(
            float(base_daily.loc[date, "turnover"]) > 0.0
        )
        trend_recovered = bool(
            pd.notna(signal["qqq_63d_return"])
            and pd.notna(signal["semis_63d_return"])
            and pd.notna(signal["vix_term_ratio"])
            and float(signal["qqq_63d_return"]) > 0.0
            and float(signal["semis_63d_return"]) > 0.0
            and float(signal["vix_term_ratio"])
            <= NORMAL_VIX_TERM_RATIO
        )

        if triggered:
            active = True
            confirmations = 0
        elif (
            active
            and candidate is not None
            and candidate.release_mode == "reentry_brake"
        ):
            if date.weekday() == 3:
                recovery_observation = True
                if trend_recovered:
                    active = False
                    engaged = False
                    released = True
                    if overlay_had_effect:
                        adjusted_daily.loc[date, "turnover"] = max(
                            float(
                                adjusted_daily.loc[date, "turnover"]
                            ),
                            1.0,
                        )
                    overlay_had_effect = False
            if (
                active
                and not engaged
                and base_trade
                and previous_baseline_growth is not None
                and baseline_growth > candidate.growth_cap
                and baseline_growth
                > previous_baseline_growth + 1e-12
            ):
                engaged = True
        elif active and date.weekday() == 3:
            recovery_observation = True
            normal_term_structure = bool(
                pd.notna(signal["vix_term_ratio"])
                and float(signal["vix_term_ratio"])
                <= NORMAL_VIX_TERM_RATIO
            )
            if candidate is None:
                recovered = False
            elif candidate.release_mode == "base_growth":
                recovered = bool(
                    baseline_growth >= RECOVERY_GROWTH_WEIGHT
                    and normal_term_structure
                )
            elif candidate.release_mode == "trend_recovery":
                recovered = trend_recovered
            else:
                raise ValueError(
                    f"Unknown release_mode: {candidate.release_mode}"
                )
            confirmations = confirmations + 1 if recovered else 0
            if confirmations >= RECOVERY_CONFIRMATIONS:
                active = False
                engaged = False
                confirmations = 0
                released = True
                if overlay_had_effect:
                    adjusted_daily.loc[date, "turnover"] = max(
                        float(adjusted_daily.loc[date, "turnover"]),
                        1.0,
                    )
                overlay_had_effect = False

        capped_growth = baseline_growth
        changed = False
        should_cap = bool(
            active
            and candidate is not None
            and (
                candidate.release_mode != "reentry_brake"
                or engaged
            )
        )
        if should_cap:
            assert candidate is not None
            capped = cap_growth_to_cash(
                adjusted_weights.loc[date],
                candidate.growth_cap,
            )
            changed = not np.allclose(
                capped.to_numpy(dtype=float),
                adjusted_weights.loc[date].to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            )
            if changed:
                adjusted_weights.loc[date] = capped
                capped_growth = float(
                    capped.loc[list(GROWTH_ASSETS)].sum()
                )
                overlay_had_effect = True
                if triggered or base_trade:
                    adjusted_daily.loc[date, "turnover"] = max(
                        float(adjusted_daily.loc[date, "turnover"]),
                        1.0,
                    )

        diagnostics.append(
            {
                "previous_growth_weight": signal[
                    "previous_growth_weight"
                ],
                "previous_weighted_growth_return": signal[
                    "previous_weighted_growth_return"
                ],
                "qqq_63d_return": signal["qqq_63d_return"],
                "semis_63d_return": signal["semis_63d_return"],
                "semis_126d_return": signal["semis_126d_return"],
                "vix_term_ratio": signal["vix_term_ratio"],
                "structural_condition": structural,
                "triggered": triggered,
                "active": active,
                "engaged": engaged,
                "recovery_observation": recovery_observation,
                "recovery_confirmations": confirmations,
                "released": released,
                "changed": changed,
                "baseline_growth_weight": baseline_growth,
                "implemented_growth_weight": capped_growth,
            }
        )
        previous_baseline_growth = baseline_growth

    return (
        adjusted_weights,
        adjusted_daily,
        pd.DataFrame(diagnostics, index=dates),
    )


def load_market_prices() -> dict[str, pd.DataFrame]:
    normal = pd.read_csv(NORMAL_PRICES, index_col=0, parse_dates=True)
    proxy = pd.read_csv(PROXY_PRICES, index_col=0, parse_dates=True)
    return {
        "normal_synthetic": normal,
        "proxy_synthetic": proxy,
        "normal_live": normal,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    candidates = candidate_family()
    metric_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    current_paths: dict[str, pd.Series] = {}
    candidate_paths: dict[str, dict[str, pd.Series]] = {}
    saved_payloads: dict[
        tuple[str, str, str],
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ] = {}

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

        sample_candidate_paths: dict[str, pd.Series] = {}
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
            signals = build_causal_signals(
                guard_weights,
                closes,
                market_prices[sample],
            )
            if scenario.name == "current_liquidity":
                current_paths[sample] = current["net_return"]

            for candidate in candidates:
                (
                    memory_weights,
                    memory_daily,
                    diagnostics,
                ) = apply_selective_shock_memory(
                    guard_weights,
                    guard_daily,
                    signals,
                    candidate,
                )
                trial = simulate_gde_substitution(
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
                    weights_override=memory_weights,
                    daily_override=memory_daily,
                    extra_slippage=memory_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )
                common = (
                    r9.index.intersection(current.index)
                    .intersection(trial.index)
                )
                for rollout_share in ROLLOUT_SHARES:
                    current_staged = (
                        (1.0 - rollout_share)
                        * r9.loc[common, "net_return"]
                        + rollout_share
                        * current.loc[common, "net_return"]
                    )
                    trial_staged = (
                        (1.0 - rollout_share)
                        * r9.loc[common, "net_return"]
                        + rollout_share
                        * trial.loc[common, "net_return"]
                    )
                    for period, (
                        period_start,
                        period_end,
                    ) in periods.items():
                        selected = common[
                            (common >= period_start)
                            & (common <= period_end)
                        ]
                        metric_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "candidate": candidate.name,
                                "gate_mode": candidate.gate_mode,
                                "release_mode": candidate.release_mode,
                                "growth_cap": candidate.growth_cap,
                                "rollout_share": rollout_share,
                                "period": period,
                                **metric_delta(
                                    current_staged.loc[selected],
                                    trial_staged.loc[selected],
                                ),
                            }
                        )
                    for year in (
                        2008,
                        2011,
                        2015,
                        2018,
                        2020,
                        2022,
                        2025,
                        2026,
                    ):
                        selected = common[common.year == year]
                        if selected.empty:
                            continue
                        event_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "candidate": candidate.name,
                                "rollout_share": rollout_share,
                                "year": year,
                                "current_return": compounded_return(
                                    current_staged.loc[selected]
                                ),
                                "candidate_return": compounded_return(
                                    trial_staged.loc[selected]
                                ),
                                "relative_log_return": float(
                                    (
                                        np.log1p(
                                            trial_staged.loc[selected]
                                        )
                                        - np.log1p(
                                            current_staged.loc[selected]
                                        )
                                    ).sum()
                                ),
                            }
                        )
                if scenario.name == "current_liquidity":
                    sample_candidate_paths[candidate.name] = trial[
                        "net_return"
                    ]
                    saved_payloads[
                        (sample, scenario.name, candidate.name)
                    ] = (trial, memory_weights, diagnostics)
        candidate_paths[sample] = sample_candidate_paths

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    events.to_csv(OUTPUT / "event_metrics.csv", index=False)
    selected_name, selection = select_candidate(metrics)
    selection.to_csv(OUTPUT / "selection.csv", index=False)

    family_rows: list[dict[str, object]] = []
    cumulative_trials = PREVIOUS_FIXED_COOLDOWN_TRIALS + len(candidates)
    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = current_paths[sample]
        names = list(candidate_paths[sample])
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    candidate_paths[sample][name],
                )
                for name in names
            ]
        )
        selected_index = names.index(selected_name)
        for block_days in (21, 63, 126):
            check = circular_family_reality_check(
                matrix,
                selected_index,
                block_days,
            )
            family_p = float(
                check["familywise_reality_check_p_value"]
            )
            family_rows.append(
                {
                    "sample": sample,
                    "candidate": selected_name,
                    "current_family_size": len(names),
                    "cumulative_trials": cumulative_trials,
                    "block_days": block_days,
                    **check,
                    "cumulative_trial_adjusted_p_value": min(
                        1.0,
                        family_p * cumulative_trials / len(names),
                    ),
                }
            )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    for sample in samples:
        payload = saved_payloads[
            (sample, "current_liquidity", selected_name)
        ]
        payload[0].to_csv(
            OUTPUT / f"{sample}_{selected_name}_daily.csv",
            index_label="date",
        )
        payload[1].to_csv(
            OUTPUT / f"{sample}_{selected_name}_weights.csv",
            index_label="date",
        )
        payload[2].to_csv(
            OUTPUT / f"{sample}_{selected_name}_diagnostics.csv",
            index_label="date",
        )

    print(f"Selected candidate: {selected_name}")
    print(
        metrics.loc[
            metrics["candidate"].eq(selected_name)
            & metrics["rollout_share"].eq(1.0)
            & metrics["period"].isin(
                [
                    "complete_2015_2026",
                    "complete_2006_2026",
                    "live_2022_2026",
                ]
            ),
            [
                "sample",
                "scenario",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "max_drawdown_delta",
                "candidate_max_drawdown",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
