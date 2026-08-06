from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    GDE_FRACTION,
    robustness_rows,
    simulate_fixed_r11,
)
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
from tools.evaluate_r38_crowding_fragility_cap import (
    CASH_FLOOR,
    EVENT_WINDOWS,
    _period_row,
    _window_metrics,
    causal_crowding_fragility_state,
)
from tools.evaluate_r38_targeted_crowding_hysteresis import (
    _base_reference_schedule,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_crowding_confirmed_rotation")
CANDIDATE = "r38_crowding_confirmed_half_rotation"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
OVERLAY_FRACTION = 0.10
CROWDING_PERCENTILE = 0.90
MEMORY_DAYS = 63
PEAK_LOOKBACK_DAYS = 21
ENTRY_DRAWDOWN = -0.05
EXIT_DRAWDOWN = -0.025
RECOVERY_MA_DAYS = 20
QQQ_ROTATION_FRACTION = 0.50
CUMULATIVE_TRIALS = 138
MAX_DRAWDOWN_TOLERANCE = 0.005


def causal_confirmed_unwind_state(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    memory_days: int = MEMORY_DAYS,
    entry_drawdown: float = ENTRY_DRAWDOWN,
    exit_drawdown: float = EXIT_DRAWDOWN,
    recovery_ma_days: int = RECOVERY_MA_DAYS,
) -> pd.DataFrame:
    if memory_days < 1:
        raise ValueError("memory days must be positive")
    if not -1.0 < entry_drawdown < exit_drawdown <= 0.0:
        raise ValueError("entry drawdown must be below exit drawdown")
    if recovery_ma_days < 2:
        raise ValueError("recovery moving average must be at least two")
    crowding = causal_crowding_fragility_state(
        closes,
        index,
        percentile=CROWDING_PERCENTILE,
    )
    crowding_memory = (
        crowding["crowding_active"]
        .astype(int)
        .rolling(memory_days, min_periods=1)
        .max()
        .astype(bool)
    )
    semis = closes["SEMIS"].astype(float)
    prior_close = semis.shift(1).reindex(index)
    prior_peak = (
        semis.rolling(
            PEAK_LOOKBACK_DAYS,
            min_periods=PEAK_LOOKBACK_DAYS,
        )
        .max()
        .shift(1)
        .reindex(index)
    )
    prior_recovery_average = (
        semis.rolling(
            recovery_ma_days,
            min_periods=recovery_ma_days,
        )
        .mean()
        .shift(1)
        .reindex(index)
    )
    drawdown = prior_close.div(prior_peak) - 1.0
    entry = crowding_memory & drawdown.le(entry_drawdown)
    exit_ready = (
        drawdown.ge(exit_drawdown)
        & prior_close.ge(prior_recovery_average)
    ).fillna(False)
    active_values: list[bool] = []
    active = False
    for date in index:
        if not active and bool(entry.loc[date]):
            active = True
        elif active and bool(exit_ready.loc[date]):
            active = False
        active_values.append(active)
    active_state = pd.Series(
        active_values,
        index=index,
        dtype=bool,
        name="confirmed_unwind_active",
    )
    diagnostics = crowding.add_prefix("crowding_")
    diagnostics["crowding_memory_active"] = crowding_memory
    diagnostics["prior_semis_close"] = prior_close
    diagnostics["prior_semis_peak_21"] = prior_peak
    diagnostics["prior_semis_drawdown_21"] = drawdown
    diagnostics["prior_recovery_moving_average"] = (
        prior_recovery_average
    )
    diagnostics["confirmed_unwind_entry"] = entry
    diagnostics["confirmed_unwind_exit_ready"] = exit_ready
    diagnostics["confirmed_unwind_active"] = active_state
    diagnostics["confirmed_unwind_state_change"] = active_state.ne(
        active_state.shift(1)
    )
    return diagnostics


def apply_confirmed_rotation(
    full_weights: pd.DataFrame,
    base_reference_weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    memory_days: int = MEMORY_DAYS,
    entry_drawdown: float = ENTRY_DRAWDOWN,
    exit_drawdown: float = EXIT_DRAWDOWN,
    recovery_ma_days: int = RECOVERY_MA_DAYS,
    qqq_rotation_fraction: float = QQQ_ROTATION_FRACTION,
    rotation_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= qqq_rotation_fraction <= 1.0:
        raise ValueError("QQQ rotation fraction must be in [0, 1]")
    index = (
        full_weights.index.intersection(base_reference_weights.index)
        .intersection(execution_daily.index)
        .intersection(closes.index)
    )
    diagnostics = causal_confirmed_unwind_state(
        closes.reindex(index),
        index,
        memory_days=memory_days,
        entry_drawdown=entry_drawdown,
        exit_drawdown=exit_drawdown,
        recovery_ma_days=recovery_ma_days,
    )
    active = diagnostics["confirmed_unwind_active"].astype(bool)
    if not rotation_enabled:
        active = pd.Series(False, index=index)
    full = full_weights.loc[index].copy()
    reference = base_reference_weights.loc[index].copy()
    adjusted = full.copy()
    removable = (
        full["SEMIS"] - reference["SEMIS"]
    ).clip(lower=0.0).where(active, 0.0)
    adjusted["SEMIS"] -= removable
    adjusted["QQQ"] += removable * qqq_rotation_fraction
    adjusted["CASH"] += removable * (
        1.0 - qqq_rotation_fraction
    )
    non_growth = [
        column
        for column in adjusted
        if column not in ("QQQ", "SEMIS", "CASH")
    ]
    non_growth_error = (
        adjusted[non_growth] - full[non_growth]
    ).abs().max(axis=1)
    growth_increase = (
        adjusted["QQQ"]
        + adjusted["SEMIS"]
        - full["QQQ"]
        - full["SEMIS"]
    ).clip(lower=0.0)
    qqq_routing_error = (
        adjusted["QQQ"]
        - full["QQQ"]
        - removable * qqq_rotation_fraction
    ).abs()
    cash_routing_error = (
        adjusted["CASH"]
        - full["CASH"]
        - removable * (1.0 - qqq_rotation_fraction)
    ).abs()
    conservation_error = (adjusted.sum(axis=1) - 1.0).abs()
    if float(non_growth_error.max()) > 1e-12:
        raise AssertionError("Confirmed rotation changed other assets")
    if float(growth_increase.max()) > 1e-12:
        raise AssertionError("Confirmed rotation increased growth risk")
    if float(qqq_routing_error.max()) > 1e-12:
        raise AssertionError("QQQ routing was not exact")
    if float(cash_routing_error.max()) > 1e-12:
        raise AssertionError("Cash routing was not exact")
    if float(conservation_error.max()) > 1e-12:
        raise AssertionError("Confirmed rotation did not conserve weight")

    state_change = active.ne(active.shift(1))
    amount_change = removable.ne(removable.shift(1))
    update = state_change | amount_change
    updated_daily = execution_daily.loc[index].copy()
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics["confirmed_unwind_active"] = active
    diagnostics["removed_semis_weight"] = removable
    diagnostics["added_qqq_weight"] = (
        removable * qqq_rotation_fraction
    )
    diagnostics["added_cash_weight"] = (
        removable * (1.0 - qqq_rotation_fraction)
    )
    diagnostics["rotation_update"] = update
    diagnostics["rotation_non_growth_error"] = non_growth_error
    diagnostics["rotation_growth_increase_error"] = growth_increase
    diagnostics["rotation_qqq_routing_error"] = qqq_routing_error
    diagnostics["rotation_cash_routing_error"] = cash_routing_error
    diagnostics["rotation_conservation_error"] = conservation_error
    return adjusted, updated_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    memory_days: int = MEMORY_DAYS,
    entry_drawdown: float = ENTRY_DRAWDOWN,
    exit_drawdown: float = EXIT_DRAWDOWN,
    recovery_ma_days: int = RECOVERY_MA_DAYS,
    qqq_rotation_fraction: float = QQQ_ROTATION_FRACTION,
    rotation_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    r38._configure(
        base_multiplier=BASE_MULTIPLIER,
        active_multiplier=ACTIVE_MULTIPLIER,
        shock_multiplier=SHOCK_MULTIPLIER,
    )
    scheduled, execution_daily, risk_diagnostics = (
        r33.one_session_shock_schedule(
            weights,
            daily,
            closes,
            cash_floor=CASH_FLOOR,
        )
    )
    full_tilted, full_daily, overlay_diagnostics = (
        r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution_daily,
            closes,
            max_tilt=OVERLAY_FRACTION,
        )
    )
    base_reference, _ = _base_reference_schedule(
        weights,
        daily,
        closes,
    )
    rotated, rotated_daily, rotation_diagnostics = (
        apply_confirmed_rotation(
            full_tilted,
            base_reference,
            full_daily,
            closes,
            memory_days=memory_days,
            entry_drawdown=entry_drawdown,
            exit_drawdown=exit_drawdown,
            recovery_ma_days=recovery_ma_days,
            qqq_rotation_fraction=qqq_rotation_fraction,
            rotation_enabled=rotation_enabled,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        rotated,
        rotated_daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    limited, limited_daily, limit_diagnostics = (
        enforce_daily_cash_target_limit(
            guard_weights,
            guard_daily,
            cash_floor=CASH_FLOOR,
        )
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
        base_one_way_cost_bps=(
            scenario.base_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        gde_one_way_cost_bps=(
            scenario.gde_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        financing_spread_bps=(
            scenario.financing_spread_bps  # type: ignore[attr-defined]
        ),
        weights_override=limited,
        daily_override=limited_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = risk_diagnostics.reindex(trial.index).join(
        overlay_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_overlay",
    ).join(
        rotation_diagnostics.reindex(trial.index),
        how="left",
    ).join(
        limit_diagnostics.reindex(trial.index),
        how="left",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["pre_gde_cash_weight"] = limited.reindex(
        trial.index
    )["CASH"]
    diagnostics["post_guard_semis_weight"] = guard_weights.reindex(
        trial.index
    )["SEMIS"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def _definitions() -> list[dict[str, object]]:
    return [
        {"label": "central"},
        {"label": "memory42", "memory_days": 42},
        {"label": "memory84", "memory_days": 84},
        {"label": "entry04", "entry_drawdown": -0.04},
        {"label": "entry06", "entry_drawdown": -0.06},
        {"label": "exit015", "exit_drawdown": -0.015},
        {"label": "exit035", "exit_drawdown": -0.035},
        {"label": "recovery10", "recovery_ma_days": 10},
        {"label": "recovery50", "recovery_ma_days": 50},
        {"label": "all_qqq", "qqq_rotation_fraction": 1.0},
        {"label": "all_cash", "qqq_rotation_fraction": 0.0},
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {}
    diagnostics_by_sample: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            r11_path, _, _ = simulate_fixed_r11(settings, scenario)
            r38_path, _ = r38.simulate_candidate(settings, scenario)
            candidate_path, diagnostics = simulate_candidate(
                settings,
                scenario,
            )
            common = (
                r11_path.index.intersection(r38_path.index)
                .intersection(candidate_path.index)
            )
            for period, (start, end) in periods.items():
                selected = common[
                    (common >= start) & (common <= end)
                ]
                metric_rows.append(
                    _period_row(
                        sample,
                        scenario.name,
                        period,
                        r11_path.loc[selected, "net_return"],
                        r38_path.loc[selected, "net_return"],
                        candidate_path.loc[selected, "net_return"],
                    )
                )
            if scenario.name == "current_liquidity":
                paths[sample] = {
                    "r11": r11_path["net_return"],
                    "r38": r38_path["net_return"],
                    "candidate": candidate_path["net_return"],
                }
                diagnostics_by_sample[sample] = diagnostics
                for label, frame in (
                    ("r11", r11_path),
                    ("r38", r38_path),
                    ("candidate", candidate_path),
                ):
                    frame.to_csv(
                        OUTPUT / f"{sample}_{label}_daily.csv",
                        index_label="date",
                    )
                diagnostics.to_csv(
                    OUTPUT / f"{sample}_diagnostics.csv",
                    index_label="date",
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    normal = paths["normal_synthetic"]
    event_rows: list[dict[str, object]] = []
    for event, (start, end) in EVENT_WINDOWS.items():
        selected = normal["candidate"].index[
            (normal["candidate"].index >= start)
            & (normal["candidate"].index <= end)
        ]
        parent = _window_metrics(normal["r38"].reindex(selected))
        candidate = _window_metrics(
            normal["candidate"].reindex(selected)
        )
        event_rows.append(
            {
                "event": event,
                "start": start,
                "end": end,
                "r38_total_return": parent["total_return"],
                "candidate_total_return": candidate["total_return"],
                "total_return_delta": (
                    candidate["total_return"] - parent["total_return"]
                ),
                "r38_max_drawdown": parent["max_drawdown"],
                "candidate_max_drawdown": candidate["max_drawdown"],
                "max_drawdown_delta": (
                    candidate["max_drawdown"]
                    - parent["max_drawdown"]
                ),
            }
        )
    events = pd.DataFrame(event_rows)
    events.to_csv(OUTPUT / "event_windows.csv", index=False)

    years_rows: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        years, drawdowns = robustness_rows(
            sample,
            CANDIDATE,
            paths[sample]["r11"],
            paths[sample]["candidate"],
        )
        years_rows.extend(years)
        drawdown_rows.extend(drawdowns)
    years = pd.DataFrame(years_rows)
    drawdowns = pd.DataFrame(drawdown_rows)
    years.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    drawdowns.to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )

    central_settings = samples["normal_synthetic"]
    central_scenario = COST_SCENARIOS[0]
    r11_center, _, _ = simulate_fixed_r11(
        central_settings,
        central_scenario,
    )
    r38_center, _ = r38.simulate_candidate(
        central_settings,
        central_scenario,
    )
    neighborhood_rows: list[dict[str, object]] = []
    family_paths: list[np.ndarray] = []
    for definition in _definitions():
        label = str(definition["label"])
        arguments = {
            key: value
            for key, value in definition.items()
            if key != "label"
        }
        trial, diagnostics = simulate_candidate(
            central_settings,
            central_scenario,
            **arguments,
        )
        common = r38_center.index.intersection(trial.index)
        family_paths.append(
            (
                np.log1p(trial.loc[common, "net_return"])
                - np.log1p(r38_center.loc[common, "net_return"])
            ).to_numpy(dtype=float)
        )
        parent_delta = metric_delta(
            r38_center.loc[common, "net_return"],
            trial.loc[common, "net_return"],
        )
        r11_delta = metric_delta(
            r11_center.loc[common, "net_return"],
            trial.loc[common, "net_return"],
        )
        neighborhood_rows.append(
            {
                "neighborhood": label,
                "memory_days": arguments.get(
                    "memory_days", MEMORY_DAYS
                ),
                "entry_drawdown": arguments.get(
                    "entry_drawdown", ENTRY_DRAWDOWN
                ),
                "exit_drawdown": arguments.get(
                    "exit_drawdown", EXIT_DRAWDOWN
                ),
                "recovery_ma_days": arguments.get(
                    "recovery_ma_days", RECOVERY_MA_DAYS
                ),
                "qqq_rotation_fraction": arguments.get(
                    "qqq_rotation_fraction",
                    QQQ_ROTATION_FRACTION,
                ),
                "active_days": int(
                    diagnostics["confirmed_unwind_active"].sum()
                ),
                **{
                    f"vs_r38_{key}": value
                    for key, value in parent_delta.items()
                },
                **{
                    f"vs_r11_{key}": value
                    for key, value in r11_delta.items()
                },
            }
        )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    family = np.column_stack(family_paths)
    reality_rows: list[dict[str, object]] = []
    for block_days in (21, 63, 126):
        result = circular_family_reality_check(
            family,
            0,
            block_days,
        )
        reality_rows.append(
            {
                "block_days": block_days,
                "declared_family_size": family.shape[1],
                "cumulative_trials": CUMULATIVE_TRIALS,
                **result,
                "cumulative_trial_adjusted_p_value": min(
                    1.0,
                    float(result["familywise_reality_check_p_value"])
                    * CUMULATIVE_TRIALS
                    / family.shape[1],
                ),
            }
        )
    reality = pd.DataFrame(reality_rows)
    reality.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    def metric(
        sample: str,
        scenario: str,
        period: str,
        column: str,
    ) -> float:
        selected = metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["scenario"].eq(scenario)
            & metrics["period"].eq(period)
        ]
        return float(selected.iloc[0][column])

    frozen_periods = [
        ("normal_synthetic", "current_liquidity", "development_2015_2021"),
        ("normal_synthetic", "current_liquidity", "holdout_2022_2025"),
        ("normal_synthetic", "current_liquidity", "recent_2026"),
        ("normal_synthetic", "cost_stress", "complete_2015_2026"),
        ("proxy_synthetic", "current_liquidity", "early_2006_2014"),
        ("proxy_synthetic", "current_liquidity", "late_2015_2026"),
        ("normal_live", "current_liquidity", "live_2022_2026"),
    ]
    periods_nonnegative = all(
        metric(*key, "annualized_log_return_vs_r38") >= 0.0
        for key in frozen_periods
    )
    complete_cagr = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_candidate_cagr",
    )
    r38_cagr = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_baseline_cagr",
    )
    complete_mdd = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_candidate_max_drawdown",
    )
    r38_mdd = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_baseline_max_drawdown",
    )
    r11_mdd = metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r11_baseline_max_drawdown",
    )
    event_improvements = int(events["max_drawdown_delta"].gt(0.0).sum())
    event_average = float(events["max_drawdown_delta"].mean())
    event_worst = float(events["max_drawdown_delta"].min())
    diagnostics = diagnostics_by_sample["normal_synthetic"]
    current_active = bool(
        diagnostics.loc["2026-07-28", "confirmed_unwind_active"]
    )
    neighbors = neighborhoods.loc[
        ~neighborhoods["neighborhood"].eq("central")
    ]
    acceptance_rows = [
        ("complete_cagr_not_below_r38", complete_cagr >= r38_cagr),
        ("complete_drawdown_not_worse_than_r38", complete_mdd >= r38_mdd),
        (
            "complete_drawdown_within_r11_tolerance",
            complete_mdd >= r11_mdd - MAX_DRAWDOWN_TOLERANCE,
        ),
        ("all_frozen_periods_nonnegative_vs_r38", periods_nonnegative),
        ("four_of_six_event_windows_improve", event_improvements >= 4),
        ("average_event_drawdown_improves", event_average > 0.0),
        ("no_event_worsens_over_50bps", event_worst >= -0.005),
        ("current_2026_unwind_remains_active", current_active),
        (
            "leave_one_year_out_positive_vs_r11",
            years["annualized_relative_log_return"].gt(0.0).all(),
        ),
        (
            "leave_one_drawdown_out_positive_vs_r11",
            drawdowns["annualized_relative_log_return"].gt(0.0).all(),
        ),
        (
            "parameter_neighborhood_within_r11_tolerance",
            neighbors["vs_r11_candidate_max_drawdown"].ge(
                neighbors["vs_r11_baseline_max_drawdown"]
                - MAX_DRAWDOWN_TOLERANCE
            ).all(),
        ),
        (
            "rotation_non_growth_unchanged",
            diagnostics["rotation_non_growth_error"].le(1e-12).all(),
        ),
        (
            "rotation_growth_never_increases",
            diagnostics["rotation_growth_increase_error"]
            .le(1e-12)
            .all(),
        ),
        (
            "rotation_routing_exact",
            diagnostics[
                [
                    "rotation_qqq_routing_error",
                    "rotation_cash_routing_error",
                    "rotation_conservation_error",
                ]
            ].le(1e-12).all().all(),
        ),
        (
            "cash_hard_limit_pass",
            diagnostics["pre_gde_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all(),
        ),
        (
            "cumulative_trial_adjusted_significance",
            reality["cumulative_trial_adjusted_p_value"]
            .lt(0.05)
            .all(),
        ),
    ]
    acceptance = pd.DataFrame(
        acceptance_rows,
        columns=["gate", "passed"],
    )
    production_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = {
        "gate": "production_pass",
        "passed": production_pass,
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    summary = pd.DataFrame(
        {
            "value": {
                "candidate": CANDIDATE,
                "complete_cagr": complete_cagr,
                "r38_cagr": r38_cagr,
                "complete_max_drawdown": complete_mdd,
                "r38_max_drawdown": r38_mdd,
                "event_windows_improved": event_improvements,
                "average_event_drawdown_delta": event_average,
                "active_days": int(
                    diagnostics["confirmed_unwind_active"].sum()
                ),
                "current_unwind_active": current_active,
                "production_pass": production_pass,
            }
        }
    )
    summary.to_csv(OUTPUT / "summary.csv", index_label="metric")
    print(summary.to_string())
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print("\nEvent windows:")
    print(events.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
