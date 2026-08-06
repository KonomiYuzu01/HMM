from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    scale_non_cash_weights_by_series,
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
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
    causal_trend_permission,
)
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_crowding_fragility_cap")
CANDIDATE = "r38_with_causal_crowding_fragility_cap"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
OVERLAY_FRACTION = 0.10
PERCENTILE = 0.90
CROWDING_CAP = 1.070
FRAGILITY_CAP = 1.000
THRESHOLD_LOOKBACK = 756
MINIMUM_THRESHOLD_OBSERVATIONS = 504
CUMULATIVE_TRIALS = 119
MAX_DRAWDOWN_TOLERANCE = 0.005

EVENT_WINDOWS = {
    "covid_acceleration_2020": ("2020-02-18", "2020-03-23"),
    "momentum_unwind_2020": ("2020-09-03", "2020-10-30"),
    "growth_unwind_2021": ("2021-02-17", "2021-03-08"),
    "semis_unwind_2024": ("2024-07-09", "2024-08-07"),
    "semis_shock_2026": ("2026-06-05", "2026-06-19"),
    "semis_unwind_2026": ("2026-06-22", "2026-07-28"),
}


def _causal_quantile_threshold(
    values: pd.Series,
    *,
    percentile: float,
    lookback: int = THRESHOLD_LOOKBACK,
    minimum_observations: int = MINIMUM_THRESHOLD_OBSERVATIONS,
) -> pd.Series:
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must be between zero and one")
    if minimum_observations > lookback:
        raise ValueError("minimum observations cannot exceed lookback")
    return (
        values.shift(2)
        .rolling(lookback, min_periods=minimum_observations)
        .quantile(percentile)
    )


def causal_crowding_fragility_state(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    percentile: float = PERCENTILE,
) -> pd.DataFrame:
    if "SEMIS" not in closes:
        raise ValueError("SEMIS is required for crowding diagnostics")
    semis = closes["SEMIS"].astype(float)
    log_returns = np.log(semis.div(semis.shift(1)))
    raw = pd.DataFrame(index=semis.index)
    raw["semis_return_126"] = np.log(semis.div(semis.shift(126)))
    raw["semis_extension_200"] = semis.div(
        semis.rolling(200, min_periods=200).mean()
    ) - 1.0
    realized_20 = (
        log_returns.rolling(20, min_periods=20).std(ddof=1)
        * np.sqrt(252.0)
    )
    realized_63 = (
        log_returns.rolling(63, min_periods=63).std(ddof=1)
        * np.sqrt(252.0)
    )
    raw["semis_volatility_ratio"] = realized_20.div(
        realized_63.clip(lower=1e-8)
    )

    diagnostics = pd.DataFrame(index=index)
    for column in raw:
        diagnostics[column] = raw[column].shift(1).reindex(index)
        diagnostics[f"{column}_threshold"] = (
            _causal_quantile_threshold(
                raw[column],
                percentile=percentile,
            ).reindex(index)
        )
    sufficient = diagnostics[
        [
            "semis_return_126_threshold",
            "semis_extension_200_threshold",
            "semis_volatility_ratio_threshold",
        ]
    ].notna().all(axis=1)
    crowding = (
        sufficient
        & diagnostics["semis_return_126"].ge(
            diagnostics["semis_return_126_threshold"]
        )
        & diagnostics["semis_extension_200"].ge(
            diagnostics["semis_extension_200_threshold"]
        )
    )
    fragility = (
        crowding
        & diagnostics["semis_volatility_ratio"].ge(
            diagnostics["semis_volatility_ratio_threshold"]
        )
    )
    diagnostics["crowding_history_sufficient"] = sufficient
    diagnostics["crowding_active"] = crowding
    diagnostics["fragility_active"] = fragility
    diagnostics["crowding_state_change"] = crowding.ne(
        crowding.shift(1)
    )
    diagnostics["fragility_state_change"] = fragility.ne(
        fragility.shift(1)
    )
    return diagnostics


def crowding_fragility_risk_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    percentile: float = PERCENTILE,
    crowding_cap: float = CROWDING_CAP,
    fragility_cap: float = FRAGILITY_CAP,
    fragility_enabled: bool = True,
    crowding_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if min(
        base_multiplier,
        active_multiplier,
        shock_multiplier,
        crowding_cap,
        fragility_cap,
    ) < 0.0:
        raise ValueError("risk multipliers and caps must be non-negative")
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    base_weights = weights.loc[index].copy()
    trend = causal_trend_permission(
        closes.reindex(index),
        r30.DEFINITION,
    )
    shock_diagnostics = r33.causal_one_session_shock(
        closes.reindex(index),
        index,
    )
    shock = shock_diagnostics["pulse_veto_active"].astype(bool)
    vulnerability = causal_crowding_fragility_state(
        closes.reindex(index),
        index,
        percentile=percentile,
    )
    crowding = vulnerability["crowding_active"].astype(bool)
    fragility = vulnerability["fragility_active"].astype(bool)
    if not crowding_enabled:
        crowding = pd.Series(False, index=index)
        fragility = pd.Series(False, index=index)
    elif not fragility_enabled:
        fragility = pd.Series(False, index=index)

    requested_absolute = pd.Series(
        np.where(
            shock,
            shock_multiplier,
            np.where(
                trend,
                base_multiplier * active_multiplier,
                base_multiplier,
            ),
        ),
        index=index,
        dtype=float,
    )
    requested_absolute = requested_absolute.where(
        ~crowding,
        np.minimum(requested_absolute, crowding_cap),
    )
    requested_absolute = requested_absolute.where(
        ~fragility,
        np.minimum(requested_absolute, fragility_cap),
    )

    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_change = trend.ne(trend.shift(1))
    shock_change = shock.ne(shock.shift(1))
    crowding_change = crowding.ne(crowding.shift(1))
    fragility_change = fragility.ne(fragility.shift(1))
    update = (
        base_trade
        | trend_change
        | shock_change
        | crowding_change
        | fragility_change
    )
    accepted_absolute = (
        requested_absolute.where(update)
        .ffill()
        .fillna(base_multiplier)
    )
    requested = scale_non_cash_weights_by_series(
        base_weights,
        accepted_absolute,
    )
    implemented, cap_scale = cap_weights_at_cash_floor(
        requested,
        cash_floor,
    )
    expected, _ = cap_weights_at_cash_floor(
        scale_non_cash_weights_by_series(
            base_weights,
            requested_absolute,
        ),
        cash_floor,
    )
    non_cash = [column for column in implemented if column != "CASH"]
    semantic_error = (
        implemented[non_cash] - expected[non_cash]
    ).abs().max(axis=1)
    if implemented["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("Crowding schedule exceeded its cash floor")
    if float(semantic_error.max()) > 1e-12:
        raise AssertionError("Crowding schedule did not implement its cap")

    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": trend,
            "pulse_veto_active": shock,
            "crowding_active": crowding,
            "fragility_active": fragility,
            "requested_absolute_multiplier": requested_absolute,
            "accepted_absolute_multiplier": accepted_absolute,
            "base_trade": base_trade,
            "trend_state_change": trend_change,
            "pulse_state_change": shock_change,
            "crowding_state_change": crowding_change,
            "fragility_state_change": fragility_change,
            "execution_update": update,
            "pre_cap_cash_weight": requested["CASH"],
            "cash_cap_scale": cap_scale,
            "implemented_cash_weight": implemented["CASH"],
            "implemented_non_cash_weight": implemented[
                non_cash
            ].sum(axis=1),
            "state_multiplier_semantic_error": semantic_error,
        },
        index=index,
    )
    for column in vulnerability:
        if column not in diagnostics:
            diagnostics[column] = vulnerability[column]
    for column in (
        "high_volatility_episode_entry",
        "pulse_veto_recovered",
        "pulse_veto_expired",
        "pulse_veto_remaining_sessions",
        "pair_realized_volatility",
        "high_volatility_threshold",
        "prior_pair_return",
    ):
        diagnostics[column] = shock_diagnostics[column]
    return implemented, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    percentile: float = PERCENTILE,
    crowding_cap: float = CROWDING_CAP,
    fragility_cap: float = FRAGILITY_CAP,
    fragility_enabled: bool = True,
    crowding_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, risk_diagnostics = (
        crowding_fragility_risk_schedule(
            weights,
            daily,
            closes,
            percentile=percentile,
            crowding_cap=crowding_cap,
            fragility_cap=fragility_cap,
            fragility_enabled=fragility_enabled,
            crowding_enabled=crowding_enabled,
        )
    )
    tilted, tilted_daily, overlay_diagnostics = (
        r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution_daily,
            closes,
            max_tilt=OVERLAY_FRACTION,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
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


def _period_row(
    sample: str,
    scenario: str,
    period: str,
    r11_returns: pd.Series,
    r38_returns: pd.Series,
    candidate_returns: pd.Series,
) -> dict[str, object]:
    common = (
        r11_returns.index.intersection(r38_returns.index)
        .intersection(candidate_returns.index)
    )
    r11_selected = r11_returns.reindex(common)
    r38_selected = r38_returns.reindex(common)
    candidate_selected = candidate_returns.reindex(common)
    parent_delta = metric_delta(r38_selected, candidate_selected)
    r11_delta = metric_delta(r11_selected, candidate_selected)
    return {
        "sample": sample,
        "scenario": scenario,
        "period": period,
        "annualized_log_return_vs_r38": float(
            (
                np.log1p(candidate_selected)
                - np.log1p(r38_selected)
            ).mean()
            * 252.0
        ),
        "annualized_log_return_vs_r11": float(
            (
                np.log1p(candidate_selected)
                - np.log1p(r11_selected)
            ).mean()
            * 252.0
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


def _window_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    metrics = performance_metrics(clean)
    return {
        "total_return": float(np.prod(1.0 + clean) - 1.0),
        "max_drawdown": float(metrics["max_drawdown"]),
    }


def _neighborhood_definitions() -> list[dict[str, object]]:
    return [
        {"label": "central"},
        {"label": "threshold85", "percentile": 0.85},
        {"label": "threshold95", "percentile": 0.95},
        {"label": "crowding100", "crowding_cap": 1.00},
        {"label": "crowding114", "crowding_cap": 1.14},
        {"label": "fragility095", "fragility_cap": 0.95},
        {"label": "fragility105", "fragility_cap": 1.05},
        {"label": "no_fragility", "fragility_enabled": False},
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    current_paths: dict[str, dict[str, pd.Series]] = {}
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
                current_paths[sample] = {
                    "r11": r11_path["net_return"],
                    "r38": r38_path["net_return"],
                    "candidate": candidate_path["net_return"],
                }
                diagnostics_by_sample[sample] = diagnostics
                r11_path.to_csv(
                    OUTPUT / f"{sample}_r11_daily.csv",
                    index_label="date",
                )
                r38_path.to_csv(
                    OUTPUT / f"{sample}_r38_daily.csv",
                    index_label="date",
                )
                candidate_path.to_csv(
                    OUTPUT / f"{sample}_candidate_daily.csv",
                    index_label="date",
                )
                diagnostics.to_csv(
                    OUTPUT / f"{sample}_diagnostics.csv",
                    index_label="date",
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    event_rows: list[dict[str, object]] = []
    normal = current_paths["normal_synthetic"]
    for event, (start, end) in EVENT_WINDOWS.items():
        selected = normal["candidate"].index[
            (normal["candidate"].index >= start)
            & (normal["candidate"].index <= end)
        ]
        parent_metrics = _window_metrics(normal["r38"].reindex(selected))
        candidate_metrics = _window_metrics(
            normal["candidate"].reindex(selected)
        )
        event_rows.append(
            {
                "event": event,
                "start": start,
                "end": end,
                "r38_total_return": parent_metrics["total_return"],
                "candidate_total_return": (
                    candidate_metrics["total_return"]
                ),
                "total_return_delta": (
                    candidate_metrics["total_return"]
                    - parent_metrics["total_return"]
                ),
                "r38_max_drawdown": parent_metrics["max_drawdown"],
                "candidate_max_drawdown": (
                    candidate_metrics["max_drawdown"]
                ),
                "max_drawdown_delta": (
                    candidate_metrics["max_drawdown"]
                    - parent_metrics["max_drawdown"]
                ),
            }
        )
    events = pd.DataFrame(event_rows)
    events.to_csv(OUTPUT / "event_windows.csv", index=False)

    year_rows: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        years, drawdowns = robustness_rows(
            sample,
            CANDIDATE,
            current_paths[sample]["r11"],
            current_paths[sample]["candidate"],
        )
        year_rows.extend(years)
        drawdown_rows.extend(drawdowns)
    years = pd.DataFrame(year_rows)
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
    relative_family: list[np.ndarray] = []
    for definition in _neighborhood_definitions():
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
        relative_family.append(
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
                "percentile": arguments.get("percentile", PERCENTILE),
                "crowding_cap": arguments.get(
                    "crowding_cap", CROWDING_CAP
                ),
                "fragility_cap": arguments.get(
                    "fragility_cap", FRAGILITY_CAP
                ),
                "fragility_enabled": arguments.get(
                    "fragility_enabled", True
                ),
                "crowding_days": int(
                    diagnostics["crowding_active"].sum()
                ),
                "fragility_days": int(
                    diagnostics["fragility_active"].sum()
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

    family_array = np.column_stack(relative_family)
    reality_rows: list[dict[str, object]] = []
    for block_days in (21, 63, 126):
        result = circular_family_reality_check(
            family_array,
            0,
            block_days,
        )
        reality_rows.append(
            {
                "block_days": block_days,
                "declared_family_size": family_array.shape[1],
                "cumulative_trials": CUMULATIVE_TRIALS,
                **result,
                "cumulative_trial_adjusted_p_value": min(
                    1.0,
                    float(result["familywise_reality_check_p_value"])
                    * CUMULATIVE_TRIALS
                    / family_array.shape[1],
                ),
            }
        )
    reality = pd.DataFrame(reality_rows)
    reality.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    def selected_metric(
        sample: str,
        scenario: str,
        period: str,
        column: str,
    ) -> float:
        row = metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["scenario"].eq(scenario)
            & metrics["period"].eq(period)
        ]
        return float(row.iloc[0][column])

    complete_cagr = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_candidate_cagr",
    )
    complete_mdd = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_candidate_max_drawdown",
    )
    r38_complete_mdd = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r38_baseline_max_drawdown",
    )
    r11_complete_mdd = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
        "vs_r11_baseline_max_drawdown",
    )
    holdout_cagr = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
        "vs_r38_candidate_cagr",
    )
    holdout_mdd_delta = selected_metric(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
        "vs_r38_max_drawdown_delta",
    )
    proxy_cagr = selected_metric(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
        "vs_r38_candidate_cagr",
    )
    proxy_mdd_delta = selected_metric(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
        "vs_r38_max_drawdown_delta",
    )
    event_improvements = int(events["max_drawdown_delta"].gt(0.0).sum())
    event_worst = float(events["max_drawdown_delta"].min())
    diagnostics = diagnostics_by_sample["normal_synthetic"]
    central_neighbors = neighborhoods.loc[
        ~neighborhoods["neighborhood"].eq("central")
    ]
    acceptance_rows = [
        ("complete_cagr_at_least_25pct", complete_cagr >= 0.25),
        (
            "complete_drawdown_improves_r38",
            complete_mdd > r38_complete_mdd,
        ),
        (
            "complete_drawdown_within_r11_tolerance",
            complete_mdd >= r11_complete_mdd - MAX_DRAWDOWN_TOLERANCE,
        ),
        ("holdout_cagr_at_least_24pct", holdout_cagr >= 0.24),
        ("holdout_drawdown_not_worse_than_r38", holdout_mdd_delta >= 0.0),
        ("proxy_cagr_at_least_17_5pct", proxy_cagr >= 0.175),
        ("proxy_drawdown_not_worse_than_r38", proxy_mdd_delta >= 0.0),
        ("four_of_six_event_windows_improve", event_improvements >= 4),
        ("no_event_worsens_over_50bps", event_worst >= -0.005),
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
            central_neighbors[
                "vs_r11_candidate_max_drawdown"
            ].ge(
                central_neighbors["vs_r11_baseline_max_drawdown"]
                - MAX_DRAWDOWN_TOLERANCE
            ).all(),
        ),
        (
            "cash_hard_limit_pass",
            diagnostics["pre_gde_cash_weight"].ge(
                CASH_FLOOR - 1e-12
            ).all(),
        ),
        (
            "state_multiplier_semantics_pass",
            diagnostics["state_multiplier_semantic_error"]
            .le(1e-12)
            .all(),
        ),
        (
            "growth_budget_conservation_pass",
            diagnostics["growth_budget_error"].le(1e-12).all(),
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
                "complete_max_drawdown": complete_mdd,
                "r38_max_drawdown": r38_complete_mdd,
                "r11_max_drawdown": r11_complete_mdd,
                "holdout_cagr": holdout_cagr,
                "proxy_cagr": proxy_cagr,
                "event_windows_improved": event_improvements,
                "crowding_days": int(
                    diagnostics["crowding_active"].sum()
                ),
                "fragility_days": int(
                    diagnostics["fragility_active"].sum()
                ),
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
