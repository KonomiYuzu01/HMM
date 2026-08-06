from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
    causal_trend_permission,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r33_one_session_unlevered_shock_brake as r33
from tools.evaluate_r38_active135_capacity_fill import _event_rows
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_volatility_conditioned_capacity_fill")
CANDIDATE = "r38_volatility_conditioned_capacity_fill"
BASE_MULTIPLIER = 1.070
LOW_VOL_ACTIVE_MULTIPLIER = 1.35
HIGH_VOL_ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
OVERLAY_FRACTION = 0.10
CUMULATIVE_TRIALS = 166
R38_DRAWDOWN_TOLERANCE = 0.0025
R11_DRAWDOWN_TOLERANCE = 0.005


def volatility_conditioned_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cash_floor: float = CASH_FLOOR,
    low_vol_active_multiplier: float | pd.Series = (
        LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    base_multiplier: float = BASE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    low_multiplier = (
        low_vol_active_multiplier.reindex(index).ffill()
        if isinstance(low_vol_active_multiplier, pd.Series)
        else pd.Series(float(low_vol_active_multiplier), index=index)
    )
    if low_multiplier.isna().any():
        raise ValueError("Low-volatility multiplier is missing at the sample start")
    if min(
        float(low_multiplier.min()),
        high_vol_active_multiplier,
        base_multiplier,
        shock_multiplier,
    ) < 0.0:
        raise ValueError("Risk multipliers must be non-negative")
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
    high_volatility = shock_diagnostics[
        "volatility_gate_active"
    ].astype(bool)
    state_active_multiplier = pd.Series(
        np.where(
            high_volatility,
            high_vol_active_multiplier,
            low_multiplier,
        ),
        index=index,
        dtype=float,
    )
    requested_absolute = pd.Series(
        np.where(
            shock,
            shock_multiplier,
            np.where(
                trend,
                base_multiplier * state_active_multiplier,
                base_multiplier,
            ),
        ),
        index=index,
        dtype=float,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_change = trend.ne(trend.shift(1))
    shock_change = shock.ne(shock.shift(1))
    low_multiplier_change = low_multiplier.ne(low_multiplier.shift(1))
    high_volatility_change = (
        high_volatility.ne(high_volatility.shift(1))
        & (low_multiplier - high_vol_active_multiplier).abs().gt(1e-12)
    )
    state_multiplier_change = high_volatility_change | (
        ~high_volatility & low_multiplier_change
    )
    update = (
        base_trade
        | trend_change
        | shock_change
        | state_multiplier_change
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
    expected_requested = scale_non_cash_weights_by_series(
        base_weights,
        requested_absolute,
    )
    expected_implemented, _ = cap_weights_at_cash_floor(
        expected_requested,
        cash_floor,
    )
    non_cash = [
        column for column in implemented if column != "CASH"
    ]
    semantic_error = (
        implemented[non_cash] - expected_implemented[non_cash]
    ).abs().max(axis=1)
    if implemented["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("Candidate exceeded its cash floor")
    if float(semantic_error.max()) > 1e-12:
        raise AssertionError("Candidate state was not implemented")

    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(
            dtype=float
        ),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": trend,
            "pulse_veto_active": shock,
            "effective_incremental_permission": (
                trend & ~shock & ~high_volatility
            ),
            "requested_absolute_multiplier": requested_absolute,
            "accepted_absolute_multiplier": accepted_absolute,
            "base_trade": base_trade,
            "trend_state_change": trend_change,
            "pulse_state_change": shock_change,
            "high_volatility_state_change": (
                high_volatility_change
            ),
            "state_multiplier_change": state_multiplier_change,
            "low_volatility_active_multiplier": low_multiplier,
            "execution_update": update,
            "pre_cap_cash_weight": requested["CASH"],
            "cash_cap_scale": cap_scale,
            "implemented_cash_weight": implemented["CASH"],
            "implemented_non_cash_weight": implemented[
                non_cash
            ].sum(axis=1),
            "base_r11_non_cash_weight": np.nan,
            "state_multiplier_semantic_error": semantic_error,
            "high_volatility_active": high_volatility,
            "state_active_multiplier": state_active_multiplier,
        },
        index=index,
    )
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
    low_vol_active_multiplier: float = (
        LOW_VOL_ACTIVE_MULTIPLIER
    ),
    high_vol_active_multiplier: float = (
        HIGH_VOL_ACTIVE_MULTIPLIER
    ),
    cash_floor: float = CASH_FLOOR,
    overlay_fraction: float = OVERLAY_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = r38.r37.r33.one_session_shock_schedule

    def schedule(
        weights: pd.DataFrame,
        base_daily: pd.DataFrame,
        closes: pd.DataFrame,
        *,
        cash_floor: float = cash_floor,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return volatility_conditioned_schedule(
            weights,
            base_daily,
            closes,
            cash_floor=cash_floor,
            low_vol_active_multiplier=low_vol_active_multiplier,
            high_vol_active_multiplier=high_vol_active_multiplier,
        )

    r38.r37.r33.one_session_shock_schedule = schedule
    try:
        return r38.simulate_candidate(
            settings,
            scenario,
            base_multiplier=BASE_MULTIPLIER,
            active_multiplier=low_vol_active_multiplier,
            shock_multiplier=SHOCK_MULTIPLIER,
            cash_floor=cash_floor,
            overlay_fraction=overlay_fraction,
        )
    finally:
        r38.r37.r33.one_session_shock_schedule = original


def _definitions() -> list[tuple[str, float, float]]:
    return [
        ("central", 1.35, 1.30),
        ("low1325", 1.325, 1.30),
        ("low1375", 1.375, 1.30),
        ("high1275", 1.35, 1.275),
        ("high1325", 1.35, 1.325),
        ("identity_r38", 1.30, 1.30),
        ("identity_fixed135", 1.35, 1.35),
    ]


def _metric_rows(
    samples: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    reality_rows: list[dict[str, object]] = []
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in r30.COST_SCENARIOS:
            baseline, _ = r38.simulate_candidate(
                settings,
                scenario,
            )
            candidate, _ = simulate_candidate(
                settings,
                scenario,
            )
            common = baseline.index.intersection(candidate.index)
            for period, (start, end) in periods.items():
                selected = common[
                    (common >= start) & (common <= end)
                ]
                baseline_returns = baseline.loc[
                    selected, "net_return"
                ]
                candidate_returns = candidate.loc[
                    selected, "net_return"
                ]
                rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "period": period,
                        "annualized_relative_log_return": float(
                            (
                                np.log1p(candidate_returns)
                                - np.log1p(baseline_returns)
                            ).mean()
                            * 252.0
                        ),
                        **metric_delta(
                            baseline_returns,
                            candidate_returns,
                        ),
                    }
                )
            if (
                scenario.name == "current_liquidity"
                and sample in ("normal_synthetic", "proxy_synthetic")
            ):
                relative = relative_log_return(
                    baseline["net_return"],
                    candidate["net_return"],
                )[:, None]
                for block_days in (21, 63, 126):
                    check = circular_family_reality_check(
                        relative,
                        0,
                        block_days,
                    )
                    raw_p = float(
                        check["familywise_reality_check_p_value"]
                    )
                    reality_rows.append(
                        {
                            "sample": sample,
                            "block_days": block_days,
                            "cumulative_trials": CUMULATIVE_TRIALS,
                            **check,
                            "cumulative_trial_adjusted_p_value": min(
                                1.0,
                                raw_p * CUMULATIVE_TRIALS,
                            ),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(reality_rows)


def _neighborhood_rows(
    settings: dict[str, object],
    scenario: object,
    r11_path: pd.DataFrame,
    r38_path: pd.DataFrame,
    fixed_path: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, low, high in _definitions():
        trial, diagnostics = simulate_candidate(
            settings,
            scenario,
            low_vol_active_multiplier=low,
            high_vol_active_multiplier=high,
        )
        versus_r11 = metric_delta(
            r11_path["net_return"],
            trial["net_return"],
        )
        versus_r38 = metric_delta(
            r38_path["net_return"],
            trial["net_return"],
        )
        rows.append(
            {
                "neighborhood": label,
                "low_vol_active_multiplier": low,
                "high_vol_active_multiplier": high,
                "observed_maximum_non_cash_weight": float(
                    diagnostics["implemented_non_cash_weight"].max()
                ),
                "r38_identity_max_daily_error": float(
                    (
                        trial["net_return"]
                        - r38_path["net_return"].reindex(trial.index)
                    )
                    .abs()
                    .max()
                ),
                "fixed_active_identity_max_daily_error": float(
                    (
                        trial["net_return"]
                        - fixed_path["net_return"].reindex(trial.index)
                    )
                    .abs()
                    .max()
                ),
                **{
                    f"vs_r11_{key}": value
                    for key, value in versus_r11.items()
                },
                **{
                    f"vs_r38_{key}": value
                    for key, value in versus_r38.items()
                },
                "point_target_pass": bool(
                    versus_r11["candidate_cagr"] >= 0.25
                    and versus_r11["candidate_max_drawdown"]
                    >= (
                        versus_r11["baseline_max_drawdown"]
                        - R11_DRAWDOWN_TOLERANCE
                    )
                ),
                "hard_limit_pass": bool(
                    diagnostics["implemented_non_cash_weight"]
                    .le(1.20 + 1e-12)
                    .all()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    r11_path, _, _ = simulate_fixed_r11(settings, scenario)
    r38_path, _ = r38.simulate_candidate(settings, scenario)
    central_low_multiplier = _definitions()[0][1]
    fixed_path, _ = r38.simulate_candidate(
        settings,
        scenario,
        active_multiplier=central_low_multiplier,
    )
    candidate, diagnostics = simulate_candidate(
        settings,
        scenario,
    )
    metrics, reality = _metric_rows(samples)
    neighborhoods = _neighborhood_rows(
        settings,
        scenario,
        r11_path,
        r38_path,
        fixed_path,
    )
    events = _event_rows(r38_path, candidate)
    fixed_events = _event_rows(r38_path, fixed_path).rename(
        columns={
            "candidate_max_drawdown": (
                "fixed_active_candidate_max_drawdown"
            ),
            "max_drawdown_delta": (
                "fixed_active_max_drawdown_delta"
            ),
        }
    )
    events = events.merge(
        fixed_events[
            [
                "event",
                "fixed_active_candidate_max_drawdown",
                "fixed_active_max_drawdown_delta",
            ]
        ],
        on="event",
        how="left",
    )
    candidate.to_csv(
        OUTPUT / "normal_synthetic_candidate_daily.csv",
        index_label="date",
    )
    diagnostics.to_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv",
        index_label="date",
    )
    metrics.to_csv(
        OUTPUT / "relative_to_r38_metrics.csv",
        index=False,
    )
    reality.to_csv(
        OUTPUT / "relative_to_r38_reality_check.csv",
        index=False,
    )
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    events.to_csv(OUTPUT / "event_windows.csv", index=False)

    complete = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["scenario"].eq("current_liquidity")
        & metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    r38_identity = neighborhoods.loc[
        neighborhoods["neighborhood"].eq("identity_r38")
    ].iloc[0]
    fixed_identity = neighborhoods.loc[
        neighborhoods["neighborhood"].str.startswith(
            "identity_fixed"
        )
    ].iloc[0]
    major = events["event"].isin(
        (
            "growth_unwind_2021",
            "semis_unwind_2024",
            "semis_unwind_2026",
        )
    )
    economic_neighbors = neighborhoods.loc[
        ~neighborhoods["neighborhood"].str.startswith("identity")
    ]
    gates = {
        "complete_cagr_above_r38": float(
            complete["cagr_delta"]
        )
        > 0.0,
        "complete_drawdown_within_r38_tolerance": float(
            complete["max_drawdown_delta"]
        )
        >= -R38_DRAWDOWN_TOLERANCE,
        "all_frozen_periods_positive_vs_r38": bool(
            metrics["annualized_relative_log_return"].gt(0.0).all()
        ),
        "fast_selloff_tolerance_pass": bool(
            events["max_drawdown_delta"]
            .ge(-R38_DRAWDOWN_TOLERANCE)
            .all()
        ),
        "major_selloffs_not_worse_than_fixed_active": bool(
            events.loc[major, "candidate_max_drawdown"].mean()
            >= events.loc[
                major,
                "fixed_active_candidate_max_drawdown",
            ].mean()
        ),
        "parameter_neighborhood_pass": bool(
            economic_neighbors["point_target_pass"].all()
        ),
        "r38_identity_pass": bool(
            float(r38_identity["r38_identity_max_daily_error"])
            <= 1e-12
        ),
        "fixed_active_identity_pass": bool(
            float(
                fixed_identity[
                    "fixed_active_identity_max_daily_error"
                ]
            )
            <= 1e-12
        ),
        "cash_and_non_cash_limits_pass": bool(
            diagnostics["implemented_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all()
            and diagnostics["implemented_non_cash_weight"]
            .le(1.20 + 1e-12)
            .all()
            and neighborhoods["hard_limit_pass"].all()
        ),
        "state_multiplier_semantics_pass": bool(
            diagnostics["state_multiplier_semantic_error"]
            .le(1e-12)
            .all()
        ),
    }
    acceptance = pd.DataFrame(
        [
            {"gate": gate, "passed": bool(passed)}
            for gate, passed in gates.items()
        ]
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "research_pass",
        "passed": bool(all(gates.values())),
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    pd.Series(
        {
            "candidate": CANDIDATE,
            "complete_cagr": complete["candidate_cagr"],
            "r38_cagr": complete["baseline_cagr"],
            "complete_max_drawdown": (
                complete["candidate_max_drawdown"]
            ),
            "r38_max_drawdown": complete[
                "baseline_max_drawdown"
            ],
            "high_volatility_days": int(
                diagnostics["high_volatility_active"].sum()
            ),
            "extra_capacity_days": int(
                diagnostics[
                    "effective_incremental_permission"
                ].sum()
            ),
            "research_pass": bool(all(gates.values())),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")
    print(pd.read_csv(OUTPUT / "summary.csv").to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print("\nEvent windows:")
    print(events.round(6).to_string(index=False))
    print("\nParameter neighborhood:")
    print(neighborhoods.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
