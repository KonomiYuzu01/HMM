from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    relative_log_return,
)
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
from tools.evaluate_r38_active135_capacity_fill import _event_rows
import tools.evaluate_r38_accelerating_volatility_capacity_fill as accel
import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as current
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_stable_capacity_extension")
CANDIDATE = "r38_stable_capacity_extension"
BASE_CASH_FLOOR = -0.20
STABLE_CASH_FLOORS = (-0.21, -0.22, -0.23, -0.25)
IDENTITY_CASH_FLOOR = -0.20
ROLLOUT_SHARES = (0.25, 0.50, 1.00)
CURRENT_PRODUCTION_SHARE = 0.25
TARGET_ROLLOUT_SHARE = 0.50
CUMULATIVE_TRIALS = 188
DRAWDOWN_TOLERANCE = 0.0025
TARGET_CAGR = 0.25
LEAVE_ONE_YEAR_CAGR = 0.245


def stable_capacity_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    stable_cash_floor: float,
    base_cash_floor: float = BASE_CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stable_cash_floor > base_cash_floor:
        raise ValueError(
            "stable_cash_floor cannot be tighter than base_cash_floor"
        )
    implemented, execution_daily, diagnostics = (
        accel.accelerating_volatility_schedule(
            weights,
            base_daily,
            closes,
            cash_floor=stable_cash_floor,
            low_vol_active_multiplier=(
                current.LOW_VOL_ACTIVE_MULTIPLIER
            ),
            high_vol_active_multiplier=(
                current.HIGH_VOL_ACTIVE_MULTIPLIER
            ),
        )
    )
    eligible = diagnostics[
        "effective_incremental_permission"
    ].astype(bool)
    base_capped, base_cap_scale = cap_weights_at_cash_floor(
        implemented,
        base_cash_floor,
    )
    recapped = implemented.copy()
    recapped.loc[~eligible] = base_capped.loc[~eligible]
    state_floor = pd.Series(
        np.where(eligible, stable_cash_floor, base_cash_floor),
        index=recapped.index,
        dtype=float,
    )
    non_cash = [column for column in recapped if column != "CASH"]
    maximum_non_cash = 1.0 - state_floor
    semantic_error = (
        recapped[non_cash].sum(axis=1) - maximum_non_cash
    ).clip(lower=0.0)
    if recapped["CASH"].lt(state_floor - 1e-12).any():
        raise AssertionError("State-specific cash floor was exceeded")
    if float(semantic_error.max()) > 1e-12:
        raise AssertionError("State-specific capacity was not enforced")

    diagnostics = diagnostics.copy()
    diagnostics["stable_extra_capacity_eligible"] = eligible
    diagnostics["state_cash_floor"] = state_floor
    diagnostics["base_state_recap_scale"] = base_cap_scale
    diagnostics["implemented_cash_weight"] = recapped["CASH"]
    diagnostics["implemented_non_cash_weight"] = recapped[
        non_cash
    ].sum(axis=1)
    diagnostics["state_capacity_semantic_error"] = semantic_error
    return recapped, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    stable_cash_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = r38.r37.r33.one_session_shock_schedule

    def schedule(
        weights: pd.DataFrame,
        base_daily: pd.DataFrame,
        closes: pd.DataFrame,
        *,
        cash_floor: float = stable_cash_floor,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        del cash_floor
        return stable_capacity_schedule(
            weights,
            base_daily,
            closes,
            stable_cash_floor=stable_cash_floor,
        )

    r38.r37.r33.one_session_shock_schedule = schedule
    try:
        return r38.simulate_candidate(
            settings,
            scenario,
            base_multiplier=accel.vc.BASE_MULTIPLIER,
            active_multiplier=current.LOW_VOL_ACTIVE_MULTIPLIER,
            shock_multiplier=accel.vc.SHOCK_MULTIPLIER,
            cash_floor=stable_cash_floor,
            overlay_fraction=accel.vc.OVERLAY_FRACTION,
        )
    finally:
        r38.r37.r33.one_session_shock_schedule = original


def rollout_returns(
    r11_returns: pd.Series,
    candidate_returns: pd.Series,
    share: float,
) -> pd.Series:
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be between zero and one")
    aligned = pd.concat(
        [
            r11_returns.rename("r11"),
            candidate_returns.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    return (
        (1.0 - share) * aligned["r11"]
        + share * aligned["candidate"]
    ).rename("net_return")


def _metric_record(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, float]:
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    base = performance_metrics(aligned["baseline"])
    trial = performance_metrics(aligned["candidate"])
    return {
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"candidate_{key}": value for key, value in trial.items()},
        "annualized_relative_log_return": float(
            (
                np.log1p(aligned["candidate"])
                - np.log1p(aligned["baseline"])
            ).mean()
            * 252.0
        ),
        "cagr_delta": trial["cagr"] - base["cagr"],
        "max_drawdown_delta": (
            trial["max_drawdown"] - base["max_drawdown"]
        ),
    }


def _evaluate_paths(
    samples: dict[str, dict[str, object]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
]:
    rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    normal_paths: dict[float, pd.DataFrame] = {}
    normal_diagnostics: dict[float, pd.DataFrame] = {}
    floors = (IDENTITY_CASH_FLOOR, *STABLE_CASH_FLOORS)
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in r30.COST_SCENARIOS:
            r11_path, _, _ = simulate_fixed_r11(settings, scenario)
            current_path, _ = current.simulate_candidate(
                settings,
                scenario,
            )
            common_base = r11_path.index.intersection(
                current_path.index
            )
            r11_returns = r11_path.loc[
                common_base, "net_return"
            ]
            current_returns = current_path.loc[
                common_base, "net_return"
            ]
            current_production = rollout_returns(
                r11_returns,
                current_returns,
                CURRENT_PRODUCTION_SHARE,
            )
            for stable_cash_floor in floors:
                if stable_cash_floor == IDENTITY_CASH_FLOOR:
                    candidate_path = current_path.copy()
                    candidate_diagnostics = pd.DataFrame(
                        index=current_path.index
                    )
                else:
                    candidate_path, candidate_diagnostics = (
                        simulate_candidate(
                            settings,
                            scenario,
                            stable_cash_floor=stable_cash_floor,
                        )
                    )
                common = common_base.intersection(
                    candidate_path.index
                )
                candidate_returns = candidate_path.loc[
                    common, "net_return"
                ]
                r11_common = r11_returns.reindex(common)
                current_common = current_returns.reindex(common)
                for share in ROLLOUT_SHARES:
                    current_rollout = rollout_returns(
                        r11_common,
                        current_common,
                        share,
                    )
                    candidate_rollout = rollout_returns(
                        r11_common,
                        candidate_returns,
                        share,
                    )
                    for period, (start, end) in periods.items():
                        rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "stable_cash_floor": (
                                    stable_cash_floor
                                ),
                                "rollout_share": share,
                                "period": period,
                                **_metric_record(
                                    current_rollout.loc[start:end],
                                    candidate_rollout.loc[start:end],
                                ),
                                "current_production_max_drawdown": (
                                    performance_metrics(
                                        current_production.loc[
                                            start:end
                                        ]
                                    )["max_drawdown"]
                                ),
                            }
                        )
                for year in sorted(set(common.year)):
                    selected = common[common.year == year]
                    relative = (
                        np.log1p(candidate_returns.loc[selected])
                        - np.log1p(current_common.loc[selected])
                    )
                    annual_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "stable_cash_floor": stable_cash_floor,
                            "year": int(year),
                            "annual_relative_log_return": float(
                                relative.sum()
                            ),
                        }
                    )
                if (
                    sample == "normal_synthetic"
                    and scenario.name == "current_liquidity"
                ):
                    normal_paths[stable_cash_floor] = (
                        candidate_path
                    )
                    normal_diagnostics[stable_cash_floor] = (
                        candidate_diagnostics
                    )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(annual_rows),
        normal_paths,
        normal_diagnostics,
    )


def _event_metrics(
    current_path: pd.DataFrame,
    candidate_paths: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stable_cash_floor in STABLE_CASH_FLOORS:
        event = _event_rows(
            current_path,
            candidate_paths[stable_cash_floor],
        )
        event["stable_cash_floor"] = stable_cash_floor
        rows.append(event)
    return pd.concat(rows, ignore_index=True)


def _leave_one_year_metrics(
    r11_returns: pd.Series,
    current_returns: pd.Series,
    candidate_paths: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common = r11_returns.index.intersection(current_returns.index)
    years = sorted(set(common.year))
    for stable_cash_floor in STABLE_CASH_FLOORS:
        candidate = candidate_paths[stable_cash_floor][
            "net_return"
        ].reindex(common)
        current_rollout = rollout_returns(
            r11_returns.reindex(common),
            current_returns.reindex(common),
            TARGET_ROLLOUT_SHARE,
        )
        candidate_rollout = rollout_returns(
            r11_returns.reindex(common),
            candidate,
            TARGET_ROLLOUT_SHARE,
        )
        for omitted_year in years:
            selected = common[common.year != omitted_year]
            rows.append(
                {
                    "stable_cash_floor": stable_cash_floor,
                    "omitted_year": int(omitted_year),
                    **_metric_record(
                        current_rollout.reindex(selected),
                        candidate_rollout.reindex(selected),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reality_checks(
    current_path: pd.DataFrame,
    candidate_paths: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    current_returns = current_path["net_return"]
    matrix = np.column_stack(
        [
            relative_log_return(
                current_returns,
                candidate_paths[floor]["net_return"],
            )
            for floor in STABLE_CASH_FLOORS
        ]
    )
    rows: list[dict[str, object]] = []
    for selected_index, floor in enumerate(STABLE_CASH_FLOORS):
        for block_days in (21, 63, 126):
            rows.append(
                {
                    "stable_cash_floor": floor,
                    "cumulative_trials": CUMULATIVE_TRIALS,
                    **circular_family_reality_check(
                        matrix,
                        selected_index,
                        block_days,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _row(
    metrics: pd.DataFrame,
    *,
    sample: str,
    scenario: str,
    floor: float,
    share: float,
    period: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["sample"].eq(sample)
        & metrics["scenario"].eq(scenario)
        & metrics["stable_cash_floor"].eq(floor)
        & metrics["rollout_share"].eq(share)
        & metrics["period"].eq(period)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Expected one metric row, received {len(selected)}"
        )
    return selected.iloc[0]


def _candidate_gates(
    floor: float,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    events: pd.DataFrame,
    leave_one_year: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, bool]:
    target = _row(
        metrics,
        sample="normal_synthetic",
        scenario="current_liquidity",
        floor=floor,
        share=TARGET_ROLLOUT_SHARE,
        period="complete_2015_2026",
    )
    full = _row(
        metrics,
        sample="normal_synthetic",
        scenario="current_liquidity",
        floor=floor,
        share=1.0,
        period="complete_2015_2026",
    )
    cost = _row(
        metrics,
        sample="normal_synthetic",
        scenario="cost_stress",
        floor=floor,
        share=TARGET_ROLLOUT_SHARE,
        period="complete_2015_2026",
    )
    frozen = metrics.loc[
        metrics["stable_cash_floor"].eq(floor)
        & metrics["rollout_share"].eq(TARGET_ROLLOUT_SHARE)
        & metrics["scenario"].eq("current_liquidity")
        & ~metrics["period"].isin(
            ("complete_2015_2026", "complete_2006_2026")
        )
    ]
    annual_selected = annual.loc[
        annual["sample"].eq("normal_synthetic")
        & annual["scenario"].eq("current_liquidity")
        & annual["stable_cash_floor"].eq(floor)
    ]
    development_positive = annual_selected.loc[
        annual_selected["year"].between(2015, 2021),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    holdout_positive = annual_selected.loc[
        annual_selected["year"].between(2022, 2025),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    event_selected = events.loc[
        events["stable_cash_floor"].eq(floor)
    ]
    loo = leave_one_year.loc[
        leave_one_year["stable_cash_floor"].eq(floor)
    ]
    eligible = diagnostics[
        "stable_extra_capacity_eligible"
    ].astype(bool)
    non_cash = diagnostics["implemented_non_cash_weight"]
    return {
        "target_cagr_pass": bool(
            target["candidate_cagr"] >= TARGET_CAGR
        ),
        "target_vs_current_production_drawdown_pass": bool(
            target["candidate_max_drawdown"]
            >= (
                target["current_production_max_drawdown"]
                - DRAWDOWN_TOLERANCE
            )
        ),
        "full_r38_drawdown_pass": bool(
            full["max_drawdown_delta"] >= -DRAWDOWN_TOLERANCE
        ),
        "all_frozen_periods_nonnegative_pass": bool(
            frozen["annualized_relative_log_return"]
            .ge(-1e-12)
            .all()
        ),
        "cost_stress_pass": bool(
            cost["cagr_delta"] >= -1e-12
            and cost["max_drawdown_delta"]
            >= -DRAWDOWN_TOLERANCE
        ),
        "fast_selloff_pass": bool(
            event_selected["max_drawdown_delta"]
            .ge(-DRAWDOWN_TOLERANCE)
            .all()
        ),
        "leave_one_year_pass": bool(
            loo["candidate_cagr"].ge(LEAVE_ONE_YEAR_CAGR).all()
            and loo["annualized_relative_log_return"]
            .ge(-1e-12)
            .all()
        ),
        "year_breadth_pass": bool(
            development_positive >= 5 and holdout_positive >= 3
        ),
        "state_capacity_limits_pass": bool(
            non_cash.loc[eligible]
            .le(1.0 - floor + 1e-12)
            .all()
            and non_cash.loc[~eligible]
            .le(1.20 + 1e-12)
            .all()
            and diagnostics["state_capacity_semantic_error"]
            .le(1e-12)
            .all()
        ),
    }


def select_smallest_passing_floor(
    gates: pd.DataFrame,
) -> float | None:
    for floor in STABLE_CASH_FLOORS:
        selected = gates.loc[gates["stable_cash_floor"].eq(floor)]
        if len(selected) != 1:
            raise AssertionError("Each cash floor needs one gate row")
        gate_columns = [
            column
            for column in selected
            if column.endswith("_pass")
        ]
        if selected.iloc[0][gate_columns].astype(bool).all():
            return floor
    return None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metrics, annual, paths, diagnostics = _evaluate_paths(samples)
    current_path = paths[IDENTITY_CASH_FLOOR]
    events = _event_metrics(current_path, paths)
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    r11_path, _, _ = simulate_fixed_r11(settings, scenario)
    common = r11_path.index.intersection(current_path.index)
    leave_one_year = _leave_one_year_metrics(
        r11_path.loc[common, "net_return"],
        current_path.loc[common, "net_return"],
        paths,
    )
    reality = _reality_checks(current_path, paths)
    identity_error = float(
        (
            paths[IDENTITY_CASH_FLOOR]["net_return"]
            - current_path["net_return"]
        )
        .abs()
        .max()
    )

    gate_rows: list[dict[str, object]] = []
    for floor in STABLE_CASH_FLOORS:
        gates = _candidate_gates(
            floor,
            metrics,
            annual,
            events,
            leave_one_year,
            diagnostics[floor],
        )
        gate_rows.append(
            {
                "stable_cash_floor": floor,
                **gates,
                "identity_path_pass": identity_error <= 1e-12,
            }
        )
    gate_frame = pd.DataFrame(gate_rows)
    selected_floor = select_smallest_passing_floor(gate_frame)

    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    annual.to_csv(OUTPUT / "annual_relative_returns.csv", index=False)
    events.to_csv(OUTPUT / "event_windows.csv", index=False)
    leave_one_year.to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )
    reality.to_csv(OUTPUT / "family_reality_check.csv", index=False)
    gate_frame.to_csv(OUTPUT / "candidate_gates.csv", index=False)
    for floor in STABLE_CASH_FLOORS:
        label = int(round(abs(floor) * 100))
        paths[floor].to_csv(
            OUTPUT / f"normal_synthetic_floor{label}_daily.csv",
            index_label="date",
        )
        diagnostics[floor].to_csv(
            OUTPUT / f"normal_synthetic_floor{label}_diagnostics.csv",
            index_label="date",
        )

    if selected_floor is not None:
        selected = _row(
            metrics,
            sample="normal_synthetic",
            scenario="current_liquidity",
            floor=selected_floor,
            share=TARGET_ROLLOUT_SHARE,
            period="complete_2015_2026",
        )
        selected_label = int(round(abs(selected_floor) * 100))
        paths[selected_floor].to_csv(
            OUTPUT / "normal_synthetic_selected_daily.csv",
            index_label="date",
        )
        selected_cagr = float(selected["candidate_cagr"])
        selected_drawdown = float(
            selected["candidate_max_drawdown"]
        )
    else:
        selected_label = ""
        selected_cagr = np.nan
        selected_drawdown = np.nan
    pd.Series(
        {
            "candidate": CANDIDATE,
            "cumulative_trials": CUMULATIVE_TRIALS,
            "selected_stable_cash_floor": selected_floor,
            "selected_non_cash_cap_percent": selected_label,
            "target_rollout_share": TARGET_ROLLOUT_SHARE,
            "selected_target_cagr": selected_cagr,
            "selected_target_max_drawdown": selected_drawdown,
            "research_pass_pending_spa_and_audit": (
                selected_floor is not None
            ),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")

    columns = [
        "sample",
        "scenario",
        "stable_cash_floor",
        "rollout_share",
        "period",
        "candidate_cagr",
        "candidate_max_drawdown",
        "annualized_relative_log_return",
    ]
    print(metrics[columns].round(6).to_string(index=False))
    print("\nCandidate gates:")
    print(gate_frame.to_string(index=False))
    print(f"\nSelected stable cash floor: {selected_floor}")
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
