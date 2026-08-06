from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    GDE_FRACTION,
    simulate_fixed_r11,
)
from tools.evaluate_r14_incremental_trend_permission import (
    TRADING_EPSILON,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r24_declared_cash_hard_limit import (
    enforce_daily_cash_target_limit,
)
import tools.evaluate_r30_trend_risk_budget_pulse as r30
from tools.evaluate_r38_crowding_fragility_cap import EVENT_WINDOWS
import tools.evaluate_r38_accelerating_volatility_capacity_fill as accel
import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as current
import tools.evaluate_r38_convex_semiconductor_overlay as r38
from tools.evaluate_r38_stable_capacity_extension import (
    _metric_record,
)


OUTPUT = Path("output/r38_state_dependent_rollout")
BASE_SHARE = 0.25
STABLE_SHARES = (0.60, 0.70, 0.80)
TARGET_CAGR = 0.25
LEAVE_ONE_YEAR_CAGR = 0.245
DRAWDOWN_TOLERANCE = 0.0025
CUMULATIVE_TRIALS = 192


def build_r38_targets(
    settings: dict[str, object],
    scenario: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, risk_diagnostics = (
        accel.accelerating_volatility_schedule(
            weights,
            daily,
            closes,
            cash_floor=accel.vc.CASH_FLOOR,
            low_vol_active_multiplier=(
                current.LOW_VOL_ACTIVE_MULTIPLIER
            ),
            high_vol_active_multiplier=(
                current.HIGH_VOL_ACTIVE_MULTIPLIER
            ),
        )
    )
    tilted, tilted_daily, tilt_diagnostics = (
        r38.apply_convex_semiconductor_overlay(
            scheduled,
            execution_daily,
            closes,
            max_tilt=accel.vc.OVERLAY_FRACTION,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
        opens,
        closes,
        guard_for_multiplier(
            accel.vc.BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    limited, limited_daily, limit_diagnostics = (
        enforce_daily_cash_target_limit(
            guard_weights,
            guard_daily,
            cash_floor=accel.vc.CASH_FLOOR,
        )
    )
    diagnostics = risk_diagnostics.join(
        tilt_diagnostics,
        how="left",
        rsuffix="_tilt",
    ).join(
        limit_diagnostics,
        how="left",
    )
    return limited, limited_daily, diagnostics


def state_dependent_share(
    diagnostics: pd.DataFrame,
    stable_share: float,
    *,
    base_share: float = BASE_SHARE,
) -> pd.Series:
    if not 0.0 <= base_share <= stable_share <= 1.0:
        raise ValueError(
            "Shares must satisfy 0 <= base <= stable <= 1"
        )
    stable = diagnostics[
        "effective_incremental_permission"
    ].fillna(False).astype(bool)
    return pd.Series(
        np.where(stable, stable_share, base_share),
        index=diagnostics.index,
        dtype=float,
        name="r38_share",
    )


def simulate_blended_account(
    settings: dict[str, object],
    scenario: object,
    r11_weights: pd.DataFrame,
    r11_daily: pd.DataFrame,
    r38_weights: pd.DataFrame,
    r38_daily: pd.DataFrame,
    share: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    index = (
        r11_weights.index.intersection(r11_daily.index)
        .intersection(r38_weights.index)
        .intersection(r38_daily.index)
        .intersection(share.index)
    )
    aligned_share = share.reindex(index)
    blended_weights = (
        r11_weights.reindex(index).mul(
            1.0 - aligned_share,
            axis=0,
        )
        + r38_weights.reindex(index).mul(
            aligned_share,
            axis=0,
        )
    )
    r11_trade = (
        r11_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    r38_trade = (
        r38_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    share_change = aligned_share.ne(aligned_share.shift(1))
    update = r11_trade | r38_trade | share_change
    blended_daily = r11_daily.reindex(index).copy()
    blended_daily["turnover"] = np.where(update, 1e-12, 0.0)
    r11_slippage = r11_daily.reindex(index)[
        "slippage_cost"
    ].fillna(0.0)
    r38_slippage = r38_daily.reindex(index)[
        "slippage_cost"
    ].fillna(0.0)
    blended_slippage = (
        (1.0 - aligned_share) * r11_slippage
        + aligned_share * r38_slippage
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
        weights_override=blended_weights,
        daily_override=blended_daily,
        extra_slippage=blended_slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = pd.DataFrame(
        {
            "r38_share": aligned_share,
            "share_change": share_change,
            "r11_trade": r11_trade,
            "r38_trade": r38_trade,
            "execution_update": update,
            "blended_target_sum": blended_weights.sum(axis=1),
            "blended_target_cash": blended_weights["CASH"],
            "blended_slippage_cost": blended_slippage,
        },
        index=index,
    ).reindex(trial.index)
    return trial, diagnostics


def simulate_paths(
    settings: dict[str, object],
    scenario: object,
) -> tuple[
    pd.DataFrame,
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
    pd.DataFrame,
]:
    _, r11_weights, r11_daily = simulate_fixed_r11(
        settings,
        scenario,
    )
    r38_weights, r38_daily, r38_diagnostics = (
        build_r38_targets(settings, scenario)
    )
    constant_share = pd.Series(
        BASE_SHARE,
        index=r38_diagnostics.index,
        dtype=float,
    )
    baseline, baseline_diagnostics = simulate_blended_account(
        settings,
        scenario,
        r11_weights,
        r11_daily,
        r38_weights,
        r38_daily,
        constant_share,
    )
    paths: dict[float, pd.DataFrame] = {}
    diagnostics: dict[float, pd.DataFrame] = {}
    for stable_share in STABLE_SHARES:
        share = state_dependent_share(
            r38_diagnostics,
            stable_share,
        )
        path, account_diagnostics = simulate_blended_account(
            settings,
            scenario,
            r11_weights,
            r11_daily,
            r38_weights,
            r38_daily,
            share,
        )
        paths[stable_share] = path
        diagnostics[stable_share] = account_diagnostics.join(
            r38_diagnostics[
                [
                    "effective_incremental_permission",
                    "pulse_veto_active",
                    "volatility_acceleration_block",
                ]
            ],
            how="left",
        )
    return baseline, paths, diagnostics, baseline_diagnostics


def _evaluate() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
]:
    samples = r21._build_samples()
    rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    leave_rows: list[dict[str, object]] = []
    normal_paths: dict[float, pd.DataFrame] = {}
    normal_diagnostics: dict[float, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in r30.COST_SCENARIOS:
            baseline, paths, diagnostics, _ = simulate_paths(
                settings,
                scenario,
            )
            for stable_share, path in paths.items():
                common = baseline.index.intersection(path.index)
                for period, (start, end) in periods.items():
                    rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "stable_share": stable_share,
                            "period": period,
                            **_metric_record(
                                baseline.loc[start:end, "net_return"],
                                path.loc[start:end, "net_return"],
                            ),
                        }
                    )
                relative = (
                    np.log1p(path.loc[common, "net_return"])
                    - np.log1p(
                        baseline.loc[common, "net_return"]
                    )
                )
                for year, values in relative.groupby(
                    relative.index.year
                ):
                    annual_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "stable_share": stable_share,
                            "year": int(year),
                            "annual_relative_log_return": float(
                                values.sum()
                            ),
                        }
                    )
                if (
                    sample == "normal_synthetic"
                    and scenario.name == "current_liquidity"
                ):
                    normal_paths[stable_share] = path
                    normal_diagnostics[stable_share] = diagnostics[
                        stable_share
                    ]
                    for event, (start, end) in EVENT_WINDOWS.items():
                        event_rows.append(
                            {
                                "stable_share": stable_share,
                                "event": event,
                                "start": start,
                                "end": end,
                                **_metric_record(
                                    baseline.loc[
                                        start:end,
                                        "net_return",
                                    ],
                                    path.loc[
                                        start:end,
                                        "net_return",
                                    ],
                                ),
                            }
                        )
                    for omitted_year in sorted(set(common.year)):
                        selected = common[
                            common.year != omitted_year
                        ]
                        leave_rows.append(
                            {
                                "stable_share": stable_share,
                                "omitted_year": int(omitted_year),
                                **_metric_record(
                                    baseline.loc[
                                        selected,
                                        "net_return",
                                    ],
                                    path.loc[
                                        selected,
                                        "net_return",
                                    ],
                                ),
                            }
                        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(annual_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(leave_rows),
        normal_paths,
        normal_diagnostics,
    )


def _row(
    metrics: pd.DataFrame,
    share: float,
    sample: str,
    scenario: str,
    period: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["stable_share"].eq(share)
        & metrics["sample"].eq(sample)
        & metrics["scenario"].eq(scenario)
        & metrics["period"].eq(period)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Expected one metric row, received {len(selected)}"
        )
    return selected.iloc[0]


def _candidate_gates(
    share: float,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    events: pd.DataFrame,
    leave: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, bool]:
    complete = _row(
        metrics,
        share,
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
    )
    cost = _row(
        metrics,
        share,
        "normal_synthetic",
        "cost_stress",
        "complete_2015_2026",
    )
    frozen = metrics.loc[
        metrics["stable_share"].eq(share)
        & metrics["scenario"].eq("current_liquidity")
        & ~metrics["period"].isin(
            ("complete_2015_2026", "complete_2006_2026")
        )
    ]
    selected_events = events.loc[
        events["stable_share"].eq(share)
    ]
    selected_leave = leave.loc[
        leave["stable_share"].eq(share)
    ]
    selected_annual = annual.loc[
        annual["stable_share"].eq(share)
        & annual["sample"].eq("normal_synthetic")
        & annual["scenario"].eq("current_liquidity")
    ]
    development_positive = selected_annual.loc[
        selected_annual["year"].between(2015, 2021),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    holdout_positive = selected_annual.loc[
        selected_annual["year"].between(2022, 2025),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    stable = diagnostics[
        "effective_incremental_permission"
    ].fillna(False).astype(bool)
    implemented_share = diagnostics["r38_share"]
    return {
        "target_cagr_pass": bool(
            complete["candidate_cagr"] >= TARGET_CAGR
        ),
        "complete_drawdown_pass": bool(
            complete["max_drawdown_delta"]
            >= -DRAWDOWN_TOLERANCE
        ),
        "all_frozen_periods_positive_pass": bool(
            frozen["annualized_relative_log_return"].gt(0.0).all()
        ),
        "cost_stress_pass": bool(
            cost["annualized_relative_log_return"] > 0.0
            and cost["max_drawdown_delta"]
            >= -DRAWDOWN_TOLERANCE
        ),
        "event_drawdown_pass": bool(
            len(selected_events) == len(EVENT_WINDOWS)
            and selected_events["max_drawdown_delta"]
            .ge(-DRAWDOWN_TOLERANCE)
            .all()
        ),
        "leave_one_year_pass": bool(
            selected_leave["candidate_cagr"]
            .ge(LEAVE_ONE_YEAR_CAGR)
            .all()
            and selected_leave[
                "annualized_relative_log_return"
            ]
            .gt(0.0)
            .all()
        ),
        "year_breadth_pass": bool(
            development_positive >= 5 and holdout_positive >= 3
        ),
        "state_share_semantics_pass": bool(
            implemented_share.loc[stable].eq(share).all()
            and implemented_share.loc[~stable].eq(BASE_SHARE).all()
            and diagnostics["blended_target_sum"]
            .sub(1.0)
            .abs()
            .le(1e-12)
            .all()
        ),
    }


def select_smallest_passing_share(
    gates: pd.DataFrame,
) -> float | None:
    for share in STABLE_SHARES:
        selected = gates.loc[gates["stable_share"].eq(share)]
        if len(selected) != 1:
            raise AssertionError("Each share needs one gate row")
        columns = [
            column
            for column in selected
            if column.endswith("_pass")
        ]
        if selected.iloc[0][columns].astype(bool).all():
            return share
    return None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        metrics,
        annual,
        events,
        leave,
        paths,
        diagnostics,
    ) = _evaluate()
    gate_rows: list[dict[str, object]] = []
    for share in STABLE_SHARES:
        gate_rows.append(
            {
                "stable_share": share,
                **_candidate_gates(
                    share,
                    metrics,
                    annual,
                    events,
                    leave,
                    diagnostics[share],
                ),
            }
        )
    gates = pd.DataFrame(gate_rows)
    selected_share = select_smallest_passing_share(gates)

    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    annual.to_csv(
        OUTPUT / "annual_relative_returns.csv",
        index=False,
    )
    events.to_csv(OUTPUT / "event_windows.csv", index=False)
    leave.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    gates.to_csv(OUTPUT / "candidate_gates.csv", index=False)
    for share in STABLE_SHARES:
        label = int(round(share * 100))
        paths[share].to_csv(
            OUTPUT / f"normal_synthetic_stable{label}_daily.csv",
            index_label="date",
        )
        diagnostics[share].to_csv(
            OUTPUT
            / f"normal_synthetic_stable{label}_diagnostics.csv",
            index_label="date",
        )
    if selected_share is not None:
        paths[selected_share].to_csv(
            OUTPUT / "normal_synthetic_selected_daily.csv",
            index_label="date",
        )
        complete = _row(
            metrics,
            selected_share,
            "normal_synthetic",
            "current_liquidity",
            "complete_2015_2026",
        )
        selected_cagr = complete["candidate_cagr"]
        selected_drawdown = complete["candidate_max_drawdown"]
    else:
        selected_cagr = np.nan
        selected_drawdown = np.nan
    pd.Series(
        {
            "candidate": "r38_state_dependent_rollout",
            "cumulative_trials": CUMULATIVE_TRIALS,
            "selected_stable_share": selected_share,
            "base_share": BASE_SHARE,
            "selected_cagr": selected_cagr,
            "selected_max_drawdown": selected_drawdown,
            "research_pass_pending_spa_and_audit": (
                selected_share is not None
            ),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")
    print("Metrics:")
    print(metrics.round(6).to_string(index=False))
    print("\nEvents:")
    print(events.round(6).to_string(index=False))
    print("\nCandidate gates:")
    print(gates.to_string(index=False))
    print(f"\nSelected stable share: {selected_share}")
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
