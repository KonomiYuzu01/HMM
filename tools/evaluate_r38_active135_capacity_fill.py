from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
)
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
from tools.evaluate_r38_crowding_fragility_cap import EVENT_WINDOWS
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_active135_capacity_fill")
CANDIDATE = "r38_active135_capacity_fill"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.35
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
OVERLAY_FRACTION = 0.10
CUMULATIVE_TRIALS = 160
R38_DRAWDOWN_TOLERANCE = 0.0025
R11_DRAWDOWN_TOLERANCE = 0.005


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    overlay_fraction: float = OVERLAY_FRACTION,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    return r38.simulate_candidate(
        settings,
        scenario,
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
        cash_floor=cash_floor,
        overlay_fraction=overlay_fraction,
    )


def _definitions() -> list[
    tuple[str, str, float, float, float, float, float]
]:
    return [
        ("base1065", "base", 1.065, 1.35, 1.00, -0.20, 0.10),
        ("base1075", "base", 1.075, 1.35, 1.00, -0.20, 0.10),
        ("active130", "active", 1.070, 1.30, 1.00, -0.20, 0.10),
        ("active140", "active", 1.070, 1.40, 1.00, -0.20, 0.10),
        ("shock095", "shock", 1.070, 1.35, 0.95, -0.20, 0.10),
        ("shock105", "shock", 1.070, 1.35, 1.05, -0.20, 0.10),
        ("cash18", "cash", 1.070, 1.35, 1.00, -0.18, 0.10),
        ("cash22", "cash", 1.070, 1.35, 1.00, -0.22, 0.10),
        ("overlay05", "overlay", 1.070, 1.35, 1.00, -0.20, 0.05),
        ("overlay15", "overlay", 1.070, 1.35, 1.00, -0.20, 0.15),
    ]


def _neighborhood_rows(
    settings: dict[str, object],
    scenario: object,
    r11_path: pd.DataFrame,
    r38_path: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, family, base, active, shock, cash, overlay in (
        _definitions()
    ):
        trial, diagnostics = simulate_candidate(
            settings,
            scenario,
            base_multiplier=base,
            active_multiplier=active,
            shock_multiplier=shock,
            cash_floor=cash,
            overlay_fraction=overlay,
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
                "family": family,
                "base_multiplier": base,
                "active_multiplier": active,
                "shock_multiplier": shock,
                "cash_floor": cash,
                "overlay_fraction": overlay,
                "maximum_non_cash_weight": 1.0 - cash,
                "observed_maximum_non_cash_weight": float(
                    diagnostics["implemented_non_cash_weight"].max()
                ),
                "identity_max_daily_return_error": float(
                    (
                        trial["net_return"]
                        - r38_path["net_return"].reindex(trial.index)
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
                "relative_positive": bool(
                    versus_r11["candidate_cagr"]
                    > versus_r11["baseline_cagr"]
                ),
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
                    .le(1.0 - cash + 1e-12)
                    .all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _relative_to_r38_rows(
    samples: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
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
            for period, (period_start, period_end) in periods.items():
                selected = common[
                    (common >= period_start) & (common <= period_end)
                ]
                baseline_returns = baseline.loc[
                    selected, "net_return"
                ]
                candidate_returns = candidate.loc[
                    selected, "net_return"
                ]
                metric_rows.append(
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
                    family_rows.append(
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
    return pd.DataFrame(metric_rows), pd.DataFrame(family_rows)


def _event_rows(
    r38_path: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event, (start, end) in EVENT_WINDOWS.items():
        baseline_returns = r38_path.loc[start:end, "net_return"]
        candidate_returns = candidate.loc[start:end, "net_return"]
        baseline_metrics = performance_metrics(baseline_returns)
        candidate_metrics = performance_metrics(candidate_returns)
        rows.append(
            {
                "event": event,
                "start": start,
                "end": end,
                "r38_total_return": float(
                    (1.0 + baseline_returns).prod() - 1.0
                ),
                "candidate_total_return": float(
                    (1.0 + candidate_returns).prod() - 1.0
                ),
                "total_return_delta": float(
                    (1.0 + candidate_returns).prod()
                    - (1.0 + baseline_returns).prod()
                ),
                "r38_max_drawdown": float(
                    baseline_metrics["max_drawdown"]
                ),
                "candidate_max_drawdown": float(
                    candidate_metrics["max_drawdown"]
                ),
                "max_drawdown_delta": float(
                    candidate_metrics["max_drawdown"]
                    - baseline_metrics["max_drawdown"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _rewrite_candidate_labels() -> None:
    for path in OUTPUT.glob("*.csv"):
        frame = pd.read_csv(path)
        changed = False
        if "candidate" in frame:
            frame["candidate"] = CANDIDATE
            changed = True
        if path.name == "summary.csv" and set(frame.columns) >= {
            "Unnamed: 0",
            "value",
        }:
            selected = frame["Unnamed: 0"].eq("candidate")
            frame.loc[selected, "value"] = CANDIDATE
            changed = True
        if changed:
            frame.to_csv(path, index=False)


def _rewrite_acceptance(
    neighborhoods: pd.DataFrame,
    relative_metrics: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    acceptance.loc[
        acceptance["gate"].eq("pulse_duration_pass"),
        "gate",
    ] = "one_session_shock_duration_pass"
    acceptance.loc[
        acceptance["gate"].eq(
            "disabled_increment_equals_r11_pass"
        ),
        "gate",
    ] = "state_multiplier_semantics_pass"
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "state_multiplier_semantics_pass"
        ),
        "passed",
    ] = bool(
        diagnostics["state_multiplier_semantic_error"]
        .fillna(0.0)
        .le(1e-12)
        .all()
    )
    family_pass = bool(
        set(neighborhoods["family"])
        == {"base", "active", "shock", "cash", "overlay"}
        and neighborhoods.groupby("family")[
            "relative_positive"
        ].all().all()
        and neighborhoods.groupby("family")[
            "point_target_pass"
        ].any().all()
    )
    acceptance.loc[
        acceptance["gate"].eq("parameter_neighborhood_pass"),
        "passed",
    ] = family_pass
    complete = relative_metrics.loc[
        relative_metrics["sample"].eq("normal_synthetic")
        & relative_metrics["scenario"].eq("current_liquidity")
        & relative_metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    identity = neighborhoods.loc[
        neighborhoods["neighborhood"].eq("active130")
    ].iloc[0]
    extra = {
        "complete_cagr_above_r38": float(
            complete["cagr_delta"]
        )
        > 0.0,
        "complete_drawdown_within_r38_tolerance": float(
            complete["max_drawdown_delta"]
        )
        >= -R38_DRAWDOWN_TOLERANCE,
        "all_frozen_periods_positive_vs_r38": bool(
            relative_metrics["annualized_relative_log_return"]
            .gt(0.0)
            .all()
        ),
        "fast_selloff_tolerance_pass": bool(
            events["max_drawdown_delta"]
            .ge(-R38_DRAWDOWN_TOLERANCE)
            .all()
        ),
        "identity_equals_frozen_r38": bool(
            float(identity["identity_max_daily_return_error"])
            <= 1e-12
        ),
        "absolute_risk_cap_pass": bool(
            diagnostics["implemented_non_cash_weight"]
            .le(1.20 + 1e-12)
            .all()
            and diagnostics["pre_gde_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all()
        ),
        "growth_budget_conservation_pass": bool(
            diagnostics["growth_budget_error"]
            .fillna(0.0)
            .le(1e-12)
            .all()
        ),
        "convex_overlay_budget_pass": bool(
            diagnostics["overlay_absolute_deviation"]
            .le(OVERLAY_FRACTION + 1e-12)
            .all()
        ),
    }
    for gate, passed in extra.items():
        acceptance.loc[len(acceptance)] = {
            "gate": gate,
            "passed": passed,
        }
    ignored = {"multiple_testing_pass", "production_pass"}
    research_pass = bool(
        acceptance.loc[
            ~acceptance["gate"].isin(ignored),
            "passed",
        ]
        .astype(bool)
        .all()
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "research_pass_pending_full_library_spa",
        "passed": research_pass,
    }
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = False
    acceptance.to_csv(path, index=False)
    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("production_pass"),
        "value",
    ] = "False"
    summary.loc[len(summary)] = {
        "Unnamed: 0": "research_pass_pending_full_library_spa",
        "value": str(research_pass),
    }
    summary.to_csv(summary_path, index=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    r11_path, _, _ = simulate_fixed_r11(settings, scenario)
    r38_path, _ = r38.simulate_candidate(settings, scenario)

    r30.OUTPUT = OUTPUT
    r30.CASH_FLOOR = CASH_FLOOR
    r30.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r30.simulate_candidate = simulate_candidate
    r30.main()

    neighborhoods = _neighborhood_rows(
        settings,
        scenario,
        r11_path,
        r38_path,
    )
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    relative_metrics, relative_reality = _relative_to_r38_rows(
        samples
    )
    relative_metrics.to_csv(
        OUTPUT / "relative_to_r38_metrics.csv",
        index=False,
    )
    relative_reality.to_csv(
        OUTPUT / "relative_to_r38_reality_check.csv",
        index=False,
    )
    central = pd.read_csv(
        OUTPUT / "normal_synthetic_candidate_daily.csv",
        index_col="date",
        parse_dates=True,
    )
    events = _event_rows(r38_path, central)
    events.to_csv(OUTPUT / "event_windows.csv", index=False)
    _rewrite_candidate_labels()
    _rewrite_acceptance(
        neighborhoods,
        relative_metrics,
        events,
    )
    print("\nR38 active-1.35 neighborhood:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nRelative-to-R38 metrics:")
    print(
        relative_metrics[
            [
                "sample",
                "scenario",
                "period",
                "annualized_relative_log_return",
                "cagr_delta",
                "max_drawdown_delta",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nEvent windows:")
    print(events.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

