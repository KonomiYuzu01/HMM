from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
)
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r38_capacity_rebalanced_frontier")
CANDIDATE = "r38_capacity_rebalanced_135_118"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.35
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.18
OVERLAY_FRACTION = 0.10
CUMULATIVE_TRIALS = 146
MAX_DRAWDOWN_TOLERANCE = 0.005


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


def _candidate_definitions() -> list[
    tuple[str, float, float, str]
]:
    return [
        ("identity_r38", 1.30, -0.20, "identity"),
        ("active130", 1.30, -0.18, "active"),
        ("central", 1.35, -0.18, "central"),
        ("active140", 1.40, -0.18, "active"),
        ("cash16", 1.35, -0.16, "cash"),
        ("cash20", 1.35, -0.20, "cash"),
        ("ridge_low", 1.30, -0.16, "ridge"),
        ("ridge_high", 1.40, -0.20, "ridge"),
    ]


def _neighborhood_rows(
    settings: dict[str, object],
    scenario: object,
    r11: pd.DataFrame,
    r38_path: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, active, cash, family in _candidate_definitions():
        trial, diagnostics = simulate_candidate(
            settings,
            scenario,
            active_multiplier=active,
            cash_floor=cash,
        )
        versus_r11 = metric_delta(
            r11["net_return"],
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
                "active_multiplier": active,
                "cash_floor": cash,
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
                "point_target_pass": bool(
                    versus_r11["candidate_cagr"] >= 0.25
                    and versus_r11["candidate_max_drawdown"]
                    >= (
                        versus_r11["baseline_max_drawdown"]
                        - MAX_DRAWDOWN_TOLERANCE
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
    r38_path: pd.DataFrame,
) -> None:
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    metrics = pd.read_csv(OUTPUT / "metrics_by_period.csv")
    central_complete = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["scenario"].eq("current_liquidity")
        & metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    central_daily = pd.read_csv(
        OUTPUT / "normal_synthetic_candidate_daily.csv",
        index_col="date",
        parse_dates=True,
    )
    identity = neighborhoods.loc[
        neighborhoods["neighborhood"].eq("identity_r38")
    ].iloc[0]
    relative_metrics = pd.read_csv(
        OUTPUT / "relative_to_r38_metrics.csv"
    )
    one_dimensional = neighborhoods.loc[
        neighborhoods["family"].isin(("active", "cash", "central"))
    ]
    ridge = neighborhoods.loc[
        neighborhoods["family"].isin(("ridge", "central"))
    ]
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    extra = {
        "complete_cagr_above_r38": float(
            central_complete["candidate_cagr"]
        )
        > float(
            metric_delta(
                r38_path["net_return"],
                central_daily["net_return"],
            )["baseline_cagr"]
        ),
        "complete_drawdown_not_worse_than_r38": float(
            central_complete["candidate_max_drawdown"]
        )
        >= float(
            metric_delta(
                r38_path["net_return"],
                central_daily["net_return"],
            )["baseline_max_drawdown"]
        ),
        "all_frozen_periods_nonnegative_vs_r38": bool(
            relative_metrics["annualized_relative_log_return"]
            .ge(0.0)
            .all()
        ),
        "one_dimensional_neighborhood_pass": int(
            one_dimensional["point_target_pass"].astype(bool).sum()
        )
        >= 4,
        "paired_ridge_pass": bool(
            ridge["point_target_pass"].astype(bool).all()
        ),
        "identity_equals_frozen_r38": bool(
            float(identity["identity_max_daily_return_error"])
            <= 1e-12
        ),
        "candidate_hard_limit_pass": bool(
            diagnostics["implemented_non_cash_weight"]
            .le(1.18 + 1e-12)
            .all()
        ),
        "all_neighborhood_hard_limits_pass": bool(
            neighborhoods["hard_limit_pass"].astype(bool).all()
        ),
    }
    for gate, passed in extra.items():
        acceptance.loc[len(acceptance)] = {
            "gate": gate,
            "passed": passed,
        }
    non_production = acceptance["gate"].ne("production_pass")
    production = bool(
        acceptance.loc[non_production, "passed"].astype(bool).all()
    )
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = production
    acceptance.to_csv(path, index=False)
    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("production_pass"),
        "value",
    ] = str(production)
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
    _rewrite_candidate_labels()
    _rewrite_acceptance(neighborhoods, r38_path)
    print("\nCapacity-rebalanced parameter neighborhood:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
