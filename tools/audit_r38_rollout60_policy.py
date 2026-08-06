from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
from tools.evaluate_r38_crowding_fragility_cap import EVENT_WINDOWS
import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as r38
from tools.evaluate_r38_stable_capacity_extension import (
    _metric_record,
    rollout_returns,
)


OUTPUT = Path("output/r38_rollout60_policy")
CURRENT_SHARE = 0.25
TARGET_SHARE = 0.60
TARGET_CAGR = 0.25
LEAVE_ONE_YEAR_CAGR = 0.245
DRAWDOWN_TOLERANCE = 0.0025
CUMULATIVE_TRIALS = 189


def _evaluate_metrics(
    samples: dict[str, dict[str, object]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[pd.Series, pd.Series]],
]:
    rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    current_paths: dict[str, tuple[pd.Series, pd.Series]] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in r30.COST_SCENARIOS:
            r11_path, _, _ = simulate_fixed_r11(settings, scenario)
            r38_path, _ = r38.simulate_candidate(settings, scenario)
            common = r11_path.index.intersection(r38_path.index)
            r11_returns = r11_path.loc[common, "net_return"]
            r38_returns = r38_path.loc[common, "net_return"]
            current = rollout_returns(
                r11_returns,
                r38_returns,
                CURRENT_SHARE,
            )
            target = rollout_returns(
                r11_returns,
                r38_returns,
                TARGET_SHARE,
            )
            for period, (start, end) in periods.items():
                rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "period": period,
                        "current_share": CURRENT_SHARE,
                        "target_share": TARGET_SHARE,
                        **_metric_record(
                            current.loc[start:end],
                            target.loc[start:end],
                        ),
                    }
                )
            for year in sorted(set(common.year)):
                selected = common[common.year == year]
                relative = (
                    np.log1p(target.loc[selected])
                    - np.log1p(current.loc[selected])
                )
                annual_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "year": int(year),
                        "annual_relative_log_return": float(
                            relative.sum()
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                current_paths[sample] = (current, target)
    return (
        pd.DataFrame(rows),
        pd.DataFrame(annual_rows),
        current_paths,
    )


def _event_rows(
    current: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event, (start, end) in EVENT_WINDOWS.items():
        baseline = current.loc[start:end]
        candidate = target.loc[start:end]
        if baseline.empty or candidate.empty:
            continue
        metrics = _metric_record(baseline, candidate)
        rows.append(
            {
                "event": event,
                "start": start,
                "end": end,
                "current_total_return": float(
                    (1.0 + baseline).prod() - 1.0
                ),
                "target_total_return": float(
                    (1.0 + candidate).prod() - 1.0
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_year_rows(
    current: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    common = current.index.intersection(target.index)
    rows: list[dict[str, object]] = []
    for omitted_year in sorted(set(common.year)):
        selected = common[common.year != omitted_year]
        rows.append(
            {
                "omitted_year": int(omitted_year),
                **_metric_record(
                    current.loc[selected],
                    target.loc[selected],
                ),
            }
        )
    return pd.DataFrame(rows)


def _selected_row(
    metrics: pd.DataFrame,
    sample: str,
    scenario: str,
    period: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["sample"].eq(sample)
        & metrics["scenario"].eq(scenario)
        & metrics["period"].eq(period)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Expected one metric row, received {len(selected)}"
        )
    return selected.iloc[0]


def _acceptance(
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    events: pd.DataFrame,
    leave_one_year: pd.DataFrame,
) -> pd.DataFrame:
    complete = _selected_row(
        metrics,
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
    )
    cost = _selected_row(
        metrics,
        "normal_synthetic",
        "cost_stress",
        "complete_2015_2026",
    )
    frozen = metrics.loc[
        metrics["scenario"].eq("current_liquidity")
        & ~metrics["period"].isin(
            ("complete_2015_2026", "complete_2006_2026")
        )
    ]
    annual_normal = annual.loc[
        annual["sample"].eq("normal_synthetic")
        & annual["scenario"].eq("current_liquidity")
    ]
    development_positive = annual_normal.loc[
        annual_normal["year"].between(2015, 2021),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    holdout_positive = annual_normal.loc[
        annual_normal["year"].between(2022, 2025),
        "annual_relative_log_return",
    ].gt(0.0).sum()
    gates = {
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
            len(events) == len(EVENT_WINDOWS)
            and events["max_drawdown_delta"]
            .ge(-DRAWDOWN_TOLERANCE)
            .all()
        ),
        "leave_one_year_pass": bool(
            leave_one_year["candidate_cagr"]
            .ge(LEAVE_ONE_YEAR_CAGR)
            .all()
            and leave_one_year["annualized_relative_log_return"]
            .gt(0.0)
            .all()
        ),
        "year_breadth_pass": bool(
            development_positive >= 5 and holdout_positive >= 3
        ),
    }
    rows = [
        {"gate": gate, "passed": passed}
        for gate, passed in gates.items()
    ]
    rows.append(
        {
            "gate": "research_pass_pending_spa_and_audit",
            "passed": bool(all(gates.values())),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metrics, annual, paths = _evaluate_metrics(samples)
    normal_current, normal_target = paths["normal_synthetic"]
    events = _event_rows(normal_current, normal_target)
    leave_one_year = _leave_one_year_rows(
        normal_current,
        normal_target,
    )
    acceptance = _acceptance(
        metrics,
        annual,
        events,
        leave_one_year,
    )
    complete = _selected_row(
        metrics,
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
    )

    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    annual.to_csv(
        OUTPUT / "annual_relative_returns.csv",
        index=False,
    )
    events.to_csv(OUTPUT / "event_windows.csv", index=False)
    leave_one_year.to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    pd.DataFrame(
        {
            "current_25_net_return": normal_current,
            "target_60_net_return": normal_target,
        }
    ).to_csv(
        OUTPUT / "normal_synthetic_rollout_daily.csv",
        index_label="date",
    )
    pd.DataFrame(
        {
            "net_return": normal_target,
        }
    ).to_csv(
        OUTPUT / "normal_synthetic_target60_daily.csv",
        index_label="date",
    )
    pd.Series(
        {
            "candidate": "r38_rollout60_policy",
            "cumulative_trials": CUMULATIVE_TRIALS,
            "target_share": TARGET_SHARE,
            "target_cagr": complete["candidate_cagr"],
            "target_max_drawdown": (
                complete["candidate_max_drawdown"]
            ),
            "current_25_cagr": complete["baseline_cagr"],
            "current_25_max_drawdown": (
                complete["baseline_max_drawdown"]
            ),
            "research_pass_pending_spa_and_audit": bool(
                acceptance.loc[
                    acceptance["gate"].eq(
                        "research_pass_pending_spa_and_audit"
                    ),
                    "passed",
                ].iloc[0]
            ),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")

    print("Metrics:")
    print(metrics.round(6).to_string(index=False))
    print("\nEvents:")
    print(events.round(6).to_string(index=False))
    print("\nLeave one year out:")
    print(leave_one_year.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
