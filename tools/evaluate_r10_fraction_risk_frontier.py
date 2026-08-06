from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r10_combined_tail_capital import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    metrics_for_period,
    selected_guard,
)
from tools.evaluate_r10_gde_capital_efficiency import (
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r10_gde_tracking_uncertainty import (
    ADJUSTED_CLOSE,
    bootstrap_tracking_uncertainty,
    gde_implementation_residual,
)
from tools.evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    load_inputs as load_normal_guard_inputs,
    simulate as simulate_guard,
)
from tools.evaluate_smh_long_proxy_validation import (
    load_inputs as load_proxy_guard_inputs,
)


OUTPUT = Path("output/r10_fraction_risk_frontier")
FRACTIONS = (0.25, 0.33, 0.40, 0.50)
GUARD_CAP = 0.15
RELATIVE_GAP_TRIGGER = -0.005
MINIMUM_OBJECTIVE_PROBABILITY = 0.95


def select_largest_feasible(
    tracking: pd.DataFrame,
    *,
    minimum_probability: float = MINIMUM_OBJECTIVE_PROBABILITY,
) -> float:
    feasible = tracking.loc[
        (tracking["probability_all_objectives"] >= minimum_probability)
        & (tracking["cagr_delta_p05"] > 0.0)
        & (tracking["max_drawdown_p05"] >= -0.18)
    ]
    if feasible.empty:
        raise RuntimeError("No fraction satisfies the declared risk budget")
    return float(feasible["substitution_fraction"].max())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily, _, _ = load_normal_guard_inputs()
    proxy_weights, proxy_daily, _, _ = load_proxy_guard_inputs(
        Path("output") / PROXY_DIRECTORY
    )
    samples = {
        "normal_synthetic": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": (
                    "2022-01-01",
                    "2025-12-31",
                ),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": (
                    "2015-01-01",
                    "2026-12-31",
                ),
            },
        },
        "proxy_synthetic": {
            "directory": PROXY_DIRECTORY,
            "weights": proxy_weights,
            "daily": proxy_daily,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "mode": "synthetic",
            "start": "2006-08-01",
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": (
                    "2006-08-01",
                    "2026-12-31",
                ),
            },
        },
        "normal_live": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": live_opens,
            "closes": live_closes,
            "mode": "live",
            "start": "2022-03-17",
            "periods": {
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
            },
        },
    }
    scenarios = {
        "current_liquidity": {
            "emergency_slippage_bps": 20.0,
            "gde_cost_bps": 40.0,
        },
        "cost_stress": {
            "emergency_slippage_bps": 50.0,
            "gde_cost_bps": 75.0,
        },
    }
    metric_rows: list[dict[str, float | int | str]] = []
    candidates: dict[tuple[str, float], pd.DataFrame] = {}
    baselines: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        start = str(settings["start"])
        periods = settings["periods"]
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        baseline_guard, _ = simulate_guard(
            weights,
            daily,
            opens,
            closes,
            OpenGapGuard("baseline"),
            start_date=start,
        )
        baseline = simulate_gde_substitution(
            str(settings["directory"]),
            opens,
            closes,
            substitution_fraction=0.0,
            gate_mode="always",
            gde_return_mode=str(settings["mode"]),
            start_date=start,
            end_date=None,
            gde_one_way_cost_bps=40.0,
        )
        common_baseline = baseline.index.intersection(
            baseline_guard.index
        )
        reconstruction_error = float(
            (
                baseline.loc[common_baseline, "net_return"]
                - baseline_guard.loc[common_baseline, "net_return"]
            )
            .abs()
            .max()
        )
        baselines[sample] = baseline
        for scenario, assumptions in scenarios.items():
            guard_daily, guard_weights = simulate_guard(
                weights,
                daily,
                opens,
                closes,
                selected_guard(
                    float(assumptions["emergency_slippage_bps"]),
                    post_trigger_cap=GUARD_CAP,
                    relative_gap_trigger=RELATIVE_GAP_TRIGGER,
                ),
                start_date=start,
            )
            for fraction in FRACTIONS:
                candidate = simulate_gde_substitution(
                    str(settings["directory"]),
                    opens,
                    closes,
                    substitution_fraction=fraction,
                    gate_mode="growth_linked",
                    gde_return_mode=str(settings["mode"]),
                    start_date=start,
                    end_date=None,
                    gde_one_way_cost_bps=float(
                        assumptions["gde_cost_bps"]
                    ),
                    weights_override=guard_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                )
                common = baseline.index.intersection(candidate.index)
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "substitution_fraction": fraction,
                            "period": period,
                            "baseline_reconstruction_error": (
                                reconstruction_error
                            ),
                            **metrics_for_period(
                                baseline.loc[selected, "net_return"],
                                candidate.loc[selected, "net_return"],
                            ),
                            "guard_triggers": int(
                                guard_daily.loc[
                                    selected,
                                    "triggered",
                                ].sum()
                            ),
                        }
                    )
                if scenario == "current_liquidity":
                    candidates[(sample, fraction)] = candidate
                    label = int(round(fraction * 100))
                    candidate.to_csv(
                        OUTPUT
                        / f"{sample}_fraction{label}_daily.csv",
                        index_label="date",
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    residual = gde_implementation_residual(prices)
    proxy_baseline = baselines["proxy_synthetic"]["net_return"]
    tracking_rows: list[dict[str, float | int | str]] = []
    for fraction in FRACTIONS:
        candidate_key = ("proxy_synthetic", fraction)
        if candidate_key not in candidates:
            raise RuntimeError(
                f"Missing current-liquidity path: {candidate_key}"
            )
        tracking_rows.append(
            {
                "substitution_fraction": fraction,
                **bootstrap_tracking_uncertainty(
                    proxy_baseline,
                    candidates[candidate_key],
                    residual,
                ),
            }
        )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(
        OUTPUT / "tracking_risk_frontier.csv",
        index=False,
    )
    selected_fraction = select_largest_feasible(tracking)
    pd.Series(
        {
            "selection_rule": (
                "largest_fraction_with_at_least_95pct_probability_"
                "of_positive_cagr_nonworse_sharpe_and_mdd_within_18pct"
            ),
            "selected_fraction": selected_fraction,
            "minimum_objective_probability": (
                MINIMUM_OBJECTIVE_PROBABILITY
            ),
            "guard_cap": GUARD_CAP,
            "relative_gap_trigger": RELATIVE_GAP_TRIGGER,
            "gde_one_way_cost_bps": 40.0,
        },
        name="value",
    ).to_csv(OUTPUT / "selection.csv")
    print("Metrics:")
    print(
        metrics[
            [
                "sample",
                "scenario",
                "substitution_fraction",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nTracking-risk frontier:")
    print(tracking.round(6).to_string(index=False))
    print(f"\nSelected fraction: {selected_fraction:.0%}")
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
