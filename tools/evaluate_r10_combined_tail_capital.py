from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    join_live_gde,
    load_adjusted_open_close,
    paired_block_bootstrap,
    simulate_gde_substitution,
)
from tools.evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    load_inputs as load_normal_guard_inputs,
    simulate as simulate_guard,
)
from tools.evaluate_smh_long_proxy_validation import (
    load_inputs as load_proxy_guard_inputs,
)


OUTPUT = Path("output/r10_combined_tail_capital")
SUBSTITUTION_FRACTIONS = (0.25, 0.50, 0.75)
NORMAL_DIRECTORY = (
    "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
PROXY_DIRECTORY = "experiment_r9_broad50_stage35_d10_20y_proxy"
NORMAL_OPEN_CLOSE = Path("data/adjusted_open_close_2011_present.csv")
PROXY_OPEN_CLOSE = Path("data/adjusted_open_close_20y_proxy.csv")


def selected_guard(
    emergency_slippage_bps: float,
    post_trigger_cap: float = 0.25,
    relative_gap_trigger: float | None = None,
    absolute_gap_trigger: float = -0.025,
    minimum_semis_weight: float = 0.60,
) -> OpenGapGuard:
    cap_label = int(round(post_trigger_cap * 100))
    gap_label = int(round(abs(absolute_gap_trigger) * 10_000))
    minimum_label = int(round(minimum_semis_weight * 100))
    relative_label = (
        ""
        if relative_gap_trigger is None
        else (
            f"_rel{int(round(abs(relative_gap_trigger) * 10_000))}bp"
        )
    )
    return OpenGapGuard(
        name=(
            f"causal_gap{gap_label}bp_min{minimum_label}"
            f"_cap{cap_label}_confirm2_hold4"
            f"{relative_label}"
            f"_slip{int(emergency_slippage_bps)}"
        ),
        absolute_gap_trigger=absolute_gap_trigger,
        relative_gap_trigger=relative_gap_trigger,
        post_trigger_cap=post_trigger_cap,
        trigger_delay_days=1,
        minimum_semis_weight=minimum_semis_weight,
        recovery_signal="relative",
        recovery_confirmations=2,
        maximum_hold_days=4,
        overflow_asset="CASH",
        emergency_slippage_bps=emergency_slippage_bps,
    )


def metrics_for_period(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, float]:
    base = performance_metrics(baseline)
    combined = performance_metrics(candidate)
    return {
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"candidate_{key}": value for key, value in combined.items()},
        "cagr_delta": combined["cagr"] - base["cagr"],
        "sharpe_delta": combined["sharpe"] - base["sharpe"],
        "max_drawdown_delta": (
            combined["max_drawdown"] - base["max_drawdown"]
        ),
    }


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
                "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": ("2015-01-01", "2026-12-31"),
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
                "complete_2006_2026": ("2006-08-01", "2026-12-31"),
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
        "primary": {
            "emergency_slippage_bps": 20.0,
            "gde_cost_bps": 30.0,
        },
        "cost_stress": {
            "emergency_slippage_bps": 50.0,
            "gde_cost_bps": 75.0,
        },
    }
    metric_rows: list[dict[str, float | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    reconstruction_rows: list[dict[str, float | str]] = []
    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        start = str(settings["start"])
        mode = str(settings["mode"])
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
            gde_return_mode=mode,
            start_date=start,
            end_date=None,
            gde_one_way_cost_bps=30.0,
        )
        common_baseline = baseline.index.intersection(baseline_guard.index)
        reconstruction_rows.append(
            {
                "sample": sample,
                "maximum_daily_return_error": float(
                    (
                        baseline.loc[common_baseline, "net_return"]
                        - baseline_guard.loc[
                            common_baseline,
                            "net_return",
                        ]
                    ).abs().max()
                ),
            }
        )
        for scenario, assumptions in scenarios.items():
            guard_daily, guard_weights = simulate_guard(
                weights,
                daily,
                opens,
                closes,
                selected_guard(
                    float(assumptions["emergency_slippage_bps"])
                ),
                start_date=start,
            )
            for substitution_fraction in SUBSTITUTION_FRACTIONS:
                combined = simulate_gde_substitution(
                    str(settings["directory"]),
                    opens,
                    closes,
                    substitution_fraction=substitution_fraction,
                    gate_mode="growth_linked",
                    gde_return_mode=mode,
                    start_date=start,
                    end_date=None,
                    gde_one_way_cost_bps=float(
                        assumptions["gde_cost_bps"]
                    ),
                    weights_override=guard_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                )
                common = baseline.index.intersection(combined.index)
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start) & (common <= period_end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "substitution_fraction": substitution_fraction,
                            "period": period,
                            "emergency_slippage_bps": assumptions[
                                "emergency_slippage_bps"
                            ],
                            "gde_one_way_cost_bps": assumptions[
                                "gde_cost_bps"
                            ],
                            **metrics_for_period(
                                baseline.loc[selected, "net_return"],
                                combined.loc[selected, "net_return"],
                            ),
                            "guard_triggers": int(
                                guard_daily.loc[selected, "triggered"].sum()
                            ),
                        }
                    )
                if scenario == "primary":
                    fraction_label = int(
                        round(substitution_fraction * 100)
                    )
                    baseline.to_csv(
                        OUTPUT / f"{sample}_baseline_daily.csv",
                        index_label="date",
                    )
                    combined.to_csv(
                        OUTPUT
                        / (
                            f"{sample}_primary_fraction"
                            f"{fraction_label}_daily.csv"
                        ),
                        index_label="date",
                    )
                    if substitution_fraction == 0.50:
                        combined.to_csv(
                            OUTPUT / f"{sample}_primary_daily.csv",
                            index_label="date",
                        )
                    for block_days in (21, 63, 126):
                        bootstrap_rows.append(
                            {
                                "sample": sample,
                                "substitution_fraction": (
                                    substitution_fraction
                                ),
                                **paired_block_bootstrap(
                                    baseline.loc[common, "net_return"],
                                    combined.loc[common, "net_return"],
                                    block_days=block_days,
                                ),
                            }
                        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "paired_bootstrap.csv",
        index=False,
    )
    reconstruction = pd.DataFrame(reconstruction_rows)
    reconstruction.to_csv(
        OUTPUT / "baseline_reconstruction.csv",
        index=False,
    )
    print(
        metrics[
            [
                "sample",
                "scenario",
                "substitution_fraction",
                "period",
                "baseline_cagr",
                "candidate_cagr",
                "cagr_delta",
                "baseline_sharpe",
                "candidate_sharpe",
                "candidate_max_drawdown",
                "guard_triggers",
            ]
        ].to_string(index=False)
    )
    print("\nBaseline reconstruction:")
    print(reconstruction.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
