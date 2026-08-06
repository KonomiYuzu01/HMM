from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r10_combined_tail_capital import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    OUTPUT as COMBINED_OUTPUT,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    metrics_for_period,
    selected_guard,
)
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


OUTPUT = Path("output/r10_guard_cap_neighborhood")
CAPS = (0.15, 0.20, 0.25, 0.30)
SUBSTITUTION_FRACTION = 0.25


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
        "primary": (20.0, 30.0),
        "cost_stress": (50.0, 75.0),
    }
    metric_rows: list[dict[str, float | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
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
        common_baseline = baseline.index.intersection(
            baseline_guard.index
        )
        maximum_reconstruction_error = float(
            (
                baseline.loc[common_baseline, "net_return"]
                - baseline_guard.loc[common_baseline, "net_return"]
            )
            .abs()
            .max()
        )
        for scenario, (
            emergency_slippage_bps,
            gde_cost_bps,
        ) in scenarios.items():
            for cap in CAPS:
                guard_daily, guard_weights = simulate_guard(
                    weights,
                    daily,
                    opens,
                    closes,
                    selected_guard(
                        emergency_slippage_bps,
                        post_trigger_cap=cap,
                    ),
                    start_date=start,
                )
                candidate = simulate_gde_substitution(
                    str(settings["directory"]),
                    opens,
                    closes,
                    substitution_fraction=SUBSTITUTION_FRACTION,
                    gate_mode="growth_linked",
                    gde_return_mode=mode,
                    start_date=start,
                    end_date=None,
                    gde_one_way_cost_bps=gde_cost_bps,
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
                            "post_trigger_cap": cap,
                            "period": period,
                            "emergency_slippage_bps": (
                                emergency_slippage_bps
                            ),
                            "gde_one_way_cost_bps": gde_cost_bps,
                            "baseline_reconstruction_error": (
                                maximum_reconstruction_error
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
                if scenario == "primary":
                    cap_label = int(round(cap * 100))
                    candidate.to_csv(
                        OUTPUT
                        / f"{sample}_cap{cap_label}_daily.csv",
                        index_label="date",
                    )
                    for block_days in (21, 63, 126):
                        bootstrap_rows.append(
                            {
                                "sample": sample,
                                "post_trigger_cap": cap,
                                **paired_block_bootstrap(
                                    baseline.loc[common, "net_return"],
                                    candidate.loc[common, "net_return"],
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
    columns = [
        "sample",
        "scenario",
        "post_trigger_cap",
        "period",
        "cagr_delta",
        "sharpe_delta",
        "candidate_max_drawdown",
        "guard_triggers",
    ]
    print(metrics[columns].round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
