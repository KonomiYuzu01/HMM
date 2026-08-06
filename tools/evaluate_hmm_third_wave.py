from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_hmm_order_ensemble import (
    DEFINITIONS,
    doubled_trading_cost,
    state_switch_rows,
    with_strategy_inputs,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_third_wave_validation")
CANDIDATES = {
    "macro_only": {
        "normal": Path("output/research_hmm_macro_only"),
        "proxy": Path("output/research_hmm_macro_only_20y_proxy"),
    },
    "slow_features": {
        "normal": Path("output/research_hmm_slow_features"),
        "proxy": Path("output/research_hmm_slow_features_20y_proxy"),
    },
    "refit63": {
        "normal": Path("output/research_hmm_refit63"),
        "proxy": Path("output/research_hmm_refit63_20y_proxy"),
    },
    "clip005": {
        "normal": Path("output/research_hmm_clip005"),
        "proxy": Path("output/research_hmm_clip005_20y_proxy"),
    },
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    switch_frames: list[pd.DataFrame] = []
    for sample, definition in DEFINITIONS.items():
        sample_name = str(definition["sample"])
        baseline_directory = definition["baseline"]
        periods = definition["periods"]
        assert isinstance(baseline_directory, Path)
        assert isinstance(periods, dict)
        baseline_settings = with_strategy_inputs(
            samples[sample_name], baseline_directory
        )
        baseline_paths: dict[str, pd.DataFrame] = {}
        for scenario in COST_SCENARIOS:
            baseline_paths[scenario.name], _ = r39.simulate_candidate(
                baseline_settings, scenario
            )
        for candidate, directories in CANDIDATES.items():
            candidate_directory = directories[sample]
            candidate_settings = with_strategy_inputs(
                samples[sample_name], candidate_directory
            )
            for scenario in COST_SCENARIOS:
                baseline = baseline_paths[scenario.name]
                trial, _ = r39.simulate_candidate(candidate_settings, scenario)
                common = baseline.index.intersection(trial.index)
                for period, (start, end) in periods.items():
                    selected = common[(common >= start) & (common <= end)]
                    for cost_view, transform in (
                        ("reported", lambda frame: frame["net_return"]),
                        ("double_trading_cost", doubled_trading_cost),
                    ):
                        metric_rows.append(
                            {
                                "candidate": candidate,
                                "sample": sample,
                                "scenario": scenario.name,
                                "cost_view": cost_view,
                                "period": period,
                                **metric_delta(
                                    transform(baseline.loc[selected]),
                                    transform(trial.loc[selected]),
                                ),
                            }
                        )
            switch_frame = pd.DataFrame(
                state_switch_rows(
                    sample, baseline_directory, candidate_directory
                )
            )
            switch_frame.insert(0, "candidate", candidate)
            switch_frames.append(switch_frame)

    metrics = pd.DataFrame(metric_rows)
    switches = pd.concat(switch_frames, ignore_index=True)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    switches.to_csv(OUTPUT / "state_switches.csv", index=False)

    def delta(candidate: str, sample: str, period: str, field: str, *, cost_view: str = "reported") -> float:
        return float(
            metrics.loc[
                metrics["candidate"].eq(candidate)
                & metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["scenario"].eq("current_liquidity")
                & metrics["cost_view"].eq(cost_view),
                field,
            ].iloc[0]
        )

    candidate_gates: dict[str, dict[str, bool]] = {}
    for candidate in CANDIDATES:
        selected_switches = switches.loc[switches["candidate"].eq(candidate)]
        candidate_gates[candidate] = {
            "development_cagr_nonnegative": delta(
                candidate, "normal", "development_2015_2021", "cagr_delta"
            )
            >= 0.0,
            "holdout_cagr_nonnegative": delta(
                candidate, "normal", "holdout_2022_2025", "cagr_delta"
            )
            >= 0.0,
            "normal_drawdown_within_50bp": delta(
                candidate, "normal", "complete_2015_2025", "max_drawdown_delta"
            )
            >= -0.005,
            "proxy_early_cagr_nonnegative": delta(
                candidate, "proxy", "early_2006_2014", "cagr_delta"
            )
            >= 0.0,
            "proxy_late_cagr_nonnegative": delta(
                candidate, "proxy", "late_2015_2025", "cagr_delta"
            )
            >= 0.0,
            "proxy_complete_cagr_nonnegative": delta(
                candidate, "proxy", "complete_2006_2025", "cagr_delta"
            )
            >= 0.0,
            "double_cost_complete_cagr_nonnegative": delta(
                candidate,
                "normal",
                "complete_2015_2025",
                "cagr_delta",
                cost_view="double_trading_cost",
            )
            >= 0.0,
            "state_switches_not_increased": bool(
                selected_switches["candidate_switches"].sum()
                <= selected_switches["selected_order_switches"].sum()
            ),
        }
    passes = {
        candidate: bool(all(gates.values()))
        for candidate, gates in candidate_gates.items()
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "candidate_gates": candidate_gates,
        "research_pass": passes,
        "decision": {
            candidate: (
                "continue_robustness_audit"
                if passed
                else "reject_current_specification"
            )
            for candidate, passed in passes.items()
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
