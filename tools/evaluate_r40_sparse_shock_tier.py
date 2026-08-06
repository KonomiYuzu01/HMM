from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r40_sparse_shock_strength import simulate_strength_candidate


OUTPUT = Path("output/r40_sparse_shock_tier")
TIER_SIZES = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15)


@dataclass(frozen=True)
class SparseShockTierCandidate:
    name: str
    tier_size: float
    bear_multiplier: float = 11.0
    calm_multiplier: float = 30.0
    one_day_stress_threshold: float = -0.05
    five_day_stress_threshold: float = -0.09
    normal_non_cash_cap: float = 1.22
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    settings_by_sample = {
        sample: samples[sample]
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        SparseShockTierCandidate(
            f"tier_{str(tier).replace('.', '_')}", tier
        )
        for tier in TIER_SIZES
    )
    rows: list[dict[str, object]] = []
    paths: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_strength_candidate(
                settings,
                scenario,
                components_by_sample[sample],
                candidate,
            )
            baseline = components_by_sample[sample]["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            period, bounds = complete_bounds(settings)
            rows.append(
                {
                    "candidate": candidate.name,
                    "tier_size": candidate.tier_size,
                    "sample": sample,
                    "controlled_fraction": float(
                        diagnostics["state"].ne("normal").mean()
                    ),
                    **metric_row(
                        sample,
                        period,
                        bounds,
                        baseline["net_return"],
                        trial["net_return"],
                    ),
                }
            )
            if sample == "normal_synthetic":
                paths[candidate.name] = trial
    screen = pd.DataFrame(rows)
    screen.to_csv(OUTPUT / "screen_metrics.csv", index=False)
    modern = screen.loc[screen["sample"].eq("normal_synthetic")]
    selected = set(
        modern.loc[
            modern["candidate_cagr"].between(0.24, 0.25, inclusive="both"),
            "candidate",
        ]
    )

    pre_rows: list[dict[str, object]] = []
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_components = base_components(pre_settings, scenario)
        for candidate in candidates:
            if candidate.name not in selected:
                continue
            print(f"crash replay {candidate.name}", flush=True)
            trial, diagnostics = simulate_strength_candidate(
                pre_settings, scenario, pre_components, candidate
            )
            baseline = pre_components["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            pre_rows.append(
                {
                    "candidate": candidate.name,
                    "tier_size": candidate.tier_size,
                    "controlled_fraction": float(
                        diagnostics["state"].ne("normal").mean()
                    ),
                    **metric_row(
                        "pre2008",
                        "complete_1931_2007",
                        ("1931-01-02", "2007-12-31"),
                        baseline["net_return"],
                        trial["net_return"],
                    ),
                }
            )
            trial.to_csv(OUTPUT / f"pre2008_{candidate.name}_daily.csv")
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(OUTPUT / "pre2008_metrics.csv", index=False)
    proxy = screen.loc[screen["sample"].eq("proxy_synthetic")]
    frontier = modern.merge(
        proxy[[
            "candidate",
            "candidate_cagr",
            "cagr_delta",
            "candidate_max_drawdown",
        ]],
        on="candidate",
        suffixes=("_modern", "_proxy"),
    )
    if pre.empty:
        frontier["pre2008_max_drawdown"] = float("nan")
    else:
        frontier = frontier.merge(
            pre[["candidate", "candidate_max_drawdown"]],
            on="candidate",
            how="left",
        ).rename(columns={"candidate_max_drawdown": "pre2008_max_drawdown"})
    frontier["target_pass"] = (
        frontier["candidate_cagr_modern"].between(0.24, 0.25, inclusive="both")
        & frontier["pre2008_max_drawdown"].ge(-0.20)
    )
    frontier.to_csv(OUTPUT / "frontier.csv", index=False)
    for candidate in frontier.loc[frontier["target_pass"], "candidate"]:
        paths[candidate].to_csv(OUTPUT / f"normal_synthetic_{candidate}_daily.csv")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "pre2008_replay_count": len(pre),
        "target_candidates": frontier.loc[
            frontier["target_pass"], "candidate"
        ].tolist(),
        "production_changed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(frontier.round(6).to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
