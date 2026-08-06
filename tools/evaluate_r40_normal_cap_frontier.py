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
import tools.evaluate_recursive_cushion_budget as cushion


OUTPUT = Path("output/r40_normal_cap_frontier")
NORMAL_CAPS = (1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20)
MODERN_PREFILTER_CAGR = 0.235


@dataclass(frozen=True)
class NormalCapCandidate:
    name: str
    normal_non_cash_cap: float
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def effective_non_cash_cap(requested_cap: float, normal_cap: float) -> float:
    if not 0.0 <= requested_cap <= 1.0:
        raise ValueError("requested_cap must be in [0, 1]")
    if not 1.0 <= normal_cap <= 1.20:
        raise ValueError("normal_cap must be in [1, 1.20]")
    return normal_cap if requested_cap >= 1.0 - 1e-12 else requested_cap


def simulate_candidate(settings, scenario, components, candidate):
    original = cushion.cap_total_non_cash

    def cap_with_normal_extension(target: pd.Series, cap: float) -> pd.Series:
        effective = effective_non_cash_cap(cap, candidate.normal_non_cash_cap)
        if effective <= 1.0:
            return original(target, effective)
        adjusted = target.astype(float).copy()
        non_cash = [asset for asset in cushion.ASSETS if asset != "CASH"]
        total = float(adjusted.loc[non_cash].sum())
        if total > effective + 1e-12:
            adjusted.loc[non_cash] *= effective / total
            adjusted["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
        return adjusted

    cushion.cap_total_non_cash = cap_with_normal_extension
    try:
        return cushion.simulate_cushion_candidate(
            settings, scenario, components, candidate
        )
    finally:
        cushion.cap_total_non_cash = original


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    screen_settings = {
        sample: samples[sample]
        for sample in ("normal_synthetic", "proxy_synthetic", "normal_live")
    }
    screen_components = {
        sample: base_components(settings, scenario)
        for sample, settings in screen_settings.items()
    }
    candidates = tuple(
        NormalCapCandidate(f"normal_cap_{int(cap * 100):03d}", cap)
        for cap in NORMAL_CAPS
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in screen_settings.items():
            trial, diagnostics = simulate_candidate(
                settings, scenario, screen_components[sample], candidate
            )
            baseline = screen_components[sample]["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            if sample == "normal_live":
                period, bounds = "live_2022_2026", settings["periods"]["live_2022_2026"]
            else:
                period, bounds = complete_bounds(settings)
            rows.append(
                {
                    "candidate": candidate.name,
                    "normal_non_cash_cap": candidate.normal_non_cash_cap,
                    "sample": sample,
                    "active_fraction": float(diagnostics["state"].ne("normal").mean()),
                    **metric_row(
                        sample,
                        period,
                        bounds,
                        baseline["net_return"],
                        trial["net_return"],
                    ),
                }
            )
    screen = pd.DataFrame(rows)
    screen.to_csv(OUTPUT / "screen_metrics.csv", index=False)
    modern = screen.loc[screen["sample"].eq("normal_synthetic")]
    selected_names = set(
        modern.loc[modern["candidate_cagr"].ge(MODERN_PREFILTER_CAGR), "candidate"]
    )
    selected_names.add("normal_cap_100")

    pre_settings, _ = build_pre2008_settings()
    pre_components = base_components(pre_settings, scenario)
    pre_rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.name not in selected_names:
            continue
        print(f"crash replay {candidate.name}", flush=True)
        trial, diagnostics = simulate_candidate(
            pre_settings, scenario, pre_components, candidate
        )
        baseline = pre_components["baseline"]
        assert isinstance(baseline, pd.DataFrame)
        pre_rows.append(
            {
                "candidate": candidate.name,
                "normal_non_cash_cap": candidate.normal_non_cash_cap,
                "active_fraction": float(diagnostics["state"].ne("normal").mean()),
                **metric_row(
                    "pre2008",
                    "complete_1931_2007",
                    ("1931-01-02", "2007-12-31"),
                    baseline["net_return"],
                    trial["net_return"],
                ),
            }
        )
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(OUTPUT / "pre2008_metrics.csv", index=False)
    combined = modern.merge(
        screen.loc[screen["sample"].eq("proxy_synthetic"), [
            "candidate", "candidate_cagr", "cagr_delta", "candidate_max_drawdown"
        ]],
        on="candidate",
        suffixes=("_modern", "_proxy"),
    ).merge(
        pre[["candidate", "candidate_max_drawdown"]],
        on="candidate",
        how="left",
    ).rename(columns={"candidate_max_drawdown": "pre2008_max_drawdown"})
    combined["target_pass"] = (
        combined["candidate_cagr_modern"].between(0.24, 0.25, inclusive="both")
        & combined["pre2008_max_drawdown"].ge(-0.20)
    )
    combined.to_csv(OUTPUT / "frontier.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "pre2008_replay_count": len(pre),
        "target_candidates": combined.loc[combined["target_pass"], "candidate"].tolist(),
        "production_changed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(combined.round(6).to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
