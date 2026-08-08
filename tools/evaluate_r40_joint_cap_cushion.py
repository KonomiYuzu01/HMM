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
from tools.evaluate_r40_normal_cap_frontier import simulate_candidate


OUTPUT = Path("output/r40_joint_cap_cushion")
NORMAL_CAPS = (1.20,)
FLOOR_DRAWDOWNS = (-0.17, -0.175, -0.18, -0.185, -0.19)
BEAR_MULTIPLIERS = (9.5, 12.0, 15.0, 20.0)


@dataclass(frozen=True)
class JointCandidate:
    name: str
    normal_non_cash_cap: float
    floor_drawdown: float
    bull_multiplier: float
    bear_multiplier: float
    tier_size: float


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def evaluate(
    sample: str,
    settings: dict[str, object],
    scenario: object,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: JointCandidate,
) -> dict[str, object]:
    trial, diagnostics = simulate_candidate(
        settings,
        scenario,
        components,
        candidate,
    )
    baseline = components["baseline"]
    assert isinstance(baseline, pd.DataFrame)
    period, bounds = complete_bounds(settings)
    return {
        "candidate": candidate.name,
        "sample": sample,
        "normal_non_cash_cap": candidate.normal_non_cash_cap,
        "floor_drawdown": candidate.floor_drawdown,
        "bear_multiplier": candidate.bear_multiplier,
        "tier_size": candidate.tier_size,
        "active_fraction": float(diagnostics["state"].ne("normal").mean()),
        **metric_row(
            sample,
            period,
            bounds,
            baseline["net_return"],
            trial["net_return"],
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    modern_settings = samples["normal_synthetic"]
    modern_components = base_components(modern_settings, scenario)
    candidates = tuple(
        JointCandidate(
            f"cap{int(cap * 100):03d}_floor{int(abs(floor) * 1000):03d}_bear{int(bear * 10):03d}",
            cap,
            floor,
            100.0,
            bear,
            0.05,
        )
        for cap in NORMAL_CAPS
        for floor in FLOOR_DRAWDOWNS
        for bear in BEAR_MULTIPLIERS
    )
    modern_rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"modern screening {candidate.name}", flush=True)
        modern_rows.append(
            evaluate(
                "normal_synthetic",
                modern_settings,
                scenario,
                modern_components,
                candidate,
            )
        )
    modern = pd.DataFrame(modern_rows)
    modern.to_csv(OUTPUT / "modern_metrics.csv", index=False)
    selected = set(
        modern.loc[
            modern["candidate_cagr"].between(0.24, 0.25)
            & modern["candidate_max_drawdown"].ge(-0.20),
            "candidate",
        ]
    )
    validation_rows: list[dict[str, object]] = []
    if selected:
        validation_settings = {
            "proxy_synthetic": samples["proxy_synthetic"],
        }
        pre_settings, _ = build_pre2008_settings()
        pre_settings = {
            **pre_settings,
            "periods": {
                "complete_1931_2007": ("1931-01-02", "2007-12-31")
            },
        }
        validation_settings["pre2008"] = pre_settings
        validation_components = {
            sample: base_components(settings, scenario)
            for sample, settings in validation_settings.items()
        }
        for candidate in candidates:
            if candidate.name not in selected:
                continue
            for sample, settings in validation_settings.items():
                print(f"{sample} replay {candidate.name}", flush=True)
                validation_rows.append(
                    evaluate(
                        sample,
                        settings,
                        scenario,
                        validation_components[sample],
                        candidate,
                    )
                )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUTPUT / "validation_metrics.csv", index=False)
    frontier = modern.copy()
    if validation.empty:
        frontier["proxy_cagr"] = float("nan")
        frontier["proxy_max_drawdown"] = float("nan")
        frontier["pre2008_max_drawdown"] = float("nan")
    else:
        proxy = validation.loc[validation["sample"].eq("proxy_synthetic")]
        pre = validation.loc[validation["sample"].eq("pre2008")]
        frontier = frontier.merge(
            proxy[["candidate", "candidate_cagr", "candidate_max_drawdown"]],
            on="candidate",
            how="left",
        ).rename(
            columns={
                "candidate_cagr_y": "proxy_cagr",
                "candidate_max_drawdown_y": "proxy_max_drawdown",
                "candidate_cagr_x": "candidate_cagr",
                "candidate_max_drawdown_x": "candidate_max_drawdown",
            }
        )
        frontier = frontier.merge(
            pre[["candidate", "candidate_max_drawdown"]],
            on="candidate",
            how="left",
        ).rename(columns={"candidate_max_drawdown_y": "pre2008_max_drawdown"})
        if "candidate_max_drawdown_x" in frontier:
            frontier = frontier.rename(
                columns={"candidate_max_drawdown_x": "candidate_max_drawdown"}
            )
    frontier["target_pass"] = (
        frontier["candidate"].isin(selected)
        & frontier["proxy_cagr"].ge(0.14)
        & frontier["proxy_max_drawdown"].ge(-0.20)
        & frontier["pre2008_max_drawdown"].ge(-0.20)
    )
    frontier.to_csv(OUTPUT / "frontier.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "modern_target_count": len(selected),
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
