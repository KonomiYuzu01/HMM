from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_recursive_cushion_budget import (
    CushionCandidate,
    simulate_cushion_candidate,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/recursive_trend_cushion_frontier")
TARGET_CEILINGS = tuple(value / 1000 for value in range(160, 201, 5))

# The already-computed 14.5%, 15.0%, 18.5%, 18.75%, and 19.0% floor points
# are merged into the final analysis separately. This grid fills the interior
# and tests alternative discretizations near the 20% maximum-drawdown boundary.
CANDIDATES = (
    *(
        CushionCandidate(
            f"floor_{int(round(abs(floor) * 1000)):03d}",
            floor_drawdown=floor,
            bull_multiplier=100.0,
            bear_multiplier=9.5,
            tier_size=0.05,
        )
        for floor in (-0.155, -0.160, -0.165, -0.170, -0.175, -0.180)
    ),
    CushionCandidate("boundary_190_b950_t060", -0.190, 100.0, 9.50, 0.06),
    CushionCandidate("boundary_1925_b925_t060", -0.1925, 100.0, 9.25, 0.06),
    CushionCandidate("boundary_195_b900_t060", -0.195, 100.0, 9.00, 0.06),
    CushionCandidate("boundary_1975_b875_t060", -0.1975, 100.0, 8.75, 0.06),
    CushionCandidate("boundary_200_b850_t060", -0.200, 100.0, 8.50, 0.06),
    CushionCandidate("boundary_1925_b925_t050", -0.1925, 100.0, 9.25, 0.05),
    CushionCandidate("boundary_195_b900_t050", -0.195, 100.0, 9.00, 0.05),
    CushionCandidate("floor_190_b950_t050", -0.190, 100.0, 9.50, 0.05),
)
PRIOR_POINTS = (
    (
        "output/recursive_trend_cushion_16pct",
        "central",
        "prior_floor_1475_b950_t050",
        -0.1475,
        9.50,
        0.05,
    ),
    (
        "output/recursive_trend_cushion_16pct",
        "floor_145",
        "prior_floor_145_b950_t050",
        -0.145,
        9.50,
        0.05,
    ),
    (
        "output/recursive_trend_cushion_16pct",
        "floor_150",
        "prior_floor_150_b950_t050",
        -0.150,
        9.50,
        0.05,
    ),
)


def complete_period(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    names = [name for name in periods if name.startswith("complete_")]
    if not names:
        names = [name for name in periods if name.startswith("live_")]
    if len(names) != 1:
        raise ValueError("Expected exactly one complete or live period")
    name = names[0]
    return name, periods[name]


def select_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="candidate", columns="sample")
    rows: list[dict[str, object]] = []
    for ceiling in TARGET_CEILINGS:
        feasible = wide.loc[
            wide[("candidate_max_drawdown", "pre2008")].ge(-ceiling)
        ].copy()
        if feasible.empty:
            rows.append(
                {
                    "drawdown_ceiling": ceiling,
                    "candidate": None,
                    "feasible": False,
                }
            )
            continue
        candidate = feasible[("candidate_cagr", "normal_synthetic")].idxmax()
        row = feasible.loc[candidate]
        rows.append(
            {
                "drawdown_ceiling": ceiling,
                "candidate": candidate,
                "feasible": True,
                "pre2008_max_drawdown": row[
                    ("candidate_max_drawdown", "pre2008")
                ],
                "modern_cagr": row[("candidate_cagr", "normal_synthetic")],
                "modern_cagr_delta": row[("cagr_delta", "normal_synthetic")],
                "proxy_cagr": row[("candidate_cagr", "proxy_synthetic")],
                "proxy_cagr_delta": row[("cagr_delta", "proxy_synthetic")],
            }
        )
    return pd.DataFrame(rows)


def load_prior_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for directory, source_name, name, floor, bear, tier in PRIOR_POINTS:
        path = Path(directory)
        reported = pd.read_csv(path / "metrics.csv")
        complete = reported.loc[
            reported["sample"].eq("pre2008")
            | reported["period"].str.startswith("complete_")
        ].copy()
        if source_name == "central":
            selected = complete
        else:
            neighbors = pd.read_csv(path / "parameter_neighborhood.csv")
            selected = neighbors.loc[neighbors["candidate"].eq(source_name)].copy()
            baselines = complete.set_index("sample")["baseline_cagr"]
            selected["candidate_cagr"] = selected.apply(
                lambda row: baselines.loc[row["sample"]] + row["cagr_delta"],
                axis=1,
            )
        for _, row in selected.iterrows():
            rows.append(
                {
                    "candidate": name,
                    "floor_drawdown": floor,
                    "bear_multiplier": bear,
                    "tier_size": tier,
                    "sample": row["sample"],
                    "candidate_cagr": row["candidate_cagr"],
                    "cagr_delta": row["cagr_delta"],
                    "candidate_max_drawdown": row["candidate_max_drawdown"],
                    "source": str(path),
                }
            )
    return pd.DataFrame(rows)


def assemble_results(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    current = metrics.loc[metrics["sample"].ne("normal_live")].copy()
    current["source"] = str(OUTPUT)
    combined = pd.concat([current, load_prior_points()], ignore_index=True)
    frontier = select_frontier(combined)
    qualified = frontier.loc[
        frontier["feasible"].astype(bool)
        & frontier["modern_cagr_delta"].ge(-0.02)
        & frontier["proxy_cagr_delta"].ge(-0.02)
    ]
    best_name = (
        qualified.sort_values("modern_cagr", ascending=False).iloc[0]["candidate"]
        if not qualified.empty
        else None
    )
    combined.to_csv(OUTPUT / "combined_metrics.csv", index=False)
    frontier.to_csv(OUTPUT / "combined_frontier.csv", index=False)
    return combined, frontier, best_name


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    pre_settings, _ = build_pre2008_settings()
    samples = {
        "pre2008": pre_settings,
        **{
            key: value
            for key, value in r21._build_samples().items()
            if key in {"normal_synthetic", "proxy_synthetic", "normal_live"}
        },
    }
    components = {
        sample: base_components(settings, scenario)
        for sample, settings in samples.items()
    }
    metric_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        print(f"evaluating {candidate.name}", flush=True)
        for sample, settings in samples.items():
            trial, diagnostics = simulate_cushion_candidate(
                settings,
                scenario,
                components[sample],
                candidate,
            )
            reference = components[sample]["baseline"]
            assert isinstance(reference, pd.DataFrame)
            if sample == "pre2008":
                period = "complete_1931_2007"
                bounds = ("1931-01-02", "2007-12-31")
            else:
                period, bounds = complete_period(settings)
            row = metric_row(
                sample,
                period,
                bounds,
                reference["net_return"],
                trial["net_return"],
            )
            metric_rows.append(
                {
                    "candidate": candidate.name,
                    "floor_drawdown": candidate.floor_drawdown,
                    "bear_multiplier": candidate.bear_multiplier,
                    "tier_size": candidate.tier_size,
                    "active_fraction": float(
                        diagnostics["state"].ne("normal").mean()
                    ),
                    **row,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    _, frontier, best_name = assemble_results(metrics)
    if best_name is not None:
        best_candidate = next(
            candidate for candidate in CANDIDATES if candidate.name == best_name
        )
        for sample, settings in samples.items():
            trial, diagnostics = simulate_cushion_candidate(
                settings,
                scenario,
                components[sample],
                best_candidate,
            )
            trial.to_csv(
                OUTPUT / f"{sample}_best_daily.csv", index_label="date"
            )
            diagnostics.to_csv(
                OUTPUT / f"{sample}_best_diagnostics.csv", index_label="date"
            )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_count": len(CANDIDATES),
        "drawdown_range": [0.16, 0.20],
        "ranking_objective": "maximum modern CAGR after drawdown feasibility",
        "qualification_requires": {
            "modern_cagr_delta_at_least": -0.02,
            "proxy_cagr_delta_at_least": -0.02,
        },
        "best_qualified_candidate": best_name,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(frontier.to_string(index=False))


if __name__ == "__main__":
    if "--assemble-only" in sys.argv:
        existing = pd.read_csv(OUTPUT / "metrics.csv")
        _, report, winner = assemble_results(existing)
        print(json.dumps({"best_qualified_candidate": winner}, indent=2))
        print(report.to_string(index=False))
    else:
        main()
