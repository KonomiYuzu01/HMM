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


OUTPUT = Path("output/r40_risk_weighted_cap")
NORMAL_CAPS = (1.00, 1.10, 1.20)
DEFENSIVE_RISK_COEFFICIENTS = (0.00, 0.25, 0.50, 0.75, 1.00)
MODERN_PREFILTER_CAGR = 0.23
GROWTH_ASSETS = ("SPX", "QQQ", "SEMIS", "OIL")
DEFENSIVE_ASSETS = ("BOND", "GOLD", "USD", "VIX_HEDGE")


@dataclass(frozen=True)
class RiskWeightedCapCandidate:
    name: str
    normal_non_cash_cap: float
    defensive_risk_coefficient: float
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def apply_risk_weighted_cap(
    target: pd.Series,
    *,
    accepted_cap: float,
    normal_non_cash_cap: float,
    defensive_risk_coefficient: float,
) -> pd.Series:
    if not 0.0 <= accepted_cap <= 1.0:
        raise ValueError("accepted_cap must be in [0, 1]")
    if not 1.0 <= normal_non_cash_cap <= 1.20:
        raise ValueError("normal_non_cash_cap must be in [1, 1.20]")
    if not 0.0 <= defensive_risk_coefficient <= 1.0:
        raise ValueError("defensive_risk_coefficient must be in [0, 1]")

    adjusted = target.loc[cushion.ASSETS].astype(float).copy()
    if abs(float(adjusted.sum()) - 1.0) > 1e-9:
        raise ValueError("target weights must sum to one")
    non_cash_assets = [asset for asset in cushion.ASSETS if asset != "CASH"]
    if (adjusted.loc[non_cash_assets] < -1e-12).any():
        raise ValueError("non-cash target weights cannot be negative")

    effective_cap = (
        normal_non_cash_cap
        if accepted_cap >= 1.0 - 1e-12
        else accepted_cap
    )
    defensive = float(adjusted.loc[list(DEFENSIVE_ASSETS)].sum())
    defensive_risk = defensive_risk_coefficient * defensive
    growth = float(adjusted.loc[list(GROWTH_ASSETS)].sum())
    if defensive_risk > effective_cap + 1e-12:
        adjusted.loc[list(GROWTH_ASSETS)] = 0.0
        adjusted.loc[list(DEFENSIVE_ASSETS)] *= (
            effective_cap / defensive_risk
        )
    elif growth + defensive_risk > effective_cap + 1e-12:
        allowed_growth = max(effective_cap - defensive_risk, 0.0)
        if growth > 1e-12:
            adjusted.loc[list(GROWTH_ASSETS)] *= allowed_growth / growth
    adjusted["CASH"] = 1.0 - float(
        adjusted.loc[non_cash_assets].sum()
    )
    return adjusted


def simulate_candidate(settings, scenario, components, candidate):
    original = cushion.cap_total_non_cash

    def risk_weighted_cap(target: pd.Series, cap: float) -> pd.Series:
        return apply_risk_weighted_cap(
            target,
            accepted_cap=cap,
            normal_non_cash_cap=candidate.normal_non_cash_cap,
            defensive_risk_coefficient=candidate.defensive_risk_coefficient,
        )

    cushion.cap_total_non_cash = risk_weighted_cap
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
    settings_by_sample = {
        sample: samples[sample]
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        RiskWeightedCapCandidate(
            f"normal{int(normal_cap * 100):03d}_defrisk{int(defensive_risk * 100):03d}",
            normal_cap,
            defensive_risk,
        )
        for normal_cap in NORMAL_CAPS
        for defensive_risk in DEFENSIVE_RISK_COEFFICIENTS
    )

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_candidate(
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
                    "normal_non_cash_cap": candidate.normal_non_cash_cap,
                    "defensive_risk_coefficient": candidate.defensive_risk_coefficient,
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
    screen = pd.DataFrame(rows)
    screen.to_csv(OUTPUT / "screen_metrics.csv", index=False)

    modern = screen.loc[screen["sample"].eq("normal_synthetic")].copy()
    selected = set(
        modern.loc[
            modern["candidate_cagr"].ge(MODERN_PREFILTER_CAGR), "candidate"
        ]
    )
    selected.add("normal100_defrisk100")
    pre_settings, _ = build_pre2008_settings()
    pre_components = base_components(pre_settings, scenario)
    pre_rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.name not in selected:
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
                "defensive_risk_coefficient": candidate.defensive_risk_coefficient,
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
    ).merge(
        pre[["candidate", "candidate_max_drawdown"]],
        on="candidate",
        how="left",
    ).rename(columns={"candidate_max_drawdown": "pre2008_max_drawdown"})
    frontier["target_pass"] = (
        frontier["candidate_cagr_modern"].between(0.24, 0.25, inclusive="both")
        & frontier["pre2008_max_drawdown"].ge(-0.20)
    )
    frontier.to_csv(OUTPUT / "frontier.csv", index=False)
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
