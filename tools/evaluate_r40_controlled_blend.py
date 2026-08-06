from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r40_normal_cap_frontier import simulate_candidate
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/r40_controlled_blend")
CONTROLLED_R38_SHARES = (0.00, 0.25, 0.50, 0.75, 1.00)
NORMAL_NON_CASH_CAP = 1.20
MODERN_PREFILTER_CAGR = 0.235


@dataclass(frozen=True)
class ControlledBlendCandidate:
    name: str
    controlled_r38_share: float
    normal_non_cash_cap: float = NORMAL_NON_CASH_CAP
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def blend_controlled_targets(
    r11_weights: pd.DataFrame,
    staged_weights: pd.DataFrame,
    r38_share: float,
) -> pd.DataFrame:
    if not 0.0 <= r38_share <= 1.0:
        raise ValueError("r38_share must be in [0, 1]")
    left, right = r11_weights.align(staged_weights, join="inner", axis=0)
    left, right = left.align(right, join="inner", axis=1)
    blended = (1.0 - r38_share) * left + r38_share * right
    if not blended.sum(axis=1).sub(1.0).abs().le(1e-9).all():
        raise ValueError("controlled target weights must sum to one")
    return blended


def candidate_components(
    components: dict[str, pd.DataFrame | pd.Series],
    r38_share: float,
) -> dict[str, pd.DataFrame | pd.Series]:
    result = dict(components)
    r11 = result["r11_weights"]
    staged = result["blended_weights"]
    assert isinstance(r11, pd.DataFrame)
    assert isinstance(staged, pd.DataFrame)
    result["r11_weights"] = blend_controlled_targets(r11, staged, r38_share)
    return result


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
    base_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        ControlledBlendCandidate(
            f"controlled_r38_{int(share * 100):03d}", share
        )
        for share in CONTROLLED_R38_SHARES
    )

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            components = candidate_components(
                base_by_sample[sample], candidate.controlled_r38_share
            )
            trial, diagnostics = simulate_candidate(
                settings, scenario, components, candidate
            )
            baseline = base_by_sample[sample]["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            period, bounds = complete_bounds(settings)
            rows.append(
                {
                    "candidate": candidate.name,
                    "controlled_r38_share": candidate.controlled_r38_share,
                    "normal_non_cash_cap": candidate.normal_non_cash_cap,
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

    modern = screen.loc[screen["sample"].eq("normal_synthetic")]
    selected = set(
        modern.loc[
            modern["candidate_cagr"].ge(MODERN_PREFILTER_CAGR), "candidate"
        ]
    )
    selected.add("controlled_r38_000")
    pre_settings, _ = build_pre2008_settings()
    pre_base = base_components(pre_settings, scenario)
    pre_rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.name not in selected:
            continue
        print(f"crash replay {candidate.name}", flush=True)
        components = candidate_components(
            pre_base, candidate.controlled_r38_share
        )
        trial, diagnostics = simulate_candidate(
            pre_settings, scenario, components, candidate
        )
        baseline = pre_base["baseline"]
        assert isinstance(baseline, pd.DataFrame)
        pre_rows.append(
            {
                "candidate": candidate.name,
                "controlled_r38_share": candidate.controlled_r38_share,
                "normal_non_cash_cap": candidate.normal_non_cash_cap,
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
