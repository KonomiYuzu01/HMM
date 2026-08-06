from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_bear_recovery_governor import base_components, metric_row
import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r40_shock_aware_multiplier as shock


OUTPUT = Path("output/r40_sparse_shock_multiplier")
ONE_DAY_THRESHOLDS = (-0.035, -0.04, -0.05)
FIVE_DAY_THRESHOLDS = (-0.09, -0.12)
CALM_MULTIPLIER = 30.0
NORMAL_NON_CASH_CAP = 1.22
MODERN_PREFILTER_CAGR = 0.24


@dataclass(frozen=True)
class SparseShockCandidate:
    name: str
    one_day_stress_threshold: float
    five_day_stress_threshold: float
    calm_multiplier: float = CALM_MULTIPLIER
    normal_non_cash_cap: float = NORMAL_NON_CASH_CAP
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def causal_sparse_stress_signals(
    closes: pd.DataFrame,
    *,
    one_day_stress_threshold: float,
    five_day_stress_threshold: float,
) -> pd.DataFrame:
    if not -1.0 < one_day_stress_threshold < 0.0:
        raise ValueError("one_day_stress_threshold must be in (-1, 0)")
    if not -1.0 < five_day_stress_threshold < 0.0:
        raise ValueError("five_day_stress_threshold must be in (-1, 0)")
    prices = closes.loc[:, list(bear.GROWTH_ASSETS)].astype(float)
    growth_return = prices.pct_change(fill_method=None).mean(axis=1)
    one_day = growth_return.shift(1)
    five_day = (
        (1.0 + growth_return)
        .rolling(5, min_periods=5)
        .apply(lambda values: values.prod(), raw=True)
        .sub(1.0)
        .shift(1)
    )
    stress = (
        one_day.le(one_day_stress_threshold)
        | five_day.le(five_day_stress_threshold)
    ).fillna(False)
    return pd.DataFrame(
        {
            "stress": stress,
            "prior_one_day_growth_return": one_day.fillna(0.0),
            "prior_five_day_growth_return": five_day.fillna(0.0),
            "prior_volatility_ratio": 0.0,
        },
        index=closes.index,
    )


def simulate_sparse_candidate(settings, scenario, components, candidate):
    original = shock.causal_stress_signals

    def sparse(closes, *, five_day_stress_threshold):
        return causal_sparse_stress_signals(
            closes,
            one_day_stress_threshold=candidate.one_day_stress_threshold,
            five_day_stress_threshold=five_day_stress_threshold,
        )

    shock.causal_stress_signals = sparse
    try:
        return shock.simulate_shock_candidate(
            settings, scenario, components, candidate
        )
    finally:
        shock.causal_stress_signals = original


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
        SparseShockCandidate(
            f"day{abs(int(one * 1000)):03d}_week{abs(int(five * 100)):02d}",
            one,
            five,
        )
        for one in ONE_DAY_THRESHOLDS
        for five in FIVE_DAY_THRESHOLDS
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_sparse_candidate(
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
                    "one_day_stress_threshold": candidate.one_day_stress_threshold,
                    "five_day_stress_threshold": candidate.five_day_stress_threshold,
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

    pre_rows: list[dict[str, object]] = []
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_components = base_components(pre_settings, scenario)
        for candidate in candidates:
            if candidate.name not in selected:
                continue
            print(f"crash replay {candidate.name}", flush=True)
            trial, diagnostics = simulate_sparse_candidate(
                pre_settings, scenario, pre_components, candidate
            )
            baseline = pre_components["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            pre_rows.append(
                {
                    "candidate": candidate.name,
                    "one_day_stress_threshold": candidate.one_day_stress_threshold,
                    "five_day_stress_threshold": candidate.five_day_stress_threshold,
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
