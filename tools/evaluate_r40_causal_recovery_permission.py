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
from tools.evaluate_r40_normal_cap_frontier import simulate_candidate
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/r40_causal_recovery_permission")
FAST_SMA_WINDOWS = (10, 20, 40)
REBOUND_THRESHOLDS = (0.03, 0.05, 0.07)
RECOVERY_DRAWDOWN_TRIGGER = -0.08
RECENT_LOW_WINDOW = 63
NORMAL_NON_CASH_CAP = 1.20
MODERN_PREFILTER_CAGR = 0.24


@dataclass(frozen=True)
class RecoveryPermissionCandidate:
    name: str
    fast_sma_window: int
    rebound_threshold: float
    normal_non_cash_cap: float = NORMAL_NON_CASH_CAP
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def causal_recovery_signals(
    closes: pd.DataFrame,
    *,
    fast_sma_window: int,
    rebound_threshold: float,
) -> pd.DataFrame:
    if fast_sma_window < 2:
        raise ValueError("fast_sma_window must be at least two")
    if not 0.0 < rebound_threshold < 1.0:
        raise ValueError("rebound_threshold must be in (0, 1)")
    base = bear.causal_bear_signals(closes)
    prices = closes.loc[:, list(bear.GROWTH_ASSETS)].astype(float)
    growth_return = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    growth_index = (1.0 + growth_return).cumprod()
    rolling_peak = growth_index.rolling(252, min_periods=252).max()
    growth_drawdown = growth_index / rolling_peak - 1.0
    rolling_low = growth_index.rolling(
        RECENT_LOW_WINDOW,
        min_periods=fast_sma_window,
    ).min()
    rebound = growth_index / rolling_low - 1.0
    prior_prices = prices.shift(1)
    prior_fast_sma = prices.rolling(
        fast_sma_window,
        min_periods=fast_sma_window,
    ).mean().shift(1)
    both_above_fast = prior_prices.gt(prior_fast_sma).all(axis=1)
    fast_recovery = (
        growth_drawdown.shift(1).le(RECOVERY_DRAWDOWN_TRIGGER)
        & rebound.shift(1).ge(rebound_threshold)
        & both_above_fast
    ).fillna(False)
    result = base.copy()
    result["long_trend_positive"] = base["both_above_sma"]
    result["fast_recovery_permission"] = fast_recovery
    result["prior_growth_rebound"] = rebound.shift(1).fillna(0.0)
    result["both_above_sma"] = (
        base["both_above_sma"] | fast_recovery
    )
    return result


def simulate_recovery_candidate(settings, scenario, components, candidate):
    original = bear.causal_bear_signals

    def candidate_signals(closes: pd.DataFrame) -> pd.DataFrame:
        bear.causal_bear_signals = original
        try:
            return causal_recovery_signals(
                closes,
                fast_sma_window=candidate.fast_sma_window,
                rebound_threshold=candidate.rebound_threshold,
            )
        finally:
            bear.causal_bear_signals = candidate_signals

    bear.causal_bear_signals = candidate_signals
    try:
        return simulate_candidate(settings, scenario, components, candidate)
    finally:
        bear.causal_bear_signals = original


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
        RecoveryPermissionCandidate(
            f"sma{sma:02d}_rebound{int(rebound * 100):02d}",
            sma,
            rebound,
        )
        for sma in FAST_SMA_WINDOWS
        for rebound in REBOUND_THRESHOLDS
    )

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_recovery_candidate(
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
                    "fast_sma_window": candidate.fast_sma_window,
                    "rebound_threshold": candidate.rebound_threshold,
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
    pre_settings, _ = build_pre2008_settings()
    pre_components = base_components(pre_settings, scenario)
    pre_rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.name not in selected:
            continue
        print(f"crash replay {candidate.name}", flush=True)
        trial, diagnostics = simulate_recovery_candidate(
            pre_settings, scenario, pre_components, candidate
        )
        baseline = pre_components["baseline"]
        assert isinstance(baseline, pd.DataFrame)
        pre_rows.append(
            {
                "candidate": candidate.name,
                "fast_sma_window": candidate.fast_sma_window,
                "rebound_threshold": candidate.rebound_threshold,
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
