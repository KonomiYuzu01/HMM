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


OUTPUT = Path("output/r40_latched_recovery")
FAST_SMA_WINDOW = 20
ENTRY_REBOUNDS = (0.03, 0.05, 0.07)
TRAILING_STOPS = (0.02, 0.03, 0.04)
RECOVERY_DRAWDOWN_TRIGGER = -0.08
RECENT_LOW_WINDOW = 63
NORMAL_NON_CASH_CAP = 1.20
MODERN_PREFILTER_CAGR = 0.24


@dataclass(frozen=True)
class LatchedRecoveryCandidate:
    name: str
    entry_rebound: float
    trailing_stop: float
    normal_non_cash_cap: float = NORMAL_NON_CASH_CAP
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def causal_latched_recovery_signals(
    closes: pd.DataFrame,
    *,
    entry_rebound: float,
    trailing_stop: float,
) -> pd.DataFrame:
    if not 0.0 < entry_rebound < 1.0:
        raise ValueError("entry_rebound must be in (0, 1)")
    if not 0.0 < trailing_stop < 1.0:
        raise ValueError("trailing_stop must be in (0, 1)")
    base = bear.causal_bear_signals(closes)
    prices = closes.loc[:, list(bear.GROWTH_ASSETS)].astype(float)
    growth_return = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    growth_index = (1.0 + growth_return).cumprod()
    rolling_peak = growth_index.rolling(252, min_periods=252).max()
    drawdown = growth_index / rolling_peak - 1.0
    rolling_low = growth_index.rolling(
        RECENT_LOW_WINDOW,
        min_periods=FAST_SMA_WINDOW,
    ).min()
    rebound = growth_index / rolling_low - 1.0
    prior_prices = prices.shift(1)
    prior_sma = prices.rolling(
        FAST_SMA_WINDOW,
        min_periods=FAST_SMA_WINDOW,
    ).mean().shift(1)
    fast_positive = prior_prices.gt(prior_sma).all(axis=1)
    prior_index = growth_index.shift(1)
    prior_drawdown = drawdown.shift(1)
    prior_rebound = rebound.shift(1)

    active = False
    recovery_peak = float("nan")
    active_rows: list[bool] = []
    entry_rows: list[bool] = []
    stop_rows: list[bool] = []
    trailing_drawdown_rows: list[float] = []
    for date in closes.index:
        value = float(prior_index.loc[date])
        long_positive = bool(base.loc[date, "both_above_sma"])
        entered = False
        stopped = False
        trailing_drawdown = 0.0
        if active and pd.notna(value):
            recovery_peak = max(recovery_peak, value)
            trailing_drawdown = value / recovery_peak - 1.0
            if trailing_drawdown <= -trailing_stop:
                active = False
                stopped = True
        if long_positive:
            active = False
        elif (
            not active
            and not stopped
            and bool(fast_positive.loc[date])
            and float(prior_drawdown.loc[date]) <= RECOVERY_DRAWDOWN_TRIGGER
            and float(prior_rebound.loc[date]) >= entry_rebound
        ):
            active = True
            recovery_peak = value
            trailing_drawdown = 0.0
            entered = True
        active_rows.append(active)
        entry_rows.append(entered)
        stop_rows.append(stopped)
        trailing_drawdown_rows.append(trailing_drawdown)

    result = base.copy()
    result["long_trend_positive"] = base["both_above_sma"]
    result["latched_recovery_permission"] = active_rows
    result["recovery_entry"] = entry_rows
    result["recovery_stop"] = stop_rows
    result["recovery_trailing_drawdown"] = trailing_drawdown_rows
    result["prior_growth_rebound"] = prior_rebound.fillna(0.0)
    result["both_above_sma"] = (
        base["both_above_sma"] | result["latched_recovery_permission"]
    )
    return result


def simulate_latched_candidate(settings, scenario, components, candidate):
    original = bear.causal_bear_signals

    def candidate_signals(closes: pd.DataFrame) -> pd.DataFrame:
        bear.causal_bear_signals = original
        try:
            return causal_latched_recovery_signals(
                closes,
                entry_rebound=candidate.entry_rebound,
                trailing_stop=candidate.trailing_stop,
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
        LatchedRecoveryCandidate(
            f"rebound{int(rebound * 100):02d}_stop{int(stop * 100):02d}",
            rebound,
            stop,
        )
        for rebound in ENTRY_REBOUNDS
        for stop in TRAILING_STOPS
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_latched_candidate(
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
                    "entry_rebound": candidate.entry_rebound,
                    "trailing_stop": candidate.trailing_stop,
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
            trial, diagnostics = simulate_latched_candidate(
                pre_settings, scenario, pre_components, candidate
            )
            baseline = pre_components["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            pre_rows.append(
                {
                    "candidate": candidate.name,
                    "entry_rebound": candidate.entry_rebound,
                    "trailing_stop": candidate.trailing_stop,
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
