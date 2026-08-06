from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r40_term_structure_capacity as term


OUTPUT = Path("output/r40_term_trend_capacity")
TERM_PREMIUM_THRESHOLDS = (1.00,)
CAPACITY_SCALES = (1.04, 1.045, 1.05, 1.055, 1.06)
BASE_TERM_PERMISSION = term.causal_term_permission


def causal_term_trend_permission(
    closes: pd.DataFrame,
    *,
    term_premium_threshold: float,
) -> pd.Series:
    term_permission = BASE_TERM_PERMISSION(
        closes,
        term_premium_threshold=term_premium_threshold,
    )
    trend_permission = bear.causal_bear_signals(closes)["both_above_sma"]
    result = (term_permission & trend_permission).fillna(False)
    result.name = "term_trend_capacity_permission"
    return result


def simulate_candidate(settings, scenario, components, candidate):
    original = term.causal_term_permission
    term.causal_term_permission = causal_term_trend_permission
    try:
        return term.simulate_candidate(settings, scenario, components, candidate)
    finally:
        term.causal_term_permission = original


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def attach_term_prices(
    settings: dict[str, object],
    term_prices: pd.DataFrame,
) -> dict[str, object]:
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    signals = closes.loc[:, ["QQQ", "SEMIS"]].join(
        term_prices.loc[:, ["VIX", "VIX3M"]], how="left"
    )
    return {**settings, "term_closes": signals}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    modern_term = pd.read_csv(
        term.MODERN_TERM_PRICES, index_col="date", parse_dates=True
    )
    settings_by_sample = {
        sample: attach_term_prices(samples[sample], modern_term)
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        term.TermCapacityCandidate(
            f"term{int(round((premium - 1) * 1000)):03d}_trend_scale{int(round((scale - 1) * 1000)):03d}",
            premium,
            scale,
        )
        for premium in TERM_PREMIUM_THRESHOLDS
        for scale in CAPACITY_SCALES
    )
    rows: list[dict[str, object]] = []
    paths: dict[str, pd.DataFrame] = {}
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
                    "term_premium_threshold": candidate.term_premium_threshold,
                    "capacity_scale": candidate.capacity_scale,
                    "sample": sample,
                    "permission_fraction": float(
                        diagnostics["term_capacity_permission"].mean()
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
    proxy = screen.loc[screen["sample"].eq("proxy_synthetic")]
    proxy_by_name = proxy.set_index("candidate")
    selected = set(
        modern.loc[
            modern["candidate_cagr"].between(0.24, 0.25, inclusive="both")
            & modern["candidate"].map(proxy_by_name["candidate_cagr"]).ge(0.14),
            "candidate",
        ]
    )
    pre_rows: list[dict[str, object]] = []
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_term = pd.read_csv(
            term.PRE2008_TERM_PRICES, index_col="date", parse_dates=True
        )
        pre_settings = attach_term_prices(pre_settings, pre_term)
        pre_components = base_components(pre_settings, scenario)
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
                    "term_premium_threshold": candidate.term_premium_threshold,
                    "capacity_scale": candidate.capacity_scale,
                    "permission_fraction": float(
                        diagnostics["term_capacity_permission"].mean()
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
    frontier = modern.merge(
        proxy[["candidate", "candidate_cagr", "candidate_max_drawdown"]],
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
        & frontier["candidate_cagr_proxy"].ge(0.14)
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
