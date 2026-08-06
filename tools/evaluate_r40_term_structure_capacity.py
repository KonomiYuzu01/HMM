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


OUTPUT = Path("output/r40_term_structure_capacity")
TERM_PREMIUM_THRESHOLDS = (1.00, 1.05)
CAPACITY_SCALES = (1.02, 1.04, 1.06)
MAX_NORMAL_NON_CASH = 1.35
MODERN_TERM_PRICES = Path("data/prices_vix_hedge.csv")
PRE2008_TERM_PRICES = Path("data/prices_pre2008_crash_proxy.csv")


@dataclass(frozen=True)
class TermCapacityCandidate:
    name: str
    term_premium_threshold: float
    capacity_scale: float
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = 9.5
    tier_size: float = 0.05


def causal_term_permission(
    closes: pd.DataFrame,
    *,
    term_premium_threshold: float,
) -> pd.Series:
    if term_premium_threshold < 1.0:
        raise ValueError("term_premium_threshold must be at least one")
    missing = {"VIX", "VIX3M"}.difference(closes.columns)
    if missing:
        raise ValueError(f"missing volatility term prices: {sorted(missing)}")
    prior_ratio = (closes["VIX3M"] / closes["VIX"]).shift(1)
    permission = prior_ratio.gt(term_premium_threshold).fillna(False)
    permission.name = "term_capacity_permission"
    return permission


def term_conditioned_weights(
    weights: pd.DataFrame,
    permission: pd.Series,
    *,
    capacity_scale: float,
) -> pd.DataFrame:
    if not 1.0 <= capacity_scale <= 1.10:
        raise ValueError("capacity_scale must be in [1, 1.10]")
    adjusted = weights.astype(float).copy()
    non_cash = [asset for asset in adjusted.columns if asset != "CASH"]
    allow = permission.reindex(adjusted.index).fillna(False)
    adjusted.loc[allow, non_cash] *= capacity_scale
    adjusted.loc[allow, "CASH"] = 1.0 - adjusted.loc[allow, non_cash].sum(axis=1)
    denied_total = adjusted.loc[~allow, non_cash].sum(axis=1)
    denied_scale = (1.0 / denied_total).clip(upper=1.0).fillna(1.0)
    adjusted.loc[~allow, non_cash] = adjusted.loc[~allow, non_cash].mul(
        denied_scale, axis=0
    )
    adjusted.loc[~allow, "CASH"] = 1.0 - adjusted.loc[~allow, non_cash].sum(axis=1)
    total = adjusted[non_cash].sum(axis=1)
    maximum_scale = (MAX_NORMAL_NON_CASH / total).clip(upper=1.0).fillna(1.0)
    adjusted[non_cash] = adjusted[non_cash].mul(maximum_scale, axis=0)
    adjusted["CASH"] = 1.0 - adjusted[non_cash].sum(axis=1)
    if not adjusted.sum(axis=1).sub(1.0).abs().le(1e-9).all():
        raise AssertionError("term-conditioned weights do not sum to one")
    return adjusted


def simulate_candidate(settings, scenario, components, candidate):
    closes = settings["closes"]
    term_closes = settings.get("term_closes", closes)
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(term_closes, pd.DataFrame)
    permission = causal_term_permission(
        term_closes,
        term_premium_threshold=candidate.term_premium_threshold,
    )
    trial_components = dict(components)
    staged = trial_components["blended_weights"]
    assert isinstance(staged, pd.DataFrame)
    trial_components["blended_weights"] = term_conditioned_weights(
        staged,
        permission,
        capacity_scale=candidate.capacity_scale,
    )
    original_cap = cushion.cap_total_non_cash

    def cap_with_extended_normal(target: pd.Series, cap: float) -> pd.Series:
        effective_cap = MAX_NORMAL_NON_CASH if cap >= 1.0 - 1e-12 else cap
        adjusted = target.loc[cushion.ASSETS].astype(float).copy()
        non_cash = [asset for asset in cushion.ASSETS if asset != "CASH"]
        total = float(adjusted.loc[non_cash].sum())
        if total > effective_cap + 1e-12:
            adjusted.loc[non_cash] *= effective_cap / total
            adjusted["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
        return adjusted

    cushion.cap_total_non_cash = cap_with_extended_normal
    try:
        trial, diagnostics = cushion.simulate_cushion_candidate(
            settings, scenario, trial_components, candidate
        )
    finally:
        cushion.cap_total_non_cash = original_cap
    diagnostics = diagnostics.copy()
    diagnostics["term_capacity_permission"] = permission.reindex(
        diagnostics.index
    ).fillna(False)
    return trial, diagnostics


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    modern_term = pd.read_csv(
        MODERN_TERM_PRICES, index_col="date", parse_dates=True
    ).loc[:, ["VIX", "VIX3M"]]
    settings_by_sample = {
        sample: {
            **samples[sample],
            "term_closes": modern_term.reindex(samples[sample]["closes"].index),
        }
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        TermCapacityCandidate(
            f"term{int((term - 1) * 100):02d}_scale{int((scale - 1) * 100):02d}",
            term,
            scale,
        )
        for term in TERM_PREMIUM_THRESHOLDS
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
    selected = set(
        modern.loc[
            modern["candidate_cagr"].between(0.24, 0.25, inclusive="both"),
            "candidate",
        ]
    )
    pre_rows: list[dict[str, object]] = []
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_term = pd.read_csv(
            PRE2008_TERM_PRICES, index_col="date", parse_dates=True
        ).loc[:, ["VIX", "VIX3M"]]
        pre_settings["term_closes"] = pre_term.reindex(
            pre_settings["closes"].index
        )
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
    proxy = screen.loc[screen["sample"].eq("proxy_synthetic")]
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
