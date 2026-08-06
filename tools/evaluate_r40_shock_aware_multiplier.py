from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, sqrt
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_bear_recovery_governor import base_components, metric_row
import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_recursive_cushion_budget as cushion


OUTPUT = Path("output/r40_shock_aware_multiplier")
CALM_MULTIPLIERS = (15.0, 20.0, 30.0)
FIVE_DAY_STRESS_THRESHOLDS = (-0.04, -0.05, -0.06)
ONE_DAY_STRESS_THRESHOLD = -0.03
VOLATILITY_RATIO_THRESHOLD = 1.50
STRESS_MULTIPLIER = 9.5
NORMAL_NON_CASH_CAP = 1.22
MODERN_PREFILTER_CAGR = 0.24


@dataclass(frozen=True)
class ShockAwareCandidate:
    name: str
    calm_multiplier: float
    five_day_stress_threshold: float
    normal_non_cash_cap: float = NORMAL_NON_CASH_CAP
    floor_drawdown: float = -0.19
    bull_multiplier: float = 100.0
    bear_multiplier: float = STRESS_MULTIPLIER
    tier_size: float = 0.05


def causal_stress_signals(
    closes: pd.DataFrame,
    *,
    five_day_stress_threshold: float,
) -> pd.DataFrame:
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
    realized_volatility = growth_return.rolling(20, min_periods=20).std(ddof=1) * sqrt(252.0)
    volatility_reference = realized_volatility.rolling(
        252, min_periods=126
    ).median()
    volatility_ratio = realized_volatility.shift(1) / volatility_reference.shift(1)
    stress = (
        one_day.le(ONE_DAY_STRESS_THRESHOLD)
        | five_day.le(five_day_stress_threshold)
        | volatility_ratio.ge(VOLATILITY_RATIO_THRESHOLD)
    ).fillna(False)
    return pd.DataFrame(
        {
            "stress": stress,
            "prior_one_day_growth_return": one_day.fillna(0.0),
            "prior_five_day_growth_return": five_day.fillna(0.0),
            "prior_volatility_ratio": volatility_ratio.fillna(0.0),
        },
        index=closes.index,
    )


class ShockAwarePolicy:
    def __init__(
        self,
        candidate: ShockAwareCandidate,
        trend_signals: pd.DataFrame,
        stress_signals: pd.DataFrame,
        r11_weights: pd.DataFrame,
    ) -> None:
        self.candidate = candidate
        self.trend_signals = trend_signals
        self.stress_signals = stress_signals
        self.r11_weights = r11_weights
        self.accepted_cap = 1.0
        self.previous_stress = False

    def __call__(
        self,
        date: pd.Timestamp,
        blended_target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | int | bool | str]]:
        floor_equity = prior_peak * (1.0 + self.candidate.floor_drawdown)
        cushion_value = max(prior_equity - floor_equity, 0.0)
        bull = bool(self.trend_signals.loc[date, "both_above_sma"])
        stress = bool(self.stress_signals.loc[date, "stress"])
        multiplier = (
            self.candidate.bull_multiplier
            if bull
            else self.candidate.bear_multiplier
            if stress
            else self.candidate.calm_multiplier
        )
        requested_cap = min(
            max(multiplier * cushion_value / prior_equity, 0.0), 1.0
        )
        accepted_cap = (
            1.0
            if requested_cap >= 1.0 - 1e-12
            else floor(
                (requested_cap + 1e-12) / self.candidate.tier_size
            )
            * self.candidate.tier_size
        )
        accepted_cap = min(max(accepted_cap, 0.0), 1.0)
        previous_cap = self.accepted_cap
        self.accepted_cap = accepted_cap
        constrained = accepted_cap < 1.0 - 1e-12
        base_target = (
            self.r11_weights.loc[date, cushion.ASSETS]
            if constrained
            else blended_target
        )
        target = cushion.cap_total_non_cash(base_target, accepted_cap)
        state = (
            "floor"
            if accepted_cap <= 1e-12
            else "controlled"
            if constrained
            else "normal"
        )
        stress_normal_cap = getattr(
            self.candidate, "stress_normal_non_cash_cap", None
        )
        if not constrained and stress and stress_normal_cap is not None:
            target = cap_non_cash_exact(target, float(stress_normal_cap))
        force_trade = abs(accepted_cap - previous_cap) > 1e-12 or (
            stress_normal_cap is not None
            and stress != self.previous_stress
            and not constrained
        )
        self.previous_stress = stress
        return target, force_trade, {
            "state": state,
            "entry": force_trade and constrained,
            "state_change": force_trade,
            "stage_change": force_trade,
            "recovery_run": 0,
            "ramp_day": 0,
            "growth_cap": accepted_cap,
            "r38_incremental_enabled": not constrained,
            "implemented_growth_weight": float(
                target.loc[list(bear.GROWTH_ASSETS)].sum()
            ),
            "implemented_cash_weight": float(target["CASH"]),
            "recursive_prior_drawdown": prior_equity / prior_peak - 1.0,
            "floor_equity": floor_equity,
            "cushion": cushion_value,
            "risk_multiplier": multiplier,
            "requested_non_cash_cap": requested_cap,
            "accepted_non_cash_cap": accepted_cap,
            "dual_trend_positive": bull,
            "stress_active": stress,
        }


def cap_non_cash_exact(target: pd.Series, cap: float) -> pd.Series:
    if not 0.0 <= cap <= 1.22:
        raise ValueError("cap must be in [0, 1.22]")
    adjusted = target.loc[cushion.ASSETS].astype(float).copy()
    non_cash = [asset for asset in cushion.ASSETS if asset != "CASH"]
    total = float(adjusted.loc[non_cash].sum())
    if total > cap + 1e-12:
        adjusted.loc[non_cash] *= cap / total
        adjusted["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
    return adjusted


def simulate_shock_candidate(settings, scenario, components, candidate):
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    stress_signals = causal_stress_signals(
        closes,
        five_day_stress_threshold=candidate.five_day_stress_threshold,
    )
    original_policy = cushion.RecursiveCushionPolicy
    original_cap = cushion.cap_total_non_cash

    def policy_factory(candidate_value, trend_signals, r11_weights):
        return ShockAwarePolicy(
            candidate_value,
            trend_signals,
            stress_signals,
            r11_weights,
        )

    def cap_with_normal_extension(target: pd.Series, cap: float) -> pd.Series:
        effective_cap = (
            candidate.normal_non_cash_cap
            if cap >= 1.0 - 1e-12
            else cap
        )
        adjusted = target.loc[cushion.ASSETS].astype(float).copy()
        non_cash = [asset for asset in cushion.ASSETS if asset != "CASH"]
        total = float(adjusted.loc[non_cash].sum())
        if total > effective_cap + 1e-12:
            adjusted.loc[non_cash] *= effective_cap / total
            adjusted["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
        return adjusted

    cushion.RecursiveCushionPolicy = policy_factory
    cushion.cap_total_non_cash = cap_with_normal_extension
    try:
        return cushion.simulate_cushion_candidate(
            settings, scenario, components, candidate
        )
    finally:
        cushion.RecursiveCushionPolicy = original_policy
        cushion.cap_total_non_cash = original_cap


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
        ShockAwareCandidate(
            f"calm{int(calm):02d}_stress{abs(int(threshold * 100)):02d}",
            calm,
            threshold,
        )
        for calm in CALM_MULTIPLIERS
        for threshold in FIVE_DAY_STRESS_THRESHOLDS
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, diagnostics = simulate_shock_candidate(
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
                    "calm_multiplier": candidate.calm_multiplier,
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
            trial, diagnostics = simulate_shock_candidate(
                pre_settings, scenario, pre_components, candidate
            )
            baseline = pre_components["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            pre_rows.append(
                {
                    "candidate": candidate.name,
                    "calm_multiplier": candidate.calm_multiplier,
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
