from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r10_gde_capital_efficiency import ASSETS, simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
)
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r40_price_shock_multiplier import causal_price_stress_signals
from tools.evaluate_recursive_cushion_budget import RecursiveCushionPolicy


OUTPUT = Path("output/r40_trend_conditioned_cap")
NORMAL_CAPS = (1.20,)
FLOOR_DRAWDOWNS = (-0.185, -0.19)
BEAR_MULTIPLIERS = (18.0, 20.0, 22.0)
FIVE_DAY_STRESS_THRESHOLDS = (-0.05, -0.07, -0.09)
EXTENSION_DRAWDOWN_GATES = (-0.075, -0.08, -0.085, -0.09, -0.095)


@dataclass(frozen=True)
class TrendCapCandidate:
    name: str
    normal_non_cash_cap: float
    floor_drawdown: float
    bull_multiplier: float
    bear_multiplier: float
    tier_size: float
    five_day_stress_threshold: float = -0.07
    shock_bear_multiplier: float = 9.5
    extension_drawdown_gate: float = -0.05


def cap_non_cash(target: pd.Series, cap: float) -> pd.Series:
    if not 0.0 <= cap <= 1.20:
        raise ValueError("non-cash cap must be in [0, 1.20]")
    adjusted = target.loc[ASSETS].astype(float).copy()
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    total = float(adjusted.loc[non_cash].sum())
    if total > cap + 1e-12:
        adjusted.loc[non_cash] *= cap / total
        adjusted.loc["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
    return adjusted


class TrendConditionedCapPolicy:
    def __init__(
        self,
        candidate: TrendCapCandidate,
        trend_signals: pd.DataFrame,
        r11_weights: pd.DataFrame,
    ) -> None:
        self.candidate = candidate
        self.trend_signals = trend_signals
        self.base_policy = RecursiveCushionPolicy(
            candidate,  # type: ignore[arg-type]
            trend_signals,
            r11_weights,
        )
        self.extended = False

    def __call__(
        self,
        date: pd.Timestamp,
        blended_target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | int | bool | str]]:
        prior_drawdown = prior_equity / prior_peak - 1.0
        dual_trend_positive = bool(
            date in self.trend_signals.index
            and self.trend_signals.loc[date, "both_above_sma"]
        )
        shock_guard = bool(
            not dual_trend_positive
            and date in self.trend_signals.index
            and self.trend_signals.loc[date, "stress"]
        )
        effective_bear_multiplier = (
            self.candidate.shock_bear_multiplier
            if shock_guard
            else self.candidate.bear_multiplier
        )
        self.base_policy.candidate = replace(
            self.candidate,
            bear_multiplier=effective_bear_multiplier,
        )
        target, base_force, metadata = self.base_policy(
            date,
            blended_target,
            prior_equity,
            prior_peak,
        )
        extended = bool(
            metadata["state"] == "normal"
            and metadata["dual_trend_positive"]
            and float(self.trend_signals.loc[date, "prior_growth_drawdown"])
            >= self.candidate.extension_drawdown_gate
        )
        if extended:
            target = cap_non_cash(
                blended_target,
                self.candidate.normal_non_cash_cap,
            )
        extension_changed = extended != self.extended
        self.extended = extended
        result_metadata = dict(metadata)
        result_metadata.update(
            {
                "normal_cap_extended": extended,
                "effective_normal_non_cash_cap": (
                    self.candidate.normal_non_cash_cap if extended else 1.0
                ),
                "implemented_non_cash_weight": float(
                    target.drop(labels="CASH").sum()
                ),
                "shock_bear_guard": shock_guard,
                "effective_bear_multiplier": effective_bear_multiplier,
                "extension_growth_drawdown_gate": (
                    self.candidate.extension_drawdown_gate
                ),
            }
        )
        return target, bool(base_force or extension_changed), result_metadata


def simulate_candidate(settings, scenario, components, candidate):
    closes = settings["closes"]
    opens = settings["opens"]
    weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    daily = components["blended_daily"]
    slippage = components["blended_slippage"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(slippage, pd.Series)
    trend_signals = bear.causal_bear_signals(closes).join(
        causal_price_stress_signals(
            closes,
            five_day_stress_threshold=candidate.five_day_stress_threshold,
        )[["stress"]]
    )
    policy = TrendConditionedCapPolicy(
        candidate,
        trend_signals,
        r11_weights,
    )
    return simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=scenario.base_one_way_cost_bps,
        gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
        financing_spread_bps=scenario.financing_spread_bps,
        weights_override=weights,
        daily_override=daily,
        extra_slippage=slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
        target_policy=policy,
    )


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def evaluate(sample, settings, scenario, components, candidate):
    trial = simulate_candidate(settings, scenario, components, candidate)
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
        "five_day_stress_threshold": candidate.five_day_stress_threshold,
        "shock_bear_multiplier": candidate.shock_bear_multiplier,
        "extension_drawdown_gate": candidate.extension_drawdown_gate,
        "controlled_fraction": float(trial["state"].ne("normal").mean()),
        "extended_fraction": float(trial["normal_cap_extended"].mean()),
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
    candidates = tuple(
        TrendCapCandidate(
            f"trendcap{int(cap * 100):03d}_floor{int(abs(floor) * 1000):03d}_bear{int(bear * 10):03d}_shock{int(abs(stress_threshold) * 100):02d}_extdd{int(abs(extension_gate) * 100):02d}",
            cap,
            floor,
            100.0,
            bear,
            0.05,
            stress_threshold,
            9.5,
            extension_gate,
        )
        for cap in NORMAL_CAPS
        for floor in (-0.19,)
        for bear in (20.0,)
        for stress_threshold in (-0.07,)
        for extension_gate in EXTENSION_DRAWDOWN_GATES
    )
    modern_settings = samples["normal_synthetic"]
    modern_components = base_components(modern_settings, scenario)
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
        pre_settings, _ = build_pre2008_settings()
        pre_settings = {
            **pre_settings,
            "periods": {
                "complete_1931_2007": ("1931-01-02", "2007-12-31")
            },
        }
        validation_settings = {
            "proxy_synthetic": samples["proxy_synthetic"],
            "pre2008": pre_settings,
        }
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
            suffixes=("", "_proxy"),
        ).rename(
            columns={
                "candidate_cagr_proxy": "proxy_cagr",
                "candidate_max_drawdown_proxy": "proxy_max_drawdown",
            }
        )
        frontier = frontier.merge(
            pre[["candidate", "candidate_max_drawdown"]],
            on="candidate",
            how="left",
            suffixes=("", "_pre2008"),
        ).rename(
            columns={"candidate_max_drawdown_pre2008": "pre2008_max_drawdown"}
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
