from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_bear_recovery_governor import cap_growth_to_cash
import tools.evaluate_reference_drawdown_governor as reference
from tools.evaluate_r10_gde_capital_efficiency import simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import GDE_NO_TRADE_BAND
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION


OUTPUT = Path("output/recursive_equity_risk_budget")
RAMP_CAPS = (0.20, 0.40, 0.60, 0.80)


@dataclass(frozen=True)
class RecursiveRiskCandidate:
    name: str
    soft_drawdown: float = -0.06
    hard_drawdown: float = -0.10
    soft_release_drawdown: float = -0.03
    soft_growth_cap: float = 0.35
    recovery_confirmation_days: int = 5
    ramp_stage_sessions: int = 5

    def __post_init__(self) -> None:
        if not -1.0 < self.hard_drawdown < self.soft_drawdown < 0.0:
            raise ValueError("hard_drawdown must be below soft_drawdown")
        if not self.soft_drawdown < self.soft_release_drawdown < 0.0:
            raise ValueError("soft release must provide hysteresis")
        if not 0.0 <= self.soft_growth_cap <= 1.0:
            raise ValueError("soft_growth_cap must be in [0, 1]")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = RecursiveRiskCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="hard_08", hard_drawdown=-0.08),
    replace(CENTRAL, name="hard_12", hard_drawdown=-0.12),
    replace(CENTRAL, name="soft_cap_25", soft_growth_cap=0.25),
    replace(CENTRAL, name="soft_cap_45", soft_growth_cap=0.45),
    replace(CENTRAL, name="confirm_03", recovery_confirmation_days=3),
    replace(CENTRAL, name="confirm_10", recovery_confirmation_days=10),
    replace(CENTRAL, name="ramp_03", ramp_stage_sessions=3),
    replace(CENTRAL, name="ramp_07", ramp_stage_sessions=7),
)


class RecursiveEquityPolicy:
    def __init__(
        self,
        candidate: RecursiveRiskCandidate,
        trend_signals: pd.DataFrame,
        r11_weights: pd.DataFrame,
    ) -> None:
        self.candidate = candidate
        self.trend_signals = trend_signals
        self.r11_weights = r11_weights
        self.state = "normal"
        self.recovery_run = 0
        self.ramp_day = 0

    def __call__(
        self,
        date: pd.Timestamp,
        blended_target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | int | bool | str]]:
        drawdown = prior_equity / prior_peak - 1.0
        recovery_quality = bool(
            date in self.trend_signals.index
            and self.trend_signals.loc[date, "both_above_sma"]
        )
        previous_state = self.state
        previous_stage = self._ramp_stage()

        if self.state == "normal":
            if drawdown <= self.candidate.hard_drawdown:
                self.state = "defense"
                self.recovery_run = 0
                self.ramp_day = 0
            elif drawdown <= self.candidate.soft_drawdown:
                self.state = "soft"
        elif self.state == "soft":
            if drawdown <= self.candidate.hard_drawdown:
                self.state = "defense"
                self.recovery_run = 0
                self.ramp_day = 0
            elif drawdown >= self.candidate.soft_release_drawdown:
                self.state = "normal"
        elif self.state == "defense":
            self.recovery_run = self.recovery_run + 1 if recovery_quality else 0
            if self.recovery_run >= self.candidate.recovery_confirmation_days:
                self.state = "ramp"
                self.ramp_day = 0
        elif self.state == "ramp":
            if not recovery_quality or drawdown <= self.candidate.hard_drawdown - 0.02:
                self.state = "defense"
                self.recovery_run = 0
                self.ramp_day = 0
            else:
                self.ramp_day += 1
                if self.ramp_day >= self.candidate.ramp_stage_sessions * len(
                    RAMP_CAPS
                ):
                    self.state = (
                        "soft"
                        if drawdown <= self.candidate.soft_release_drawdown
                        else "normal"
                    )
                    self.recovery_run = 0
                    self.ramp_day = 0

        growth_cap = self._growth_cap()
        base_target = blended_target
        if self.state != "normal":
            base_target = self.r11_weights.loc[date, bear.ASSETS]
        target = cap_growth_to_cash(base_target, growth_cap)
        stage = self._ramp_stage()
        state_change = self.state != previous_state
        stage_change = self.state == "ramp" and stage != previous_stage
        force_trade = state_change or stage_change
        return target, force_trade, {
            "state": self.state,
            "entry": state_change and self.state in {"soft", "defense"},
            "state_change": state_change,
            "stage_change": stage_change,
            "recovery_quality": recovery_quality,
            "recovery_run": self.recovery_run,
            "ramp_day": self.ramp_day,
            "growth_cap": growth_cap,
            "r38_incremental_enabled": self.state == "normal",
            "implemented_growth_weight": float(
                target.loc[list(bear.GROWTH_ASSETS)].sum()
            ),
            "implemented_cash_weight": float(target["CASH"]),
            "recursive_prior_drawdown": drawdown,
        }

    def _ramp_stage(self) -> int:
        if self.state != "ramp":
            return -1
        return min(
            self.ramp_day // self.candidate.ramp_stage_sessions,
            len(RAMP_CAPS) - 1,
        )

    def _growth_cap(self) -> float:
        if self.state == "normal":
            return 1.0
        if self.state == "soft":
            return self.candidate.soft_growth_cap
        if self.state == "defense":
            return 0.0
        return RAMP_CAPS[self._ramp_stage()]


def simulate_recursive_candidate(
    settings: dict[str, object],
    scenario: object,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: RecursiveRiskCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = settings["closes"]
    opens = settings["opens"]
    blended_weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    blended_daily = components["blended_daily"]
    blended_slippage = components["blended_slippage"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(blended_weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(blended_daily, pd.DataFrame)
    assert isinstance(blended_slippage, pd.Series)
    policy = RecursiveEquityPolicy(
        candidate,
        bear.causal_bear_signals(closes),
        r11_weights,
    )
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,  # type: ignore[attr-defined]
        financing_spread_bps=scenario.financing_spread_bps,  # type: ignore[attr-defined]
        weights_override=blended_weights,
        daily_override=blended_daily,
        extra_slippage=blended_slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
        target_policy=policy,
    )
    diagnostics = trial[
        [
            "state",
            "entry",
            "state_change",
            "stage_change",
            "recovery_quality",
            "recovery_run",
            "ramp_day",
            "growth_cap",
            "r38_incremental_enabled",
            "implemented_growth_weight",
            "implemented_cash_weight",
            "recursive_prior_drawdown",
            "policy_prior_equity",
            "policy_prior_peak",
            "policy_force_trade",
        ]
    ].copy()
    active = diagnostics["state"].ne("normal")
    if (
        diagnostics.loc[active, "implemented_growth_weight"]
        > diagnostics.loc[active, "growth_cap"] + 1e-10
    ).any():
        raise AssertionError("recursive policy exceeded its growth cap")
    if diagnostics.loc[active, "r38_incremental_enabled"].any():
        raise AssertionError("recursive policy left R38 enabled while active")
    return trial, diagnostics


def write_strict_acceptance(output: Path = OUTPUT) -> bool:
    metrics = pd.read_csv(output / "metrics.csv")
    events = pd.read_csv(output / "pre2008_events.csv")
    neighborhood = pd.read_csv(output / "parameter_neighborhood.csv")
    pre = metrics.loc[metrics["sample"].eq("pre2008")].iloc[0]
    normal = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    proxy = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    neighbor_pre = neighborhood.loc[neighborhood["sample"].eq("pre2008")]
    gates = [
        ("all_named_events_mdd_at_or_above_minus_20pct", events["candidate_max_drawdown"].ge(-0.20).all()),
        ("complete_pre2008_mdd_at_or_above_minus_20pct", pre["candidate_max_drawdown"] >= -0.20),
        ("modern_cagr_cost_at_most_2pct", normal["cagr_delta"] >= -0.02),
        ("proxy_cagr_cost_at_most_2pct", proxy["cagr_delta"] >= -0.02),
        ("modern_drawdown_not_worse", normal["max_drawdown_delta"] >= 0.0),
        ("proxy_drawdown_not_worse", proxy["max_drawdown_delta"] >= 0.0),
        ("neighbors_pre2008_mdd_at_or_above_minus_22pct", neighbor_pre["candidate_max_drawdown"].ge(-0.22).all()),
        ("neighbors_dotcom_mdd_at_or_above_minus_22pct", neighbor_pre["dotcom_max_drawdown"].ge(-0.22).all()),
        ("neighbors_modern_cagr_cost_at_most_2_5pct", neighborhood.loc[neighborhood["sample"].eq("normal_synthetic"), "cagr_delta"].ge(-0.025).all()),
    ]
    acceptance = pd.DataFrame(gates, columns=["gate", "passed"])
    research_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = ["research_pass", research_pass]
    acceptance.to_csv(output / "acceptance.csv", index=False)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_pass": research_pass,
            "production_eligible": False,
            "production_changed": False,
            "orders_generated": False,
            "strict_complete_pre2008_mdd": float(pre["candidate_max_drawdown"]),
            "strict_modern_cagr_delta": float(normal["cagr_delta"]),
            "strict_proxy_cagr_delta": float(proxy["cagr_delta"]),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return research_pass


def main() -> None:
    original_output = reference.OUTPUT
    original_central = reference.CENTRAL
    original_neighbors = reference.NEIGHBORS
    original_simulate = reference.simulate_candidate
    try:
        reference.OUTPUT = OUTPUT
        reference.CENTRAL = CENTRAL  # type: ignore[assignment]
        reference.NEIGHBORS = NEIGHBORS  # type: ignore[assignment]
        reference.simulate_candidate = simulate_recursive_candidate  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.simulate_candidate = original_simulate
    research_pass = write_strict_acceptance()
    print((OUTPUT / "summary.json").read_text())
    print("\nStrict acceptance:")
    print(pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False))
    if not research_pass:
        print("\nCandidate remains research-only and failed strict qualification.")


if __name__ == "__main__":
    main()
