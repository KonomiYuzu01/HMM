from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
import tools.evaluate_recursive_equity_risk_budget as recursive
import tools.evaluate_reference_drawdown_governor as reference
from tools.evaluate_r10_gde_capital_efficiency import ASSETS, simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import GDE_NO_TRADE_BAND
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION


OUTPUT = Path("output/recursive_cushion_budget")


@dataclass(frozen=True)
class CushionCandidate:
    name: str
    floor_drawdown: float = -0.18
    bull_multiplier: float = 7.0
    bear_multiplier: float = 4.0
    tier_size: float = 0.20

    def __post_init__(self) -> None:
        if not -1.0 < self.floor_drawdown < 0.0:
            raise ValueError("floor_drawdown must be between -1 and 0")
        if self.bull_multiplier <= 0.0:
            raise ValueError("bear_multiplier must be positive")
        if self.bull_multiplier < self.bear_multiplier:
            raise ValueError("bull_multiplier cannot be below bear_multiplier")
        if not 0.0 < self.tier_size <= 1.0:
            raise ValueError("tier_size must be in (0, 1]")


CENTRAL = CushionCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="floor_16", floor_drawdown=-0.16),
    replace(CENTRAL, name="floor_20", floor_drawdown=-0.20),
    replace(CENTRAL, name="bull_06", bull_multiplier=6.0),
    replace(CENTRAL, name="bull_08", bull_multiplier=8.0),
    replace(CENTRAL, name="bear_03", bear_multiplier=3.0),
    replace(CENTRAL, name="bear_05", bear_multiplier=5.0),
    replace(CENTRAL, name="tier_10", tier_size=0.10),
    replace(CENTRAL, name="tier_25", tier_size=0.25),
)


def cap_total_non_cash(target: pd.Series, cap: float) -> pd.Series:
    if not 0.0 <= cap <= 1.0:
        raise ValueError("non-cash cap must be in [0, 1]")
    adjusted = target.loc[ASSETS].astype(float).copy()
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    total = float(adjusted.loc[non_cash].sum())
    if total > cap + 1e-12:
        adjusted.loc[non_cash] *= cap / total
        adjusted["CASH"] = 1.0 - float(adjusted.loc[non_cash].sum())
    return adjusted


class RecursiveCushionPolicy:
    def __init__(
        self,
        candidate: CushionCandidate,
        trend_signals: pd.DataFrame,
        r11_weights: pd.DataFrame,
    ) -> None:
        self.candidate = candidate
        self.trend_signals = trend_signals
        self.r11_weights = r11_weights
        self.accepted_cap = 1.0

    def __call__(
        self,
        date: pd.Timestamp,
        blended_target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | int | bool | str]]:
        floor_equity = prior_peak * (1.0 + self.candidate.floor_drawdown)
        cushion = max(prior_equity - floor_equity, 0.0)
        bull = bool(
            date in self.trend_signals.index
            and self.trend_signals.loc[date, "both_above_sma"]
        )
        multiplier = (
            self.candidate.bull_multiplier
            if bull
            else self.candidate.bear_multiplier
        )
        requested_cap = float(
            np.clip(multiplier * cushion / prior_equity, 0.0, 1.0)
        )
        tier = self.candidate.tier_size
        accepted_cap = (
            1.0
            if requested_cap >= 1.0 - 1e-12
            else float(np.floor((requested_cap + 1e-12) / tier) * tier)
        )
        accepted_cap = float(np.clip(accepted_cap, 0.0, 1.0))
        previous_cap = self.accepted_cap
        self.accepted_cap = accepted_cap
        constrained = accepted_cap < 1.0 - 1e-12
        base_target = (
            self.r11_weights.loc[date, ASSETS]
            if constrained
            else blended_target
        )
        target = cap_total_non_cash(base_target, accepted_cap)
        state = (
            "floor"
            if accepted_cap <= 1e-12
            else "controlled"
            if constrained
            else "normal"
        )
        force_trade = abs(accepted_cap - previous_cap) > 1e-12
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
            "cushion": cushion,
            "risk_multiplier": multiplier,
            "requested_non_cash_cap": requested_cap,
            "accepted_non_cash_cap": accepted_cap,
            "dual_trend_positive": bull,
        }


def simulate_cushion_candidate(
    settings: dict[str, object],
    scenario: object,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: CushionCandidate,
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
    policy = RecursiveCushionPolicy(
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
    diagnostic_columns = [
        "state",
        "entry",
        "state_change",
        "stage_change",
        "recovery_run",
        "ramp_day",
        "growth_cap",
        "r38_incremental_enabled",
        "implemented_growth_weight",
        "implemented_cash_weight",
        "recursive_prior_drawdown",
        "floor_equity",
        "cushion",
        "risk_multiplier",
        "requested_non_cash_cap",
        "accepted_non_cash_cap",
        "dual_trend_positive",
        "policy_prior_equity",
        "policy_prior_peak",
        "policy_force_trade",
    ]
    diagnostics = trial[diagnostic_columns].copy()
    active = diagnostics["state"].ne("normal")
    if diagnostics.loc[active, "r38_incremental_enabled"].any():
        raise AssertionError("R38 remained enabled under a cushion cap")
    if (diagnostics["policy_prior_equity"] + 1e-12 < diagnostics["floor_equity"]).any():
        diagnostics["floor_breach_observed"] = (
            diagnostics["policy_prior_equity"] < diagnostics["floor_equity"]
        )
    else:
        diagnostics["floor_breach_observed"] = False
    return trial, diagnostics


def main() -> None:
    original_output = reference.OUTPUT
    original_central = reference.CENTRAL
    original_neighbors = reference.NEIGHBORS
    original_simulate = reference.simulate_candidate
    try:
        reference.OUTPUT = OUTPUT
        reference.CENTRAL = CENTRAL  # type: ignore[assignment]
        reference.NEIGHBORS = NEIGHBORS  # type: ignore[assignment]
        reference.simulate_candidate = simulate_cushion_candidate  # type: ignore[assignment]
        reference.main()
    finally:
        reference.OUTPUT = original_output
        reference.CENTRAL = original_central
        reference.NEIGHBORS = original_neighbors
        reference.simulate_candidate = original_simulate
    passed = recursive.write_strict_acceptance(OUTPUT)
    print((OUTPUT / "summary.json").read_text())
    print("\nStrict acceptance:")
    print(pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False))
    if not passed:
        print("\nCandidate remains research-only and failed strict qualification.")


if __name__ == "__main__":
    main()
