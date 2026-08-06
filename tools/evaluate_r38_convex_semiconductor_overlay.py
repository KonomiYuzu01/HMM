from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r23_high_volatility_tilt_gate as r23
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r37_bounded_semiconductor_tilt as r37


OUTPUT = Path("output/r38_convex_semiconductor_overlay")
CANDIDATE = "r35_with_10pct_convex_semiconductor_overlay"
BASE_MULTIPLIER = 1.070
ACTIVE_MULTIPLIER = 1.30
SHOCK_MULTIPLIER = 1.00
CASH_FLOOR = -0.20
OVERLAY_FRACTION = 0.10
BASE_NEIGHBORS = (1.065, 1.075)
ACTIVE_NEIGHBORS = (1.25, 1.35)
SHOCK_NEIGHBORS = (0.95, 1.05)
CASH_NEIGHBORS = (-0.18, -0.22)
OVERLAY_NEIGHBORS = (0.05, 0.15)
CUMULATIVE_TRIALS = 111
MAX_DRAWDOWN_TOLERANCE = 0.005

_ORIGINAL_APPLY_TILT = r37.r23.apply_gated_industry_momentum


def apply_convex_semiconductor_overlay(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    rebalance_days: int = r21.REBALANCE_DAYS,
    max_tilt: float = OVERLAY_FRACTION,
    high_quantile: float = r23.HIGH_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlay_fraction = max_tilt
    if not 0.0 <= overlay_fraction <= 1.0:
        raise ValueError("overlay_fraction must be between zero and one")
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    signal = r23.causal_gated_industry_momentum(
        closes,
        index,
        rebalance_days=rebalance_days,
        max_tilt=r21.MAX_TILT,
        high_quantile=high_quantile,
    )
    adjusted = weights.loc[index].copy()
    growth_total = adjusted["QQQ"] + adjusted["SEMIS"]
    valid_growth = growth_total.abs().gt(1e-12)
    base_share = adjusted["SEMIS"].div(
        growth_total.where(valid_growth)
    ).fillna(0.50)
    full_signal_share = signal["semis_growth_share"]
    implemented_share = (
        (1.0 - overlay_fraction) * base_share
        + overlay_fraction * full_signal_share
    )
    adjusted["SEMIS"] = growth_total * implemented_share
    adjusted["QQQ"] = growth_total - adjusted["SEMIS"]
    budget_error = (
        adjusted["QQQ"] + adjusted["SEMIS"] - growth_total
    ).abs()
    overlay_deviation = (implemented_share - base_share).abs()
    endpoint_low = pd.concat(
        [base_share, full_signal_share],
        axis=1,
    ).min(axis=1)
    endpoint_high = pd.concat(
        [base_share, full_signal_share],
        axis=1,
    ).max(axis=1)
    if float(budget_error.max()) > 1e-12:
        raise AssertionError("Convex overlay changed growth budget")
    if (
        implemented_share.lt(endpoint_low - 1e-12).any()
        or implemented_share.gt(endpoint_high + 1e-12).any()
    ):
        raise AssertionError("Convex overlay left its endpoints")
    if overlay_deviation.gt(overlay_fraction + 1e-12).any():
        raise AssertionError("Convex overlay exceeded its risk budget")

    updated_daily = execution_daily.loc[index].copy()
    update = signal["momentum_update"].astype(bool)
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = signal.copy()
    diagnostics["base_semis_growth_share"] = base_share
    diagnostics["full_signal_semis_growth_share"] = (
        full_signal_share
    )
    diagnostics["semis_growth_share"] = implemented_share
    diagnostics["overlay_fraction"] = overlay_fraction
    diagnostics["overlay_absolute_deviation"] = overlay_deviation
    diagnostics["overlay_endpoint_low"] = endpoint_low
    diagnostics["overlay_endpoint_high"] = endpoint_high
    diagnostics["growth_budget_before"] = growth_total
    diagnostics["growth_budget_after"] = (
        adjusted["QQQ"] + adjusted["SEMIS"]
    )
    diagnostics["growth_budget_error"] = budget_error
    return adjusted, updated_daily, diagnostics


def _configure(
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
) -> None:
    r37._configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    base_multiplier: float = BASE_MULTIPLIER,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    shock_multiplier: float = SHOCK_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    overlay_fraction: float = OVERLAY_FRACTION,
    pulse_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del pulse_enabled
    _configure(
        base_multiplier=base_multiplier,
        active_multiplier=active_multiplier,
        shock_multiplier=shock_multiplier,
    )
    r37.r23.apply_gated_industry_momentum = (
        apply_convex_semiconductor_overlay
    )
    try:
        return r37.simulate_candidate(
            settings,
            scenario,
            base_multiplier=base_multiplier,
            active_multiplier=active_multiplier,
            shock_multiplier=shock_multiplier,
            cash_floor=cash_floor,
            max_tilt=overlay_fraction,
        )
    finally:
        r37.r23.apply_gated_industry_momentum = _ORIGINAL_APPLY_TILT


def _rewrite_candidate_labels() -> None:
    for path in OUTPUT.glob("*.csv"):
        frame = pd.read_csv(path)
        changed = False
        if "candidate" in frame:
            frame["candidate"] = CANDIDATE
            changed = True
        if path.name == "summary.csv" and set(frame.columns) >= {
            "Unnamed: 0",
            "value",
        }:
            selected = frame["Unnamed: 0"].eq("candidate")
            frame.loc[selected, "value"] = CANDIDATE
            changed = True
        if changed:
            frame.to_csv(path, index=False)


def _neighborhood_rows() -> pd.DataFrame:
    settings = r21._build_samples()["normal_synthetic"]
    scenario = r30.COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    definitions = [
        ("base1065", 1.065, 1.30, 1.00, -0.20, 0.10, "base"),
        ("base1075", 1.075, 1.30, 1.00, -0.20, 0.10, "base"),
        ("active125", 1.070, 1.25, 1.00, -0.20, 0.10, "active"),
        ("active135", 1.070, 1.35, 1.00, -0.20, 0.10, "active"),
        ("shock095", 1.070, 1.30, 0.95, -0.20, 0.10, "shock"),
        ("shock105", 1.070, 1.30, 1.05, -0.20, 0.10, "shock"),
        ("cash18", 1.070, 1.30, 1.00, -0.18, 0.10, "cash"),
        ("cash22", 1.070, 1.30, 1.00, -0.22, 0.10, "cash"),
        ("overlay05", 1.070, 1.30, 1.00, -0.20, 0.05, "overlay"),
        ("overlay15", 1.070, 1.30, 1.00, -0.20, 0.15, "overlay"),
    ]
    rows: list[dict[str, object]] = []
    try:
        for (
            label,
            base,
            active,
            shock,
            cash,
            overlay,
            family,
        ) in definitions:
            trial, _ = simulate_candidate(
                settings,
                scenario,
                base_multiplier=base,
                active_multiplier=active,
                shock_multiplier=shock,
                cash_floor=cash,
                overlay_fraction=overlay,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            rows.append(
                {
                    "neighborhood": label,
                    "family": family,
                    "base_multiplier": base,
                    "active_multiplier": active,
                    "shock_multiplier": shock,
                    "cash_floor": cash,
                    "overlay_fraction": overlay,
                    **delta,
                    "relative_positive": bool(
                        delta["candidate_cagr"]
                        > delta["baseline_cagr"]
                    ),
                    "point_target_pass": bool(
                        delta["candidate_cagr"] >= 0.25
                        and delta["candidate_max_drawdown"]
                        >= (
                            delta["baseline_max_drawdown"]
                            - MAX_DRAWDOWN_TOLERANCE
                        )
                    ),
                }
            )
    finally:
        _configure()
    return pd.DataFrame(rows)


def _rewrite_acceptance(neighborhoods: pd.DataFrame) -> None:
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    neighborhood_pass = True
    for family in ("base", "active", "shock", "cash", "overlay"):
        selected = neighborhoods.loc[
            neighborhoods["family"].eq(family)
        ]
        neighborhood_pass = bool(
            neighborhood_pass
            and selected["relative_positive"].astype(bool).all()
            and selected["point_target_pass"].astype(bool).any()
        )
    acceptance.loc[
        acceptance["gate"].eq("parameter_neighborhood_pass"),
        "passed",
    ] = neighborhood_pass
    acceptance.loc[
        acceptance["gate"].eq("pulse_duration_pass"),
        "gate",
    ] = "one_session_shock_duration_pass"
    acceptance.loc[
        acceptance["gate"].eq(
            "disabled_increment_equals_r11_pass"
        ),
        "gate",
    ] = "state_multiplier_semantics_pass"
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "one_session_shock_duration_pass"
        ),
        "passed",
    ] = (
        r30.maximum_true_run(
            diagnostics["pulse_veto_active"]
        )
        <= 1
    )
    acceptance.loc[
        acceptance["gate"].eq(
            "state_multiplier_semantics_pass"
        ),
        "passed",
    ] = bool(
        diagnostics["state_multiplier_semantic_error"]
        .le(1e-12)
        .all()
    )
    extra = {
        "absolute_risk_cap_pass": bool(
            diagnostics["implemented_non_cash_weight"]
            .le(1.20 + 1e-12)
            .all()
            and diagnostics["pre_gde_cash_weight"]
            .ge(CASH_FLOOR - 1e-12)
            .all()
        ),
        "growth_budget_conservation_pass": bool(
            diagnostics["growth_budget_error"]
            .fillna(0.0)
            .le(1e-12)
            .all()
        ),
        "convex_overlay_budget_pass": bool(
            diagnostics["overlay_absolute_deviation"]
            .le(OVERLAY_FRACTION + 1e-12)
            .all()
            and diagnostics["semis_growth_share"]
            .ge(diagnostics["overlay_endpoint_low"] - 1e-12)
            .all()
            and diagnostics["semis_growth_share"]
            .le(diagnostics["overlay_endpoint_high"] + 1e-12)
            .all()
        ),
    }
    for gate, passed in extra.items():
        acceptance.loc[len(acceptance)] = {
            "gate": gate,
            "passed": passed,
        }
    non_production = acceptance["gate"].ne("production_pass")
    production = bool(
        acceptance.loc[non_production, "passed"].astype(bool).all()
    )
    acceptance.loc[
        acceptance["gate"].eq("production_pass"),
        "passed",
    ] = production
    acceptance.to_csv(path, index=False)
    summary_path = OUTPUT / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[
        summary["Unnamed: 0"].eq("production_pass"),
        "value",
    ] = str(production)
    summary.to_csv(summary_path, index=False)


def main() -> None:
    _configure()
    r30.OUTPUT = OUTPUT
    r30.CASH_FLOOR = CASH_FLOOR
    r30.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r30.simulate_candidate = simulate_candidate
    r30.main()
    neighborhoods = _neighborhood_rows()
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )
    _rewrite_candidate_labels()
    _rewrite_acceptance(neighborhoods)
    print("\nR38 parameter neighborhoods:")
    print(neighborhoods.round(6).to_string(index=False))
    print("\nR38 acceptance:")
    print(
        pd.read_csv(OUTPUT / "acceptance.csv").to_string(index=False)
    )
    print(f"\nR38 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
